"""Chunk planning must never emit a chunk above the Cloudflare cache ceiling."""

import importlib.util
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("_cdn_mirror", HERE / "cdn_mirror.py")
cdn_mirror = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cdn_mirror)

plan_chunks = cdn_mirror.plan_chunks

# Measured, inclusive: 536870912 caches and 536870913 never does.
CEILING = 512 * 1024 * 1024


def test_default_chunk_is_under_the_measured_ceiling():
    assert cdn_mirror.DEFAULT_CHUNK_BYTES <= CEILING


@pytest.mark.parametrize("size", [
    1, 100, CEILING - 1, CEILING, CEILING + 1,
    4548221488,            # qwen35-2b's single safetensors
    49889939456,           # gemma4-31b's 47 GB shard
])
def test_chunks_tile_the_object_exactly(size):
    chunk = cdn_mirror.DEFAULT_CHUNK_BYTES
    ranges = plan_chunks(size, chunk)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == size - 1
    for (_, prev_hi), (lo, _) in zip(ranges, ranges[1:]):
        assert lo == prev_hi + 1, "gap or overlap between chunks"
    assert sum(hi - lo + 1 for lo, hi in ranges) == size


@pytest.mark.parametrize("size", [CEILING + 1, 4548221488, 49889939456])
def test_no_chunk_exceeds_the_ceiling(size):
    # The failure this guards is silent: an over-ceiling object still returns
    # 200 with correct bytes, it just never populates the edge cache.
    for lo, hi in plan_chunks(size, cdn_mirror.DEFAULT_CHUNK_BYTES):
        assert hi - lo + 1 <= CEILING


def test_a_small_object_stays_a_single_unsuffixed_part():
    assert len(plan_chunks(1000, cdn_mirror.DEFAULT_CHUNK_BYTES)) == 1


# --- `--models all`: enumerate the estate instead of hand-listing it ---------

class _FakeB2:
    """Just enough of the B2 client for list_slugs: B2 returns each 'folder' as
    a synthetic file whose name ends in '/' when a delimiter is set."""

    def __init__(self, pages):
        self.pages = list(pages)
        self.bodies = []

    def call(self, endpoint, body):
        self.bodies.append(body)
        return self.pages[len(self.bodies) - 1]


def test_list_slugs_reads_folders_and_pages():
    fake = _FakeB2([
        {"files": [{"fileName": "base-models/qwen35-2b/"},
                   {"fileName": "base-models/qwen35-9b/"},
                   {"fileName": "base-models/LATEST"}],       # not a folder
         "nextFileName": "base-models/qwen36-27b/"},
        {"files": [{"fileName": "base-models/qwen36-27b/"},
                   {"fileName": "base-models/"}],             # the prefix itself
         "nextFileName": None},
    ])
    got = cdn_mirror.B2.list_slugs(fake, "bid", "base-models/")
    assert got == ["qwen35-2b", "qwen35-9b", "qwen36-27b"]
    # the delimiter is what keeps this off the per-shard listing path
    assert all(b["delimiter"] == "/" for b in fake.bodies)
    assert fake.bodies[1]["startFileName"] == "base-models/qwen36-27b/"
