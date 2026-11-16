import wave

import pytest

from kvedio_agent1.core.asr.chunking import MIN_TAIL_MS, plan_chunks, slice_audio

RATE = 16_000


def test_short_audio_is_one_chunk(make_wav):
    audio = make_wav([(2_000, 8_000)])
    chunks = plan_chunks(audio, chunk_ms=5_000)
    assert [chunk.start_ms for chunk in chunks] == [0]
    assert chunks[0].end_ms == 2_000


def test_a_boundary_is_moved_into_the_silence_near_the_target(make_wav):
    audio = make_wav(
        [(1_000, 0), (3_000, 8_000), (1_000, 0), (4_000, 8_000), (1_000, 0)]
    )
    chunks = plan_chunks(audio, chunk_ms=4_000, search_ms=2_000)
    assert len(chunks) > 1
    assert 4_000 <= chunks[0].end_ms <= 5_000


def test_chunks_cover_the_whole_file_without_gaps(make_wav):
    audio = make_wav([(1_000, 8_000), (200, 0)] * 4)
    chunks = plan_chunks(audio, chunk_ms=1_000, search_ms=300)
    assert chunks[0].start_ms == 0
    assert chunks[-1].end_ms == 4_800
    for previous, following in zip(chunks, chunks[1:], strict=False):
        assert previous.end_ms == following.start_ms
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_a_sliver_at_the_end_is_merged_into_its_predecessor(make_wav):
    audio = make_wav([(4_000, 8_000), (400, 0)])
    chunks = plan_chunks(audio, chunk_ms=3_000, search_ms=200)
    assert chunks[-1].end_ms - chunks[-1].start_ms >= MIN_TAIL_MS


def test_slicing_copies_only_the_requested_span(make_wav, tmp_path):
    audio = make_wav([(5_000, 8_000)])
    target = tmp_path / "slice.wav"
    chunks = plan_chunks(audio, chunk_ms=2_000, search_ms=100)
    slice_audio(audio, chunks[0], target)

    with wave.open(str(target), "rb") as sliced:
        assert sliced.getframerate() == RATE
        assert sliced.getnchannels() == 1
        duration_ms = sliced.getnframes() * 1000 // RATE
    assert duration_ms == pytest.approx(chunks[0].end_ms - chunks[0].start_ms, abs=2)
