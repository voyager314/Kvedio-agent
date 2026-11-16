
import av
import pytest

from kvedio_agent1.core.media.scenes import detect_scenes, grab_frame


def test_a_cut_is_found_where_the_picture_changes(two_shot_video):
    scenes = detect_scenes(two_shot_video, sample_ms=250, min_ms=400, threshold=0.12)
    assert [(scene.start_ms, scene.end_ms) for scene in scenes] == [(0, 1000), (1000, 2000)]


def test_a_single_shot_stays_one_scene(sample_video):
    scenes = detect_scenes(sample_video, sample_ms=250, min_ms=400)
    assert len(scenes) == 1
    assert scenes[0].start_ms == 0
    assert scenes[0].end_ms >= 900


def test_two_cuts_too_close_together_keep_one_shot(two_shot_video):
    scenes = detect_scenes(two_shot_video, sample_ms=250, min_ms=1_500, threshold=0.12)
    assert len(scenes) == 1


def test_a_frame_is_written_as_a_jpeg(two_shot_video, tmp_path):
    target = tmp_path / "frame.jpg"
    assert grab_frame(two_shot_video, 1_500, target) is True
    assert target.exists()
    assert target.read_bytes()[:2] == b"\xff\xd8"  # JPEG start of image


def test_a_file_without_a_picture_track_is_reported(tmp_path):
    audio_only = tmp_path / "sound.mp4"
    container = av.open(str(audio_only), "w")
    stream = container.add_stream("aac", rate=48_000)
    stream.layout = "mono"
    for index in range(10):
        frame = av.AudioFrame(format="s16", layout="mono", samples=1024)
        frame.sample_rate = 48_000
        frame.pts = index * 1024
        for plane in frame.planes:
            plane.update(bytes(plane.buffer_size))
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()

    with pytest.raises(ValueError, match="视频轨"):
        detect_scenes(audio_only)

def test_scene_detection_reports_progress_and_checks_between_samples(two_shot_video):
    progress: list[float] = []
    checks = 0

    def check() -> None:
        nonlocal checks
        checks += 1

    detect_scenes(
        two_shot_video,
        sample_ms=250,
        min_ms=400,
        progress=progress.append,
        check=check,
    )

    assert progress[-1] == pytest.approx(1.0)
    assert checks > 1


def test_scene_detection_stops_when_the_check_raises(two_shot_video):
    class Stop(Exception):
        pass

    def stop() -> None:
        raise Stop()

    with pytest.raises(Stop):
        detect_scenes(two_shot_video, sample_ms=250, min_ms=400, check=stop)


def test_a_still_camera_is_sampled_along_its_length(sample_video):
    scenes = detect_scenes(sample_video, sample_ms=250, min_ms=400, max_ms=300)
    assert len(scenes) == 4
    assert scenes[0].start_ms == 0
    assert scenes[-1].end_ms >= 900
