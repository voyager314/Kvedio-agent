from kvedio_agent1.core.config.schema import RuntimeLimits
from kvedio_agent1.core.tasks import stage_budget


def test_a_long_stage_scales_with_the_recording():
    runtime = RuntimeLimits()

    assert stage_budget("transcribe", 3_600_000, runtime) == 5_400
    assert stage_budget("describe", 3_600_000, runtime) == 5_400


def test_a_short_stage_uses_the_floor():
    runtime = RuntimeLimits()

    assert stage_budget("probe", 3_600_000, runtime) == 300
    assert stage_budget("transcribe", 0, runtime) == 300


def test_the_answer_scales_with_the_request_timeout_not_the_recording():
    runtime = RuntimeLimits()

    assert stage_budget("answer", 7_200_000, runtime) == 720
    assert stage_budget("answer_preliminary", 0, runtime) == 360


def test_the_scale_multiplies_every_budget():
    runtime = RuntimeLimits(stage_budget_scale=2.0)

    assert stage_budget("transcribe", 3_600_000, runtime) == 10_800
    assert stage_budget("answer", 0, runtime) == 1_440


def test_a_ceiling_caps_every_budget():
    runtime = RuntimeLimits(stage_budget_ceiling_seconds=900)

    assert stage_budget("transcribe", 3_600_000, runtime) == 900
    assert stage_budget("answer", 0, runtime) == 720


def test_an_unknown_action_scales_like_the_recording():
    runtime = RuntimeLimits()

    assert stage_budget("something_new", 3_600_000, runtime) == 3_600
