from cantera_model.cli.reduce_validate import _evaluate_metric_replay_health_trust


def test_replay_health_trust_requires_guardrail_trigger() -> None:
    detail = _evaluate_metric_replay_health_trust(
        metric_clip_ratio=1.0,
        guardrail_trigger_ratio=0.001,
        max_metric_clip_ratio=0.80,
        min_guardrail_trigger_ratio=0.02,
    )
    assert detail["clip_exceeded"] is True
    assert detail["trigger_reliable"] is False
    assert detail["replay_health_trust_invalid"] is False


def test_replay_health_trust_invalid_when_both_conditions_met() -> None:
    detail = _evaluate_metric_replay_health_trust(
        metric_clip_ratio=1.0,
        guardrail_trigger_ratio=0.10,
        max_metric_clip_ratio=0.80,
        min_guardrail_trigger_ratio=0.02,
    )
    assert detail["clip_exceeded"] is True
    assert detail["trigger_reliable"] is True
    assert detail["replay_health_trust_invalid"] is True
