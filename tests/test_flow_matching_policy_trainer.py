from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("zarr")
pytest.importorskip("tensorboard")

from lv_flow_matching.trainers import policy_trainer
from lv_flow_matching.trainers.policy_trainer import (
    build_scheduler,
    distributed_all_finite,
    get_checkpoint_state,
    preflight_resume_checkpoint,
    train_one_epoch,
)


class _Normalizer:
    def state_dict(self):
        return {"mode": "test"}


class _Dataset:
    normalizer = _Normalizer()


class _LossPolicy(torch.nn.Module):
    def __init__(self, *, loss_value: float = 1.0, metric_value: float = 1.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.loss_value = loss_value
        self.metric_value = metric_value

    def forward(self, _batch):
        loss = self.weight * 0.0 + self.loss_value
        return {"loss": loss, "metrics": {"aux": self.metric_value}}


class _FiniteLossNanGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value * 0.0

    @staticmethod
    def backward(ctx, grad_output):
        return torch.full_like(grad_output, float("nan"))


class _NanGradientPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, _batch):
        return {"loss": _FiniteLossNanGradient.apply(self.weight), "metrics": {}}


def _loader():
    return [{"action": torch.zeros((2, 1), dtype=torch.float32)}]


@pytest.mark.parametrize(
    ("loss_value", "metric_value", "detail"),
    [
        (float("nan"), 1.0, "loss=nan"),
        (1.0, float("inf"), "metrics="),
    ],
)
def test_train_one_epoch_rejects_nonfinite_outputs_before_backward(
    loss_value, metric_value, detail
):
    policy = _LossPolicy(loss_value=loss_value, metric_value=metric_value)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)

    with pytest.raises(FloatingPointError, match=detail):
        train_one_epoch(
            policy,
            _loader(),
            optimizer,
            torch.device("cpu"),
            grad_clip=1.0,
            is_main=False,
        )

    assert policy.weight.grad is None


@pytest.mark.parametrize("grad_clip", [None, 0.0, 1.0])
def test_train_one_epoch_rejects_nonfinite_gradients_without_relying_on_clip(grad_clip):
    policy = _NanGradientPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)

    with pytest.raises(FloatingPointError, match="non-finite training gradients"):
        train_one_epoch(
            policy,
            _loader(),
            optimizer,
            torch.device("cpu"),
            grad_clip=grad_clip,
            is_main=False,
        )


def test_distributed_finite_flag_all_reduces_to_rank_consistent_failure(monkeypatch):
    calls = []

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fake_all_reduce(flag, *, op):
        calls.append((flag.item(), op))
        flag.zero_()  # Simulate one other rank reporting non-finite.

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    assert distributed_all_finite(True, torch.device("cpu")) is False
    assert calls == [(1, torch.distributed.ReduceOp.MIN)]


def test_remote_nonfinite_output_raises_before_local_backward(monkeypatch):
    policy = _LossPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    monkeypatch.setattr(policy_trainer, "distributed_all_finite", lambda *_args: False)

    with pytest.raises(FloatingPointError, match="another distributed rank"):
        train_one_epoch(
            policy,
            _loader(),
            optimizer,
            torch.device("cpu"),
            grad_clip=None,
            is_main=False,
        )

    assert policy.weight.grad is None


def test_remote_nonfinite_gradient_raises_on_all_ranks(monkeypatch):
    policy = _LossPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    responses = iter([True, False])
    monkeypatch.setattr(
        policy_trainer,
        "distributed_all_finite",
        lambda *_args: next(responses),
    )

    with pytest.raises(FloatingPointError, match="another distributed rank"):
        train_one_epoch(
            policy,
            _loader(),
            optimizer,
            torch.device("cpu"),
            grad_clip=None,
            is_main=False,
        )


def test_checkpoint_includes_scheduler_and_scaler_state():
    policy = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.1)
    scheduler = build_scheduler(
        optimizer,
        {"scheduler": "cosine", "min_lr": 0.01},
        epochs=10,
        start_epoch=3,
    )
    assert scheduler is not None
    scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=128.0)

    state = get_checkpoint_state(
        policy,
        optimizer,
        _Dataset(),
        epoch=3,
        global_step=12,
        cfg={"train": {}},
        scheduler=scheduler,
        scaler=scaler,
    )

    assert state["scheduler_state_dict"] == scheduler.state_dict()
    assert state["scaler_state_dict"] == scaler.state_dict()
    assert state["scaler_state_dict"]["scale"] == 128.0
    assert state["normalizer_state_dict"] == {"mode": "test"}
    assert state["flow_matching_source"]["distribution"] == (
        "lv-robotics-flow-matching"
    )
    assert "git_commit" in state["flow_matching_source"]


def test_restored_cosine_scheduler_continues_the_same_lr_sequence():
    policy = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    scheduler = build_scheduler(
        optimizer,
        {"scheduler": "cosine", "min_lr": 0.01},
        epochs=5,
        start_epoch=1,
    )
    assert scheduler is not None
    for _ in range(2):
        optimizer.step()
        scheduler.step()

    resumed_policy = torch.nn.Linear(2, 1)
    resumed_optimizer = torch.optim.SGD(resumed_policy.parameters(), lr=0.1)
    resumed_optimizer.load_state_dict(optimizer.state_dict())
    resumed_scheduler = build_scheduler(
        resumed_optimizer,
        {"scheduler": "cosine", "min_lr": 0.01},
        epochs=5,
        start_epoch=3,
    )
    assert resumed_scheduler is not None
    resumed_scheduler.load_state_dict(scheduler.state_dict())

    optimizer.step()
    scheduler.step()
    resumed_optimizer.step()
    resumed_scheduler.step()

    assert resumed_scheduler.last_epoch == scheduler.last_epoch
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(
        optimizer.param_groups[0]["lr"]
    )


def test_scheduler_config_is_explicit_and_fail_closed():
    policy = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.1)

    assert build_scheduler(optimizer, {}, epochs=10, start_epoch=1) is None
    with pytest.raises(ValueError, match="unsupported optimizer scheduler"):
        build_scheduler(
            optimizer,
            {"scheduler": "mystery"},
            epochs=10,
            start_epoch=1,
        )


def _resume_config(checkpoint, **train_overrides):
    train = {
        "epochs": 10,
        "resume_path": str(checkpoint),
        "optimizer": {"scheduler": "none"},
        "use_amp": False,
    }
    train.update(train_overrides)
    return {"train": train}


def _minimal_resume_state(*, epoch=2):
    return {
        "epoch": epoch,
        "global_step": 12,
        "policy_state_dict": {},
        "normalizer_state_dict": {},
    }


def test_full_resume_is_default_and_requires_optimizer_state(tmp_path):
    checkpoint = tmp_path / "missing-optimizer.pt"
    torch.save(_minimal_resume_state(), checkpoint)

    with pytest.raises(ValueError, match="optimizer_state_dict"):
        preflight_resume_checkpoint(_resume_config(checkpoint))


def test_resume_requires_global_step_in_all_modes(tmp_path):
    checkpoint = tmp_path / "missing-global-step.pt"
    state = _minimal_resume_state()
    state.pop("global_step")
    torch.save(state, checkpoint)

    with pytest.raises(ValueError, match="global_step"):
        preflight_resume_checkpoint(
            _resume_config(checkpoint, resume_mode="weights-only")
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epoch", True),
        ("epoch", 2.7),
        ("epoch", "2"),
        ("global_step", False),
        ("global_step", 12.5),
        ("global_step", "12"),
    ],
)
def test_resume_rejects_noninteger_checkpoint_counters(tmp_path, field, value):
    checkpoint = tmp_path / f"invalid-{field}.pt"
    state = _minimal_resume_state()
    state[field] = value
    torch.save(state, checkpoint)

    with pytest.raises(TypeError, match="resume counters must be integers"):
        preflight_resume_checkpoint(
            _resume_config(checkpoint, resume_mode="weights-only")
        )


def test_resume_rejects_negative_global_step(tmp_path):
    checkpoint = tmp_path / "negative-global-step.pt"
    state = _minimal_resume_state()
    state["global_step"] = -1
    torch.save(state, checkpoint)

    with pytest.raises(ValueError, match="global_step must be non-negative"):
        preflight_resume_checkpoint(
            _resume_config(checkpoint, resume_mode="weights-only")
        )


@pytest.mark.parametrize("epochs", [True, 10.5, "10"])
def test_resume_rejects_noninteger_train_epochs(tmp_path, epochs):
    checkpoint = tmp_path / "invalid-epochs.pt"
    torch.save(_minimal_resume_state(), checkpoint)

    with pytest.raises(TypeError, match="resume counters must be integers"):
        preflight_resume_checkpoint(
            _resume_config(
                checkpoint,
                epochs=epochs,
                resume_mode="weights-only",
            )
        )


@pytest.mark.parametrize(
    ("train_overrides", "missing_key"),
    [
        ({"optimizer": {"scheduler": "cosine"}}, "scheduler_state_dict"),
        ({"use_amp": True}, "scaler_state_dict"),
    ],
)
def test_full_resume_requires_configured_scheduler_and_amp_state(
    tmp_path, train_overrides, missing_key
):
    checkpoint = tmp_path / f"missing-{missing_key}.pt"
    state = _minimal_resume_state()
    state["optimizer_state_dict"] = {"state": {}, "param_groups": []}
    torch.save(state, checkpoint)

    with pytest.raises(ValueError, match=missing_key):
        preflight_resume_checkpoint(_resume_config(checkpoint, **train_overrides))


def test_full_resume_accepts_complete_training_state(tmp_path):
    checkpoint = tmp_path / "full.pt"
    state = _minimal_resume_state()
    state.update(
        {
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "scheduler_state_dict": {"last_epoch": 2},
            "scaler_state_dict": {"scale": 128.0},
        }
    )
    torch.save(state, checkpoint)

    loaded, mode = preflight_resume_checkpoint(
        _resume_config(
            checkpoint,
            optimizer={"scheduler": "cosine"},
            use_amp=True,
        )
    )

    assert mode == "full"
    assert loaded is not None and loaded["epoch"] == 2


def test_full_resume_without_amp_or_scheduler_needs_no_optional_state(tmp_path):
    checkpoint = tmp_path / "cpu-full.pt"
    state = _minimal_resume_state()
    state["optimizer_state_dict"] = {"state": {}, "param_groups": []}
    torch.save(state, checkpoint)

    loaded, mode = preflight_resume_checkpoint(_resume_config(checkpoint))

    assert mode == "full"
    assert loaded is not None
    assert "scheduler_state_dict" not in loaded
    assert "scaler_state_dict" not in loaded


def test_weights_only_resume_must_be_explicit_and_accepts_minimal_state(tmp_path):
    checkpoint = tmp_path / "weights-only.pt"
    torch.save(_minimal_resume_state(), checkpoint)

    loaded, mode = preflight_resume_checkpoint(
        _resume_config(checkpoint, resume_mode="weights-only")
    )

    assert mode == "weights-only"
    assert loaded is not None and "optimizer_state_dict" not in loaded


def test_legacy_resume_optimizer_flag_is_rejected(tmp_path):
    checkpoint = tmp_path / "legacy.pt"
    torch.save(_minimal_resume_state(), checkpoint)

    with pytest.raises(ValueError, match="resume_optimizer is no longer supported"):
        preflight_resume_checkpoint(
            _resume_config(checkpoint, resume_optimizer=False)
        )


def test_resume_rejects_checkpoint_with_no_remaining_epochs(tmp_path):
    checkpoint = tmp_path / "finished.pt"
    state = _minimal_resume_state(epoch=10)
    state["optimizer_state_dict"] = {"state": {}, "param_groups": []}
    torch.save(state, checkpoint)

    with pytest.raises(ValueError, match="no remaining work"):
        preflight_resume_checkpoint(_resume_config(checkpoint))


def test_resume_preflight_runs_before_distributed_or_data_initialization(monkeypatch):
    initialized = False

    def fail_preflight(_cfg):
        raise RuntimeError("preflight stopped launch")

    def mark_initialized():
        nonlocal initialized
        initialized = True

    monkeypatch.setattr(policy_trainer, "preflight_resume_checkpoint", fail_preflight)
    monkeypatch.setattr(policy_trainer, "init_distributed", mark_initialized)

    with pytest.raises(RuntimeError, match="preflight stopped launch"):
        policy_trainer.main({})

    assert initialized is False
