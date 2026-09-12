"""Busy timing: the deadline model, per-op overrides, and the global
enable."""

from __future__ import annotations

import pytest

from flash_model import profiles
from flash_model.timing import BUSY_KEYS, LIMIT_KEYS, Timing


@pytest.fixture
def timing() -> Timing:
    profile = profiles.build()
    return Timing(profile["timing"], profile["limits"])


def test_every_contract_key_is_present(timing: Timing) -> None:
    assert BUSY_KEYS == (
        "tPP",
        "tSE",
        "tBE32",
        "tBE64",
        "tCE",
        "tW",
        "tRST",
        "tRES1",
        "tRES2",
    )
    for key in BUSY_KEYS:
        assert timing.busy_seconds(key) > 0.0


def test_limit_order_is_the_contract_order(timing: Timing) -> None:
    assert LIMIT_KEYS[0] == "t_sck_min_ps"
    assert LIMIT_KEYS[5] == "t_shsl_ps"
    assert LIMIT_KEYS[8:] == ("t_clqv_ps", "t_shqz_ps")
    assert len(timing.limits_ps()) == 10
    assert all(isinstance(v, int) for v in timing.limits_ps())


def test_busy_is_a_deadline_not_a_flag(timing: Timing) -> None:
    duration = timing.start_busy(1.0, "tPP")
    assert duration == pytest.approx(700e-6)
    assert timing.is_busy(1.0)
    assert timing.is_busy(1.0 + 699e-6)
    # The deadline is exclusive: at exactly the deadline the device is free.
    assert not timing.is_busy(1.0 + duration)
    assert not timing.is_busy(2.0)
    # Asking twice does not consume anything -- nothing is stateful but t.
    assert timing.is_busy(1.0 + 100e-6)


def test_a_later_command_rearms_from_its_own_now(timing: Timing) -> None:
    timing.start_busy(0.0, "tPP")
    timing.start_busy(5.0, "tSE")
    assert timing.is_busy(5.0 + 44e-3)
    assert not timing.is_busy(5.1)


def test_override_one_op(timing: Timing) -> None:
    timing.set_busy("tSE", 1e-9)
    assert timing.start_busy(0.0, "tSE") == pytest.approx(1e-9)
    assert timing.busy_seconds("tPP") == pytest.approx(700e-6)  # untouched


def test_override_rejects_unknown_names_and_negative_times(timing: Timing) -> None:
    with pytest.raises(KeyError):
        timing.set_busy("tPp", 1e-6)
    with pytest.raises(KeyError):
        timing.set_busy("tERASE", 1e-6)
    with pytest.raises(ValueError):
        timing.set_busy("tPP", -1e-6)


def test_disable_collapses_every_single_op(timing: Timing) -> None:
    timing.set_busy("tCE", 60.0)
    timing.set_enable(False)
    for key in BUSY_KEYS:
        assert timing.busy_seconds(key) == 0.0
        assert timing.start_busy(0.0, key) == 0.0
        assert not timing.is_busy(0.0)


def test_enable_restores_the_table_including_overrides(timing: Timing) -> None:
    timing.set_busy("tPP", 1e-3)
    timing.set_enable(False)
    assert timing.start_busy(0.0, "tPP") == 0.0
    timing.set_enable(True)
    assert timing.start_busy(0.0, "tPP") == pytest.approx(1e-3)


def test_a_command_with_no_busy_key_never_arms_the_deadline(timing: Timing) -> None:
    timing.start_busy(0.0, "tCE")
    assert timing.is_busy(1.0)
    assert timing.busy_seconds(None) == 0.0


def test_a_profile_missing_a_busy_key_is_rejected() -> None:
    profile = profiles.build()
    del profile["timing"]["tRES2"]
    with pytest.raises(ValueError, match="tRES2"):
        Timing(profile["timing"], profile["limits"])


def test_unspecified_limits_report_zero() -> None:
    profile = profiles.build()
    timing = Timing(profile["timing"], {})
    assert timing.limits_ps() == [0] * 10
