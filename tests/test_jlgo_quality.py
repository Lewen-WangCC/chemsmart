"""Quality policy controls acceptance without bypassing hard constraints."""

from dataclasses import replace

import numpy as np
import pytest

from chemsmart.jobs.iterate.jlgo import (
    JointLagrangeConfig,
    JointLagrangeOptimizer,
    Mol,
    StartBuildStats,
    _CandidateSolution,
    _StartRecord,
)


def candidate(index, **overrides):
    return replace(
        _CandidateSolution(
            x=np.array([float(index)]),
            objective=-float(index),
            raw_ok=True,
            equality_error=0.0,
            inequality_slack=1.0,
            message="test candidate",
            iterations=1,
            function_evaluations=1,
            detected_early=False,
        ),
        **overrides,
    )


def controlled_search(monkeypatch, mode, stages, passing=(), repairs=(), **kw):
    """Supply solved candidates; exercise real search/repair/acceptance flow."""
    optimizer = JointLagrangeOptimizer(
        Mol([[0, 0, 0], [0, 0, 4]], ["C", "H"]),
        [Mol([[0, 0, 0]], ["C"])],
        [0],
        [0],
        config=JointLagrangeConfig(
            quality_mode=mode, use_convergence_detection=False, **kw
        ),
    )
    repair_queue = iter(repairs)
    calls = []
    solutions = {id(s): s for stage in stages for s in stage}

    def build(level):
        return [
            _StartRecord(np.array([id(s)]), 0, 1, np.empty(0))
            for s in stages[level]
        ], StartBuildStats()

    def solve(record, use_detection, enforce_quality=False):
        calls.append(enforce_quality)
        if enforce_quality:
            return next(repair_queue, candidate(0, raw_ok=False))
        return solutions[int(record.x0[0])]

    monkeypatch.setattr(
        optimizer, "_sampling_levels", lambda: range(len(stages))
    )
    monkeypatch.setattr(optimizer, "_build_starts", build)
    monkeypatch.setattr(optimizer, "_solve_one", solve)
    monkeypatch.setattr(
        optimizer,
        "_make_result_molecule",
        lambda x: (Mol([[x[0], 0, 0]], ["C"]), {0: (0, 1)}),
    )
    monkeypatch.setattr(
        optimizer,
        "_quality_metrics",
        lambda mol, ranges: {
            "min_nonbond": 2.1 if mol.positions[0, 0] in passing else 1.8,
            "n_close_20": 0 if mol.positions[0, 0] in passing else 1,
            "vdw075_overlap_sum": 0.1,
        },
    )
    monkeypatch.setattr(
        optimizer, "_quality_violation", lambda s: 100 - s.x[0]
    )
    return optimizer, calls


@pytest.mark.parametrize("mode,success", [("strict", False), ("warn", True)])
def test_quality_reject_search_exhausted(monkeypatch, mode, success):
    optimizer, calls = controlled_search(
        monkeypatch,
        mode,
        [[candidate(1)], [candidate(2)]],
        repairs=[candidate(3), candidate(4)],
    )
    result = optimizer.optimize()
    assert result.success is success
    assert not result.quality_ok
    assert result.stats.build.adaptive_levels == 2
    assert calls == [False, True, False, True]
    if success:
        # The best candidate came from repair, not an ordinary start.
        assert result.final.positions[0, 0] == 4
        assert result.stats.accepted_successes == 1
        assert "quality warning" in result.message
    else:
        assert result.final is None


def test_warn_prefers_later_passing_candidate(monkeypatch):
    optimizer, calls = controlled_search(
        monkeypatch, "warn", [[candidate(90)], [candidate(2)]], passing=(2,)
    )
    result = optimizer.optimize()
    assert result.success and result.quality_ok
    assert result.final.positions[0, 0] == 2
    assert calls == [False, True, False]


def test_off_does_not_repair_or_reject_quality(monkeypatch):
    optimizer, calls = controlled_search(
        monkeypatch, "off", [[candidate(1)], [candidate(2)]]
    )
    result = optimizer.optimize()
    assert result.success and not result.quality_ok
    assert result.final.positions[0, 0] == 1
    assert calls == [False]
    assert not result.stats.quality_repair_triggered


@pytest.mark.parametrize("mode", ["warn", "off"])
@pytest.mark.parametrize(
    "invalid",
    [
        {"equality_error": 0.1},
        {"inequality_slack": -0.1},
        {"equality_error": float("nan")},
        {"x": np.array([float("nan")])},
        {"objective": float("inf")},
        {"raw_ok": False},
    ],
)
def test_relaxed_modes_still_reject_invalid_candidates(
    monkeypatch, mode, invalid
):
    optimizer, calls = controlled_search(
        monkeypatch, mode, [[candidate(1, **invalid)]]
    )
    result = optimizer.optimize()
    assert not result.success
    assert result.final is None
    assert calls == [False]


@pytest.mark.parametrize("mode", ["strict", "warn", "off"])
def test_no_starts_remains_failure(monkeypatch, mode):
    optimizer, calls = controlled_search(monkeypatch, mode, [[]])
    assert not optimizer.optimize().success
    assert calls == []


def test_strict_gate_is_independent_of_early_stop(monkeypatch):
    optimizer, _ = controlled_search(
        monkeypatch,
        "strict",
        [[candidate(1)]],
        use_early_stop=False,
        use_quality_early_stop=False,
    )
    assert not optimizer.optimize().success


def test_off_respects_disabled_early_stop(monkeypatch):
    optimizer, calls = controlled_search(
        monkeypatch,
        "off",
        [[candidate(1), candidate(2)]],
        passing=(1,),
        use_early_stop=False,
    )
    result = optimizer.optimize()
    assert result.final.positions[0, 0] == 2
    assert calls == [False, False]


def test_invalid_core_quality_mode():
    with pytest.raises(ValueError, match="quality_mode"):
        JointLagrangeConfig(quality_mode="invalid").validate()
