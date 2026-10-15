from types import SimpleNamespace

import numpy as np
import pytest
import torch
from timelyhelp.simulation import FEATURE_NAMES

from timelyhelp.learning import (
    Candidate,
    ConsequenceEnsemble,
    LearnedScheduler,
    LatencyDistribution,
    ModelConfig,
    PairedDataset,
    SchedulerConfig,
    cluster_bootstrap_indices,
    predict_useful_window,
    shuffled_pair_records,
    useful_window,
    validate_splits,
)


def records(split="train", layout="train_layout", snapshots=3, replications=2):
    result = []
    for snapshot in range(snapshots):
        for source in ("infrastructure", "supervisor"):
            for bridge in ("cautious", "hold"):
                for delay in (0.0, 1.0):
                    for replication in range(replications):
                        # Histories are identical across candidate branches and
                        # repetitions; source usefulness includes all signs.
                        h = np.zeros((8, len(FEATURE_NAMES)), dtype=np.float32)
                        h[:, 0] = 0.01 * snapshot
                        h[:, 1] = snapshot / max(1, snapshots)
                        sign = (-1, 0, 1)[snapshot % 3]
                        a, w = 3 + 0.1 * replication, 2 + 0.1 * replication
                        q = w - sign * (0.5 if source == "infrastructure" else 0.25)
                        outcomes = {}
                        for name, cost in zip(("A", "W", "Q"), (a, w, q)):
                            outcomes[name] = dict(
                                cost=cost,
                                collision=False,
                                completion=True,
                                completion_time=5 + delay + replication,
                            )
                        result.append(
                            dict(
                                history=h,
                                source=source,
                                bridge=bridge,
                                delay=delay,
                                layout=layout,
                                rollout_id=f"rollout-{snapshot // 2}",
                                snapshot_id=f"snapshot-{snapshot}",
                                split=split,
                                replication=replication,
                                outcomes=outcomes,
                                info=w - q,
                                motion=a - w,
                            )
                        )
    return result


def test_signed_monte_carlo_labels_identity_and_cache_integrity(tmp_path):
    source = records()
    dataset = PairedDataset.from_records(source)
    signs = dataset.targets[
        [i for i, c in enumerate(dataset.candidates) if c.source == "infrastructure"], 0
    ]
    assert np.any(signs < 0) and np.any(signs == 0) and np.any(signs > 0)
    assert all(m["replications"] == 2 for m in dataset.metadata)
    for i, (candidate, metadata) in enumerate(zip(dataset.candidates, dataset.metadata)):
        a, w, q = metadata["mean_costs"]
        assert dataset.targets[i, 0] + dataset.targets[i, 1] == pytest.approx(a - q)
        if candidate.source == "none":
            assert dataset.targets[i, 0] == 0
            assert dataset.targets[i, 4] == dataset.targets[i, 3]
    path = tmp_path / "features.npz"
    dataset.save(path)
    loaded = PairedDataset.load(path)
    np.testing.assert_array_equal(dataset.histories, loaded.histories)
    np.testing.assert_array_equal(dataset.targets, loaded.targets)
    assert loaded.provenance["filtering"] == "none"
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    arrays["targets"] = arrays["targets"].copy()
    arrays["targets"][0, 0] += 1
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="integrity"):
        PairedDataset.load(path)


def test_layout_rollout_and_sibling_split_leakage_rejected():
    train, test = records(snapshots=1), records("test", "train_layout", snapshots=1)
    with pytest.raises(ValueError, match="layout"):
        validate_splits(train + test)
    wrong = records(snapshots=1)
    wrong[-1]["split"] = "calibration"
    with pytest.raises(ValueError, match="leakage"):
        PairedDataset.from_records(wrong)
    altered = records(snapshots=1)
    altered[1]["history"] = altered[1]["history"].copy()
    altered[1]["history"][0, 0] = 100
    with pytest.raises(ValueError, match="conditioning history"):
        PairedDataset.from_records(altered)


def test_shuffled_control_preserves_interaction_budget_and_Q_marginals():
    source = records(snapshots=6)
    shuffled = shuffled_pair_records(source, 17)
    assert len(shuffled) == len(source)
    assert sorted(r["outcomes"]["Q"]["cost"] for r in shuffled) == sorted(
        r["outcomes"]["Q"]["cost"] for r in source
    )
    assert any(a["info"] != b["info"] for a, b in zip(source, shuffled))
    ds = PairedDataset.from_records(source)
    changed = ds.shuffled_pairs(17)
    assert changed.provenance["interaction_count"] == ds.provenance["interaction_count"]
    assert sorted(m["mean_costs"][2] for m in changed.metadata) == sorted(
        m["mean_costs"][2] for m in ds.metadata
    )


def test_bootstrap_samples_complete_rollouts():
    ds = PairedDataset.from_records(records(snapshots=4))
    indices = cluster_bootstrap_indices(ds, np.random.default_rng(4))
    counts = np.bincount(indices, minlength=len(ds))
    for rollout in {m["rollout_id"] for m in ds.metadata}:
        cluster = [i for i, m in enumerate(ds.metadata) if m["rollout_id"] == rollout]
        assert len(set(counts[cluster])) == 1


def test_time_heads_retain_collision_timeout_and_snapshot_remaining_budget():
    rows = records(snapshots=1, replications=1)
    for row in rows:
        row["history"][:, 0] = 0.2
        row["observation"] = {"time": 2.0, "remaining_time": 8.0}
        row["outcomes"]["A"].update(collision=True, completion=False, completion_time=3.0, time=3.0)
        row["outcomes"]["W"].update(
            collision=False, completion=False, completion_time=10.0, time=10.0
        )
        row["outcomes"]["Q"].update(collision=False, completion=True, completion_time=6.0, time=6.0)
    dataset = PairedDataset.from_records(rows, include_no_query=False)
    np.testing.assert_array_equal(
        dataset.targets[:, 8:], np.tile([10.0, 10.0, 6.0], (len(dataset), 1))
    )
    assert all(row["outcomes"]["A"]["time"] == 3.0 for row in rows)
    invalid = [{k: v for k, v in row.items() if k != "observation"} for row in rows]
    with pytest.raises(ValueError, match="Mission horizon"):
        PairedDataset.from_records(invalid)
    model = ConsequenceEnsemble(
        ModelConfig(epochs=1, ensemble_size=1, time_scale=10.0, hidden_dim=8)
    )
    model.fit(dataset)
    for fraction in (0.0, 0.2, 0.9, 1.0):
        current = dataset.histories[0].copy()
        current[-1, 0] = fraction
        prediction = model.predict(current, [Candidate("infrastructure", "hold", 0.5)])
        assert (prediction["restricted_completion_time"] >= fraction * 10.0 - 1e-6).all()
        assert (prediction["restricted_completion_time"] <= 10.0 + 1e-6).all()
    current[-1, 0] = 1.0
    np.testing.assert_allclose(
        model.predict(current, [Candidate("infrastructure", "hold", 0.5)])["time"], 10.0
    )


def test_learning_training_calibration_save_reload_and_non_oracle_inputs(tmp_path):
    torch.set_num_threads(1)
    train = PairedDataset.from_records(records())
    development = PairedDataset.from_records(records("development", "dev_layout"))
    calibration = PairedDataset.from_records(records("calibration", "cal_layout"))
    model = ConsequenceEnsemble(
        ModelConfig(
            epochs=2, ensemble_size=2, hidden_dim=8, batch_size=16, seed=41, time_scale=10.0
        )
    )
    report = model.fit(train, development)
    assert all(np.isfinite(row["train_loss"]) for trace in report["members"] for row in trace)
    assert any(row["ranking"] > 0 for trace in report["members"] for row in trace)
    fitted = model.calibrate(calibration)
    assert fitted["episodes"] == 2
    assert "no safety" in fitted["warning"]
    prediction = model.predict(train.histories[0], train.candidates[:4])
    for name in ("info", "motion", "collision", "completion", "time"):
        assert np.isfinite(prediction[name]).all()
        assert (prediction[f"{name}_uncertainty"] >= prediction[f"{name}_epistemic"]).all()
    assert ((prediction["collision"] >= 0) & (prediction["collision"] <= 1)).all()
    assert (prediction["time"] > 0).all()
    with pytest.raises(ValueError, match="independent"):
        model.calibrate(train)
    model.save(tmp_path / "gate.pt")
    loaded = ConsequenceEnsemble.load(tmp_path / "gate.pt")
    for name, values in prediction.items():
        np.testing.assert_array_equal(
            values, loaded.predict(train.histories[0], train.candidates[:4])[name]
        )
    independent = ConsequenceEnsemble(model.config)
    independent.fit(train, development)
    independent.calibrate(calibration)
    np.testing.assert_array_equal(
        model.predict(train.histories[0], train.candidates[:4])["info"],
        independent.predict(train.histories[0], train.candidates[:4])["info"],
    )


def test_ablation_masks_latency_reliability_and_W_training():
    torch.set_num_threads(1)
    train = PairedDataset.from_records(records())
    model = ConsequenceEnsemble(
        ModelConfig(
            epochs=1,
            ensemble_size=1,
            hidden_dim=8,
            omit_latency=True,
            omit_reliability_metadata=True,
            without_W=True,
        )
    )
    model.fit(train)
    assert model.fit_report["examples"] == sum(c.source != "none" for c in train.candidates)
    h = train.histories[0].copy()
    changed = h.copy()
    changed[:, [22, 23, 30, 31]] = 42
    changed[:, [18, 19, 20, 21, 24, 25]] = 100
    first = model.predict(h, [Candidate("infrastructure", "hold", 0.0)])
    second = model.predict(changed, [Candidate("infrastructure", "hold", 5.0)])
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


class AnalyticModel:
    config = SimpleNamespace(without_W=False)

    def predict(self, history, candidates):
        n = len(candidates)
        info = np.asarray([2 - c.delay if c.source != "none" else 0 for c in candidates])
        collision = np.asarray(
            [[0.5, 0.05, 0.05 if c.bridge == "hold" else 0.25] for c in candidates]
        )
        result = dict(
            info=info,
            motion=np.ones(n) * 0.3,
            collision=collision,
            completion=np.ones((n, 3)) * 0.9,
            time=np.ones((n, 3)) * 8,
        )
        for name, values in list(result.items()):
            result[name + "_uncertainty"] = np.zeros_like(values)
        return result


def test_empirical_latency_integral_risk_and_noresponse_fallback():
    metadata = {
        "infrastructure": {"latency_samples": [0.0, 0.0, 1.0], "response_fraction": 0.5},
        "supervisor": {"available": False},
    }
    distribution = LatencyDistribution.from_metadata(metadata, "infrastructure")
    np.testing.assert_allclose(distribution.probabilities, [2 / 3, 1 / 3])
    assert distribution.response_probability == 0.5
    first = LatencyDistribution.from_metadata(
        metadata | {"true_latency_draw": 100}, "infrastructure"
    )
    np.testing.assert_array_equal(distribution.delays, first.delays)
    scheduler = LearnedScheduler(AnalyticModel(), SchedulerConfig(risk_limit=0.1))
    decision = scheduler.select(np.zeros((8, 32)), metadata)
    assert decision.source == "infrastructure" and decision.bridge == "hold" and decision.feasible
    # Conditional reply value E[2-delay]=5/3, multiplied by response prob .5;
    # the no-response branch contributes motion .3, costs .05 always apply.
    assert decision.net_value == pytest.approx(0.5 * 5 / 3 + 0.3 - 0.05)
    assert decision.collision_upper == pytest.approx(0.05)
    exhausted = scheduler.select(np.zeros((8, 32)), metadata | {"remaining_query_budget": 0})
    assert exhausted.source == "none"
    nohold = LearnedScheduler(AnalyticModel(), SchedulerConfig(risk_limit=0.1, no_hold=True))
    assert not nohold.select(np.zeros((8, 32)), metadata).feasible
    masked_model = AnalyticModel()
    masked_model.config = SimpleNamespace(without_W=False, omit_reliability_metadata=True)
    masked = LearnedScheduler(masked_model, SchedulerConfig(risk_limit=0.1)).select(
        np.zeros((8, 32)), metadata
    )
    assert masked.net_value == pytest.approx(5 / 3 + 0.3 - 0.05)


def test_arrival_admission_only_uses_observable_headers_and_current_evidence():
    scheduler = LearnedScheduler(AnalyticModel(), SchedulerConfig(risk_limit=0.1))
    history = np.zeros((8, 32))
    packet = dict(source="infrastructure", bridge="hold", capture_time=1.0, arrival_time=2.0)
    accepted = scheduler.reassess_packet(history, packet, 2.0, {})
    assert accepted["accept"]
    corrupted = scheduler.reassess_packet(history, packet | {"corrupted": True}, 2.0, {})
    assert accepted == corrupted
    assert not scheduler.reassess_packet(history, packet, 5.0, {})["accept"]
    assert not scheduler.reassess_packet(history, packet, 2.0, {}, latest_capture_time=1.0)[
        "accept"
    ]
    assert not scheduler.reassess_packet(history, packet | {"cancelled": True}, 2.0, {})["accept"]
    with pytest.raises(ValueError, match="closed-loop"):
        scheduler.configure_thresholds({"split": "calibration", "evaluation": "paired_labels"})
    scheduler.configure_thresholds(
        {
            "split": "calibration",
            "evaluation": "closed_loop",
            "episodes": 2,
            "thresholds": {"risk_limit": 0.2},
            "collision_rate": 0.0,
        }
    )
    assert scheduler.config.risk_limit == 0.2


def test_nonmonotone_disconnected_right_censored_windows():
    disconnected = useful_window([0, 1, 2, 3, 4], [1, -1, -1, 1, -1], [0.01] * 5)
    np.testing.assert_allclose(disconnected.intervals, [(0, 0.5), (2.5, 3.5)])
    assert not disconnected.right_censored and disconnected.connected_deadline is None
    held = useful_window([0, 1, 2], [1, 1, 1], [0.01] * 3)
    assert held.right_censored and held.connected_deadline is None
    deadline = useful_window([0, 1, 2], [1, 1, -1], [0.01] * 3)
    assert deadline.connected_deadline == pytest.approx(1.5)
    # Intermediate useful interval exists even though endpoints fail different
    # conditions; implementations checking only probe signs would miss it.
    crossing = useful_window([0, 1], [1, -0.2], [0.2, 0.0], risk_limit=0.1)
    np.testing.assert_allclose(crossing.intervals, [(0.5, 5 / 6)])
    zero_hole = useful_window([0, 1, 2], [1, 0, 1], [0.01] * 3)
    assert len(zero_hole.intervals) == 2
    assert useful_window([0, 1], [-1, -1], [0.01, 0.01]).intervals == ()
    predicted = predict_useful_window(
        AnalyticModel(), np.zeros((8, 32)), "infrastructure", "hold", [0, 1, 3]
    )
    assert predicted.connected_deadline == pytest.approx(2)


def test_useful_window_retains_singleton_at_inclusive_risk_boundary():
    isolated = useful_window([0.0, 1.0, 2.0], [1.0, 1.0, 1.0], [0.2, 0.1, 0.2], risk_limit=0.1)
    assert isolated.positive == (False, True, False)
    assert isolated.intervals == ((1.0, 1.0),)
    assert isolated.connected_deadline == 1.0 and not isolated.right_censored
    # The information inequality remains strict even at the same risk limit.
    empty = useful_window([0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [0.2, 0.1, 0.2], risk_limit=0.1)
    assert empty.intervals == ()
    boundary = useful_window([0.0, 1.0], [1.0, 1.0], [0.2, 0.1], risk_limit=0.1)
    assert boundary.intervals == ((1.0, 1.0),) and boundary.right_censored
    assert boundary.connected_deadline is None
