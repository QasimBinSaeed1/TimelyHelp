"""Audited, rollout-clustered Monte Carlo targets and deterministic feature caches."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from ..types import Candidate

BRANCHES = ("A", "W", "Q")
SOURCES = ("none", "infrastructure", "supervisor")
BRIDGES = ("cautious", "hold")
TARGET_NAMES = (
    "info",
    "motion",
    *(f"collision_{b}" for b in BRANCHES),
    *(f"completion_{b}" for b in BRANCHES),
    *(f"time_{b}" for b in BRANCHES),
)


def candidate_features(
    candidates: Iterable[Candidate], time_scale: float = 30.0, omit_latency: bool = False
) -> np.ndarray:
    rows = []
    for raw in candidates:
        c = Candidate.coerce(raw)
        rows.append(
            [
                *(float(c.source == s) for s in SOURCES),
                *(float(c.bridge == b) for b in BRIDGES),
                0.0 if omit_latency else c.delay / time_scale,
            ]
        )
    return np.asarray(rows, dtype=np.float32).reshape(-1, 6)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def content_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def validate_splits(records: Iterable[Mapping[str, Any]]) -> None:
    """Prevent layout, complete-rollout and sibling snapshot leakage."""
    assignments: dict[tuple[str, str], str] = {}
    for r in records:
        split = str(r["split"])
        if split not in {"train", "development", "calibration", "test", "transfer"}:
            raise ValueError(f"Unknown split: {split}")
        for kind in ("layout", "rollout_id", "snapshot_id"):
            # Rollout/snapshot IDs are only required to be unique within a layout.
            identifier = str(r[kind]) if kind == "layout" else f"{r['layout']}::{r[kind]}"
            key = (kind, identifier)
            if key in assignments and assignments[key] != split:
                raise ValueError(f"Split leakage in {kind}: {identifier}")
            assignments[key] = split


def restricted_completion_time(
    record: Mapping[str, Any], branch: str, mission_horizon: float | None = None
) -> float:
    """min(T_complete, mission budget); every unsuccessful branch uses budget.

    Early collision termination is elapsed time, not efficient completion. Budget
    information is declared mission metadata rather than privileged world state.
    """
    outcome = record["outcomes"][branch]
    observation = record.get("observation", {})
    horizon = record.get("mission_horizon", mission_horizon)
    if horizon is None and "time" in observation and "remaining_time" in observation:
        horizon = float(observation["time"]) + float(observation["remaining_time"])
    if outcome["completion"]:
        completion = float(outcome["completion_time"])
        if not np.isfinite(completion) or completion < 0:
            raise ValueError("Invalid successful completion time")
        if horizon is not None and completion > float(horizon) + 1e-6:
            raise ValueError("Successful completion exceeds mission budget")
        return completion
    if horizon is None:
        raise ValueError("Mission horizon is required for unsuccessful completion targets")
    horizon = float(horizon)
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("Invalid mission horizon")
    return horizon


@dataclass
class PairedDataset:
    histories: np.ndarray
    candidates: list[Candidate]
    targets: np.ndarray
    metadata: list[dict[str, Any]]
    monte_carlo_std: np.ndarray
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.histories = np.asarray(self.histories, dtype=np.float32)
        self.targets = np.asarray(self.targets, dtype=np.float32)
        self.monte_carlo_std = np.asarray(self.monte_carlo_std, dtype=np.float32)
        n = len(self.candidates)
        if self.histories.ndim != 3 or self.histories.shape[0] != n:
            raise ValueError("Histories must have shape [N, history_length, feature_dim]")
        if self.targets.shape != (n, 11) or self.monte_carlo_std.shape != (n, 11):
            raise ValueError("Expected 11 targets per candidate")
        if (
            len(self.metadata) != n
            or not np.isfinite(self.histories).all()
            or not np.isfinite(self.targets).all()
        ):
            raise ValueError("Invalid feature/target data")
        validate_splits(self.metadata)

    def __len__(self) -> int:
        return len(self.candidates)

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        include_no_query: bool = True,
        mission_horizon: float | None = None,
    ) -> PairedDataset:
        records = list(records)
        if not records:
            raise ValueError("Cannot construct an empty dataset")
        validate_splits(records)
        groups: dict[tuple, list[Mapping[str, Any]]] = {}
        for r in records:
            Candidate(str(r["source"]), str(r["bridge"]), float(r["delay"]))
            key = (
                str(r["layout"]),
                str(r["rollout_id"]),
                str(r["snapshot_id"]),
                str(r["source"]),
                str(r["bridge"]),
                float(r["delay"]),
            )
            groups.setdefault(key, []).append(r)
        # Derive actual no-answer candidates from W branches. This adds no simulator
        # interactions and avoids evaluating an untrained 'none' source embedding.
        if include_no_query:
            controls: dict[tuple, dict[int, Mapping[str, Any]]] = {}
            for r in records:
                if r["source"] == "none":
                    continue
                key = (
                    str(r["layout"]),
                    str(r["rollout_id"]),
                    str(r["snapshot_id"]),
                    "none",
                    str(r["bridge"]),
                    float(r["delay"]),
                )
                replication = int(r.get("replication", 0))
                previous = controls.setdefault(key, {}).get(replication)
                if previous is not None:
                    if previous["outcomes"]["W"] != r["outcomes"]["W"]:
                        raise ValueError("Sources disagree on the paired W control")
                    continue
                copy = dict(r)
                copy["source"] = "none"
                copy["outcomes"] = {**r["outcomes"], "Q": r["outcomes"]["W"]}
                copy["info"] = 0.0
                controls[key][replication] = copy
            for key, replicates in controls.items():
                if key not in groups:
                    groups[key] = list(replicates.values())
        histories, candidates, targets, metadata, uncertainty = [], [], [], [], []
        for key in sorted(groups):
            rows = groups[key]
            h = np.asarray(rows[0]["history"], dtype=np.float32)
            if h.ndim != 2:
                raise ValueError("Record histories must have shape [history_length, feature_dim]")
            samples, replication_ids = [], set()
            for r in rows:
                replication = int(r.get("replication", 0))
                if replication in replication_ids:
                    raise ValueError("Duplicate stochastic replication")
                replication_ids.add(replication)
                if not np.array_equal(h, np.asarray(r["history"], dtype=np.float32)):
                    raise ValueError("Monte Carlo replications changed the conditioning history")
                o = r["outcomes"]
                costs = [float(o[b]["cost"]) for b in BRANCHES]
                info, motion = costs[1] - costs[2], costs[0] - costs[1]
                if r["source"] != "none" and (
                    not np.isclose(info, r.get("info", info))
                    or not np.isclose(motion, r.get("motion", motion))
                ):
                    raise ValueError("Paired difference labels disagree with branch costs")
                samples.append(
                    [
                        info,
                        motion,
                        *(float(o[b]["collision"]) for b in BRANCHES),
                        *(float(o[b]["completion"]) for b in BRANCHES),
                        *(restricted_completion_time(r, b, mission_horizon) for b in BRANCHES),
                    ]
                )
            values = np.asarray(samples, dtype=np.float32)
            histories.append(h)
            targets.append(values.mean(axis=0))
            uncertainty.append(values.std(axis=0, ddof=1) if len(rows) > 1 else np.zeros(11))
            candidates.append(Candidate(key[3], key[4], key[5]))
            metadata.append(
                {k: _jsonable(rows[0][k]) for k in ("layout", "rollout_id", "snapshot_id", "split")}
                | {
                    "replications": len(rows),
                    "mean_costs": [
                        float(np.mean([r["outcomes"][b]["cost"] for r in rows])) for b in BRANCHES
                    ],
                }
            )
        return cls(
            np.stack(histories),
            candidates,
            np.asarray(targets),
            metadata,
            np.asarray(uncertainty),
            {
                "records_hash": content_hash(records),
                "version": 3,
                "no_query_derived_from_W": include_no_query,
                "feature_schema": "frozen public RGB-D/odometry/source and received-evidence summaries",
                "interaction_count": len(records) * 3,
                "filtering": "none",
                "time_target": "restricted full-mission completion time; failures assigned mission horizon",
            },
        )

    def subset(self, indices: Iterable[int]) -> PairedDataset:
        ids = np.asarray(list(indices), dtype=int)
        return PairedDataset(
            self.histories[ids],
            [self.candidates[i] for i in ids],
            self.targets[ids],
            [self.metadata[i] for i in ids],
            self.monte_carlo_std[ids],
            dict(self.provenance),
        )

    def split(self, name: str) -> PairedDataset:
        return self.subset(i for i, m in enumerate(self.metadata) if m["split"] == name)

    def shuffled_pairs(self, seed: int) -> PairedDataset:
        """Permute Q outcomes within split/source/motion/delay; preserve data budget."""
        result = self.subset(range(len(self)))
        rng = np.random.default_rng(seed)
        buckets: dict[tuple, list[int]] = {}
        for i, (c, m) in enumerate(zip(self.candidates, self.metadata)):
            if c.source != "none":
                buckets.setdefault((m["split"], c.source, c.bridge, c.delay), []).append(i)
        for ids in buckets.values():
            permutation = rng.permutation(ids)
            for dst, src in zip(ids, permutation):
                q_cost = self.metadata[src]["mean_costs"][2]
                w_cost = self.metadata[dst]["mean_costs"][1]
                result.targets[dst, 0] = w_cost - q_cost
                result.targets[dst, [4, 7, 10]] = self.targets[src, [4, 7, 10]]
                result.metadata[dst] = dict(result.metadata[dst]) | {
                    "mean_costs": [*self.metadata[dst]["mean_costs"][:2], q_cost]
                }
        result.provenance.update(ablation="shuffled_pair", shuffle_seed=seed)
        return result

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "metadata": self.metadata,
            "candidates": [c.__dict__ for c in self.candidates],
            "provenance": self.provenance,
            "target_names": TARGET_NAMES,
        }
        manifest["cache_hash"] = content_hash(
            {
                **manifest,
                "histories": self.histories,
                "targets": self.targets,
                "mc_std": self.monte_carlo_std,
            }
        )
        np.savez_compressed(
            path,
            histories=self.histories,
            targets=self.targets,
            monte_carlo_std=self.monte_carlo_std,
            manifest=np.asarray(json.dumps(manifest, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> PairedDataset:
        with np.load(path, allow_pickle=False) as cached:
            manifest = json.loads(str(cached["manifest"]))
            if manifest.get("provenance", {}).get("version") != 3:
                raise ValueError(
                    "Feature cache predates received-evidence features and restricted-completion-time targets; regenerate it"
                )
            checksum = manifest.pop("cache_hash")
            actual = content_hash(
                {
                    **manifest,
                    "histories": cached["histories"],
                    "targets": cached["targets"],
                    "mc_std": cached["monte_carlo_std"],
                }
            )
            if checksum != actual:
                raise ValueError("Feature cache integrity check failed")
            return cls(
                cached["histories"],
                [Candidate(**c) for c in manifest["candidates"]],
                cached["targets"],
                manifest["metadata"],
                cached["monte_carlo_std"],
                manifest["provenance"],
            )


def shuffled_pair_records(records: Iterable[Mapping[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Break pairing without changing draws, interactions, marginal Q distribution or splits."""
    records = list(records)
    validate_splits(records)
    result = [dict(r) | {"outcomes": dict(r["outcomes"])} for r in records]
    buckets: dict[tuple, list[int]] = {}
    for i, r in enumerate(records):
        if r["source"] != "none":
            buckets.setdefault((r["split"], r["source"], r["bridge"], r["delay"]), []).append(i)
    rng = np.random.default_rng(seed)
    for ids in buckets.values():
        for dst, src in zip(ids, rng.permutation(ids)):
            result[dst]["outcomes"]["Q"] = records[src]["outcomes"]["Q"]
            o = result[dst]["outcomes"]
            result[dst]["info"] = o["W"]["cost"] - o["Q"]["cost"]
            result[dst]["motion"] = o["A"]["cost"] - o["W"]["cost"]
    return result
