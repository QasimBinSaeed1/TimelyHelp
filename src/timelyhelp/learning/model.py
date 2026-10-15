"""Controller-conditioned GRU ensembles with independent empirical calibration.

Intervals are empirical prediction scores, not collision-avoidance certificates.
The visual/perception pipeline remains frozen; only cached histories are learned.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import minimize

from .data import Candidate, PairedDataset, candidate_features, validate_splits


@dataclass
class ModelConfig:
    feature_dim: int = 61
    hidden_dim: int = 32
    ensemble_size: int = 5
    seed: int = 0
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 0.002
    time_scale: float = 30.0
    ranking_weight: float = 0.1
    uncertainty_multiplier: float = 1.64
    calibration_coverage: float = 0.9
    omit_latency: bool = False
    latency_feature_indices: tuple[int, ...] = (18, 19, 20, 21, 24, 25)
    without_W: bool = False
    omit_reliability_metadata: bool = False
    reliability_feature_indices: tuple[int, ...] = (22, 23, 30, 31)
    device: str = "cpu"

    def __post_init__(self):
        if (
            min(self.feature_dim, self.hidden_dim, self.ensemble_size, self.epochs, self.batch_size)
            <= 0
            or self.time_scale <= 0
        ):
            raise ValueError("Invalid model dimensions or training settings")
        if not 0 < self.calibration_coverage < 1:
            raise ValueError("Calibration coverage must be between zero and one")


class ConsequenceNetwork(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.temporal = nn.GRU(config.feature_dim, config.hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim + 6, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 11),
        )

    def forward(self, histories: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        _, encoded = self.temporal(histories)
        return self.head(torch.cat((encoded[-1], candidates), dim=-1))


def _cluster_key(m: dict[str, Any]) -> str:
    return f"{m['layout']}::{m['rollout_id']}"


def _snapshot_key(m: dict[str, Any]) -> str:
    return f"{_cluster_key(m)}::{m['snapshot_id']}"


def cluster_bootstrap_indices(dataset: PairedDataset, rng: np.random.Generator) -> np.ndarray:
    clusters: dict[str, list[int]] = {}
    for i, m in enumerate(dataset.metadata):
        clusters.setdefault(_cluster_key(m), []).append(i)
    keys = sorted(clusters)
    if not keys:
        raise ValueError("Cannot bootstrap an empty training dataset")
    selected = rng.choice(keys, len(keys), replace=True)
    return np.asarray([i for key in selected for i in clusters[key]], dtype=int)


def _quantile(values: np.ndarray, coverage: float) -> np.ndarray:
    """Finite-sample conservative empirical quantile; still no shift guarantee."""
    if not len(values):
        raise ValueError("Calibration episodes are empty")
    rank = min(len(values), int(np.ceil((len(values) + 1) * coverage)))
    return np.sort(values, axis=0)[rank - 1]


class ConsequenceEnsemble:
    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.models: list[ConsequenceNetwork] = []
        self.mean = np.zeros(self.config.feature_dim, dtype=np.float32)
        self.scale = np.ones(self.config.feature_dim, dtype=np.float32)
        self.calibration: dict[str, Any] | None = None
        self.training_manifest: dict[str, Any] = {}
        self.fit_report: dict[str, Any] = {}

    def _histories(self, histories: np.ndarray) -> torch.Tensor:
        histories = np.asarray(histories, dtype=np.float32)
        if histories.ndim == 2:
            histories = histories[None]
        if histories.ndim != 3 or histories.shape[-1] != self.config.feature_dim:
            raise ValueError("Unexpected observation history shape")
        if not np.isfinite(histories).all():
            raise ValueError("History contains nonfinite observations")
        normalized = (histories - self.mean) / self.scale
        if self.config.omit_latency:
            normalized = normalized.copy()
            normalized[..., list(self.config.latency_feature_indices)] = 0
        if self.config.omit_reliability_metadata:
            normalized = normalized.copy()
            normalized[..., list(self.config.reliability_feature_indices)] = 0
        return torch.as_tensor(normalized, device=self.config.device)

    def _candidate_tensor(self, candidates: Sequence[Candidate]) -> torch.Tensor:
        return torch.as_tensor(
            candidate_features(candidates, self.config.time_scale, self.config.omit_latency),
            device=self.config.device,
        )

    def _elapsed_fraction(self, normalized_histories: torch.Tensor) -> torch.Tensor:
        """Recover the public mission clock from the frozen feature schema.

        Feature zero is elapsed mission time / declared mission horizon. It is
        retained by both metadata ablations because the mission clock is an
        ordinary controller observation rather than a source latency draw.
        """
        return torch.clamp(
            normalized_histories[:, -1, 0] * float(self.scale[0]) + float(self.mean[0]), 0.0, 1.0
        )

    def _loss(
        self,
        raw: torch.Tensor,
        target: torch.Tensor,
        metadata: list[dict[str, Any]],
        elapsed_fraction: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        effects = target[:, :2].clone()
        branch_ids = [0, 1, 2]
        if self.config.without_W:
            effects[:, 0] = target[:, 0] + target[:, 1]
            effects[:, 1] = 0
            branch_ids = [0, 2]
        regression = F.smooth_l1_loss(raw[:, :2], effects)
        collision = F.binary_cross_entropy_with_logits(
            raw[:, [2 + b for b in branch_ids]], target[:, [2 + b for b in branch_ids]]
        )
        completion = F.binary_cross_entropy_with_logits(
            raw[:, [5 + b for b in branch_ids]], target[:, [5 + b for b in branch_ids]]
        )
        elapsed = elapsed_fraction[:, None]
        predicted_time_fraction = elapsed + (1 - elapsed) * torch.sigmoid(
            raw[:, [8 + b for b in branch_ids]]
        )
        timing = F.smooth_l1_loss(
            predicted_time_fraction, target[:, [8 + b for b in branch_ids]] / self.config.time_scale
        )
        ranking = raw.new_zeros(())
        groups: dict[str, list[int]] = {}
        for i, m in enumerate(metadata):
            groups.setdefault(_snapshot_key(m), []).append(i)
        rank_terms = []
        predicted = raw[:, 0] + raw[:, 1]
        observed = target[:, 0] + target[:, 1]
        for ids in groups.values():
            if len(ids) < 2:
                continue
            pair = torch.triu_indices(len(ids), len(ids), offset=1, device=raw.device)
            indices = torch.as_tensor(ids, device=raw.device)
            i, j = indices[pair[0]], indices[pair[1]]
            difference = observed[i] - observed[j]
            mask = difference.abs() > 1e-4
            if mask.any():
                rank_terms.append(
                    F.softplus(
                        -difference[mask].sign() * (predicted[i[mask]] - predicted[j[mask]])
                    ).mean()
                )
        if rank_terms:
            ranking = torch.stack(rank_terms).mean()
        total = regression + collision + completion + timing + self.config.ranking_weight * ranking
        return total, {
            "regression": float(regression.detach()),
            "collision_bce": float(collision.detach()),
            "completion_bce": float(completion.detach()),
            "time": float(timing.detach()),
            "ranking": float(ranking.detach()),
        }

    def fit(self, train: PairedDataset, development: PairedDataset | None = None) -> dict[str, Any]:
        if not len(train) or {m["split"] for m in train.metadata} != {"train"}:
            raise ValueError("Fit requires the nonempty training split")
        if development is not None:
            if not len(development) or {m["split"] for m in development.metadata} != {
                "development"
            }:
                raise ValueError("Development data must use the development split")
            validate_splits(train.metadata + development.metadata)
        if self.config.without_W and train.provenance.get("no_query_derived_from_W", False):
            train = train.subset(i for i, c in enumerate(train.candidates) if c.source != "none")
            if development is not None:
                development = development.subset(
                    i for i, c in enumerate(development.candidates) if c.source != "none"
                )
        if not len(train):
            raise ValueError("No training examples remain after ablation masking")
        self.mean = train.histories.mean(axis=(0, 1))
        self.scale = np.maximum(train.histories.std(axis=(0, 1)), 1e-5)
        x, c = self._histories(train.histories), self._candidate_tensor(train.candidates)
        y = torch.as_tensor(train.targets, device=self.config.device)
        self.models, traces = [], []
        self.calibration = None
        for member in range(self.config.ensemble_size):
            seed = self.config.seed + member * 1009
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            rng = np.random.default_rng(seed)
            model = ConsequenceNetwork(self.config).to(self.config.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
            bootstrap = cluster_bootstrap_indices(train, rng)
            # Keep a snapshot's candidates together so ranking compares matched
            # histories. A large snapshot group is deliberately not truncated.
            groups: dict[str, list[int]] = {}
            for i in bootstrap:
                groups.setdefault(_snapshot_key(train.metadata[i]), []).append(int(i))
            best_state, best_loss, member_trace = None, np.inf, []
            for epoch in range(self.config.epochs):
                model.train()
                batches, current = [], []
                for key in rng.permutation(list(groups)):
                    if current and len(current) + len(groups[key]) > self.config.batch_size:
                        batches.append(current)
                        current = []
                    current.extend(groups[key])
                if current:
                    batches.append(current)
                losses, components = [], {}
                for ids in batches:
                    optimizer.zero_grad(set_to_none=True)
                    raw = model(x[ids], c[ids])
                    loss, components = self._loss(
                        raw,
                        y[ids],
                        [train.metadata[i] for i in ids],
                        self._elapsed_fraction(x[ids]),
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite consequence training loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                score = float(np.mean(losses))
                if development is not None:
                    model.eval()
                    with torch.no_grad():
                        development_histories = self._histories(development.histories)
                        score = float(
                            self._loss(
                                model(
                                    development_histories,
                                    self._candidate_tensor(development.candidates),
                                ),
                                torch.as_tensor(development.targets, device=self.config.device),
                                development.metadata,
                                self._elapsed_fraction(development_histories),
                            )[0]
                        )
                if score < best_loss:
                    best_loss = score
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }
                member_trace.append(
                    {
                        "epoch": epoch + 1,
                        "train_loss": float(np.mean(losses)),
                        "selection_loss": score,
                        **components,
                    }
                )
            model.load_state_dict(best_state)
            model.eval()
            self.models.append(model)
            traces.append(member_trace)
        self.training_manifest = {
            "training_metadata": train.metadata,
            "development_metadata": [] if development is None else development.metadata,
            "dataset_provenance": train.provenance,
            "bootstrap_unit": "complete rollout",
            "training_seed": self.config.seed,
            "members": self.config.ensemble_size,
        }
        self.fit_report = {
            "members": traces,
            "examples": len(train),
            "train_rollouts": len({_cluster_key(m) for m in train.metadata}),
        }
        return self.fit_report

    def _member_predictions(
        self, history: np.ndarray, candidates: Sequence[Candidate]
    ) -> np.ndarray:
        if not self.models:
            raise RuntimeError("Consequence model has not been fitted or loaded")
        x = self._histories(history)
        if len(x) == 1:
            x = x.expand(len(candidates), -1, -1)
        if len(x) != len(candidates):
            raise ValueError("Provide one shared history or one history per candidate")
        c = self._candidate_tensor(candidates)
        predictions = []
        with torch.no_grad():
            for model in self.models:
                raw = model(x, c)
                elapsed = self._elapsed_fraction(x)[:, None]
                time_fraction = elapsed + (1 - elapsed) * torch.sigmoid(raw[:, 8:11])
                prediction = torch.cat(
                    (
                        raw[:, :2],
                        torch.sigmoid(raw[:, 2:8]),
                        time_fraction * self.config.time_scale,
                    ),
                    dim=1,
                )
                predictions.append(prediction.cpu().numpy())
        return np.stack(predictions)

    def predict(
        self, history: np.ndarray, candidates: Sequence[Candidate | dict]
    ) -> dict[str, np.ndarray]:
        candidates = [Candidate.coerce(c) for c in candidates]
        if not candidates:
            raise ValueError("At least one candidate is required")
        members = self._member_predictions(history, candidates)
        mean, epistemic = members.mean(axis=0), members.std(axis=0)
        if self.calibration is not None:
            for j in range(2, 8):
                a, b = self.calibration["probability_transform"][str(j)]
                p = np.clip(mean[:, j], 1e-6, 1 - 1e-6)
                mean[:, j] = 1 / (1 + np.exp(-np.clip(a * np.log(p / (1 - p)) + b, -30, 30)))
        residual = (
            np.zeros(11)
            if self.calibration is None
            else np.asarray(self.calibration["absolute_residual"])
        )
        total = self.config.uncertainty_multiplier * epistemic + residual
        result = {}
        slices = {
            "info": 0,
            "motion": 1,
            "collision": slice(2, 5),
            "completion": slice(5, 8),
            "time": slice(8, 11),
        }
        for name, index in slices.items():
            result[name] = mean[:, index]
            result[f"{name}_epistemic"] = epistemic[:, index]
            result[f"{name}_uncertainty"] = total[:, index]
        for suffix in ("", "_epistemic", "_uncertainty"):
            result["restricted_completion_time" + suffix] = result["time" + suffix]
        return result

    def calibrate(self, calibration: PairedDataset) -> dict[str, Any]:
        if not len(calibration) or {m["split"] for m in calibration.metadata} != {"calibration"}:
            raise ValueError("Calibration must use independent calibration episodes")
        validate_splits(
            self.training_manifest.get("training_metadata", [])
            + self.training_manifest.get("development_metadata", [])
            + calibration.metadata
        )
        if self.config.without_W and calibration.provenance.get("no_query_derived_from_W", False):
            calibration = calibration.subset(
                i for i, c in enumerate(calibration.candidates) if c.source != "none"
            )
        self.calibration = None
        predictions = self._member_predictions(calibration.histories, calibration.candidates).mean(
            axis=0
        )
        transforms = {}
        for j in range(2, 8):
            if self.config.without_W and j in (3, 6):
                transforms[str(j)] = [1.0, 0.0]
                continue
            y, p = calibration.targets[:, j], np.clip(predictions[:, j], 1e-6, 1 - 1e-6)
            logits = np.log(p / (1 - p))
            # Probability calibration is regularized and weighted equally by
            # rollout, so grids with many siblings do not dominate the fit.
            counts: dict[str, int] = {}
            for m in calibration.metadata:
                counts[_cluster_key(m)] = counts.get(_cluster_key(m), 0) + 1
            weights = np.asarray([1 / counts[_cluster_key(m)] for m in calibration.metadata])
            weights /= weights.sum()

            def objective(parameters):
                a, b = parameters
                z = np.clip(a * logits + b, -30, 30)
                loss = np.logaddexp(0, z) - y * z
                return np.dot(weights, loss) + 0.01 * ((a - 1) ** 2 + b**2)

            fitted = minimize(
                objective, np.array([1.0, 0.0]), bounds=[(0.05, 10), (-8, 8)], method="L-BFGS-B"
            )
            a, b = fitted.x if fitted.success else (1.0, 0.0)
            transforms[str(j)] = [float(a), float(b)]
            predictions[:, j] = 1 / (1 + np.exp(-np.clip(a * logits + b, -30, 30)))
        targets = calibration.targets.copy()
        if self.config.without_W:
            targets[:, 0] += targets[:, 1]
            targets[:, 1] = 0
        absolute = np.abs(targets - predictions)
        if self.config.without_W:
            absolute[:, [3, 6, 9]] = 0
        # Calibrate on complete rollout maxima rather than treating candidates
        # and MC siblings as independent calibration episodes.
        residual_clusters: dict[str, list[int]] = {}
        for i, m in enumerate(calibration.metadata):
            residual_clusters.setdefault(_cluster_key(m), []).append(i)
        maxima = np.stack([absolute[ids].max(axis=0) for ids in residual_clusters.values()])
        self.calibration = {
            "probability_transform": transforms,
            "absolute_residual": _quantile(maxima, self.config.calibration_coverage).tolist(),
            "coverage_target": self.config.calibration_coverage,
            "episodes": len(residual_clusters),
            "examples": len(calibration),
            "unit": "complete rollout maximum",
            "metadata": calibration.metadata,
            "warning": "Empirical calibration; no safety or distribution-shift guarantee",
            "mean_absolute_error": absolute.mean(axis=0).tolist(),
        }
        return self.calibration

    def save(self, path: str | Path) -> None:
        if not self.models:
            raise RuntimeError("Cannot save an unfitted model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 4,
                "config": asdict(self.config),
                "mean": torch.as_tensor(self.mean),
                "scale": torch.as_tensor(self.scale),
                "states": [{k: v.cpu() for k, v in m.state_dict().items()} for m in self.models],
                "calibration": self.calibration,
                "training_manifest": self.training_manifest,
                "fit_report": self.fit_report,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> ConsequenceEnsemble:
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload["format_version"] != 4:
            raise ValueError(
                "Consequence model predates received-evidence features or completion-clock bounds; regenerate it"
            )
        config = ModelConfig(**(payload["config"] | {"device": device}))
        model = cls(config)
        model.mean, model.scale = payload["mean"].numpy(), payload["scale"].numpy()
        for state in payload["states"]:
            member = ConsequenceNetwork(config).to(device)
            member.load_state_dict(state)
            member.eval()
            model.models.append(member)
        model.calibration, model.training_manifest = (
            payload["calibration"],
            payload["training_manifest"],
        )
        model.fit_report = payload["fit_report"]
        return model
