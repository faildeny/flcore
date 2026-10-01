import json
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import (BinaryAccuracy, BinaryAUROC,
                                         BinaryF1Score, BinaryPrecision,
                                         BinaryRecall, BinarySpecificity)
from torchmetrics.classification.stat_scores import BinaryStatScores
from torchmetrics.functional.classification.precision_recall import \
    _precision_recall_reduce
from torchmetrics.functional.classification.specificity import \
    _specificity_reduce
from torchmetrics.regression import MeanSquaredError


class BinaryBalancedAccuracy(BinaryStatScores):
    is_differentiable = False
    higher_is_better = True
    full_state_update: bool = False

    def compute(self) -> Tensor:
        """Computes balanced accuracy based on inputs passed in to ``update`` previously."""
        tp, fp, tn, fn = self._final_state()

        recall = _precision_recall_reduce(
            "recall",
            tp,
            fp,
            tn,
            fn,
            average="binary",
            multidim_average=self.multidim_average,
        )
        specificity = _specificity_reduce(
            tp, fp, tn, fn, average="binary", multidim_average=self.multidim_average
        )

        return (recall + specificity) / 2


def get_metrics_collection(task_type="binary", device="cpu", threshold=0.5):

    if task_type.lower() == "binary":
        return MetricCollection(
            {
                "accuracy": BinaryAccuracy(threshold=threshold).to(device),
                "precision": BinaryPrecision(threshold=threshold).to(device),
                "recall": BinaryRecall(threshold=threshold).to(device),
                "specificity": BinarySpecificity(threshold=threshold).to(device),
                "f1": BinaryF1Score(threshold=threshold).to(device),
                "balanced_accuracy": BinaryBalancedAccuracy(threshold=threshold).to(device),
                "auroc": BinaryAUROC().to(device),
            }
        )
    elif task_type.lower() == "reg":
        return MetricCollection({
            "mse": MeanSquaredError().to(device),
        })


def _to_tensor(values):
    if torch.is_tensor(values):
        return values
    if isinstance(values, list):
        return torch.cat(values)
    if hasattr(values, "tolist"):
        return torch.tensor(values.tolist())
    return torch.tensor(values)


def _extract_attribute_values(X: Any, attribute: Any):
    if X is None:
        return None

    if hasattr(X, "loc") and hasattr(X, "columns"):
        if attribute in X.columns:
            return np.asarray(X[attribute])
        return None

    array = np.asarray(X)
    if array.ndim < 2:
        return None

    if isinstance(attribute, int) and 0 <= attribute < array.shape[1]:
        return array[:, attribute]

    return None


def _group_rates(y_true_group: Tensor, y_pred_group: Tensor, threshold: float):
    y_pred_labels = (y_pred_group >= threshold).int()
    y_true_group = y_true_group.int()

    tp = ((y_pred_labels == 1) & (y_true_group == 1)).sum().item()
    tn = ((y_pred_labels == 0) & (y_true_group == 0)).sum().item()
    fp = ((y_pred_labels == 1) & (y_true_group == 0)).sum().item()
    fn = ((y_pred_labels == 0) & (y_true_group == 1)).sum().item()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return tpr, fpr


def _normalize_key_part(value: Any):
    return str(value).strip().replace(" ", "_")


def _group_label(index: int):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if 0 <= index < len(alphabet):
        return alphabet[index]
    return f"G{index + 1}"


def calculate_fairness_metrics(
    y_true,
    y_pred_proba,
    X,
    fairness_attributes,
    task_type="binary",
    threshold=0.5,
):
    fairness_results = {}

    y_true_tensor = _to_tensor(y_true)
    y_pred_tensor = _to_tensor(y_pred_proba)
    if y_pred_tensor.ndim > 1 and y_pred_tensor.shape[1] > 1:
        y_pred_tensor = y_pred_tensor[:, 1]

    for attribute in fairness_attributes:
        attribute_values = _extract_attribute_values(X, attribute)
        if attribute_values is None:
            continue

        attribute_values = np.asarray(attribute_values)
        unique_groups = np.unique(attribute_values)

        group_rates = []
        normalized_attribute = _normalize_key_part(attribute)

        for group_idx, group in enumerate(unique_groups):
            group_mask = torch.tensor(attribute_values == group, dtype=torch.bool)
            if group_mask.numel() != y_true_tensor.numel() or group_mask.sum().item() == 0:
                continue

            group_y_true = y_true_tensor[group_mask]
            group_y_pred = y_pred_tensor[group_mask]
            group_metrics = calculate_metrics(
                group_y_true,
                group_y_pred,
                task_type=task_type,
                threshold=threshold,
                return_sufficient_stats=False,
            )

            normalized_group = _group_label(group_idx)
            for metric_name, metric_value in group_metrics.items():
                normalized_metric = _normalize_key_part(metric_name)
                fairness_results[
                    f"{normalized_attribute}_{normalized_group}_{normalized_metric}"
                ] = metric_value

            if task_type.lower() == "binary":
                group_rates.append(_group_rates(group_y_true, group_y_pred, threshold))

        if task_type.lower() == "binary" and len(group_rates) == 2:
            tpr_diff = abs(group_rates[0][0] - group_rates[1][0])
            fpr_diff = abs(group_rates[0][1] - group_rates[1][1])
            fairness_results[f"{normalized_attribute}_EOP"] = tpr_diff
            fairness_results[f"{normalized_attribute}_EOD"] = max(tpr_diff, fpr_diff)

    return fairness_results


# Key under which each client reports the sufficient statistics needed to compute
# pooled (global) metrics on the server. The value is a JSON string, since Flower
# metrics must be scalars. Client-side prefixes (e.g. "val ") are preserved.
SUFFICIENT_STATS_KEY = "sufficient_stats"
# Number of equal-width bins on [0, 1] used for the score histograms. Global AUROC is
# exact up to ties introduced by binning (pairs falling in the same bin count as 0.5).
N_SCORE_BINS = 100


def _confusion_counts(y_true: Tensor, y_pred: Tensor, threshold: float):
    y_pred_labels = (y_pred >= threshold).int()
    y_true = y_true.int()
    tp = ((y_pred_labels == 1) & (y_true == 1)).sum().item()
    fp = ((y_pred_labels == 1) & (y_true == 0)).sum().item()
    tn = ((y_pred_labels == 0) & (y_true == 0)).sum().item()
    fn = ((y_pred_labels == 0) & (y_true == 1)).sum().item()
    return [tp, fp, tn, fn]


def _group_key(value: Any):
    # Stable across centers: 1, 1.0 and np.int64(1) all map to "1"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, float, np.integer, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def calculate_sufficient_stats(
    y_true,
    y_pred_proba,
    X: Optional[Any] = None,
    fairness_attributes: Optional[list] = None,
    threshold=0.5,
    n_bins=N_SCORE_BINS,
):
    """Per-center statistics that can be summed across centers to compute global
    AUROC (score histograms per class) and global EOp/EOd (confusion counts per
    sensitive group). Returned as a JSON string so it can travel as a Flower metric."""
    y_true = _to_tensor(y_true)
    y_pred = _to_tensor(y_pred_proba)
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = y_pred[:, 1]
    y_true_np = y_true.detach().cpu().numpy().astype(int).ravel()
    y_pred_np = y_pred.detach().cpu().numpy().astype(float).ravel()

    # Scores outside [0, 1] (e.g. decision_function) are clipped into the edge bins
    bins = np.clip((y_pred_np * n_bins).astype(int), 0, n_bins - 1)
    pos = np.bincount(bins[y_true_np == 1], minlength=n_bins)
    neg = np.bincount(bins[y_true_np == 0], minlength=n_bins)
    nonzero = np.nonzero(pos + neg)[0]

    stats = {
        "n_bins": n_bins,
        # Sparse histogram: [bin index, n positives, n negatives]
        "hist": [[int(b), int(pos[b]), int(neg[b])] for b in nonzero],
        # [tp, fp, tn, fn] at the threshold used by this center
        "confusion": _confusion_counts(y_true, y_pred, threshold),
        "groups": {},
    }

    for attribute in fairness_attributes or []:
        attribute_values = _extract_attribute_values(X, attribute)
        if attribute_values is None:
            continue
        attribute_values = np.asarray(attribute_values).ravel()
        if attribute_values.shape[0] != y_true_np.shape[0]:
            continue
        group_counts = {}
        for group in np.unique(attribute_values):
            mask = torch.tensor(attribute_values == group, dtype=torch.bool)
            group_counts[_group_key(group)] = _confusion_counts(
                y_true[mask], y_pred[mask], threshold
            )
        stats["groups"][_normalize_key_part(attribute)] = group_counts

    return json.dumps(stats, separators=(",", ":"))


def calculate_metrics(
    y_true,
    y_pred_proba,
    task_type="binary",
    threshold=0.5,
    X: Optional[Any] = None,
    fairness_attributes: Optional[list] = None,
    return_sufficient_stats: bool = True,
):
    metrics_collection = get_metrics_collection(task_type, threshold=threshold)
    y_true = _to_tensor(y_true)
    y_pred_proba = _to_tensor(y_pred_proba)
    
    # Extract probabilities for the positive class if shape>1
    if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
        y_pred_proba = y_pred_proba[:, 1]

    metrics_collection.update(y_pred_proba, y_true)

    metrics = metrics_collection.compute()
    metrics = {k: v.item() for k, v in metrics.items()}
    # Add n positives and n negatives to the metrics for better interpretability
    metrics["n positives"] = (y_true == 1).sum().item()
    metrics["n negatives"] = (y_true == 0).sum().item()

    if X is not None and fairness_attributes:
        fairness_metrics = calculate_fairness_metrics(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            X=X,
            fairness_attributes=fairness_attributes,
            task_type=task_type,
            threshold=threshold,
        )

        #Extract fairness metrics and add them to the main metrics dictionary
        metrics.update(fairness_metrics)

    if return_sufficient_stats and task_type.lower() == "binary":
        metrics[SUFFICIENT_STATS_KEY] = calculate_sufficient_stats(
            y_true,
            y_pred_proba,
            X=X,
            fairness_attributes=fairness_attributes,
            threshold=threshold,
        )

    return metrics


def _auroc_from_histogram(pos: np.ndarray, neg: np.ndarray):
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    neg_below = np.cumsum(neg) - neg
    return float((pos * (neg_below + 0.5 * neg)).sum() / (n_pos * n_neg))


def _rates(counts):
    tp, fp, tn, fn = counts
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return tpr, fpr


def aggregate_sufficient_stats(stats_list, prefix=""):
    """Pools per-center sufficient statistics into global AUROC, EOP and EOD."""
    stats_list = [json.loads(s) for s in stats_list]
    n_bins = stats_list[0]["n_bins"]
    pos = np.zeros(n_bins, dtype=np.int64)
    neg = np.zeros(n_bins, dtype=np.int64)
    confusion = np.zeros(4, dtype=np.int64)
    groups = {}

    for stats in stats_list:
        if stats["n_bins"] != n_bins:
            raise ValueError("All centers must use the same number of score bins")
        for b, p, n in stats["hist"]:
            pos[b] += p
            neg[b] += n
        confusion += np.asarray(stats["confusion"], dtype=np.int64)
        for attribute, group_counts in stats["groups"].items():
            attribute_groups = groups.setdefault(attribute, {})
            for group, counts in group_counts.items():
                attribute_groups[group] = attribute_groups.get(group, np.zeros(4, dtype=np.int64)) + np.asarray(counts)

    metrics = {
        f"{prefix}global auroc": _auroc_from_histogram(pos, neg),
        f"{prefix}global n positives": int(pos.sum()),
        f"{prefix}global n negatives": int(neg.sum()),
    }
    tp, fp, tn, fn = confusion.tolist()
    metrics[f"{prefix}global tp"] = tp
    metrics[f"{prefix}global fp"] = fp
    metrics[f"{prefix}global tn"] = tn
    metrics[f"{prefix}global fn"] = fn

    for attribute, group_counts in groups.items():
        if len(group_counts) < 2:
            continue
        rates = [_rates(counts.tolist()) for counts in group_counts.values()]
        tprs = [r[0] for r in rates]
        fprs = [r[1] for r in rates]
        # For two groups this is |TPR_A - TPR_B|, matching the per-center definition
        tpr_diff = max(tprs) - min(tprs)
        fpr_diff = max(fprs) - min(fprs)
        metrics[f"{prefix}global {attribute}_EOP"] = tpr_diff
        metrics[f"{prefix}global {attribute}_EOD"] = max(tpr_diff, fpr_diff)

    return metrics

def metrics_aggregation_fn(distributed_metrics):
    # print(distributed_metrics[0][1].keys())
    keys_names = distributed_metrics[0][1].keys()
    keys_names = list(keys_names)

    metrics ={}

    for kn in keys_names:
        results = [ evaluate_res[kn] for _, evaluate_res in distributed_metrics if kn in evaluate_res]
        if kn.endswith(SUFFICIENT_STATS_KEY):
            prefix = kn[: -len(SUFFICIENT_STATS_KEY)]
            metrics.update(aggregate_sufficient_stats(results, prefix=prefix))
            continue
        metrics[kn] = np.mean(results)
        metrics['per client ' + kn] = results
        #print(f"Metric {kn} in aggregation evaluate: {metrics[kn]}\n")

    metrics['per client n samples'] = [res[0] for res in distributed_metrics]

    return metrics

def find_best_threshold(y_true, y_pred_proba, metric="balanced_accuracy"):
    best_threshold = 0.5
    best_metric_value = 0.0

    for threshold in np.arange(0.0, 1.01, 0.01):
        metrics = calculate_metrics(
            y_true, y_pred_proba, threshold=threshold, return_sufficient_stats=False
        )
        if metrics[metric] > best_metric_value:
            best_metric_value = metrics[metric]
            best_threshold = threshold

    return best_threshold
