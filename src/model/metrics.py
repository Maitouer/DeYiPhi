"""Binary-relevance metrics on an ordered list; invalid IDs never compact ranks."""

import numpy as np

K_VALUES = (1, 5, 10, 15, 32)


def ranking_metrics(predictions, targets, k_values=K_VALUES, invalid=-1):
    result = {f"{metric}@{k}": np.zeros(len(targets), dtype=np.float64)
              for metric in ("pass", "recall", "ndcg") for k in k_values}
    width = max(k_values)
    discounts = 1 / np.log2(np.arange(width) + 2)
    ideal = np.cumsum(discounts)
    for row, (predicted, truth) in enumerate(zip(predictions, targets, strict=True)):
        relevant = set(truth) - {invalid}
        if not relevant:
            continue
        seen = set()
        relevance = np.zeros(width)
        for rank, item in enumerate(predicted[:width]):
            if item != invalid and item in relevant and item not in seen:
                relevance[rank] = 1
            seen.add(item)
        hits = np.cumsum(relevance)
        dcg = np.cumsum(relevance * discounts)
        for k in k_values:
            result[f"pass@{k}"][row] = hits[k - 1] > 0
            result[f"recall@{k}"][row] = hits[k - 1] / len(relevant)
            result[f"ndcg@{k}"][row] = dcg[k - 1] / ideal[min(k, len(relevant)) - 1]
    return result


def recommendation_metrics(predictions, targets, k_values=K_VALUES):
    """Return the complete SID/PID × metric × K per-row table."""
    return {f"{space}/{name}": values
            for space in ("sid", "pid")
            for name, values in ranking_metrics(predictions[space], targets[space], k_values).items()}
