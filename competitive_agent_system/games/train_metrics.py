from __future__ import annotations

from collections import defaultdict

import numpy as np

from competitive_agent_system.games.cournot.parser import canonicalize_cournot_invalid_reason
from verl import DataProto


def _compute_cournot_invalid_reason_metrics(batch: DataProto) -> dict[str, float]:
    metric_dict: dict[str, float] = {}

    if "is_action_valid" in batch.non_tensor_batch:
        valids = np.asarray(batch.non_tensor_batch["is_action_valid"]).astype(bool)
        invalid_mask = np.logical_not(valids)
        if np.any(invalid_mask) and "invalid_action_reason" in batch.non_tensor_batch:
            reasons = np.asarray(batch.non_tensor_batch["invalid_action_reason"], dtype=object)
            counts = defaultdict(int)
            total_invalid = 0
            for reason in reasons[invalid_mask]:
                canonical_reason = canonicalize_cournot_invalid_reason(reason)
                if canonical_reason is None:
                    continue
                counts[canonical_reason] += 1
                total_invalid += 1

            if total_invalid > 0:
                for reason_name, count in counts.items():
                    metric_dict[f"train_invalid/step_reason/{reason_name}"] = float(count / total_invalid)

    traj_uids = batch.non_tensor_batch.get("traj_uid")
    episode_failure_reasons = batch.non_tensor_batch.get("episode_failure_reason")
    if traj_uids is None or episode_failure_reasons is None:
        return metric_dict

    unique_traj_uid, unique_idx = np.unique(traj_uids, return_index=True)
    if len(unique_traj_uid) == 0:
        return metric_dict

    reasons = np.asarray(episode_failure_reasons, dtype=object)[unique_idx]
    has_failure = np.array([bool(str(reason).strip()) for reason in reasons], dtype=bool)
    if np.any(has_failure):
        counts = defaultdict(int)
        total_failures = 0
        for reason in reasons[has_failure]:
            canonical_reason = canonicalize_cournot_invalid_reason(reason)
            if canonical_reason is None:
                continue
            counts[canonical_reason] += 1
            total_failures += 1

        if total_failures > 0:
            for reason_name, count in counts.items():
                metric_dict[f"train_invalid/episode_failure_reason/{reason_name}"] = float(count / total_failures)

    return metric_dict


def compute_task_train_metrics(env_name: str, batch: DataProto) -> dict[str, float]:
    normalized_env_name = str(env_name).lower()
    if "cournot" in normalized_env_name:
        return _compute_cournot_invalid_reason_metrics(batch=batch)
    return {}
