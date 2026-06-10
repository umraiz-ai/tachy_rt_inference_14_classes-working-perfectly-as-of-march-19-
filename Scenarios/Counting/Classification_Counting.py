# Worker counting from NPU object detections
import os
import cv2
from typing import List, Optional, Tuple

MIN_DIM_FRAC = 0.07
MIN_DIM_PIX_FLOOR = 64

COUNTING_ROLE_NAMES = {
    'worker': ('worker',),
}


def _adaptive_min_person_dim(h: int, w: int) -> int:
    return int(max(MIN_DIM_FRAC * min(h, w), MIN_DIM_PIX_FLOOR))


def resolve_counting_class_ids(clss_dict: dict) -> dict:
    """Map counting role names to class IDs by looking up class dict values."""
    name_to_id = {v.lower(): int(k) for k, v in clss_dict.items()}
    class_ids = {}
    for role, aliases in COUNTING_ROLE_NAMES.items():
        found = None
        for alias in aliases:
            if alias in name_to_id:
                found = name_to_id[alias]
                break
        if found is None:
            raise KeyError(
                f"Could not resolve counting role '{role}' from class dict. "
                f"Expected one of: {aliases}. Available: {list(clss_dict.values())}"
            )
        class_ids[role] = found
    return class_ids


def read_gt_count(label_dir: str, image_filename: str) -> int:
    """Read ground-truth worker count from a label .txt file (single integer line)."""
    base, _ = os.path.splitext(image_filename)
    label_path = os.path.join(label_dir, base + ".txt")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")
    with open(label_path, "r", encoding="utf-8") as f:
        line = f.readline().strip()
    return int(line)


def detect_worker_count(
    image,
    detections: List[dict],
    class_ids: dict,
    conf_threshold: float = 0.30,
) -> Tuple[object, int, int]:
    """
    Count eligible workers from pre-computed NPU detections.

    Returns:
        annotated_image, worker_count, skipped_workers
    """
    image = image.copy()
    h, w = image.shape[:2]
    min_person_dim = _adaptive_min_person_dim(h, w)
    worker_cls = class_ids['worker']

    worker_count = 0
    skipped_workers = 0

    for det in detections:
        c = det['confidence']
        k = det['class_id']
        x1, y1, x2, y2 = det['box']
        if c < conf_threshold or k != worker_cls:
            continue

        ph = y2 - y1
        if ph < min_person_dim:
            skipped_workers += 1
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 215, 255), 2)
            cv2.putText(
                image, f"Worker (skipped <{min_person_dim}px)",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2, cv2.LINE_AA,
            )
            continue

        worker_count += 1
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.putText(
            image, "Worker",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2, cv2.LINE_AA,
        )

    badge = f"Workers: {worker_count}"
    if skipped_workers:
        badge += f"  (skipped: {skipped_workers})"
    cv2.putText(
        image, badge, (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 0), 2, cv2.LINE_AA,
    )

    return image, worker_count, skipped_workers


def draw_count_badge(image, split_name: str, gt_count: Optional[int], pred_count: int):
    """Overlay GT/pred/error badge for eval mode."""
    if gt_count is not None:
        err = pred_count - gt_count
        text = f"{split_name}  GT:{gt_count}  Pred:{pred_count}  Err:{err:+d}"
    else:
        text = f"Pred:{pred_count}"
    cv2.putText(
        image, text, (50, 90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return image


def compute_counting_metrics(y_true, y_pred):
    y_true = [float(v) for v in y_true]
    y_pred = [float(v) for v in y_pred]
    n = len(y_true)
    if n == 0:
        return None

    abs_err = [abs(p - t) for t, p in zip(y_true, y_pred)]
    sq_err = [(p - t) ** 2 for t, p in zip(y_true, y_pred)]

    mae = sum(abs_err) / n
    mse = sum(sq_err) / n
    rmse = mse ** 0.5
    max_err = max(abs_err)
    exact_match = sum(1 for e in abs_err if e == 0) / n
    within_1 = sum(1 for e in abs_err if e <= 1) / n
    within_2 = sum(1 for e in abs_err if e <= 2) / n

    return {
        'n_samples': n,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'max_abs_error': max_err,
        'exact_match_rate': exact_match,
        'within_1_rate': within_1,
        'within_2_rate': within_2,
    }


def print_counting_metrics(metrics):
    print("\nWorker counting metrics:")
    print(f"  Samples               : {metrics['n_samples']}")
    print(f"  MAE   (mean |err|)     : {metrics['mae']:.3f}")
    print(f"  RMSE (sqrt(MSE))       : {metrics['rmse']:.3f}")
    print(f"  Max abs error          : {metrics['max_abs_error']:.3f}")
    print(f"  Exact match rate       : {metrics['exact_match_rate'] * 100:.2f}%")
    print(f"  Within ±1 worker       : {metrics['within_1_rate'] * 100:.2f}%")
    print(f"  Within ±2 workers      : {metrics['within_2_rate'] * 100:.2f}%")
