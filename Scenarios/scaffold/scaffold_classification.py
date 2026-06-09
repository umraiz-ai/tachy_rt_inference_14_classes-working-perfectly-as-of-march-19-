# Scaffold safety classification from NPU object detections
import cv2
from typing import List, Tuple

MIN_DIM_FRAC = 0.07
MIN_DIM_PIX_FLOOR = 64

SCAFFOLD_ROLE_NAMES = {
    'worker': ('worker',),
    'hardhat': ('hardhat',),
    'red_hardhat': ('red_hardhat',),
    'scaffolds': ('scaffolds', 'scaffold'),
    'hook': ('hook',),
}


def _adaptive_min_person_dim(h: int, w: int) -> int:
    return int(max(MIN_DIM_FRAC * min(h, w), MIN_DIM_PIX_FLOOR))


def _box_center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _hook_near_worker(worker_box, hook_box, margin_frac=0.35):
    """True if hook center lies inside worker box expanded by margin_frac of worker size."""
    wx1, wy1, wx2, wy2 = worker_box
    ww, wh = wx2 - wx1, wy2 - wy1
    mx, my = margin_frac * ww, margin_frac * wh
    ex1, ey1, ex2, ey2 = wx1 - mx, wy1 - my, wx2 + mx, wy2 + my
    hx, hy = _box_center(hook_box)
    return ex1 <= hx <= ex2 and ey1 <= hy <= ey2


def resolve_scaffold_class_ids(clss_dict: dict) -> dict:
    """Map scaffold role names to class IDs by looking up class.json values."""
    name_to_id = {v.lower(): int(k) for k, v in clss_dict.items()}
    class_ids = {}
    for role, aliases in SCAFFOLD_ROLE_NAMES.items():
        found = None
        for alias in aliases:
            if alias in name_to_id:
                found = name_to_id[alias]
                break
        if found is None:
            raise KeyError(
                f"Could not resolve scaffold role '{role}' from class.json. "
                f"Expected one of: {aliases}. Available: {list(clss_dict.values())}"
            )
        class_ids[role] = found
    return class_ids


def detect_scaffold(
    image,
    detections: List[dict],
    class_ids: dict,
    conf_threshold: float = 0.30,
    scaffold_min_conf: float = 0.50,
    skip_hook_rule: bool = False,
) -> Tuple[object, int, List[str]]:
    """
    Scaffold safety check from pre-computed NPU detections:
      1) All eligible workers must wear helmets   -> missing_helmet
      2) When scaffold is confidently present, each eligible worker needs a nearby hook
         (skipped when skip_hook_rule=True)
      3) No vertical up/down overlap on scaffold  -> same_vertical_area
    """
    image = image.copy()
    h, w = image.shape[:2]
    min_person_dim = _adaptive_min_person_dim(h, w)

    worker_cls = class_ids['worker']
    hat_cls = class_ids['hardhat']
    red_hat_cls = class_ids['red_hardhat']
    hook_cls = class_ids['hook']
    scaffold_cls = class_ids['scaffolds']

    person_boxes: List[List[int]] = []
    hat_boxes: List[List[int]] = []
    red_hat_boxes: List[List[int]] = []
    hook_boxes: List[List[int]] = []
    scaffold_confs: List[float] = []

    reasons: List[str] = []

    for det in detections:
        c = det['confidence']
        k = det['class_id']
        x1, y1, x2, y2 = det['box']
        if c < conf_threshold:
            continue

        if k == scaffold_cls:
            scaffold_confs.append(c)
        elif k == hat_cls:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            hat_boxes.append([x1, y1, x2, y2])
        elif k == red_hat_cls:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            red_hat_boxes.append([x1, y1, x2, y2])
        elif k == worker_cls:
            person_boxes.append([x1, y1, x2, y2])
        elif k == hook_cls:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 215, 255), 2)
            hook_boxes.append([x1, y1, x2, y2])

    has_scaffold = bool(scaffold_confs) and max(scaffold_confs) >= scaffold_min_conf
    all_hat_boxes: List[List[int]] = hat_boxes + red_hat_boxes
    eligible_workers: List[List[int]] = []

    for per_box in person_boxes:
        px1, py1, px2, py2 = per_box
        ph = py2 - py1
        head_band = max(20, int(0.15 * ph))

        if ph < min_person_dim:
            cv2.rectangle(image, (px1, py1), (px2, py2), (128, 128, 128), 2)
            cv2.putText(
                image, "Worker (too small, skipped)",
                (px1, max(0, py1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (128, 128, 128), 2, cv2.LINE_AA,
            )
            continue

        eligible_workers.append(per_box)

        hat_detected = any(
            per_box[0] <= (hat_box[0] + hat_box[2]) / 2 < per_box[2]
            and hat_box[1] >= per_box[1] - head_band
            for hat_box in all_hat_boxes
        )

        if hat_detected:
            cv2.rectangle(image, (per_box[0], per_box[1]), (per_box[2], per_box[3]), (0, 180, 0), 2)
            cv2.putText(
                image, "Worker with helmet",
                (per_box[0], max(0, per_box[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2, cv2.LINE_AA,
            )
        else:
            reasons.append("missing_helmet")
            cv2.rectangle(image, (per_box[0], per_box[1]), (per_box[2], per_box[3]), (0, 0, 255), 2)
            cv2.putText(
                image, "Worker without helmet",
                (per_box[0], max(0, per_box[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
            )

    if has_scaffold and eligible_workers:
        if not skip_hook_rule:
            workers_without_hook = 0
            for worker in eligible_workers:
                if not any(_hook_near_worker(worker, hook) for hook in hook_boxes):
                    workers_without_hook += 1

            if workers_without_hook > 0:
                reasons.append("missing_hook")
                cv2.putText(
                    image, "ALERT: Missing safety hook(s)",
                    (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
                )

        vertical_person = False
        for i, per_box1 in enumerate(eligible_workers):
            for j, per_box2 in enumerate(eligible_workers):
                if i == j:
                    continue
                if ((per_box1[1] + per_box1[3]) / 2) > per_box2[3] or (
                    (per_box2[1] + per_box2[3]) / 2) > per_box1[3]:
                    if (per_box1[0] - (per_box1[2] - per_box1[0]) / 2) < per_box2[2] and (
                        per_box1[2] + (per_box1[2] - per_box1[0]) / 2) > per_box2[0]:
                        vertical_person = True

        if vertical_person:
            reasons.append("same_vertical_area")
            cv2.putText(
                image, "ALERT: Upper/Lower simultaneous work",
                (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
            )

    helmet_violation = "missing_helmet" in reasons
    hook_violation = "missing_hook" in reasons
    vertical_violation = "same_vertical_area" in reasons

    all_safe = not (helmet_violation or hook_violation or vertical_violation)
    final_status = "safe" if all_safe else "unsafe"
    status_numeric = 1 if all_safe else 0

    color_status = (0, 120, 0) if all_safe else (0, 0, 255)
    cv2.putText(
        image, final_status, (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_status, 2, cv2.LINE_AA,
    )

    reasons = list(set(reasons))
    return image, status_numeric, reasons
