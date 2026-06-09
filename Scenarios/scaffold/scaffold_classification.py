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
) -> Tuple[object, int, List[str]]:
    """
    Scaffold safety check from pre-computed NPU detections:
      1) All workers must wear helmets   -> else missing_helmet
      2) Safety hooks must be fastened   -> else missing_hook
      3) No vertical up/down overlap     -> same_vertical_area

    AND gate: if ANY violation occurs, image is unsafe.

    detections: list of {class_id, confidence, box: [x1,y1,x2,y2]}
  class_ids: from resolve_scaffold_class_ids()
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

    reasons: List[str] = []
    has_scaffold = False

    for det in detections:
        c = det['confidence']
        k = det['class_id']
        x1, y1, x2, y2 = det['box']
        if c < conf_threshold:
            continue

        if k == scaffold_cls:
            has_scaffold = True
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

    all_hat_boxes: List[List[int]] = hat_boxes + red_hat_boxes

    for per_box in person_boxes:
        px1, py1, px2, py2 = per_box
        ph = py2 - py1

        if ph < min_person_dim:
            cv2.rectangle(image, (px1, py1), (px2, py2), (128, 128, 128), 2)
            cv2.putText(
                image, "Worker (too small, skipped)",
                (px1, max(0, py1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (128, 128, 128), 2, cv2.LINE_AA,
            )
            continue

        hat_detected = any(
            per_box[0] <= (hat_box[0] + hat_box[2]) / 2 < per_box[2]
            and hat_box[1] >= per_box[1] - 20
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

    if has_scaffold and person_boxes:
        missing_hooks = max(0, len(person_boxes) - len(hook_boxes))
        if missing_hooks > 0:
            reasons.append("missing_hook")
            cv2.putText(
                image, "ALERT: Missing safety hook(s)",
                (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
            )

        vertical_person = False
        for i, per_box1 in enumerate(person_boxes):
            for j, per_box2 in enumerate(person_boxes):
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
