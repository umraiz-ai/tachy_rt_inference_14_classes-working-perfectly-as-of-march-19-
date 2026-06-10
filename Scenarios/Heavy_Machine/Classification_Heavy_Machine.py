# Heavy machine safe/unsafe classification from NPU object detections
#
# Refactored from the original Classification_Heavy_Machine.py (not TTA variant).
# Original used a separate SEG tag model for homography; on NPU we use a per-image
# default ground-plane homography fallback (same idea as TTA_Classification_HeavyMachine).
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

MIN_DIM_FRAC = 0.07
MIN_DIM_PIX_FLOOR = 64
DANGER_DIST_METERS = 2.0
TOPK_MACHINES = 3
REQUIRE_SIGNAL_ONLY_WITH_MACHINES = True

WORLD_W_M = 6.0
WORLD_H_M = 8.0

HEAVY_MACHINE_ROLE_NAMES = {
    'cement_truck': ('cement_truck',),
    'compactor': ('compactor',),
    'dump_truck': ('dump_truck',),
    'excavator': ('excavator',),
    'grader': ('grader',),
    'mobile_crane': ('mobile_crane',),
    'tower_crane': ('tower_crane',),
    'worker': ('worker',),
    'hardhat': ('hardhat',),
    'red_hardhat': ('red_hardhat',),
}

MACHINE_VIS = {
    'cement_truck': ('cement_truck', (0, 255, 255)),
    'compactor': ('compactor', (255, 0, 255)),
    'dump_truck': ('dump_truck', (0, 255, 0)),
    'excavator': ('excavator', (255, 255, 0)),
    'grader': ('grader', (200, 200, 0)),
    'mobile_crane': ('mobile_crane', (180, 0, 180)),
    'tower_crane': ('tower_crane', (255, 255, 255)),
}

_homography_matrix: Optional[np.ndarray] = None


def _adaptive_min_person_dim(h: int, w: int) -> int:
    return int(max(MIN_DIM_FRAC * min(h, w), MIN_DIM_PIX_FLOOR))


def resolve_heavy_machine_class_ids(clss_dict: dict) -> dict:
    """Map heavy-machine scenario roles to class IDs from class dict."""
    name_to_id = {v.lower(): int(k) for k, v in clss_dict.items()}
    class_ids = {}
    for role, aliases in HEAVY_MACHINE_ROLE_NAMES.items():
        found = None
        for alias in aliases:
            if alias in name_to_id:
                found = name_to_id[alias]
                break
        if found is None:
            raise KeyError(
                f"Could not resolve heavy-machine role '{role}' from class dict. "
                f"Expected one of: {aliases}. Available: {list(clss_dict.values())}"
            )
        class_ids[role] = found
    return class_ids


def set_homography(image_points, world_points_m) -> None:
    global _homography_matrix
    img = np.float32(image_points)
    wrd = np.float32(world_points_m)
    _homography_matrix = cv2.getPerspectiveTransform(img, wrd)


def _ensure_default_homography(image: np.ndarray) -> None:
    """Rough ground-plane homography from lower image region (NPU fallback, no SEG tag)."""
    global _homography_matrix
    h, w = image.shape[:2]
    img_pts = np.float32([
        (int(0.25 * w), int(0.75 * h)),
        (int(0.75 * w), int(0.75 * h)),
        (int(0.65 * w), int(0.55 * h)),
        (int(0.35 * w), int(0.55 * h)),
    ])
    wrd_pts = np.float32([
        (0.0, WORLD_H_M),
        (WORLD_W_M, WORLD_H_M),
        (WORLD_W_M, 0.0),
        (0.0, 0.0),
    ])
    _homography_matrix = cv2.getPerspectiveTransform(img_pts, wrd_pts)


def _to_world_xy(point_xy: Tuple[float, float]) -> np.ndarray:
    assert _homography_matrix is not None and _homography_matrix.shape == (3, 3)
    pt = np.array([[[point_xy[0], point_xy[1]]]], dtype=np.float32)
    return cv2.perspectiveTransform(pt, _homography_matrix)[0, 0]


def _bottom_center(box: List[int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, y2)


def _lower_half(box: List[int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    ymid = (y1 + y2) // 2
    return x1, ymid, x2, y2


def _closest_point_on_lower_half(px: int, py: int, vbox: List[int]) -> Tuple[int, int]:
    x1, ymid, x2, y2 = _lower_half(vbox)
    cx = min(max(px, x1), x2)
    cy = min(max(py, ymid), y2)
    return (cx, cy)


def _euclid_px(p: Tuple[int, int], q: Tuple[int, int]) -> float:
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))


def _worker_wears_red_hat(per_box, red_hat_boxes, head_band_frac=0.25) -> bool:
    """Looser red-hardhat-on-worker check than standard helmet matching."""
    ph = per_box[3] - per_box[1]
    head_band = max(30, int(head_band_frac * ph))
    for rb in red_hat_boxes:
        hx = (rb[0] + rb[2]) / 2
        if per_box[0] <= hx < per_box[2] and rb[1] >= per_box[1] - head_band:
            return True
    return False


def _worker_near_machine(w_foot: Tuple[int, int], v_box: List[int], margin_frac: float = 0.25) -> bool:
    """True if worker foot is inside machine bbox expanded by margin_frac."""
    x1, y1, x2, y2 = v_box
    ww, wh = x2 - x1, y2 - y1
    mx, my = margin_frac * ww, margin_frac * wh
    return (x1 - mx) <= w_foot[0] <= (x2 + mx) and (y1 - my) <= w_foot[1] <= (y2 + my)


def _workers_near_machines(eligible_workers, vehicle_boxes, margin_frac: float = 0.25) -> bool:
    if not eligible_workers or not vehicle_boxes:
        return False
    for w_box in eligible_workers:
        w_foot = _bottom_center(w_box)
        if any(_worker_near_machine(w_foot, v_box, margin_frac) for v_box in vehicle_boxes):
            return True
    return False


def detect_heavy_machine(
    image,
    detections: List[dict],
    class_ids: dict,
    conf_threshold: float = 0.30,
    danger_dist_meters: float = DANGER_DIST_METERS,
    use_default_homography: bool = True,
    require_signal_with_machines: bool = REQUIRE_SIGNAL_ONLY_WITH_MACHINES,
    skip_signal_man_rule: bool = False,
) -> Tuple[object, int, List[str]]:
    """
    Heavy machine safety check from NPU detections:
      1) Eligible workers must wear hardhat or red_hardhat (signal man) -> missing_helmet
      2) When machines present and workers are near machines, require a signal man
         (eligible worker with red_hardhat, loose match) -> no_signal_man
      3) Worker-to-machine distance < danger_dist_meters -> proximity_violation
    """
    global _homography_matrix
    image = image.copy()
    _homography_matrix = None

    h, w = image.shape[:2]
    min_person_dim = _adaptive_min_person_dim(h, w)
    reasons: List[str] = []

    worker_cls = class_ids['worker']
    hat_cls = class_ids['hardhat']
    red_hat_cls = class_ids['red_hardhat']

    person_boxes: List[List[int]] = []
    hat_boxes: List[List[int]] = []
    red_hat_boxes: List[List[int]] = []
    machine_boxes: Dict[str, List[List[int]]] = {
        role: [] for role in MACHINE_VIS if role in class_ids
    }

    machine_cls_to_role = {
        class_ids[role]: role for role in MACHINE_VIS if role in class_ids
    }

    for det in detections:
        c = det['confidence']
        k = det['class_id']
        x1, y1, x2, y2 = det['box']
        if c < conf_threshold:
            continue

        if k == hat_cls:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            hat_boxes.append([x1, y1, x2, y2])
        elif k == red_hat_cls:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            red_hat_boxes.append([x1, y1, x2, y2])
        elif k == worker_cls:
            person_boxes.append([x1, y1, x2, y2])
        elif k in machine_cls_to_role:
            role = machine_cls_to_role[k]
            label, color = MACHINE_VIS[role]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image, label, (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
            )
            machine_boxes[role].append([x1, y1, x2, y2])

    vehicle_boxes_all: List[List[int]] = []
    for boxes in machine_boxes.values():
        vehicle_boxes_all.extend(boxes)

    eligible_workers: List[List[int]] = []
    signal_man_on_worker = False

    def _is_near_machine(worker_box: List[int]) -> bool:
        if not vehicle_boxes_all:
            return False
        w_foot = _bottom_center(worker_box)
        return any(_worker_near_machine(w_foot, v_box) for v_box in vehicle_boxes_all)

    # Helmet / signal-man per worker (helmet required only when worker is near equipment)
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
            per_box[0] <= (hb[0] + hb[2]) / 2 < per_box[2]
            and hb[1] >= per_box[1] - head_band
            for hb in hat_boxes
        )
        red_hat_detected = any(
            per_box[0] <= (rb[0] + rb[2]) / 2 < per_box[2]
            and rb[1] >= per_box[1] - head_band
            for rb in red_hat_boxes
        )

        if hat_detected:
            cv2.rectangle(image, (per_box[0], per_box[1]), (per_box[2], per_box[3]), (0, 180, 0), 2)
            cv2.putText(
                image, "Worker with helmet",
                (per_box[0], max(0, per_box[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2, cv2.LINE_AA,
            )
        elif red_hat_detected:
            signal_man_on_worker = True
            cv2.rectangle(image, (per_box[0], per_box[1]), (per_box[2], per_box[3]), (0, 180, 0), 2)
            cv2.putText(
                image, "Signal Man",
                (per_box[0], max(0, per_box[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2, cv2.LINE_AA,
            )
        else:
            near_machine = _is_near_machine(per_box)
            if not vehicle_boxes_all or near_machine:
                reasons.append("missing_helmet")
                cv2.rectangle(image, (per_box[0], per_box[1]), (per_box[2], per_box[3]), (0, 0, 255), 2)
                cv2.putText(
                    image, "Worker without helmet",
                    (per_box[0], max(0, per_box[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
                )
            else:
                cv2.rectangle(image, (per_box[0], per_box[1]), (per_box[2], per_box[3]), (0, 165, 255), 2)
                cv2.putText(
                    image, "Worker (far from machine)",
                    (per_box[0], max(0, per_box[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA,
                )

    machines_present = len(vehicle_boxes_all) > 0
    signalman_present = signal_man_on_worker or any(
        _worker_wears_red_hat(w, red_hat_boxes) for w in eligible_workers
    )
    workers_near_machines = _workers_near_machines(eligible_workers, vehicle_boxes_all)

    if not skip_signal_man_rule:
        # Relaxed: require signal man only when machines present AND a worker is near equipment
        if require_signal_with_machines:
            need_signal_check = machines_present and workers_near_machines
        else:
            need_signal_check = workers_near_machines

        if need_signal_check and not signalman_present:
            reasons.append("no_signal_man")
            cv2.putText(
                image, "ALERT: No Signal Man",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
            )

    if use_default_homography:
        _ensure_default_homography(image)
        cv2.putText(
            image, "Homography: default ground plane",
            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 170, 255), 2, cv2.LINE_AA,
        )
    elif _homography_matrix is None:
        cv2.putText(
            image, "Proximity disabled (no homography)",
            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 170, 255), 2, cv2.LINE_AA,
        )

    if _homography_matrix is not None and vehicle_boxes_all:
        for w_box in person_boxes:
            if (w_box[3] - w_box[1]) < min_person_dim:
                continue

            w_foot = _bottom_center(w_box)
            try:
                w_world = _to_world_xy(w_foot)
            except Exception:
                continue

            px_dists = []
            lh_points = []
            for v_box in vehicle_boxes_all:
                q_px = _closest_point_on_lower_half(w_foot[0], w_foot[1], v_box)
                px_dists.append(_euclid_px(w_foot, q_px))
                lh_points.append(q_px)

            if not px_dists:
                continue

            order = np.argsort(px_dists)[:min(TOPK_MACHINES, len(px_dists))]
            best_m = None
            best_q = None
            for idx in order:
                q_px = lh_points[idx]
                try:
                    q_world = _to_world_xy(q_px)
                except Exception:
                    continue
                d_m = float(np.linalg.norm(w_world - q_world))
                if best_m is None or d_m < best_m:
                    best_m = d_m
                    best_q = q_px

            if best_m is None:
                continue

            color = (0, 0, 255) if best_m < danger_dist_meters else (0, 255, 0)
            cv2.line(image, w_foot, best_q, color, 2)
            mid_pt = ((w_foot[0] + best_q[0]) // 2, (w_foot[1] + best_q[1]) // 2)
            cv2.putText(
                image, f"{best_m:.2f}m", mid_pt,
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA,
            )

            if best_m < danger_dist_meters:
                if "proximity_violation" not in reasons:
                    reasons.append("proximity_violation")
                cv2.putText(
                    image, f"ALERT: {best_m:.2f}m",
                    (w_box[0], max(0, w_box[1] - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
                )

    helmet_violation = "missing_helmet" in reasons
    signalman_violation = "no_signal_man" in reasons
    proximity_violation = "proximity_violation" in reasons
    all_safe = not (helmet_violation or signalman_violation or proximity_violation)
    final_status = "safe" if all_safe else "unsafe"
    status_numeric = 1 if all_safe else 0

    color_status = (0, 120, 0) if all_safe else (0, 0, 255)
    cv2.putText(
        image, final_status, (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_status, 2, cv2.LINE_AA,
    )

    reasons = list(set(reasons))
    return image, status_numeric, reasons
