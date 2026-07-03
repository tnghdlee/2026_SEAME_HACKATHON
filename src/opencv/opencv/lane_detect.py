"""Rule-based lane offset estimation (CLAUDE.md 8.1 TODO — lane following).

Design (see the design note in inference_node.compute_control):
    Hough two-line fitting is fragile on dashed lines, shadows and sharp
    curves. For a completion-first hackathon we instead use a *band-centroid*
    method: split a bottom ROI into a few horizontal bands, binarize the lane
    pixels, and take the horizontal centroid of lane mass in each band. The
    lateral error is how far that centroid sits from image center; comparing a
    near band to a far band gives a cheap curvature (feed-forward) estimate.

This module is pure (no ROS) so it can be unit-tested and tuned offline.

Returns a LaneResult with:
    offset    normalized lateral error in [-1, 1]; <0 = lane center is to the
              LEFT of the image center (car should steer left), >0 = to the
              right. Sign→steering mapping is applied downstream (steer_sign).
    valid     True if enough lane pixels were found to trust `offset`.
    curvature far_band_offset - near_band_offset, ~ upcoming bend (can be 0).
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LaneResult:
    offset: float
    valid: bool
    curvature: float


# Polarity: are lane markings brighter or darker than the road surface?
POLARITY_LIGHT = 'light'  # bright tape/paint on a darker floor
POLARITY_DARK = 'dark'    # dark lines on a lighter floor


def _binarize_lane(gray, polarity):
    """Adaptive threshold → binary mask where 255 = lane pixel.

    Adaptive (not global Otsu) so uneven lighting / shadows across the ROI do
    not wipe out one side of the lane.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh_type = (cv2.THRESH_BINARY if polarity == POLARITY_LIGHT
                   else cv2.THRESH_BINARY_INV)
    mask = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        thresh_type,
        blockSize=25,
        C=-10 if polarity == POLARITY_LIGHT else 10,
    )
    return mask


def compute_lane_offset(
    image_bgr,
    roi_top=50,
    roi_left=0,
    num_bands=3,
    valid_min_px=40,
    polarity=POLARITY_LIGHT,
):
    """Estimate normalized lateral lane offset from a BGR frame.

    Parameters mirror ROS params on opencv_node so tuning is done from config,
    not code (CLAUDE.md 9.4). See module docstring for the returned fields.
    """
    if image_bgr is None or image_bgr.size == 0:
        return LaneResult(0.0, False, 0.0)

    h, w = image_bgr.shape[:2]
    roi_top = int(max(0, min(roi_top, h - 1)))
    roi_left = int(max(0, min(roi_left, w - 1)))

    roi = image_bgr[roi_top:h, roi_left:w]
    if roi.size == 0:
        return LaneResult(0.0, False, 0.0)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mask = _binarize_lane(gray, polarity)

    roi_h, roi_w = mask.shape[:2]
    half_w = roi_w / 2.0
    num_bands = max(1, int(num_bands))
    band_h = max(1, roi_h // num_bands)

    # Column index vector, reused per band for the mass-weighted centroid.
    cols = np.arange(roi_w, dtype=np.float32)

    band_offsets = []  # (band_index, normalized_offset); index 0 = nearest (bottom)
    for i in range(num_bands):
        # Band 0 is the bottom-most (nearest) slice of the ROI.
        y1 = roi_h - (i + 1) * band_h
        y2 = roi_h - i * band_h
        y1 = max(0, y1)
        band = mask[y1:y2, :]

        col_mass = band.sum(axis=0).astype(np.float32) / 255.0
        total = float(col_mass.sum())
        if total < valid_min_px:
            continue

        centroid_x = float((cols * col_mass).sum() / total)
        band_offsets.append((i, (centroid_x - half_w) / half_w))

    if not band_offsets:
        return LaneResult(0.0, False, 0.0)

    # Weight nearer bands more (they reflect current position); farther bands
    # act as light feed-forward. Weight = num_bands - band_index.
    weight_sum = 0.0
    offset_acc = 0.0
    near_off = None
    far_off = None
    for band_index, off in band_offsets:
        weight = float(num_bands - band_index)
        offset_acc += weight * off
        weight_sum += weight
        if band_index == band_offsets[0][0]:
            near_off = off
        far_off = off  # last kept band = farthest valid

    offset = offset_acc / weight_sum if weight_sum > 0 else 0.0
    curvature = (far_off - near_off) if (near_off is not None and far_off is not None) else 0.0

    offset = float(np.clip(offset, -1.0, 1.0))
    return LaneResult(offset, True, float(curvature))
