from typing import Literal

import cv2
import numpy as np

RiskLevel = Literal["Low", "Medium", "High"]


def build_environmental_impact(disturbed_ratio: float) -> dict[str, float]:
    """Estimate environmental impact metrics from disturbed land ratio.

    Values are returned as percentage-like scores in the range [0, 100].
    """
    severity = max(0.0, min(disturbed_ratio / 0.3, 1.0))

    vegetation_loss = min(100.0, 12 + severity * 88)
    soil_erosion_risk = min(100.0, 8 + severity * 92)
    water_pollution_risk = min(100.0, 6 + severity * 90)
    habitat_damage = min(100.0, 10 + severity * 90)

    return {
        "vegetation_loss": round(vegetation_loss, 2),
        "soil_erosion_risk": round(soil_erosion_risk, 2),
        "water_pollution_risk": round(water_pollution_risk, 2),
        "habitat_damage": round(habitat_damage, 2),
    }


def detect_mining_regions(
    original_image: np.ndarray, processed_mask: np.ndarray
) -> tuple[int, float, np.ndarray]:
    """Find likely mining disturbance contours and draw targeted bounding boxes.

    Environmental filtering logic:
    - Vegetation/crop texture can create many small edge fragments -> ignore small contours.
    - Farmland tends to have more uniform, regular parcel-like patterns.
    - Mining scars are often irregular and expose bare soil.
    - Draw boxes only for sufficiently large, bare-soil dominant disturbed regions.
    """
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = original_image.copy()
    image_area = float(original_image.shape[0] * original_image.shape[1])
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    # Bare soil profile for exposed terrain commonly seen in open-pit disturbances.
    lower_soil = np.array([5, 30, 40])
    upper_soil = np.array([35, 255, 255])
    soil_pixels_global = cv2.inRange(hsv_image, lower_soil, upper_soil)

    disturbed_pixels = 0.0
    disturbed_regions = 0

    for contour in contours:
        contour_area = cv2.contourArea(contour)

        # Reject tiny noisy structures from forest canopy textures.
        if contour_area < 500:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w * h < 900:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        # Irregular-shape indicator: lower circularity => less likely to be natural tree blobs.
        circularity = (4 * np.pi * contour_area) / (perimeter * perimeter)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = contour_area / hull_area if hull_area > 0 else 0.0

        contour_mask = np.zeros(processed_mask.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        region_pixels = float(cv2.countNonZero(contour_mask))
        if region_pixels == 0:
            continue

        # Bare-soil dominance check to separate disturbance from vegetation/farmland remnants.
        soil_in_contour = cv2.bitwise_and(soil_pixels_global, contour_mask)
        soil_ratio = cv2.countNonZero(soil_in_contour) / region_pixels

        # Keep contours that are either clearly irregular or fractured, and soil-rich.
        if soil_ratio < 0.25:
            continue
        if circularity > 0.75 and solidity > 0.9:
            continue

        disturbed_regions += 1
        disturbed_pixels += contour_area

        # Draw boxes only for larger likely open-pit areas (avoid small residual clusters).
        if contour_area >= 1200 and soil_ratio >= 0.35:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

    disturbed_ratio = disturbed_pixels / image_area if image_area > 0 else 0.0

    return disturbed_regions, disturbed_ratio, output


def classify_risk(disturbed_ratio: float, region_count: int) -> tuple[bool, RiskLevel, float]:
    """Classify risk based on disturbed area and number of active regions."""
    # Explicit no-mining condition: negligible disturbance likely from normal land cover variation.
    if disturbed_ratio < 0.02:
        return False, "Low", 0.95

    mining_detected = True

    if disturbed_ratio < 0.05:
        risk = "Low"
        confidence = min(0.82, 0.55 + disturbed_ratio * 2.8 + region_count * 0.015)
    elif disturbed_ratio < 0.20:
        risk = "Medium"
        confidence = min(0.92, 0.68 + disturbed_ratio * 1.1 + region_count * 0.01)
    else:
        risk = "High"
        confidence = min(0.99, 0.82 + disturbed_ratio * 0.7 + region_count * 0.008)

    return mining_detected, risk, round(confidence, 2)
