import cv2
import numpy as np


def load_grayscale(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")

    return image


def load_binary_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Unable to read mask: {path}")

    return (mask > 0).astype(np.uint8)


def normalize_image(image):
    image = image.astype(np.float32)
    return (image - image.min()) / (image.max() - image.min() + 1e-8)


def _kernel_from_mask(mask, factor=0.04, minimum=3):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return np.ones((minimum, minimum), dtype=np.uint8)

    width = xs.max() - xs.min() + 1
    height = ys.max() - ys.min() + 1
    size = int(max(minimum, round(min(width, height) * factor)))

    if size % 2 == 0:
        size += 1

    return np.ones((size, size), dtype=np.uint8)


def get_inner_mask(mask, factor=0.06):
    kernel = _kernel_from_mask(mask, factor=factor)
    inner = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return (inner > 0).astype(np.uint8)


def get_cortex_band(mask, inner_mask):
    cortex = mask.astype(np.uint8) - inner_mask.astype(np.uint8)
    cortex[cortex < 0] = 0
    return cortex


def get_reference_band(mask, factor=0.10):
    kernel = _kernel_from_mask(mask, factor=factor)
    outer = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    band = outer - mask.astype(np.uint8)
    band[band < 0] = 0
    return band


def masked_values(image, mask):
    values = image[mask > 0]

    if values.size == 0:
        return np.array([], dtype=np.float32)

    return values.astype(np.float32)


def intensity_features(image, mask, prefix):
    values = masked_values(image, mask)

    if values.size == 0:
        return {
            f"{prefix}_area": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_p10": np.nan,
            f"{prefix}_p50": np.nan,
            f"{prefix}_p90": np.nan,
        }

    return {
        f"{prefix}_area": int(values.size),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_p10": float(np.percentile(values, 10)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
    }


def bright_spot_features(image, mask, prefix, percentile=90):
    values = masked_values(image, mask)

    if values.size == 0:
        return {
            f"{prefix}_bright_threshold": np.nan,
            f"{prefix}_bright_ratio": np.nan,
            f"{prefix}_bright_components": 0,
            f"{prefix}_bright_mean_area": np.nan,
        }

    threshold = np.percentile(values, percentile)
    bright_mask = ((image >= threshold) & (mask > 0)).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bright_mask)

    component_areas = []
    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area > 0:
            component_areas.append(area)

    return {
        f"{prefix}_bright_threshold": float(threshold),
        f"{prefix}_bright_ratio": float(bright_mask.sum() / (mask.sum() + 1e-8)),
        f"{prefix}_bright_components": int(len(component_areas)),
        f"{prefix}_bright_mean_area": float(np.mean(component_areas)) if component_areas else 0.0,
    }


def _glcm_matrix(image, mask, levels=16):
    quantized = np.clip((image * (levels - 1)).astype(np.int32), 0, levels - 1)
    valid = mask > 0

    glcm = np.zeros((levels, levels), dtype=np.float64)
    directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]

    for dy, dx in directions:
        y_from = max(0, -dy)
        y_to = image.shape[0] - max(0, dy)
        x_from = max(0, -dx)
        x_to = image.shape[1] - max(0, dx)

        a = quantized[y_from:y_to, x_from:x_to]
        b = quantized[y_from + dy:y_to + dy, x_from + dx:x_to + dx]
        va = valid[y_from:y_to, x_from:x_to]
        vb = valid[y_from + dy:y_to + dy, x_from + dx:x_to + dx]
        joint = va & vb

        if not np.any(joint):
            continue

        a_vals = a[joint]
        b_vals = b[joint]

        for i, j in zip(a_vals, b_vals):
            glcm[i, j] += 1
            glcm[j, i] += 1

    total = glcm.sum()
    if total > 0:
        glcm /= total

    return glcm


def texture_features(image, mask, prefix):
    if mask.sum() == 0:
        return {
            f"{prefix}_glcm_contrast": np.nan,
            f"{prefix}_glcm_homogeneity": np.nan,
            f"{prefix}_glcm_energy": np.nan,
            f"{prefix}_glcm_entropy": np.nan,
        }

    glcm = _glcm_matrix(image, mask)
    levels = glcm.shape[0]
    i, j = np.indices((levels, levels))

    contrast = np.sum(glcm * ((i - j) ** 2))
    homogeneity = np.sum(glcm / (1.0 + np.abs(i - j)))
    energy = np.sum(glcm ** 2)
    entropy = -np.sum(glcm * np.log2(glcm + 1e-12))

    return {
        f"{prefix}_glcm_contrast": float(contrast),
        f"{prefix}_glcm_homogeneity": float(homogeneity),
        f"{prefix}_glcm_energy": float(energy),
        f"{prefix}_glcm_entropy": float(entropy),
    }


def ratio_features(numerator_name, numerator_value, denominator_name, denominator_value):
    if np.isnan(numerator_value) or np.isnan(denominator_value):
        ratio = np.nan
    else:
        ratio = float(numerator_value / (denominator_value + 1e-8))

    return {
        f"{numerator_name}_to_{denominator_name}_ratio": ratio
    }


def heuristic_pyramid_mask(image, kidney_mask):
    inner_mask = get_inner_mask(kidney_mask, factor=0.08)

    if inner_mask.sum() == 0:
        return inner_mask

    enhanced = cv2.GaussianBlur(image, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply((enhanced * 255).astype(np.uint8))
    enhanced = enhanced.astype(np.float32) / 255.0

    values = masked_values(enhanced, inner_mask)
    if values.size == 0:
        return inner_mask * 0

    # Heuristic: renal pyramids often appear as darker structures inside the kidney.
    threshold = np.percentile(values, 35)
    candidate = ((enhanced <= threshold) & (inner_mask > 0)).astype(np.uint8)

    kernel = _kernel_from_mask(kidney_mask, factor=0.03)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate)
    filtered = np.zeros_like(candidate)

    min_area = max(10, int(inner_mask.sum() * 0.005))
    max_area = max(min_area, int(inner_mask.sum() * 0.20))

    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            filtered[labels == idx] = 1

    return filtered.astype(np.uint8)


def extract_renal_features(image, kidney_mask, reference_mask=None):
    image = normalize_image(image)
    kidney_mask = (kidney_mask > 0).astype(np.uint8)

    inner_mask = get_inner_mask(kidney_mask)
    cortex_mask = get_cortex_band(kidney_mask, inner_mask)
    external_reference_mask = get_reference_band(kidney_mask)
    provided_reference_mask = None

    if reference_mask is not None:
        provided_reference_mask = (reference_mask > 0).astype(np.uint8)
        provided_reference_mask[kidney_mask > 0] = 0

    effective_reference_mask = (
        provided_reference_mask
        if provided_reference_mask is not None and provided_reference_mask.sum() > 0
        else external_reference_mask
    )

    pyramid_mask = heuristic_pyramid_mask(image, kidney_mask)

    features = {}
    features.update(intensity_features(image, kidney_mask, "kidney"))
    features.update(intensity_features(image, inner_mask, "inner"))
    features.update(intensity_features(image, cortex_mask, "cortex"))
    features.update(intensity_features(image, effective_reference_mask, "reference"))
    features.update(intensity_features(image, external_reference_mask, "external_reference"))
    if provided_reference_mask is not None:
        features.update(intensity_features(image, provided_reference_mask, "provided_reference"))
    features.update(intensity_features(image, pyramid_mask, "pyramid_candidate"))

    features.update(bright_spot_features(image, kidney_mask, "kidney"))
    features.update(bright_spot_features(image, inner_mask, "inner"))
    features.update(bright_spot_features(image, pyramid_mask, "pyramid_candidate"))

    features.update(texture_features(image, kidney_mask, "kidney"))
    features.update(texture_features(image, inner_mask, "inner"))
    features.update(texture_features(image, pyramid_mask, "pyramid_candidate"))

    features.update(ratio_features("kidney_mean", features["kidney_mean"], "reference_mean", features["reference_mean"]))
    features.update(ratio_features("inner_mean", features["inner_mean"], "reference_mean", features["reference_mean"]))
    features.update(ratio_features("cortex_mean", features["cortex_mean"], "inner_mean", features["inner_mean"]))
    features.update(ratio_features("pyramid_candidate_mean", features["pyramid_candidate_mean"], "cortex_mean", features["cortex_mean"]))

    if provided_reference_mask is not None:
        features.update(
            ratio_features(
                "kidney_mean",
                features["kidney_mean"],
                "provided_reference_mean",
                features["provided_reference_mean"],
            )
        )
        features.update(
            ratio_features(
                "inner_mean",
                features["inner_mean"],
                "provided_reference_mean",
                features["provided_reference_mean"],
            )
        )

    features["kidney_bbox_fill_ratio"] = float(
        kidney_mask.sum() / (cv2.boundingRect(kidney_mask)[2] * cv2.boundingRect(kidney_mask)[3] + 1e-8)
    ) if kidney_mask.sum() > 0 else np.nan

    features["pyramid_candidate_ratio"] = float(
        pyramid_mask.sum() / (inner_mask.sum() + 1e-8)
    ) if inner_mask.sum() > 0 else np.nan

    features["reference_source"] = (
        "provided_mask"
        if provided_reference_mask is not None and provided_reference_mask.sum() > 0
        else "external_band"
    )

    return features, {
        "inner_mask": inner_mask,
        "cortex_mask": cortex_mask,
        "reference_mask": effective_reference_mask,
        "external_reference_mask": external_reference_mask,
        "provided_reference_mask": provided_reference_mask if provided_reference_mask is not None else np.zeros_like(kidney_mask),
        "pyramid_candidate_mask": pyramid_mask,
    }
