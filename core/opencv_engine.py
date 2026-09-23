"""
core/opencv_engine.py
----------------------
Baseline restoration engine using classical OpenCV image processing.

Restoration sequence (per project spec): White Balance -> CLAHE.

1. White Balance (Gray-World assumption): underwater images lose red
   wavelengths fastest with depth, producing a blue/green color cast.
   Gray-World assumes a natural scene's average color should be neutral
   gray, so each channel is rescaled until its mean matches the overall
   gray mean -- this is what actually corrects the color cast.
2. CLAHE in LAB space: restores local contrast lost to turbidity/
   backscatter, applied only to the L (lightness) channel so it doesn't
   re-introduce a color shift on top of what White Balance just fixed.

White Balance must run first: CLAHE boosting contrast on a still-blue
image would just produce a more contrasty blue image.
"""

import cv2
import numpy as np


def _white_balance(img: np.ndarray) -> np.ndarray:
    """Gray-World white balance to correct underwater blue/green color cast."""
    img_float = img.astype(np.float32)
    b, g, r = cv2.split(img_float)

    b_mean, g_mean, r_mean = b.mean(), g.mean(), r.mean()
    gray_mean = (b_mean + g_mean + r_mean) / 3.0

    # max(..., 1e-6) avoids a divide-by-zero on a pathological all-black frame.
    b_gain = gray_mean / max(b_mean, 1e-6)
    g_gain = gray_mean / max(g_mean, 1e-6)
    r_gain = gray_mean / max(r_mean, 1e-6)

    b = np.clip(b * b_gain, 0, 255)
    g = np.clip(g * g_gain, 0, 255)
    r = np.clip(r * r_gain, 0, 255)

    return cv2.merge([b, g, r]).astype(np.uint8)


def _apply_clahe(img: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel in LAB space to fix low contrast from turbidity."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    merged = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def restore_image_opencv(image_bytes: bytes) -> bytes:
    """
    Accepts raw uploaded image bytes, applies White Balance then CLAHE,
    and returns the restored image as raw JPEG bytes.

    Raises:
        ValueError: if the bytes don't decode as a valid image, or if
                    re-encoding the result fails. main.py is expected to
                    catch this and translate it into a 400 response --
                    it means the client sent something, just not an image.
    """
    # 1. Decode
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(
            "Failed to decode the uploaded image. Ensure the file is a valid JPEG/PNG."
        )

    # 2. Process: White Balance -> CLAHE, in that order (see module docstring).
    balanced = _white_balance(img)
    restored_img = _apply_clahe(balanced)

    # 3. Encode
    success, buffer = cv2.imencode(".jpg", restored_img)
    if not success:
        raise ValueError("Failed to encode the restored image.")

    return buffer.tobytes()