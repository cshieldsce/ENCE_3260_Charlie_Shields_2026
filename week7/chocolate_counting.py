import sys

import cv2
import numpy as np

IMAGE_FILE = "img_3.jpg"
OUTPUT_FILE = "img_3_annotated.png"

def segment_chocolates(gray):
    """Return a binary mask where chocolates are white (255) and background is black (0)."""

    # Blur makes thresholding more stable (reduces noise / small texture).
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu automatically chooses a threshold value.
    # We use THRESH_BINARY_INV because chocolates are darker than the background.
    threshold_value, mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    # Morphology cleans up the mask (removes speckles, fills small holes).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask, float(threshold_value)


def find_chocolates(gray, mask):
    """Find chocolate blobs and return a list of (cx, cy, contour)."""

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    image_area = float(h * w)

    chocolates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 0:
            continue

        # Filter by size (reject tiny noise and huge regions).
        if area < 0.001 * image_area or area > 0.20 * image_area:
            continue

        # Bounding-box shape check (reject extremely skinny shapes).
        x, y, bw, bh = cv2.boundingRect(contour)
        if bh == 0:
            continue
        aspect_ratio = bw / float(bh)
        if aspect_ratio < 0.35 or aspect_ratio > 3.0:
            continue

        # Mask the inside of the contour so we can measure intensity.
        region_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(region_mask, [contour], -1, 255, thickness=-1)
        mean, stddev = cv2.meanStdDev(gray, mask=region_mask)
        mean_val = float(mean[0][0])
        std_val = float(stddev[0][0])

        # Chocolates are dark-ish and not super textured.
        if mean_val > 140:
            continue
        if std_val > 70:
            continue

        # Centroid (for labeling).
        m = cv2.moments(contour)
        if m["m00"] == 0:
            continue
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])

        # Keep detections in the tray area (avoid top label / bottom border artifacts).
        if cy < int(0.27 * h) or cy > int(0.82 * h):
            continue

        chocolates.append((cx, cy, contour))

    # Stable numbering: left-to-right, then top-to-bottom.
    chocolates.sort(key=lambda item: (item[0], item[1]))
    return chocolates


def draw_labels(image, chocolates):
    """Draw contour, center dot, and number label for each chocolate."""

    annotated = image.copy()
    for i, (cx, cy, contour) in enumerate(chocolates, start=1):
        cv2.drawContours(annotated, [contour], -1, (0, 255, 0), 2)
        cv2.circle(annotated, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(
            annotated,
            str(i),
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def main():
    show_windows = "--no-show" not in sys.argv
    debug = "--debug" in sys.argv

    # Load image (BGR).
    img = cv2.imread(IMAGE_FILE)
    if img is None:
        print(f"ERROR: Could not read {IMAGE_FILE}")
        return 1

    # 1) Grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) Segment chocolates.
    mask, threshold_value = segment_chocolates(gray)

    # 3) Find + count.
    chocolates = find_chocolates(gray, mask)

    # 4) Annotate.
    annotated = draw_labels(img, chocolates)
    cv2.imwrite(OUTPUT_FILE, annotated)

    # 5) Print count.
    print(f"Otsu threshold: {threshold_value:.1f}")
    print(f"Detected chocolates: {len(chocolates)}")

    if show_windows:
        cv2.imshow("Original", img)
        if debug:
            cv2.imshow("Mask (foreground=chocolate)", mask)
        cv2.imshow("Annotated", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
