# Week 7 — Chocolate Counting (OpenCV)

## What it does
- Loads `img_3.jpg`
- Converts to grayscale
- Segments chocolates (dark objects) using Otsu thresholding + morphology
- Filters blobs by size/shape and expected tray region
- Labels each detected chocolate and prints the final count

## Output
- Terminal prints the threshold used and `Detected chocolates: N`
- Annotated image is saved as `img_3_annotated.png`
