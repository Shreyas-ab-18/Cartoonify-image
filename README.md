# Cartoonifying an Image

An image-processing pipeline built with Python and OpenCV that converts a
regular photo into a cartoon-style image.

## How it works
1. **Edge extraction** — the image is grayscaled, median-blurred to remove
   noise, then run through adaptive thresholding to produce bold, ink-like
   edges.
2. **Color smoothing** — repeated bilateral filtering flattens color regions
   into smooth patches while preserving edges better than a simple blur.
3. **Combine** — a bitwise AND of the smoothed color image and the edge mask
   overlays the black edges onto the flattened colors, producing the cartoon
   effect.

## Run it
```bash
pip install -r requirements.txt
python cartoonify.py input.jpg output.jpg
```

If no output path is given, the result is saved as `cartoon_output.jpg`.

## Tuning
`Cartoonifier.__init__` exposes parameters for the number of bilateral filter
passes and the adaptive-threshold block size/constant — increase
`num_bilateral` for a flatter, more "poster-like" look.
