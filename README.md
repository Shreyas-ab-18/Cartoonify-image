# Cartoonify Image

A command-line image-processing tool that creates a cartoon-style version of a photo using OpenCV.

## Processing pipeline
1. Convert input to grayscale and apply median blur.
2. Extract bold edges using adaptive thresholding.
3. Smooth color regions with bilateral filtering.
4. Combine the smoothed image and edge mask.

## Requirements and usage
Python 3.10+ is recommended.
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python cartoonify.py input.jpg output.jpg
```
If the output path is omitted, the result is saved as `cartoon_output.jpg`.

## Tests
```bash
pip install pytest
pytest
```
The script validates input paths, decoded image data, minimum image dimensions, processing parameters, and output-write success.
