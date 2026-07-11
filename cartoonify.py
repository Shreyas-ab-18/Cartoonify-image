"""
Cartoonifying an Image
------------------------
Transforms a photo into a cartoon-style image using an OpenCV pipeline:
edge detection (adaptive threshold) + bilateral filtering for color smoothing,
then combines the two so flat color regions are preserved but bounded by
bold black edges.

Usage:
    python cartoonify.py input.jpg [output.jpg]
"""

import sys
import cv2
import numpy as np


class Cartoonifier:
    def __init__(self, num_bilateral=7, edge_blur_ksize=7, edge_block_size=9, edge_c=2):
        self.num_bilateral = num_bilateral
        self.edge_blur_ksize = edge_blur_ksize
        self.edge_block_size = edge_block_size
        self.edge_c = edge_c

    def _extract_edges(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, self.edge_blur_ksize)
        edges = cv2.adaptiveThreshold(
            gray_blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
            self.edge_block_size, self.edge_c,
        )
        return edges

    def _smooth_colors(self, img_bgr):
        color = img_bgr
        for _ in range(self.num_bilateral):
            color = cv2.bilateralFilter(color, d=9, sigmaColor=75, sigmaSpace=75)
        return color

    def cartoonify(self, img_bgr):
        edges = self._extract_edges(img_bgr)
        color = self._smooth_colors(img_bgr)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_and(color, edges_bgr)


def main():
    if len(sys.argv) < 2:
        print("Usage: python cartoonify.py <input_image> [output_image]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "cartoon_output.jpg"

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at '{input_path}'")

    cartoonifier = Cartoonifier()
    result = cartoonifier.cartoonify(img)

    cv2.imwrite(output_path, result)
    print(f"Cartoonified image saved to '{output_path}'")


if __name__ == "__main__":
    main()
