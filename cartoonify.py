"""Convert a photo to a cartoon-style image using OpenCV."""
import argparse
from pathlib import Path
import cv2


class Cartoonifier:
    def __init__(self, num_bilateral=7, edge_blur_ksize=7, edge_block_size=9, edge_c=2):
        if num_bilateral < 1: raise ValueError("num_bilateral must be at least 1.")
        if edge_blur_ksize < 3 or edge_blur_ksize % 2 == 0: raise ValueError("edge_blur_ksize must be odd and >= 3.")
        if edge_block_size < 3 or edge_block_size % 2 == 0: raise ValueError("edge_block_size must be odd and >= 3.")
        self.num_bilateral, self.edge_blur_ksize = num_bilateral, edge_blur_ksize
        self.edge_block_size, self.edge_c = edge_block_size, edge_c

    def _extract_edges(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, self.edge_blur_ksize)
        return cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,self.edge_block_size,self.edge_c)

    def _smooth_colors(self, image):
        color = image
        for _ in range(self.num_bilateral):
            color = cv2.bilateralFilter(color,9,75,75)
        return color

    def cartoonify(self, image):
        if image is None or image.size == 0: raise ValueError("Input image is empty or invalid.")
        if image.ndim != 3 or image.shape[2] != 3: raise ValueError("Input must be a three-channel BGR image.")
        edges = self._extract_edges(image)
        return cv2.bitwise_and(self._smooth_colors(image),cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR))


def main():
    parser = argparse.ArgumentParser(description="Create a cartoon-style image.")
    parser.add_argument("input",help="Path to input image.")
    parser.add_argument("output",nargs="?",default="cartoon_output.jpg",help="Output path.")
    args = parser.parse_args()
    source,target = Path(args.input),Path(args.output)
    if not source.is_file(): parser.error(f"Input file does not exist: {source}")
    image = cv2.imread(str(source),cv2.IMREAD_COLOR)
    if image is None: parser.error(f"OpenCV could not decode: {source}")
    if not cv2.imwrite(str(target),Cartoonifier().cartoonify(image)): parser.error(f"Could not write output: {target}")
    print(f"Cartoonified image saved to '{target}'")


if __name__ == "__main__":
    main()
