import numpy as np
import pytest
from cartoonify import Cartoonifier


def test_cartoonify_returns_same_dimensions():
    image = np.full((80,80,3),180,dtype=np.uint8)
    result = Cartoonifier(num_bilateral=1).cartoonify(image)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_cartoonifier_rejects_even_block_size():
    with pytest.raises(ValueError):
        Cartoonifier(edge_block_size=8)


def test_cartoonifier_rejects_empty_image():
    with pytest.raises(ValueError):
        Cartoonifier().cartoonify(np.array([]))


def test_cartoonifier_rejects_images_smaller_than_processing_windows():
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="at least"):
        Cartoonifier().cartoonify(image)
