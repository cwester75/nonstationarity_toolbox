import numpy as np
from nonlinear.applications.image_dynamics import arnold_cat_map, baker_map_image, recurrence_period


def test_arnold_cat_map_preserves_shape():
    img = np.random.randint(0, 256, (8, 8))
    result = arnold_cat_map(img, n=1)
    assert result.shape == img.shape


def test_arnold_cat_map_recurrence():
    # Arnold cat map on NxN grid has finite recurrence period
    img = np.arange(16).reshape(4, 4)
    period = recurrence_period(img, arnold_cat_map, max_iter=100)
    assert period is not None
    assert period > 0


def test_baker_map_image_preserves_shape():
    img = np.random.randint(0, 256, (8, 8))
    result = baker_map_image(img, n=1)
    assert result.shape == img.shape
