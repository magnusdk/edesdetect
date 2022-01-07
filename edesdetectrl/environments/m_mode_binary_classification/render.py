import numpy as np
from scipy import interpolate


def gray_to_rgb(arr_2d):
    # Just stack the image three times.
    return np.tile(np.expand_dims(arr_2d * 255, axis=2).astype(int), (3))


def render_observation(
    overview_data: np.ndarray,
    mmode_image_data: np.ndarray,
):
    # Render overview data (mean image overlaid with red line)
    mean_image, line = overview_data
    mean_image = gray_to_rgb(mean_image)
    mean_image[line == 1] = [255, 0, 0]  # Make line red

    mmode_image_data = mmode_image_data.reshape((3, 3) + mmode_image_data.shape[1:])

    mmode_image_data /= 2
    minv = mmode_image_data.min()
    mmode_image_data -= minv
    mmode_image_data[1, 1] += minv
    mmode_image_data[1, 1] *= 2

    mmode_image_data = np.concatenate(mmode_image_data, axis=2)
    mmode_image_data = np.concatenate(mmode_image_data, axis=0)

    f = interpolate.interp2d(
        np.arange(mmode_image_data.shape[0]),
        np.arange(mmode_image_data.shape[1]),
        mmode_image_data.T,
    )
    mmode_image_data = f(
        np.linspace(0, mmode_image_data.shape[0], mean_image.shape[0]*2),
        np.linspace(0, mmode_image_data.shape[1], mean_image.shape[1]),
    )
    mmode_image_data = gray_to_rgb(mmode_image_data)
    return np.concatenate([mean_image, mmode_image_data], axis=1)
