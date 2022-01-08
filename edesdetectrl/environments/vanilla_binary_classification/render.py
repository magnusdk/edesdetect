import numpy as np


def gray_to_rgb(arr_2d):
    # Just stack the image three times.
    return np.tile(np.expand_dims(arr_2d * 255, axis=2).astype(int), (3))


def render_observation(frames):
    """Return the frames tiled horizontally."""
    tiled_frames = np.concatenate(frames, axis=1)
    tiled_frames -= tiled_frames.min()
    tiled_frames /= tiled_frames.max()
    return gray_to_rgb(tiled_frames)
