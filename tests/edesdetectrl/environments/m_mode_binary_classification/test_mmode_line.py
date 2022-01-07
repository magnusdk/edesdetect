import random
from typing import Callable

import numpy as np
import numpy.testing as np_testing
from edesdetectrl.environments.m_mode_binary_classification.line import (
    Bounds,
    Line,
    MModeLine,
    interpolate_points,
)


def random_step(mmode_line: MModeLine):
    """Perform a random action."""
    step = random.choice(["rotate", "move_vertically", "move_horizontally"])
    amount = random.random() * 5
    if step == "rotate":
        mmode_line.rotate(amount)
    elif step == "move_vertically":
        mmode_line.move_vertically(amount)
    elif step == "move_horizontally":
        mmode_line.move_horizontally(amount)


def random_steps(
    mmode_line: MModeLine,
    invariant: Callable[[MModeLine], bool],
    num=1000,
):
    """Perform num random actions and assert invariant."""
    assert invariant(mmode_line)
    for _ in range(num):
        random_step(mmode_line)
        assert invariant(mmode_line)


def test_mmode_line_invariants():
    """Assert that no matter what actions are taken then:
    - The line remains within bounds at all times
    - An M-mode image can be created without crashing (ex. IndexError)"""
    video = np.ones((32, 10, 12))  # 32 frames, 10 by 12 pixels video

    def line_always_within_bounds(mmode_line: MModeLine):
        return mmode_line.bounds.is_within(mmode_line.line)

    def mmode_image_doesnt_crash(mmode_line: MModeLine):
        random_rotation = random.random() * 5
        random_vertical_translation = random.random() * 5
        random_horizontal_translation = random.random() * 5
        try:
            mmode_line.get_mmode_image(
                video,
                random_rotation,
                random_vertical_translation,
                random_horizontal_translation,
            )
            return True
        except:
            return False

    mmode_line = MModeLine.from_shape(video.shape[1], video.shape[2])
    random_steps(mmode_line, line_always_within_bounds)
    random_steps(mmode_line, mmode_image_doesnt_crash)


def test_interpolate_points():
    """Assert that interpolate_points gets approximately the same result as just
    naively indexing the on the rounded coordinates."""
    video_frame = np.random.rand(100, 120) * 100
    bounds = Bounds.from_shape(video_frame.shape)
    n_points = 50

    summed_difference = np.zeros((n_points))
    n_iterations = 1000
    for _ in range(n_iterations):
        line = Line.center_of_bounds(bounds, 80).rotate(random.random() * np.pi * 2)
        X, Y = line.as_points(n_points)
        naively_interpolated = video_frame[
            np.round(X).astype(int), np.round(Y).astype(int)
        ]
        true_interpolated = interpolate_points(video_frame, X, Y)
        summed_difference += naively_interpolated - true_interpolated

    np_testing.assert_allclose(
        summed_difference / n_iterations, np.zeros_like(summed_difference)
    )


def test_mmode_adjacent():
    video = np.random.random((32, 10, 12))  # 32 frames, 10 by 12 pixels video
    mmode_line = MModeLine.from_shape(video.shape[1], video.shape[2])

    mmode_image_with_adjacent = mmode_line.get_mmode_image_with_adjacent(
        video,
        rotations=[-1, 0, 1],
        vertical_translations=[-1, 0, 1],
        horizontal_translations=[-1, 0, 1],
    )
    np_testing.assert_array_equal(
        mmode_image_with_adjacent[0],
        mmode_line.get_mmode_image(video, -1, -1, -1),
    )
    np_testing.assert_array_equal(
        mmode_image_with_adjacent[1],
        mmode_line.get_mmode_image(video, -1, -1, 0),
    )
    np_testing.assert_array_equal(
        mmode_image_with_adjacent[2],
        mmode_line.get_mmode_image(video, -1, -1, 1),
    )
    np_testing.assert_array_equal(
        mmode_image_with_adjacent[3],
        mmode_line.get_mmode_image(video, -1, 0, -1),
    )

    np_testing.assert_array_equal(
        mmode_image_with_adjacent[9],
        mmode_line.get_mmode_image(video, 0, -1, -1),
    )
    np_testing.assert_array_equal(
        mmode_image_with_adjacent[-1],
        mmode_line.get_mmode_image(video, 1, 1, 1),
    )
    np_testing.assert_array_equal(
        mmode_image_with_adjacent[(3 * 3 * 3) // 2],
        mmode_line.get_mmode_image(video, 0, 0, 0),
    )

    only_rotation = mmode_line.get_mmode_image_with_adjacent(
        video, rotations=[-1, 0, 1]
    )
    assert len(only_rotation) == 3

    only_rotation_and_2_translations = mmode_line.get_mmode_image_with_adjacent(
        video, rotations=[-1, 0, 1], horizontal_translations=[-2, 2]
    )
    assert len(only_rotation_and_2_translations) == 3 * 2
