import numpy as np
import numpy.testing as np_testing
from edesdetectrl.environments.m_mode_binary_classification.line import Bounds, Line


def test_line_center_of_bounds():
    bounds = Bounds.from_shape((4, 10))

    line = Line.center_of_bounds(bounds, 5)
    assert line.magnitude == 5
    assert line.direction == (0, 1)
    assert line.perpendicular_direction == (1, 0)

    line = Line.center_of_bounds(bounds, 10)
    assert line.magnitude == 10
    assert line.direction == (0, 1)
    assert line.perpendicular_direction == (1, 0)
    np_testing.assert_allclose(
        [line.x1, line.y1, line.x2, line.y2],
        [2, 0, 2, 10],
    )


def test_rotate_line():
    line = Line(0, 0, 0, 1)
    assert line.direction == (0, 1)

    line = line.rotate(np.pi / 2)
    np_testing.assert_allclose(line.direction, [1, 0], atol=1e-10)
    line = line.rotate(np.pi / 2)
    np_testing.assert_allclose(line.direction, [0, -1], atol=1e-10)
    line = line.rotate(np.pi * 2)
    np_testing.assert_allclose(line.direction, [0, -1], atol=1e-10)


def test_move_line():
    line = Line(0, 0, 0, 1)
    assert line.direction == (0, 1)
    line = line.move_vertically(1)
    assert (line.x1, line.y1, line.x2, line.y2) == (0, 1, 0, 2)
    line = line.move_horizontally(1)
    assert (line.x1, line.y1, line.x2, line.y2) == (1, 1, 1, 2)


def test_rotate_and_move_line():
    line = Line(0, 0, 0, 1)
    assert line.direction == (0, 1)
    initial_magnitude = line.magnitude

    line = (
        line.rotate(np.pi / 4)  # Rotate 45 degrees
        .move_vertically(1)  # Move one unit in 45-degree direction
        .rotate(-np.pi / 4)  # Rotate -45 degrees
    )

    exp_t = np.sin(np.pi / 4)  # Expected translation in both x- and y-direction
    np_testing.assert_allclose(
        [line.x1, line.y1, line.x2, line.y2],
        [exp_t, exp_t, exp_t, exp_t + 1],
    )
    np_testing.assert_allclose(line.magnitude, initial_magnitude)


def test_bounds_shape():
    bounds = Bounds.from_shape((4, 10))
    assert bounds.shape == (4, 10)


def test_moving_line_within_bounds():
    bounds = Bounds.from_shape((10, 10))
    line = Line(5, 0, 5, 10)  # Horizontally centered in bounds
    assert bounds.is_within(line)
    assert not bounds.is_within(line.move_vertically(0.1))
    assert not bounds.is_within(line.move_vertically(-0.1))
    assert bounds.is_within(line.move_horizontally(5))
    assert bounds.is_within(line.move_horizontally(-5))
    assert not bounds.is_within(line.move_horizontally(5.1))
    assert not bounds.is_within(line.move_horizontally(-5.1))
