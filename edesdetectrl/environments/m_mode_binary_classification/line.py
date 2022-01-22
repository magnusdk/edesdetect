from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass
class Line:
    x1: float
    y1: float
    x2: float
    y2: float

    def rotate(self, theta: float) -> "Line":
        """Return a copy of self that is rotated around its center by theta radians."""
        cx = (self.x1 + self.x2) / 2
        cy = (self.y1 + self.y2) / 2
        # Translate line so that it's centered at (0, 0)
        tx1, ty1, tx2, ty2 = self.x1 - cx, self.y1 - cy, self.x2 - cx, self.y2 - cy

        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        # Return a new line that is rotated around (0, 0) and moved back to the original position
        return Line(
            (tx1 * cos_theta + ty1 * sin_theta) + cx,
            (-tx1 * sin_theta + ty1 * cos_theta) + cy,
            (tx2 * cos_theta + ty2 * sin_theta) + cx,
            (-tx2 * sin_theta + ty2 * cos_theta) + cy,
        )

    @property
    def magnitude(self) -> float:
        """The magnitude/length of the line"""
        return np.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)

    @property
    def direction(self) -> Tuple[float, float]:
        """The direction that the line points towards."""
        return (
            self.x2 / self.magnitude - self.x1 / self.magnitude,
            self.y2 / self.magnitude - self.y1 / self.magnitude,
        )

    @property
    def perpendicular_direction(self) -> Tuple[float, float]:
        """The direction perpendicular to the line."""
        dx, dy = self.direction
        return dy, -dx

    def move_vertically(self, amount: float) -> "Line":
        """Return a new line that has been moved in the direction that self is pointing."""
        dx, dy = self.direction
        return Line(
            self.x1 + dx * amount,
            self.y1 + dy * amount,
            self.x2 + dx * amount,
            self.y2 + dy * amount,
        )

    def move_horizontally(self, amount: float) -> "Line":
        """Return a new line that has been moved perpendicular to the direction that self is pointing."""
        dx, dy = self.perpendicular_direction
        return Line(
            self.x1 + dx * amount,
            self.y1 + dy * amount,
            self.x2 + dx * amount,
            self.y2 + dy * amount,
        )

    def as_points(self, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return a tuple of the x- and y-coordinates that represent the points along the line"""
        return (
            np.linspace(self.x1, self.x2, n_points),
            np.linspace(self.y1, self.y2, n_points),
        )

    @staticmethod
    def center_of_bounds(bounds: "Bounds", line_length: float) -> "Line":
        """Construct a Line that is centered in the bounds horizontally, points upwards, and has length line_length."""
        half_width = (bounds.max_x - bounds.min_x) / 2
        half_height = (bounds.max_y - bounds.min_y) / 2
        half_line_length = line_length / 2
        bottom = half_height - half_line_length
        top = half_height + half_line_length
        return Line(half_width, bottom, half_width, top)


@dataclass
class Bounds:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def is_within(self, line: Line) -> bool:
        """Return True if the line is within the bounds, else False. min and max values are inclusive."""
        return (
            self.min_x <= line.x1 <= self.max_x
            and self.min_x <= line.x2 <= self.max_x
            and self.min_y <= line.y1 <= self.max_y
            and self.min_y <= line.y2 <= self.max_y
        )

    @staticmethod
    def from_shape(shape: Tuple[float, float]) -> "Bounds":
        width, height = shape
        return Bounds(0, 0, width, height)

    @property
    def shape(self) -> Tuple[float, float]:
        return self.max_x - self.min_x, self.max_y - self.min_y


class MModeLine:
    def __init__(
        self,
        line: Line,
        bounds: Bounds,
        n_line_samples: int = None,
    ):
        self.line = line
        self.bounds = bounds
        self.n_line_samples = n_line_samples or np.ceil(line.magnitude).astype(int)

    def rotate(self, theta: float):
        """Rotate line around its center if it would still be within bounds."""
        new_line = self.line.rotate(theta)
        if self.bounds.is_within(new_line):
            self.line = new_line

    def move_vertically(self, amount: float):
        """Move line in the direction that it points towards if it would still be within bounds."""
        new_line = self.line.move_vertically(amount)
        if self.bounds.is_within(new_line):
            self.line = new_line

    def move_horizontally(self, amount: float):
        """Move line in the direction perpendicular to itself if it would still be within bounds."""
        new_line = self.line.move_horizontally(amount)
        if self.bounds.is_within(new_line):
            self.line = new_line

    def get_mmode_image(
        self,
        video: np.ndarray,
        rotation: float = 0,
        vertical_translation: float = 0,
        horizontal_translation: float = 0,
    ) -> np.ndarray:
        """Return the M-mode image for the video.

        video has shape (N, W, H) where N is the number of frames and W and H is width and height, respectively.
        It would make sense that the bounds of self matches the width and height of the video, but this is not required.

        Optionally rotate or translate the line before creating the M-mode image."""
        interp_line = (
            self.line.rotate(rotation)
            .move_vertically(vertical_translation)
            .move_horizontally(horizontal_translation)
        )
        Y, X = interp_line.as_points(self.n_line_samples)
        X = np.clip(np.round(X + 1), 0, video.shape[1] - 1).astype(int)
        Y = np.clip(np.round(Y + 1), 0, video.shape[2] - 1).astype(int)

        video = np.pad(video, ((0, 0), (1, 1), (1, 1)))
        return video[:, Y, X]

    def get_mmode_image_with_adjacent(
        self,
        video: np.ndarray,
        *,
        rotations: Iterable[float] = [0],
        vertical_translations: Iterable[float] = [0],
        horizontal_translations: Iterable[float] = [0],
    ) -> np.ndarray:
        """Return an array with shape (R * VT * HT, N, L) where N is the number of frames in the video and L is the length of the M-mode line,
        R is the number of rotations to be included,
        VT is the number of vertical translations to be included,
        HT is the number of horizontal translations to be included."""
        return np.array(
            [
                self.get_mmode_image(video, rot, vt, ht)
                for rot in rotations
                for vt in vertical_translations
                for ht in horizontal_translations
            ]
        )

    def visualize_line(self):
        image = np.zeros(self.bounds.shape, dtype=float)
        Y, X = self.line.as_points(self.n_line_samples)  # Video is height by width
        image[X.astype(int), Y.astype(int)] = 1
        return image

    @staticmethod
    def from_shape(
        width: float,
        height: float,
        relative_line_length: float = 0.5,
    ) -> "MModeLine":
        """Construct an MModeLine where the bounds are determined by the video width and height and the line is centered horizontally in the bounds, and points upwards.

        The length of the line is by default half the length of the diagonal of the video. This can be changed by providing the relative_line_length argument."""
        video_diag_length = np.sqrt(width ** 2 + height ** 2)
        line_length = video_diag_length * relative_line_length
        bounds = Bounds.from_shape((width, height))
        line = Line.center_of_bounds(bounds, line_length)
        assert bounds.is_within(
            line
        ), "Line must be within the bounds of the video. Try adjusting relative_line_length."
        return MModeLine(line, bounds)
