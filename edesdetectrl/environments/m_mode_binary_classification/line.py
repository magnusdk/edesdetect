from typing import Iterable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from edesdetectrl.util.pytrees import register_pytree_node_dataclass
from jax import random
from jax._src.random import KeyArray
from jax.tree_util import register_pytree_node_class, tree_flatten


class Point(NamedTuple):
    x: float
    y: float


@register_pytree_node_dataclass
class Line:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def rotation(self) -> float:
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        return jnp.where(dx != 0.0, jnp.arctan2(dx, dy), 0.0)

    @property
    def center(self) -> Point:
        return Point((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def magnitude(self) -> float:
        """The magnitude/length of the line"""
        return jnp.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)

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

    def rotate(self, theta: float) -> "Line":
        """Return a copy of self that is rotated around its center by theta radians."""
        cx = (self.x1 + self.x2) / 2
        cy = (self.y1 + self.y2) / 2
        # Translate line so that it's centered at (0, 0)
        tx1, ty1, tx2, ty2 = self.x1 - cx, self.y1 - cy, self.x2 - cx, self.y2 - cy

        cos_theta, sin_theta = jnp.cos(theta), jnp.sin(theta)
        # Return a new line that is rotated around (0, 0) and moved back to the original position
        return Line(
            (tx1 * cos_theta + ty1 * sin_theta) + cx,
            (-tx1 * sin_theta + ty1 * cos_theta) + cy,
            (tx2 * cos_theta + ty2 * sin_theta) + cx,
            (-tx2 * sin_theta + ty2 * cos_theta) + cy,
        )

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

    def as_points(self, n_points: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return a tuple of the x- and y-coordinates that represent the points along the line"""
        return (
            jnp.linspace(self.x1, self.x2, n_points),
            jnp.linspace(self.y1, self.y2, n_points),
        )

    @staticmethod
    def center_of_bounds(bounds: "Bounds", line_length: float) -> "Line":
        """Construct a Line that is centered in the bounds horizontally, points upwards, and has length line_length."""
        center_x = (bounds.max_x + bounds.min_x) / 2
        center_y = (bounds.max_y + bounds.min_y) / 2
        half_line_length = line_length / 2
        bottom = center_y - half_line_length
        top = center_y + half_line_length
        return Line(center_x, bottom, center_x, top)

    @staticmethod
    @jax.jit
    def random_within_bounds(
        key: KeyArray,
        bounds: "Bounds",
        line_length: float,
        attempts: int = 1000,
        pad_x=0,
        pad_y=0,
    ) -> "Line":
        """Construct a randomly positioned and randomly rotated Line that is within
        bounds. It does so by randomly generating a line until it generates a line that
        is within bounds.

        If no line can be generated within bounds in the given number of attempts, then
        return a line that is in the center of bounds."""

        def _random_within_bounds(key) -> Line:
            k1, k2, k3 = random.split(key, 3)
            # Generate random center
            cx = random.uniform(
                k1, minval=bounds.min_x + pad_x, maxval=bounds.max_x - pad_x
            )
            cy = random.uniform(
                k2, minval=bounds.min_y + pad_y, maxval=bounds.max_y - pad_y
            )
            # Generate random angle
            angle = random.uniform(k3, minval=-jnp.pi, maxval=jnp.pi)

            return Line(
                cx - jnp.cos(angle) * line_length / 2,
                cy - jnp.sin(angle) * line_length / 2,
                cx + jnp.cos(angle) * line_length / 2,
                cy + jnp.sin(angle) * line_length / 2,
            )

        def condition(state):
            return jnp.logical_not(
                jnp.logical_or(
                    bounds.is_within(state["line"]),
                    jnp.greater_equal(state["n"], attempts),
                )
            )

        def next_state(state):
            key, next_key = random.split(state["key"])
            line = _random_within_bounds(key)
            n = state["n"] + 1
            return {"line": line, "key": next_key, "n": n}

        initial_state = {
            "line": _random_within_bounds(key),
            "key": random.split(key, 1)[0],
            "n": 1,
        }

        result = jax.lax.while_loop(condition, next_state, initial_state)

        result_leaves, treedef = tree_flatten(result["line"])
        default_leaves, _ = tree_flatten(Line.center_of_bounds(bounds, line_length))
        return jax.tree_util.tree_unflatten(
            treedef,
            jnp.where(
                result["n"] < attempts,
                jnp.array(result_leaves),
                jnp.array(default_leaves),
            ),
        )


@register_pytree_node_dataclass
class Bounds:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def is_within(self, line: Line) -> bool:
        """Return True if the line is within the bounds, else False. min and max values are inclusive."""
        return jnp.all(
            jnp.array(
                [
                    jnp.less_equal(self.min_x, line.x1),
                    jnp.less(line.x1, self.max_x),
                    jnp.less_equal(self.min_x, line.x2),
                    jnp.less(line.x2, self.max_x),
                    jnp.less_equal(self.min_y, line.y1),
                    jnp.less(line.y1, self.max_y),
                    jnp.less_equal(self.min_y, line.y2),
                    jnp.less(line.y2, self.max_y),
                ]
            )
        )

    @staticmethod
    def from_shape(shape: Tuple[float, float]) -> "Bounds":
        width, height = shape
        return Bounds(0, 0, width, height)

    @property
    def shape(self) -> Tuple[float, float]:
        return self.max_x - self.min_x, self.max_y - self.min_y


@register_pytree_node_class
class MModeLine:
    def __init__(
        self,
        line: Line,
        bounds: Bounds,
        n_line_samples: int = -1,
    ):
        self.line = line
        self.bounds = bounds
        self.n_line_samples = jnp.where(
            n_line_samples == -1,
            jnp.ceil(line.magnitude).astype(int),
            n_line_samples,
        )

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
        video: jnp.ndarray,
        rotation: float = 0,
        vertical_translation: float = 0,
        horizontal_translation: float = 0,
    ) -> jnp.ndarray:
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
        X = jnp.clip(jnp.round(X + 1), 0, video.shape[1] - 1).astype(int)
        Y = jnp.clip(jnp.round(Y + 1), 0, video.shape[2] - 1).astype(int)

        video = jnp.pad(video, ((0, 0), (1, 1), (1, 1)))
        return video[:, Y, X]

    def get_mmode_image_with_adjacent(
        self,
        video: jnp.ndarray,
        *,
        rotations: Iterable[float] = [0],
        vertical_translations: Iterable[float] = [0],
        horizontal_translations: Iterable[float] = [0],
    ) -> jnp.ndarray:
        """Return an array with shape (R * VT * HT, N, L) where N is the number of frames in the video and L is the length of the M-mode line,
        R is the number of rotations to be included,
        VT is the number of vertical translations to be included,
        HT is the number of horizontal translations to be included."""
        return jnp.array(
            [
                self.get_mmode_image(video, rot, vt, ht)
                for rot in rotations
                for vt in vertical_translations
                for ht in horizontal_translations
            ]
        )

    def visualize_line(self):
        image = jnp.zeros(jnp.flip(jnp.array(self.bounds.shape), 0), dtype=jnp.float32)
        Y, X = self.line.as_points(self.n_line_samples)  # Video is height by width
        X, Y = X - self.bounds.min_y, Y - self.bounds.min_x
        return image.at[X.astype(int), Y.astype(int)].set(1.0)

    @staticmethod
    def from_shape(
        width: float,
        height: float,
        relative_line_length: float = 0.5,
        n_line_samples: Optional[int] = None,
    ) -> "MModeLine":
        """Construct an MModeLine where the bounds are determined by the video width and height and the line is centered horizontally in the bounds, and points upwards.

        The length of the line is by default half the length of the diagonal of the video. This can be changed by providing the relative_line_length argument."""
        video_diag_length = jnp.sqrt(width ** 2 + height ** 2)
        line_length = video_diag_length * relative_line_length
        bounds = Bounds.from_shape((width, height))
        line = Line.center_of_bounds(bounds, line_length)
        assert bounds.is_within(
            line
        ), "Line must be within the bounds of the video. Try adjusting relative_line_length."
        return MModeLine(line, bounds, n_line_samples)

    def tree_flatten(self):
        return ((self.line, self.bounds, self.n_line_samples), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
