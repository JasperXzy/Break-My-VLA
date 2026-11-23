"""A small rendered tabletop task with gripper, block, and contact dynamics."""

from collections.abc import Callable
from math import dist

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from backend.chaos_sim.env import IMAGE_SIZE, Observation, Vector3
from backend.shared.schemas import Action

TABLE_Z = 0.045
LIFT_Z = 0.50
WALL = 0.65
LOWER = (-WALL, -WALL, TABLE_Z)
UPPER = (WALL, WALL, 0.72)
Pixel = tuple[int, int]
Project = Callable[[float, float, float], Pixel]


def project_main(x: float, y: float, z: float) -> Pixel:
    return round(160 + 112 * x + 52 * y), round(155 + 44 * y - 25 * x - 128 * z)


def table_corners(project: Project) -> list[Pixel]:
    return [
        project(-WALL, -WALL, 0),
        project(WALL, -WALL, 0),
        project(WALL, WALL, 0),
        project(-WALL, WALL, 0),
    ]


def draw_gripper(draw: ImageDraw.ImageDraw, center: Pixel, *, closed: bool, scale: int) -> None:
    x, y = center
    gap = scale // 2 if closed else scale
    draw.line((x - gap, y - scale, x + gap, y - scale), fill="#dbe8ee", width=4)
    for finger in (x - gap, x + gap):
        draw.line((finger, y - scale, finger, y + scale // 2), fill="#f3fafc", width=3)


def draw_block(draw: ImageDraw.ImageDraw, center: Pixel, radius: int) -> None:
    x, y = center
    draw.rectangle(
        (x - radius, y - radius, x + radius, y + radius),
        fill="#f08a3d",
        outline="#663723",
        width=2,
    )
    draw.line((x - radius + 2, y - radius + 2, x + radius - 2, y - radius + 2), fill="#ffc380")


class CpuPickEnv:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._hand = np.array([-0.43, 0.23, 0.43])
        self._block = np.array([0.22, -0.18, TABLE_Z])
        self._hand_impulse = np.zeros(3)
        self._block_velocity = np.zeros(3)
        self._closed = False
        self._grasped = False
        self._start_distance = dist(self._hand, self._block)

    def nudge_object(self, offset: Vector3) -> None:
        self._grasped = False
        self._block = np.clip(self._block + offset, LOWER, UPPER)
        self._block_velocity = np.array([0.45 * offset[0], 0.45 * offset[1], 0.0])

    def kick_arm(self, impulse: Vector3) -> None:
        self._hand_impulse = np.clip(self._hand_impulse + impulse, -0.5, 0.5)

    def step(self, action: Action, *, friction: float) -> None:
        previous_hand = self._hand
        motion = 0.075 * np.clip(action[:3], -1.0, 1.0)
        self._hand = np.clip(self._hand + motion + self._hand_impulse, LOWER, UPPER)
        self._hand_impulse *= 0.45
        displacement = self._hand - previous_hand
        self._closed = action[6] > 0.0

        if self._grasped and not self._closed:
            self._block_velocity = displacement.copy()
        gap = dist(self._hand, self._block)
        self._grasped = self._closed and (self._grasped or gap < 0.095)
        if self._grasped:
            self._block = self._hand.copy()
            self._block_velocity.fill(0.0)
            return

        if gap < 0.095:
            self._block_velocity[:2] += displacement[:2] * 0.32
        self._block_velocity[2] -= 0.026
        next_block = self._block + self._block_velocity
        self._block = np.clip(next_block, LOWER, UPPER)
        self._block_velocity[self._block != next_block] = 0.0
        if self._block[2] <= TABLE_Z:
            self._block_velocity[:2] *= 0.6**friction

    def observe(self) -> Observation:
        if self._grasped:
            lift = (self._block[2] - TABLE_Z) / (LIFT_Z - TABLE_Z)
            progress = 0.55 + 0.45 * min(float(lift), 1.0)
        else:
            approach = dist(self._hand, self._block) / self._start_distance
            progress = 0.55 * (1.0 - min(approach, 1.0))

        return Observation(
            rgb_main=self._render_main(),
            rgb_wrist=self._render_wrist(),
            proprio=(*self._hand.tolist(), *self._block.tolist(), float(self._grasped)),
            task_progress=progress,
        )

    def _render_main(self) -> Image.Image:
        image = Image.new("RGB", IMAGE_SIZE, "#263940")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 118, 319, 239), fill="#192a31")
        rim = table_corners(project_main)
        for i in (2, 3):
            x, y = rim[i]
            draw.line((x, y, x, y + 29), fill="#34454a", width=9)
        draw.polygon(rim, fill="#728788", outline="#bac7c4", width=2)
        for coordinate in (-0.4, -0.2, 0.0, 0.2, 0.4):
            draw.line(
                [project_main(coordinate, -WALL, 0), project_main(coordinate, WALL, 0)],
                fill="#839798",
            )
            draw.line(
                [project_main(-WALL, coordinate, 0), project_main(WALL, coordinate, 0)],
                fill="#839798",
            )

        base = project_main(-0.55, 0.29, 0.02)
        self._draw_shadow(draw, self._block, radius=10)
        self._draw_shadow(draw, self._hand, radius=7)
        draw.ellipse((base[0] - 15, base[1] - 7, base[0] + 15, base[1] + 7), fill="#3a4e59")
        shoulder = project_main(-0.55, 0.29, 0.30)
        elbow = project_main(self._hand[0] - 0.19, self._hand[1] + 0.14, self._hand[2] + 0.14)
        hand = project_main(*self._hand)
        arm = [base, shoulder, elbow, (hand[0], hand[1] - 9)]
        draw.line(arm, fill="#2d4e61", width=15, joint="curve")
        draw.line(arm, fill="#74a9be", width=10, joint="curve")
        for x, y in (shoulder, elbow):
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#dbe8ee", outline="#2d4e61")
        draw_block(draw, project_main(*self._block), radius=8)
        draw_gripper(draw, hand, closed=self._closed, scale=11)
        return image

    @staticmethod
    def _draw_shadow(draw: ImageDraw.ImageDraw, position: NDArray[np.float64], radius: int) -> None:
        x, y = project_main(position[0], position[1], 0)
        draw.ellipse((x - radius, y - 4, x + radius, y + 4), fill="#566c70")

    def _render_wrist(self) -> Image.Image:
        image = Image.new("RGB", IMAGE_SIZE, "#263940")
        draw = ImageDraw.Draw(image)
        scale = 170 / (0.50 + self._hand[2])

        def project(x: float, y: float, z: float) -> Pixel:
            return round(160 + scale * (x - self._hand[0])), round(
                120 + scale * (y - self._hand[1])
            )

        draw.polygon(table_corners(project), fill="#829597", outline="#bccac9", width=3)
        for coordinate in (-0.4, -0.2, 0.0, 0.2, 0.4):
            draw.line([project(coordinate, -WALL, 0), project(coordinate, WALL, 0)], fill="#98a9aa")
            draw.line([project(-WALL, coordinate, 0), project(WALL, coordinate, 0)], fill="#98a9aa")
        draw_block(draw, project(*self._block), radius=round(0.045 * scale))
        draw_gripper(draw, (160, 120), closed=self._closed, scale=15)
        return image
