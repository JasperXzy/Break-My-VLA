"""Encode live JPEGs and keep a bounded replay of simulated frames."""

from asyncio import to_thread
from base64 import b64encode
from collections import deque
from io import BytesIO

from PIL import Image

TARGET_FPS = 20
MAX_RECORDED_FRAMES = 100


def jpeg_b64(image: Image.Image) -> str:
    with BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=80)
        return b64encode(buffer.getvalue()).decode("ascii")


def encode_gif(frames: tuple[Image.Image, ...]) -> bytes:
    with BytesIO() as buffer:
        frames[0].save(
            buffer,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=1000 // TARGET_FPS,
            loop=0,
        )
        return buffer.getvalue()


class Recording:
    def __init__(self) -> None:
        self._frames: deque[Image.Image] = deque(maxlen=MAX_RECORDED_FRAMES)

    def add(self, image: Image.Image) -> None:
        self._frames.append(image.copy())

    def clear(self) -> None:
        self._frames.clear()

    async def export(self) -> bytes | None:
        if not self._frames:
            return None
        return await to_thread(encode_gif, tuple(self._frames))
