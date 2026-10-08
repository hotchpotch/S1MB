"""Read bounded GGUF headers through HTTP ranges, without downloading weights."""

import math
import struct
from typing import Any

import httpx

MAX_HEADER = 64 * 1024 * 1024
CHUNK = 1024 * 1024


class HeaderReader:
    def __init__(self, url: str, client: Any):
        self.url, self.client = url, client
        self.data = bytearray()
        self.offset = 0

    def read(self, size: int) -> bytes:
        end = self.offset + size
        if size < 0 or end > MAX_HEADER:
            raise ValueError("GGUF header exceeds bounded read limit")
        while len(self.data) < end:
            start = len(self.data)
            stop = min(MAX_HEADER, start + CHUNK) - 1
            with self.client.stream("GET", self.url, headers={
                "Range": f"bytes={start}-{stop}", "Accept-Encoding": "identity",
            }, follow_redirects=True) as response:
                response.raise_for_status()
                if response.status_code != 206 or not response.headers.get(
                    "Content-Range", ""
                ).startswith(f"bytes {start}-"):
                    raise ValueError("Server did not honor bounded GGUF range request")
                chunk = bytearray()
                for part in response.iter_bytes():
                    chunk.extend(part)
                    if len(chunk) > CHUNK:
                        raise ValueError("GGUF range response exceeds requested size")
                if not chunk:
                    raise ValueError("Truncated GGUF header")
                self.data.extend(chunk)
        value = bytes(self.data[self.offset:end])
        self.offset = end
        return value

    def integer(self, fmt: str) -> int:
        return struct.unpack("<" + fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        size = self.integer("Q")
        return self.read(size).decode("utf-8")

    def skip_value(self, kind: int) -> None:
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                 10: 8, 11: 8, 12: 8}
        if kind in sizes:
            self.read(sizes[kind])
        elif kind == 8:
            self.read(self.integer("Q"))
        elif kind == 9:
            element, length = self.integer("I"), self.integer("Q")
            if element in sizes:
                self.read(length * sizes[element])
            elif element == 8 and length <= MAX_HEADER // 8:
                for _ in range(length):
                    self.skip_value(element)
            else:
                raise ValueError("Unsupported GGUF metadata array")
        else:
            raise ValueError("Unsupported GGUF metadata type")


def gguf_shapes(url: str, *, client: Any = None) -> dict[str, list[int]]:
    """Parse logical shapes, never quantized storage byte counts."""
    if client is None:
        with httpx.Client(timeout=30) as owned:
            return gguf_shapes(url, client=owned)
    reader = HeaderReader(url, client)
    if reader.read(4) != b"GGUF" or reader.integer("I") not in {2, 3}:
        raise ValueError("Unsupported GGUF format")
    tensor_count, metadata_count = reader.integer("Q"), reader.integer("Q")
    if not 0 < tensor_count <= 100000 or metadata_count > 100000:
        raise ValueError("Invalid GGUF inventory size")
    for _ in range(metadata_count):
        reader.string()
        reader.skip_value(reader.integer("I"))
    shapes = {}
    for _ in range(tensor_count):
        name, dimensions = reader.string(), reader.integer("I")
        if name in shapes or not 1 <= dimensions <= 4:
            raise ValueError("Invalid or duplicate GGUF tensor")
        shape = [reader.integer("Q") for _ in range(dimensions)]
        if not math.prod(shape) or any(d < 1 for d in shape):
            raise ValueError("Invalid GGUF tensor dimensions")
        reader.read(12)  # Quantization type and tensor data offset.
        shapes[name] = shape
    return shapes
