"""Synthetic GGUF files exercise range limits and logical tensor shapes."""

import re
import struct

import httpx
import pytest

from s1mb.gguf_headers import CHUNK, MAX_HEADER, HeaderReader, gguf_shapes


def string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def fixture():
    data = b"GGUF" + struct.pack("<IQQ", 3, 2, 1)
    data += string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 2)
    data += string("one") + string("two")
    for name, shape in [("token_embd.weight", [4, 10]), ("blk.0.attn_q.weight", [4, 4])]:
        data += string(name) + struct.pack("<I", len(shape))
        data += struct.pack("<" + "Q" * len(shape), *shape) + struct.pack("<IQ", 8, 0)
    return data


def test_only_bounded_ranges_and_logical_shapes_are_read():
    data = fixture() + b"\0" * (2 * CHUNK)
    requests = []

    def response(request):
        requests.append(request)
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.headers["Range"])
        assert match is not None
        start, stop = map(int, match.groups())
        assert stop - start + 1 <= CHUNK
        return httpx.Response(206, content=data[start:stop + 1],
                              headers={"Content-Range": f"bytes {start}-{stop}/{len(data)}"})

    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        assert gguf_shapes("https://example.test/model.gguf", client=client) == {
            "token_embd.weight": [4, 10], "blk.0.attn_q.weight": [4, 4]}
    assert len(requests) == 1  # The following weight bytes are never fetched.


def test_server_ignoring_ranges_is_rejected_before_reading_weights():
    with (
        httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200))) as client,
        pytest.raises(ValueError, match="honor"),
    ):
        gguf_shapes("https://example.test/model.gguf", client=client)


def test_header_limit_and_malformed_format():
    reader = HeaderReader("unused", None)
    with pytest.raises(ValueError, match="limit"):
        reader.read(MAX_HEADER + 1)
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(
        206, content=b"BAD!" + b"\0" * 20, headers={"Content-Range": "bytes 0-23/24"}
    ))) as client, pytest.raises(ValueError, match="format"):
        gguf_shapes("https://example.test/model.gguf", client=client)
