"""Native branch admission must precede GEV inference."""

import asyncio
from types import SimpleNamespace

import pytest

from s1mb.adapters.gev import GEVAdapter


def test_overflow_precedes_native_forward():
    adapter = GEVAdapter.__new__(GEVAdapter)
    adapter.tokenizer = SimpleNamespace(bos_token_id=2, encode=lambda *args, **kw: [1] * 64)
    adapter.limit = 64
    with pytest.raises(ValueError, match="Complete GEV branch"):
        asyncio.run(adapter._read(None, "choice", "state", False, "Choose", ["One", "Two"]))
