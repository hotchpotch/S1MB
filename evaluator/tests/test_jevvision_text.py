"""Text-only JEVision uses its published native text route without changing inputs."""

from unittest.mock import Mock

from test_upstream_adapters import cases

from s1mb.adapters.jevvision import JEVisionAdapter
from s1mb.adapters.kev import KevAdapter


def test_text_request_reaches_native_text_branch_unchanged():
    case = cases()
    received = []
    sentinel = object()

    backend = Mock(spec=KevAdapter)
    backend.predict.side_effect = lambda request: (received.append(request), sentinel)[1]

    adapter = object.__new__(JEVisionAdapter)
    adapter.text_adapter = backend
    assert adapter.predict(case) is sentinel
    assert received == [case]
    assert not received[0].images
    assert received[0].questions == case.questions
