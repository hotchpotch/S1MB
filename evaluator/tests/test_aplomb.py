"""Aplomb boundary checks without downloaded weights or GPU inference."""
import base64
import io
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import s1mb.adapters.aplomb as module


@pytest.mark.parametrize('image_count', [0, 3])
def test_all_images_types_numeric_values_and_anonymous_mapping(monkeypatch, image_count):
    from PIL import Image
    from test_upstream_adapters import cases

    case = cases()
    images = [Image.new('RGB', (4, 4), c) for c in ['red', 'blue', 'green']][:image_count]
    state = {'context': case.state, 'visual_inputs': {str(i): f'<image:{i}>' for i in range(image_count)}}
    monkeypatch.setattr(module, 'ordered_images', lambda _: (images, state, []))
    adapter = object.__new__(module.AplombAdapter)
    adapter.ad = object()
    adapter.calibration = {'choice': 1.255, 'noul': 1.2614, 'score': 1.4073}
    adapter.image_receipts = {}
    adapter.torch = ModuleType('torch')
    monkeypatch.setattr(adapter.torch, 'inference_mode', nullcontext, raising=False)

    def decide(ad, body, temps, **kwargs):
        assert ad is adapter.ad and temps == adapter.calibration
        assert kwargs == {'debias': True, 'batch': 1, 'abstain_bias': 0.0}
        assert 'gold-must-not-be-sent' not in str(body)
        questions = list(body['questions'].values())
        assert questions[1]['criteria'] == {'true': 'Yes', 'false': 'No'}
        assert [v['value'] for v in questions[2]['criteria']] == [10, 30]
        if image_count:
            payloads = list(body['state']['visual_inputs'].values())
            assert len(payloads) == image_count
            pixels = []
            for payload in payloads:
                assert payload['type'] == 'image' and payload['mime'] == 'image/png'
                with Image.open(io.BytesIO(base64.b64decode(payload['data']))) as image:
                    pixels.append(image.getpixel((0, 0)))
            assert pixels == [(255, 0, 0), (0, 0, 255), (0, 128, 0)]
        else:
            assert body['state'] == case.state
        return {'answers': dict(zip([q.id for q in case.questions], [
            {'type': 'choice', 'probabilities': {'option_0': 0.3, 'option_1': 0.7}},
            {'type': 'noul', 'noul': 0.8},
            {'type': 'score', 'probabilities': {'0': 0.2, '1': 0.8}},
        ], strict=True))}

    adapter.native = SimpleNamespace(decide=decide)
    result = adapter.predict(case)
    assert result[0].probabilities == {'x': 0.3, 'y': 0.7}
    assert result[1].probabilities['true'] == 0.8
    assert result[2].probabilities == {'low': 0.2, 'high': 0.8}


def test_expanded_image_tokens_reject_overflow_before_forward():
    adapter = object.__new__(module.AplombAdapter)
    adapter.context_limit = 10
    adapter.original_encode = lambda *a: SimpleNamespace(batch={
        'input_ids': SimpleNamespace(shape=(1, 10))})
    with pytest.raises(ValueError, match='refusing truncation'):
        adapter.checked_encode([])


@pytest.mark.parametrize('limit', [0, -1, True, 1.5])
def test_invalid_limit_rejected_before_loading(limit):
    with pytest.raises(ValueError, match='positive integer'):
        module.AplombAdapter('model', 'revision', 'missing', 'cuda:0', context_limit=limit)


def test_cli_dispatch_preserves_pinned_source_and_declared_context(monkeypatch, tmp_path):
    from s1mb import cli

    received = []
    adapter = SimpleNamespace(close=lambda: None)

    def construct(*args, **kwargs):
        received.append((args, kwargs))
        return adapter

    def evaluate(actual, *args, **kwargs):
        assert actual is adapter
        return SimpleNamespace(counts=SimpleNamespace(failed=0), status='complete',
                               metrics={}, elapsed_seconds=0, environment={})

    monkeypatch.setattr(module, 'AplombAdapter', construct)
    monkeypatch.setattr('s1mb.runner.evaluate', evaluate)
    monkeypatch.setattr('s1mb.dataset_source.dataset_session', lambda *a, **k: nullcontext())
    revision = 'a' * 40
    source = tmp_path / revision
    data = Path(__file__).resolve().parents[1] / 'data'
    monkeypatch.setattr('sys.argv', [
        's1mb', '--data-dir', str(data), 'run', '--adapter', 'aplomb',
        '--model', 'example/aplomb', '--revision', revision, '--source', str(source),
        '--device', 'cuda:0', '--context-limit', '12345', '--offline-dataset',
        '--benchmark', 'arc-choice-test-v1', '--output', str(tmp_path / 'results'),
    ])
    cli.main()
    assert received == [(('example/aplomb', revision, str(source), 'cuda:0'),
                         {'context_limit': 12345})]
