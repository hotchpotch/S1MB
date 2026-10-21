"""Pinned native Aplomb text/vision evaluation without hosted inference."""
import base64
import importlib
import io
import json
import re
from pathlib import Path

from s1mb.adapters.base import decode_answers
from s1mb.adapters.upstream import UpstreamAdapter, source_path
from s1mb.adapters.vision_native import ordered_images, typed_questions
from s1mb.parameters import parameter_metadata


class AplombAdapter(UpstreamAdapter):
    supports_images = True
    case_batch_size = 1

    def __init__(self, model, revision, source, device, context_limit=32768):
        if not isinstance(context_limit, int) or isinstance(context_limit, bool) or context_limit < 1:
            raise ValueError('Aplomb context_limit must be a positive integer')
        if not device.startswith('cuda'):
            raise ValueError('Aplomb requires CUDA; CPU fallback is disabled')
        self.path = Path(source).resolve(strict=True)
        if not re.fullmatch(r'[0-9a-f]{40}', revision) or self.path.name != revision:
            raise ValueError('Expected the pinned Hub snapshot as the native source')
        self.name, self.model_id, self.device, self.revision = 'aplomb', model, device, revision
        self.source, self.source_digest = source_path(source)
        self.torch = importlib.import_module('torch')
        if not self.torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; refusing CPU inference')
        self.torch.cuda.set_device(device)
        self.native = importlib.import_module('run_aplomb')
        config = json.loads((self.path / 'edm_config.json').read_text())
        self.calibration = config['temperatures']
        self.ad = self.native.DecisionAdapter(
            str(self.path), device=device, dtype=self.torch.bfloat16,
            attn_implementation='sdpa', image_tokens=config['media']['image_tokens'])
        self.engine = self.ad.model.eval()
        expected = config['label_ids']
        if (self.ad.noul_ids != expected['noul'] or self.ad.digit_ids != expected['digits']
                or self.ad.option_ids != expected['options']):
            raise ValueError('Native tokenizer labels differ from the pinned trained readout')
        # The released reference attaches this FP32 head even for text/image requests.
        self.parameter_counts = parameter_metadata(self.engine)
        head_class = importlib.import_module('aplomb.decision_head').DecisionHead
        self.ad.head = head_class.for_model(
            self.engine, self.ad.noul_ids, self.ad.digit_ids, self.ad.option_ids).to(device)
        self.ad.head.attach(self.engine)
        self.attention_model = self.engine
        self.context_limit = context_limit
        self.image_receipts = {}
        self.original_encode = self.ad.encode
        self.ad.encode = self.checked_encode
        if any(p.device.type != 'cuda' for p in self.engine.parameters()):
            raise RuntimeError('Aplomb contains non-CUDA weights')
        self.settings = {
            'dtype': 'bfloat16', 'attention_implementation': 'sdpa',
            'max_input_tokens': context_limit, 'input_length_policy': 'reject-overflow',
            'calibration_temperatures': self.calibration, 'calibration_policy': 'release-file',
            'debias': 'native-choice-reverse-log-probability-average',
            'abstain': False, 'case_batch_size': 1, 'questions_per_call': 1,
            'media': 'ordered-lossless-png-base64-native-media-objects',
            'image_tokens': config['media']['image_tokens'],
            'image_preprocessing': 'shared-full-image-policy-v1-then-native-processor',
            'components': 'native-folded-text-vision; audio-not-requested',
            'renderer': 'native-structured-anonymous-choice-numeric-score-v1',
            'output_rounding': 'native-eight-decimal-bounded-renormalization',
            'readout': 'native-attached-float32-decision-head',
        }

    def checked_encode(self, pairs, processor_kwargs=None):
        encoded = self.original_encode(pairs, processor_kwargs)
        if encoded.batch['input_ids'].shape[-1] + 1 > self.context_limit:
            raise ValueError('Aplomb input exceeds context limit; refusing truncation')
        return encoded

    def metadata(self):
        return super().metadata().model_copy(update=self.parameter_counts)

    def predict(self, case):
        definitions = typed_questions(case)
        if any(len(q.options) > (10 if q.task == 'score' else 255) for q in case.questions):
            raise ValueError('Aplomb native answer limit exceeded')
        images, state, receipts = ordered_images(case)
        if len(images) > 16:
            raise ValueError('Aplomb supports at most 16 images per request')
        if images:
            self.image_receipts[case.case_id] = receipts
            media = {}
            for name, image in zip(state['visual_inputs'], images, strict=True):
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                media[name] = {'type': 'image', 'mime': 'image/png',
                               'data': base64.b64encode(buffer.getvalue()).decode('ascii')}
            state['visual_inputs'] = media
        with self.torch.inference_mode():
            response = self.native.decide(
                self.ad, {'state': state if images else case.state, 'questions': definitions},
                self.calibration, debias=True, batch=1, abstain_bias=0.0)
        return decode_answers(case, response['answers'], rounded=True,
                              rounding_decimals=8, anonymous_choice=True)

    def close(self):
        # Native adapter and the length guard both retain the loaded model.
        if hasattr(self, 'ad'):
            del self.ad
        if hasattr(self, 'original_encode'):
            del self.original_encode
        super().close()
