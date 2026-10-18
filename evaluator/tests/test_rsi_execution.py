"""Preserve native exit values while releasing unused decoder activations."""

from types import SimpleNamespace

import pytest

from s1mb.adapters.rsi_execution import retain_rsi_exit_states


class Layer:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)

    def __call__(self, value):
        output = value + 1
        for hook in self.hooks:
            hook(self, (value,), output)
        return output


class Tower:
    def __init__(self):
        self.layers = [Layer() for _ in range(6)]
        self.fail = False
        self.capture_requests = []

    def forward(self, value, output_hidden_states=False, return_dict=True):
        self.capture_requests.append(output_hidden_states)
        states = []
        for layer in self.layers:
            if output_hidden_states:
                states.append(value)
            value = layer(value)
        if self.fail:
            raise RuntimeError("Synthetic forward failure")
        value += 100  # Distinguish the final normalized state from a decoder output.
        return SimpleNamespace(last_hidden_state=value,
                               hidden_states=(*states, value) if output_hidden_states else None)


def model():
    return SimpleNamespace(training=False, cfg=SimpleNamespace(readout_layer=-1),
                           mix_logits=None, tower=Tower(), exit_indices=lambda: [3, 6])


def test_preserves_declared_exits_and_normalized_final_without_all_layer_capture():
    native = model()
    expected = native.tower.forward(10, output_hidden_states=True)
    retain_rsi_exit_states(native)
    actual = native.tower.forward(10, output_hidden_states=True)
    assert actual.last_hidden_state == expected.last_hidden_state
    assert len(actual.hidden_states) == len(expected.hidden_states)
    assert actual.hidden_states[3] == expected.hidden_states[3]
    assert actual.hidden_states[-1] == expected.hidden_states[-1]
    assert all(value is None for i, value in enumerate(actual.hidden_states) if i not in {3, 6})
    assert native.tower.capture_requests == [True, False]
    assert native.tower.forward(20).hidden_states is None


def test_retention_handles_a_native_early_exit_and_cleans_up_after_failure():
    native = model()
    retain_rsi_exit_states(native)
    native.tower.layers = native.tower.layers[:3]
    native.tower.fail = True
    with pytest.raises(RuntimeError, match="Synthetic"):
        native.tower.forward(10, output_hidden_states=True)
    native.tower.fail = False
    actual = native.tower.forward(20, output_hidden_states=True)
    assert actual.hidden_states == (None, None, None, 123)


@pytest.mark.parametrize("change", ["training", "readout", "mixture"])
def test_rejects_unsupported_readouts_instead_of_dropping_needed_states(change):
    native = model()
    if change == "training":
        native.training = True
    elif change == "readout":
        native.cfg.readout_layer = {"choice": -1}
    else:
        native.mix_logits = object()
    with pytest.raises(ValueError, match="final-layer readout"):
        retain_rsi_exit_states(native)


def test_rejects_duplicate_installation_and_non_model_outputs():
    native = model()
    retain_rsi_exit_states(native)
    with pytest.raises(ValueError, match="already installed"):
        retain_rsi_exit_states(native)
    with pytest.raises(ValueError, match="model outputs"):
        native.tower.forward(10, output_hidden_states=True, return_dict=False)
