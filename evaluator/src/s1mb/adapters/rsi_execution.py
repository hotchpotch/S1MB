"""Bound native RSI activation retention without changing its decision readout."""

from contextvars import ContextVar


def retain_rsi_exit_states(model):
    """Keep final and declared auxiliary states for an inference-only last-layer head.

    Native RSI requests every hidden state even when its head reads only the
    final state. Long eager requests need not retain the unused decoder states.
    The internal hidden-state tuple keeps its native indices; unused entries are
    None. Auxiliary exits retain their original, unnormalized activations.
    """
    if model.training or model.cfg.readout_layer != -1 or model.mix_logits is not None:
        raise ValueError("Selective RSI retention requires inference and a final-layer readout")
    tower = model.tower
    if getattr(tower, "_s1mb_exit_state_retention", False):
        raise ValueError("Selective RSI retention is already installed")
    depths = sorted(set(model.exit_indices()))
    if any(not isinstance(depth, int) or not 1 <= depth <= len(tower.layers) for depth in depths):
        raise ValueError("Invalid native RSI exit depth")
    active: ContextVar[dict[int, object] | None] = ContextVar("rsi_retained_exit_states", default=None)

    def capture(depth):
        def hook(module, inputs, output):
            states = active.get()
            if states is not None:
                states[depth] = output

        return hook

    for depth in depths:
        tower.layers[depth - 1].register_forward_hook(capture(depth))
    forward = tower.forward

    def selective_forward(*args, **kwargs):
        if not kwargs.get("output_hidden_states", False):
            return forward(*args, **kwargs)
        if model.training or kwargs.get("return_dict") is False:
            raise ValueError("Selective RSI retention requires inference with model outputs")
        states: dict[int, object] = {}
        token = active.set(states)
        try:
            output = forward(*args, **{**kwargs, "output_hidden_states": False})
            count = len(tower.layers)
            states[count] = output.last_hidden_state
            output.hidden_states = tuple(states.get(index) for index in range(count + 1))
            return output
        finally:
            active.reset(token)

    tower.forward = selective_forward
    tower._s1mb_exit_state_retention = True
    return "native-final-and-declared-auxiliary-exits"
