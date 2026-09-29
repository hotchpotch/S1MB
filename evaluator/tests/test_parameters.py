"""Public counting tests use synthetic modules without Torch or checkpoints."""

import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from s1mb.data import ModelInfo
from s1mb.parameters import METHOD, parameter_metadata


class Parameter:
    def __init__(self, size):
        self.size = size

    def numel(self):
        return self.size


class Module:
    def __init__(self, **parameters):
        self.parameters = parameters

    def modules(self):
        return [self]

    def named_parameters(self, recurse=False):
        return self.parameters.items()


class Embedding(Module):
    pass


@pytest.fixture
def synthetic_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(nn=SimpleNamespace(
        Module=Module, Embedding=Embedding, EmbeddingBag=Embedding,
    )))


def test_complete_wrapper_and_shared_parameters(synthetic_torch):
    table, head = Parameter(100), Parameter(12)
    model = SimpleNamespace(encoder=Embedding(weight=table), head=Module(weight=head))
    model.alias = model.head
    model.cycle = model
    assert parameter_metadata(model) == {
        "total_params": 112, "active_params": 12, "parameter_count_method": METHOD,
    }
    model.output = Module(weight=table)
    assert parameter_metadata(model)["active_params"] == 112


def test_unknown_model_is_not_zero(synthetic_torch):
    with pytest.raises(ValueError, match="No torch modules"):
        parameter_metadata(object())


@pytest.mark.parametrize("counts", [
    {"total_params": -1}, {"total_params": True},
    {"active_params": 2, "total_params": 1, "parameter_count_method": METHOD},
    {"active_params": 1}, {"parameter_count_method": METHOD},
])
def test_invalid_metadata(counts):
    with pytest.raises(ValidationError):
        ModelInfo(id="test", adapter="test", revision="test", **counts)


def test_unknown_and_known_metadata():
    assert ModelInfo(id="api", adapter="api", revision="test").total_params is None
    model = ModelInfo(id="local", adapter="local", revision="test", total_params=112,
                      active_params=12, parameter_count_method=METHOD)
    assert ModelInfo.model_validate_json(model.model_dump_json()) == model
