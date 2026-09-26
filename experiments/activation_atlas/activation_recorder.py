#!/usr/bin/env python3
"""Record stable activation-site states from PyTorch models with torch.fx."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import GraphModule, Interpreter, Node, symbolic_trace


FUNCTION_FAMILIES = {
    F.relu: "relu",
    torch.relu: "relu",
    F.gelu: "gelu",
    F.silu: "silu",
    torch.sigmoid: "sigmoid",
    torch.tanh: "tanh",
}

MODULE_FAMILIES = {
    nn.ReLU: "relu",
    nn.ReLU6: "relu6",
    nn.GELU: "gelu",
    nn.SiLU: "silu",
    nn.Sigmoid: "sigmoid",
    nn.Tanh: "tanh",
}


@dataclass(frozen=True)
class ActivationSite:
    """Persistent identity and display metadata for one activation operation."""

    node_name: str
    display_name: str
    family: str
    module_path: str
    stage: str
    is_exit_candidate: bool


def _module_path(node: Node) -> str:
    stack = node.meta.get("nn_module_stack", {})
    return next(reversed(stack), "") if stack else ""


def _stage(module_path: str, node_name: str) -> str:
    if not module_path:
        return "stem"
    return module_path.split(".", 1)[0] or node_name


def _display_name(node: Node, family: str) -> str:
    path = _module_path(node)
    if not path:
        return f"stem/{family}"
    return f"{path}/{node.name}"


def _family_for_node(graph_module: GraphModule, node: Node) -> str | None:
    if node.op == "call_function":
        return FUNCTION_FAMILIES.get(node.target)
    if node.op == "call_method" and str(node.target) in {
        "relu",
        "relu_",
        "sigmoid",
        "tanh",
    }:
        return str(node.target).rstrip("_")
    if node.op == "call_module":
        module = graph_module.get_submodule(str(node.target))
        for module_type, family in MODULE_FAMILIES.items():
            if isinstance(module, module_type):
                return family
    return None


def _is_exit_candidate(node: Node) -> bool:
    """Use the stem and post-residual activations as semantically valid exits."""

    if not node.args:
        return False
    argument = node.args[0]
    if not isinstance(argument, Node):
        return False
    return argument.op == "placeholder" or argument.name.startswith("add")


class RecordingInterpreter(Interpreter):
    """FX interpreter that retains activation outputs under stable node names."""

    def __init__(self, module: GraphModule, sites: Mapping[str, ActivationSite]):
        super().__init__(module)
        self.sites = sites
        self.activations: OrderedDict[str, torch.Tensor] = OrderedDict()

    def run_node(self, node: Node):
        result = super().run_node(node)
        if node.name in self.sites:
            self.activations[node.name] = result
        return result

    def run_with_activations(self, *args, **kwargs):
        self.activations = OrderedDict()
        output = self.run(*args, **kwargs)
        return output, self.activations


class ActivationRecorder:
    """Trace a model once and record its activation operations for each batch."""

    def __init__(self, model: nn.Module):
        self.graph_module = symbolic_trace(model)
        sites: OrderedDict[str, ActivationSite] = OrderedDict()
        for node in self.graph_module.graph.nodes:
            family = _family_for_node(self.graph_module, node)
            if family is None:
                continue
            sites[node.name] = ActivationSite(
                node_name=node.name,
                display_name=_display_name(node, family),
                family=family,
                module_path=_module_path(node),
                stage=_stage(_module_path(node), node.name),
                is_exit_candidate=_is_exit_candidate(node) or node.name == "relu",
            )
        if not sites:
            raise RuntimeError("No supported activation operations were found by torch.fx")
        self.sites = sites
        self.interpreter = RecordingInterpreter(self.graph_module, sites)

    def to(self, device: torch.device | str):
        self.graph_module.to(device)
        return self

    def eval(self):
        self.graph_module.eval()
        return self

    def record(self, images: torch.Tensor):
        return self.interpreter.run_with_activations(images)


def activation_state(
    activation: torch.Tensor,
    family: str,
    threshold: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Convert an activation tensor to a discrete active/inactive state.

    ReLU-family operations have an exact gate state. Smooth activations use a
    calibrated threshold supplied by the caller; zero is only a fallback.
    """

    if family in {"relu", "relu6"}:
        return activation > 0
    if threshold is None:
        threshold = 0.0
    if isinstance(threshold, torch.Tensor):
        while threshold.ndim < activation.ndim:
            threshold = threshold.unsqueeze(-1)
    return activation > threshold


def channel_activity(state: torch.Tensor) -> torch.Tensor:
    """Return per-channel active fractions while preserving the batch axis."""

    if state.ndim <= 2:
        return state.float()
    return state.float().flatten(2).mean(2)
