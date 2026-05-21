import copy
from pathlib import Path
from typing import Any

import torch
from torch_geometric.datasets import ZINC


def adapt_zinc_item(data):
    """Adapt PyG ZINC scalar categorical fields to AttentiveFP float inputs."""
    item = copy.copy(data)
    item.x = item.x.float()
    edge_attr = item.edge_attr
    if edge_attr.dim() == 1:
        edge_attr = edge_attr.view(-1, 1)
    item.edge_attr = edge_attr.float()
    item.y = item.y.view(1, -1).float()
    return item


def load_zinc_subset(root: str | Path):
    root = Path(root)
    return {
        "train": [adapt_zinc_item(data) for data in ZINC(root=str(root), subset=True, split="train")],
        "valid": [adapt_zinc_item(data) for data in ZINC(root=str(root), subset=True, split="val")],
        "test": [adapt_zinc_item(data) for data in ZINC(root=str(root), subset=True, split="test")],
    }


def _backbone_state(state_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    result = {}
    for key, value in state_dict.items():
        if key.startswith("backbone."):
            result[key[len("backbone.") :]] = value
        else:
            result[key] = value
    return result


def _model_state(state_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "model_state_dict" in state_dict:
        return state_dict["model_state_dict"]
    if "state_dict" in state_dict:
        return state_dict["state_dict"]
    return state_dict


def load_matching_backbone_state(model, pretrained_state_dict: dict[str, Any]) -> dict[str, Any]:
    """Load only shape-compatible AttentiveFP backbone tensors.

    ZINC and MoleculeNet PyG use different raw feature dimensions, so the input
    projections and edge-conditioned layers cannot always be transferred.
    """
    source = _backbone_state(pretrained_state_dict)
    target = model.backbone.state_dict()
    merged = {}
    skipped = []
    loaded = 0
    for key, target_value in target.items():
        source_value = source.get(key)
        if source_value is not None and tuple(source_value.shape) == tuple(target_value.shape):
            merged[key] = source_value
            loaded += 1
        else:
            merged[key] = target_value
            skipped.append(f"backbone.{key}")
    model.backbone.load_state_dict(merged)
    return {"loaded": loaded, "skipped": skipped}


def load_matching_coremol_state(model, pretrained_state_dict: dict[str, Any]) -> dict[str, Any]:
    """Load shape-compatible full CoReMol tensors, including adapter weights.

    ZINC and MoleculeNet often differ in raw atom/bond feature dimensions. This
    keeps the transferable hidden-space tensors and skips incompatible input
    projections.
    """
    source = _model_state(pretrained_state_dict)
    target = model.state_dict()
    merged = {}
    skipped = []
    loaded = 0
    for key, target_value in target.items():
        source_value = source.get(key)
        if source_value is not None and tuple(source_value.shape) == tuple(target_value.shape):
            merged[key] = source_value
            loaded += 1
        else:
            merged[key] = target_value
            skipped.append(key)
    model.load_state_dict(merged)
    return {"loaded": loaded, "skipped": skipped}


def freeze_backbone(model) -> list[str]:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    trainable = []
    for name, parameter in model.named_parameters():
        if not name.startswith("backbone."):
            parameter.requires_grad = True
            trainable.append(name)
    return trainable


def freeze_attentivefp_atom_encoder(model) -> list[str]:
    atom_encoder_prefixes = (
        "backbone.lin1",
        "backbone.gate_conv",
        "backbone.gru",
        "backbone.atom_convs",
        "backbone.atom_grus",
    )
    trainable = []
    for name, parameter in model.named_parameters():
        if name.startswith(atom_encoder_prefixes):
            parameter.requires_grad = False
        else:
            parameter.requires_grad = True
            trainable.append(name)
    return trainable
