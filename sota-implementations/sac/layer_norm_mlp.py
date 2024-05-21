import torch
import numpy as np

from torch import nn
from torchrl._utils import prod
from torchrl.data import DEVICE_TYPING
from typing import Dict, List, Optional, Sequence, Tuple, Type, Union

from numbers import Number

from torchrl.modules.models.utils import (
    _find_depth,
    create_on_device,
    LazyMapping
)

class LayerNormMLP(nn.Sequential):
    """
    NOTE: If norm_class = None, behaves the same as torchRL's default MLP.

    Copied from https://pytorch.org/rl/_modules/torchrl/modules/models/models.html#MLP
    with modifications to make layer norm actually work. See docstring there.
    """

    def __init__(
        self,
        in_features: Optional[int] = None,
        out_features: Union[int, Sequence[int]] = None,
        depth: Optional[int] = None,
        num_cells: Optional[Union[Sequence, int]] = None,
        activation_class: Type[nn.Module] = nn.Tanh,
        activation_kwargs: Optional[dict] = None,
        norm_class: Optional[Type[nn.Module]] = None,
        norm_kwargs: Optional[dict] = None,
        dropout: Optional[float] = None,
        bias_last_layer: bool = True,
        single_bias_last_layer: bool = False,
        layer_class: Type[nn.Module] = nn.Linear,
        layer_kwargs: Optional[dict] = None,
        activate_last_layer: bool = False,
        last_layer_activation_class: Type[nn.Module] = nn.Tanh,
        device: Optional[DEVICE_TYPING] = None,
    ):
        if out_features is None:
            raise ValueError("out_features must be specified for MLP.")

        default_num_cells = 32
        if num_cells is None:
            if depth is None:
                num_cells = [default_num_cells] * 3
                depth = 3
            else:
                num_cells = [default_num_cells] * depth

        self.in_features = in_features

        _out_features_num = out_features
        if not isinstance(out_features, Number):
            _out_features_num = prod(out_features)
        self.out_features = out_features
        self._out_features_num = _out_features_num
        self.activation_class = activation_class
        self.activation_kwargs = (
            activation_kwargs if activation_kwargs is not None else {}
        )
        self.norm_class = norm_class
        self.norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        self.dropout = dropout
        self.bias_last_layer = bias_last_layer
        self.single_bias_last_layer = single_bias_last_layer
        self.layer_class = layer_class
        self.layer_kwargs = layer_kwargs if layer_kwargs is not None else {}
        self.activate_last_layer = activate_last_layer
        self.last_layer_activation_class = last_layer_activation_class
        if single_bias_last_layer:
            raise NotImplementedError

        if not (isinstance(num_cells, Sequence) or depth is not None):
            raise RuntimeError(
                "If num_cells is provided as an integer, \
            depth must be provided too."
            )
        self.num_cells = (
            list(num_cells) if isinstance(num_cells, Sequence) else [num_cells] * depth
        )
        self.depth = depth if depth is not None else len(self.num_cells)
        if not (len(self.num_cells) == depth or depth is None):
            raise RuntimeError(
                "depth and num_cells length conflict, \
            consider matching or specifying a constant num_cells argument together with a a desired depth"
            )
        layers = self._make_net(device)
        super().__init__(*layers)

    def _make_net(self, device: Optional[DEVICE_TYPING]) -> List[nn.Module]:
        layers = []
        in_features = [self.in_features] + self.num_cells
        out_features = self.num_cells + [self._out_features_num]
        for i, (_in, _out) in enumerate(zip(in_features, out_features)):
            _bias = self.bias_last_layer if i == self.depth else True
            if _in is not None:
                layers.append(
                    create_on_device(
                        self.layer_class,
                        device,
                        _in,
                        _out,
                        bias=_bias,
                        **self.layer_kwargs,
                    )
                )
            else:
                try:
                    lazy_version = LazyMapping[self.layer_class]
                except KeyError:
                    raise KeyError(
                        f"The lazy version of {self.layer_class.__name__} is not implemented yet. "
                        "Consider providing the input feature dimensions explicitely when creating an MLP module"
                    )
                layers.append(
                    create_on_device(
                        lazy_version, device, _out, bias=_bias, **self.layer_kwargs
                    )
                )

            # LayerNorm should be applied after the linear layer, based on:
            # - https://github.com/ikostrikov/rlpd/blob/main/rlpd/networks/mlp.py#L9 (RLPD)
            # - https://pytorch.org/vision/main/_modules/torchvision/ops/misc.html#MLP (torchvision MLP)
            if i < self.depth or self.activate_last_layer:
                if self.dropout is not None:
                    layers.append(create_on_device(nn.Dropout, device, p=self.dropout))
                if self.norm_class is not None:
                    layers.append(
                        create_on_device(self.norm_class, device, normalized_shape=_out, **self.norm_kwargs)
                    )
                layers.append(
                    create_on_device(
                        self.activation_class if (i < self.depth) else self.last_layer_activation_class, 
                        device, **self.activation_kwargs
                    )
                )

        return layers

    def forward(self, *inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        if len(inputs) > 1:
            inputs = (torch.cat([*inputs], -1),)

        out = super().forward(*inputs)
        if not isinstance(self.out_features, Number):
            out = out.view(*out.shape[:-1], *self.out_features)
        return out
