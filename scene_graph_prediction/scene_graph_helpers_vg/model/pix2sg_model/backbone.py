# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from functools import partial
from itertools import chain

import timm
import torch
import torch.nn.functional as F
import torchvision
from timm.models.efficientnet import _create_effnet

if timm.__version__ <= '0.4.12':
    from timm.models.efficientnet_builder import decode_arch_def, round_channels, resolve_act_layer, resolve_bn_args
else:
    from timm.models._efficientnet_builder import decode_arch_def, round_channels, resolve_act_layer, resolve_bn_args
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from scene_graph_prediction.scene_graph_helpers_vg.model.pix2sg_model.util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
import timm


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)

        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor, skip_pooling=False):
        if skip_pooling:
            xs = self[0](tensor_list, skip_pooling=True)
        else:
            xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        if isinstance(xs, torch.Tensor):
            mask = torch.zeros(xs.shape[0], xs.shape[2], dtype=torch.bool, device=xs.device)
            xs = {'0': NestedTensor(xs, mask)}
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


class EfficientNetBackbone(nn.Module):
    def __init__(self, backbone, freeze_first_n: int):
        super().__init__()
        arch_def = [
            ['ds_r1_k3_s1_e1_c16_se0.25'],
            ['ir_r2_k3_s2_e6_c24_se0.25'],
            ['ir_r2_k5_s2_e6_c40_se0.25'],
            ['ir_r3_k3_s2_e6_c80_se0.25'],
            ['ir_r3_k5_s1_e6_c112_se0.25'],
            ['ir_r4_k5_s2_e6_c192_se0.25'],
            ['ir_r1_k3_s1_e6_c320_se0.25'],
        ]
        variant_to_params = {
            'efficientnet_b0': {'channel_multiplier': 1.0, 'depth_multiplier': 1.0, 'resolution': 224, 'dropout_rate': 0.2},
            'efficientnet_b1': {'channel_multiplier': 1.0, 'depth_multiplier': 1.1, 'resolution': 240, 'dropout_rate': 0.2},
            'efficientnet_b2': {'channel_multiplier': 1.1, 'depth_multiplier': 1.2, 'resolution': 260, 'dropout_rate': 0.3},
            'efficientnet_b3': {'channel_multiplier': 1.2, 'depth_multiplier': 1.4, 'resolution': 300, 'dropout_rate': 0.3},
            'efficientnet_b4': {'channel_multiplier': 1.4, 'depth_multiplier': 1.8, 'resolution': 380, 'dropout_rate': 0.4},
            'efficientnet_b5': {'channel_multiplier': 1.6, 'depth_multiplier': 2.2, 'resolution': 456, 'dropout_rate': 0.4},
            'efficientnet_b6': {'channel_multiplier': 1.8, 'depth_multiplier': 2.6, 'resolution': 528, 'dropout_rate': 0.5},
            'efficientnet_b7': {'channel_multiplier': 2.0, 'depth_multiplier': 3.1, 'resolution': 600, 'dropout_rate': 0.5},
            'efficientnet_b8': {'channel_multiplier': 2.2, 'depth_multiplier': 3.6, 'resolution': 672, 'dropout_rate': 0.5},
            'efficientnet_l2': {'channel_multiplier': 4.3, 'depth_multiplier': 5.3, 'resolution': 800, 'dropout_rate': 0.5},
        }
        params = variant_to_params[backbone.replace('_ns', '').replace('tf_', '').replace('_475', '')]
        round_chs_fn = partial(round_channels, multiplier=params['channel_multiplier'])
        user_args = {'num_classes': 0, 'pad_type': 'same', 'global_pool': ''}
        model_kwargs = dict(
            block_args=decode_arch_def(arch_def, params['depth_multiplier']),
            num_features=round_chs_fn(1280),
            stem_size=32,
            round_chs_fn=round_chs_fn,
            act_layer=resolve_act_layer(user_args, 'swish'),
            norm_layer=partial(FrozenBatchNorm2d, **resolve_bn_args(user_args)),
            **user_args,
        )
        self.model = _create_effnet(backbone, pretrained=True, **model_kwargs)
        if freeze_first_n > 0:
            # Freeze the whole model
            for param in self.model.parameters():
                param.requires_grad = False
            # Unfreeze conv head, and the last N blocks
            for param in chain(self.model.conv_head.parameters(),
                               list(self.model.blocks[freeze_first_n:].parameters())):
                param.requires_grad = True
        self.num_channels = self.model.num_features
        self.model.num_channels = self.model.num_features

    def forward(self, tensor_list: NestedTensor):
        xs = self.model(tensor_list.tensors)
        xs = {'0': xs}
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class EvaBackbone(nn.Module):
    def __init__(self, name, freeze_first_n: int):
        super().__init__()
        self.model = timm.create_model(name, pretrained=True, num_classes=0)
        self.model.train()

        if freeze_first_n > 0:
            # Freeze the whole model
            for param in self.model.parameters():
                param.requires_grad = False
            for layer in self.model.blocks[freeze_first_n:]:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            for param in self.model.parameters():
                param.requires_grad = True

    def forward(self, tensor_list: NestedTensor):
        xs = self.model.forward_features(tensor_list.tensors)
        # size is 16 x 1025 x 768. Instead I want to unroll it to a  full image, essentially 16 x width x height x 768
        # start by removing the CLS token, and then unflatten it
        xs = xs[:, 1:].unflatten(1, (32, 32))
        xs = xs.permute(0, 3, 1, 2)
        xs = F.max_pool2d(xs, kernel_size=2, stride=2)

        # visualize xs to verify.
        # import matplotlib.pyplot as plt
        # plt.imshow(xs[1].mean(0).cpu().detach().numpy())
        # plt.savefig('debug_activation.jpg')
        # plt.close()

        xs = {'0': xs}
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    # timm==0.4.12
    if 'efficientnet' in args.image_backbone:
        image_backbone = EfficientNetBackbone(args.image_backbone, args.freeze_first_N)
        num_channels = image_backbone.num_channels
    elif 'eva02' in args.image_backbone:
        image_backbone = EvaBackbone(args.image_backbone, args.freeze_first_N)
        num_channels = image_backbone.model.num_features
    image_model = Joiner(image_backbone, position_embedding)
    image_model.num_channels = num_channels
    return image_model
