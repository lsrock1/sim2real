import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from mmcv.cnn import VGG, constant_init, kaiming_init, normal_init, xavier_init
from mmcv.runner import load_checkpoint

from mmdet.utils import get_root_logger
from ..builder import BACKBONES
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class Expand(nn.Module):
    def forward(self, x):
        bs, c = x.shape
        return x.reshape(bs, c, 1, 1)


class SNR(nn.Module):
    def __init__(self, dim):
        super(SNR, self).__init__()
        self.dim = dim
        self.instance_norm = nn.InstanceNorm2d(self.dim)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(dim, dim//16), nn.ReLU(True),
            nn.Linear(dim//16, dim), nn.Sigmoid(), Expand())

    def forward(self, input):
        f = self.instance_norm(input)
        r = input - f
        attn = self.se(r)
        r_plus = r * attn
        f_plus = r_plus + f

        if self.training:
            r_minus = r * (1 - attn)
            f_minus = r_minus + f#.detach()

            return [f_plus, ReverseLayerF.apply(f_minus, 0.05)]

        return f_plus


@BACKBONES.register_module()
class SSDVGGVITAL(VGG):
    """VGG Backbone network for single-shot-detection.

    Args:
        input_size (int): width and height of input, from {300, 512}.
        depth (int): Depth of vgg, from {11, 13, 16, 19}.
        out_indices (Sequence[int]): Output from which stages.

    Example:
        >>> self = SSDVGG(input_size=300, depth=11)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 300, 300)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 1024, 19, 19)
        (1, 512, 10, 10)
        (1, 256, 5, 5)
        (1, 256, 3, 3)
        (1, 256, 1, 1)
    """
    extra_setting = {
        300: (256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256),
        512: (256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128),
    }

    def __init__(self,
                 input_size,
                 depth,
                 with_last_pool=False,
                 ceil_mode=True,
                 out_indices=(3, 4),
                 out_feature_indices=(22, 34),
                 l2_norm_scale=20.):
        # TODO: in_channels for mmcv.VGG
        # super(SSDVGGVITAL, self).__init__()
        # self.features = models.vgg16().features[:-1]
        super(SSDVGGVITAL, self).__init__(
            depth,
            with_last_pool=with_last_pool,
            ceil_mode=ceil_mode,
            out_indices=out_indices)
        assert input_size in (300, 512)
        self.input_size = input_size

        self.features.add_module(
            str(len(self.features)),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        self.features.add_module(
            str(len(self.features)),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6))
        self.features.add_module(
            str(len(self.features)), nn.ReLU(inplace=True))
        self.features.add_module(
            str(len(self.features)), nn.Conv2d(1024, 1024, kernel_size=1))
        self.features.add_module(
            str(len(self.features)), nn.ReLU(inplace=True))
        self.out_feature_indices = out_feature_indices

        self.inplanes = 1024
        self.extra = self._make_extra_layers(self.extra_setting[input_size])
        self.l2_norm = L2Norm(
            self.features[out_feature_indices[0] - 1].out_channels,
            l2_norm_scale)
        self.l2_norm_minus = L2Norm(
            self.features[out_feature_indices[0] - 1].out_channels,
            l2_norm_scale)
        # self.norms = nn.ModuleList([nn.BatchNorm2d(c) for c in (512, 1024, 512, 256, 256, 256)])
        # self.norms_minus = nn.ModuleList([nn.BatchNorm2d(c) for c in (512, 1024, 512, 256, 256, 256)])
        self.snr = nn.ModuleList([SNR(c) for c in (512, 1024, 512, 256, 256, 256)])
        # self.instance_norm = nn.ModuleList([nn.InstanceNorm2d(c) for c in (512, 1024, 512, 256, 256, 256)])

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.features.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')

        for m in self.extra.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

        constant_init(self.l2_norm, self.l2_norm.scale)
        constant_init(self.l2_norm_minus, self.l2_norm_minus.scale)

    def forward(self, x):
        """Forward function."""
        outs = []
        idx = 0
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_feature_indices:
                x = self.snr[idx](x)
                outs.append(x)
                if self.training:
                    x = x[0]
                idx += 1
        for i, layer in enumerate(self.extra):
            x = F.relu(layer(x), inplace=True)
            if i % 2 == 1:
                x = self.snr[idx](x)
                outs.append(x)
                if self.training:
                    x = x[0]
                idx += 1
        if self.training:
            # outs = [[n(o[0]), n_m(o[1])] for n, n_m, o in zip(self.norms, self.norms_minus, outs)]
            outs[0][0] = self.l2_norm(outs[0][0])
            outs[0][1] = self.l2_norm_minus(outs[0][1])
        else:
            # outs = [n(o) for n, o in zip(self.norms, outs)]
            outs[0] = self.l2_norm(outs[0])
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def _make_extra_layers(self, outplanes):
        layers = []
        kernel_sizes = (1, 3)
        num_layers = 0
        outplane = None
        for i in range(len(outplanes)):
            if self.inplanes == 'S':
                self.inplanes = outplane
                continue
            k = kernel_sizes[num_layers % 2]
            if outplanes[i] == 'S':
                outplane = outplanes[i + 1]
                conv = nn.Conv2d(
                    self.inplanes, outplane, k, stride=2, padding=1)
            else:
                outplane = outplanes[i]
                conv = nn.Conv2d(
                    self.inplanes, outplane, k, stride=1, padding=0)
            layers.append(conv)
            self.inplanes = outplanes[i]
            num_layers += 1
        if self.input_size == 512:
            layers.append(nn.Conv2d(self.inplanes, 256, 4, padding=1))

        return nn.Sequential(*layers)


class L2Norm(nn.Module):

    def __init__(self, n_dims, scale=20., eps=1e-10):
        """L2 normalization layer.

        Args:
            n_dims (int): Number of dimensions to be normalized
            scale (float, optional): Defaults to 20..
            eps (float, optional): Used to avoid division by zero.
                Defaults to 1e-10.
        """
        super(L2Norm, self).__init__()
        self.n_dims = n_dims
        self.weight = nn.Parameter(torch.Tensor(self.n_dims))
        self.eps = eps
        self.scale = scale

    def forward(self, x):
        """Forward function."""
        # normalization layer convert to FP32 in FP16 training
        x_float = x.float()
        norm = x_float.pow(2).sum(1, keepdim=True).sqrt() + self.eps
        return (self.weight[None, :, None, None].float().expand_as(x_float) *
                x_float / norm).type_as(x)
