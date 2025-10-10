"""
modules.py
Contains the source code for the components of the ConvNeXt model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


class Block(nn.Module):
    """ConvNeXt Block. Implemented by:
    DwConv -> Permute axes; LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        """Initialise the ConvNeXt block.

        Args:
            dim (int): Number of input channels.
            drop_path (float): Stochastic depth rate. Default: 0.0
            layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        """
        super().__init__()

        # depth-wise convolutional layer
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        ) 

        # layer norm
        self.norm = LayerNorm(dim, eps=1e-6)

        # piece-wise feed-forward network
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )

        # gelu activation
        self.act = nn.GELU()

        # piece-wise feed-forward network
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # layer scale
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

        # stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        """Forward pass of the ConvNeXt block.

        Args:
            x (torch.tensor): Feature map

        Returns:
            torch.tensor: Result of forward pass
        """
        input = x
        x = self.dwconv(x)

        # permute axes to put channels last
        # LayerNorm and linear layers expect channels last
        x = x.permute(0, 2, 3, 1)  # (batch size, channel, height, width) -> (batch size, height, width, channel)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x

        # permute axes back to channels first for convolutional layer
        x = x.permute(0, 3, 1, 2)  # (batch size, height, width, channel) -> (batch size, channel, height, width)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    """ConvNeXt model."""

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
    ):
        """Initialise the ConvNeXt model.

            Args:
            in_chans (int): Number of input image channels. Default: 3
            num_classes (int): Number of classes for classification head. Default: 1000
            depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
            dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
            drop_path_rate (float): Stochastic depth rate. Default: 0.
            layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
            head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        """

        super().__init__()

        # create downsampling layers
        # progressively reduce image resolution while increasing the channel dimension

        # holds the downsampling layers
        self.downsample_layers = nn.ModuleList()

        # 'stem' convolutional block
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)

        # next 3 downsampling layers. each increases the channel dimension
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # create ConvNeXt stages - feature extraction blocks
        self.stages = nn.ModuleList()
        # drop rates
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(4):
            # Each stage consists of 'depths[i]' ConvNeXt blocks
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],  # number of feature channels
                        drop_path=dp_rates[cur + j],  # stochastic depth rate for this block
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]  # move to next set of drop rates

        # final norm layer and classification layer
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        # initialise weights
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        """Initialise weights of linear and convolutional layers with truncated normal distribution.

        Args:
            m (nn.Module): Module to initialise weigts to.
        """
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        """Forward pass through feature extraction layers.

        Args:
            x (torch.tensor): Feature map

        Returns:
            torch.tensor: Result of forward pass
        """
        for i in range(4):
            # downsize the dimension of x
            x = self.downsample_layers[i](x)
            # pass through the ConvNeXt stages
            x = self.stages[i](x)
        
        # return the pooled and normalized features
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        """Forward pass of the ConvNeXt model.

        Args:
            x (torch.tensor): Feature map

        Returns:
            torch.tensor: logits of the classes
        """

        # pass through feature extraction layers
        x = self.forward_features(x)
        # pass through classification head
        x = self.head(x)
        return x


class LayerNorm(nn.Module):
    """
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        """Initialise the LayerNorm module.

        Args:
            normalized_shape (int): Number of channels in the input.
            eps (float): Small constant for numerical stability Defaults to 1e-6.
            data_format (str): Input tensor format, either
            "channels_last" or "channels_first".. Defaults to "channels_last".

        Raises:
            NotImplementedError: If an unsupported data_format is provided.
        """
        super().__init__()

        # weight parameters
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

        # epsilon for numerical stability during normalization
        self.eps = eps

        # set data format
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError

        # set the normalisation dimension(s)
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        """Forward pass of the LayerNorm.

        Args:
            x (torch.tensor): Feature map

        Returns:
            torch.tensor: Result of forward pass
        """
        if self.data_format == "channels_last":
            # call built-in layer norm with stored fields
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        
        elif self.data_format == "channels_first":
            # mean across channel dimension for each pixel
            u = x.mean(1, keepdim=True)

            # variance across channels
            s = (x - u).pow(2).mean(1, keepdim=True)

            # normalise
            x = (x - u) / torch.sqrt(s + self.eps)

            # apply weight and bias parameters
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def convnext_small(**kwargs):
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
