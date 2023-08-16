import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor

import flow.utils as utils


class ResidualBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(
        self,
        features,
        context_features,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True,
    ):
        super().__init__()
        self.activation = activation

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(features, eps=1e-3) for _ in range(2)]
            )
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)
        self.linear_layers = nn.ModuleList(
            [nn.Linear(features, features) for _ in range(2)]
        )
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        return inputs + temps


class ResidualNet(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.context_features = context_features
        if context_features is not None:
            self.initial_layer = nn.Linear(
                in_features + context_features, hidden_features
            )
        else:
            self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    features=hidden_features,
                    context_features=context_features,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Linear(hidden_features, out_features)

    def forward(self, inputs, context=None):
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context=context)
        outputs = self.final_layer(temps)
        return outputs



# This code is from pytorch vision library
# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
def conv_ks3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv1d:
    """ks 3 convolution with padding"""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size = 3,
        stride = stride,
        padding = dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv_ks1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """ks 1 convolution"""
    return nn.Conv1d(
        in_planes, 
        out_planes, 
        kernel_size=1, 
        stride=stride, 
        bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3 convolution(self.conv3)
    # while original implementation places the stride at the first 1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv_ks1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv_ks3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv_ks1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv_ks3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_ks3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ConvResNet1D(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        embed_in: int = 4,
        num_classes: int = 256,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 16 #64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(embed_in, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False) # kernel_size =7
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])                                                    # 64
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2, dilate=replace_stride_with_dilation[0]) # 128
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2, dilate=replace_stride_with_dilation[1]) # 256
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2, dilate=replace_stride_with_dilation[2]) # 512
        self.avgpool = nn.AdaptiveAvgPool1d(1) #nn.MaxPool1d(32, stride=2, padding=3) # nn.AdaptiveAvgPool 1
        self.fc = nn.Linear(256 * block.expansion, num_classes) # 512

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv_ks1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)



class ConvResidualBlock(nn.Module):
    def __init__(
        self,
        channels,
        context_channels=None,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True
    ):
        super().__init__()
        self.activation = activation

        if context_channels is not None:
            self.context_layer = nn.Conv2d(
                in_channels=context_channels,
                out_channels=channels,
                kernel_size = (1,1),
                padding = 0,
            )
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm2d(channels, eps=1e-3) for _ in range(2)]
            )
        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=(1,3), padding=(0,1)) for _ in range(2)]
        )
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.conv_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.conv_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.conv_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.conv_layers[1](temps)
        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        return inputs + temps


class ConvResidualNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        context_channels=None,
        num_blocks=2,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False
    ):
        super().__init__()
        self.context_channels = context_channels
        self.hidden_channels = hidden_channels
        if context_channels is not None:
            self.initial_layer = nn.Conv2d(
                in_channels=in_channels + context_channels,
                out_channels=hidden_channels,
                kernel_size=(1,1),
                padding=0,
            )
        else:
            self.initial_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=(1,1),
                padding=0,
            )
        self.blocks = nn.ModuleList(
            [
                ConvResidualBlock(
                    channels=hidden_channels,
                    context_channels=context_channels,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_layer = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 512), padding=0),           
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 256), padding=0),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 128), padding=0),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 64), padding=0),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size = (1, 32), padding=0),
            nn.Conv2d(hidden_channels, out_channels, kernel_size = (1, 16), padding=0)          
        )


#        self.final_layer = nn.Sequential(
#            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 128), padding=0),           
#            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 128), padding=0),
#            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 128), padding=0),
#            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 256), padding=0),
#            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 512), padding=0),
#            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 128), padding=0),
#            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 16), padding=0)
#        )




    def forward(self, inputs, context=None):
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context)
        outputs = self.final_layer(temps)
        outputs = outputs.view(outputs.shape[0], -1)
        return outputs


class ConvResidualNet_VB(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        context_channels=None,
        num_blocks=2,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False
    ):
        super().__init__()
        self.context_channels = context_channels
        self.hidden_channels = hidden_channels
        if context_channels is not None:
            self.initial_layer = nn.Conv2d(
                in_channels=in_channels + context_channels,
                out_channels=hidden_channels,
                kernel_size=(1,1),
                padding=0,
            )
        else:
            self.initial_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=(1,1),
                padding=0,
            )
        self.blocks = nn.ModuleList(
            [
                ConvResidualBlock(
                    channels=hidden_channels,
                    context_channels=context_channels,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_layer =  nn.Conv2d(
            hidden_channels, out_channels, kernel_size=(1,1), padding=0
        )
        #nn.Sequential()


    def forward(self, inputs, context=None):
    
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context)
        outputs = self.final_layer(temps)
        outputs = outputs.view(outputs.shape[0], -1)
      
        return outputs






def main():
    batch_size, channels, height, width = 100, 12, 64, 64
    inputs = torch.rand(batch_size, channels, height, width)
    context = torch.rand(batch_size, channels // 2, height, width)
    net = ConvResidualNet(
        in_channels=channels,
        out_channels=2 * channels,
        hidden_channels=32,
        context_channels=channels // 2,
        num_blocks=2,
        dropout_probability=0.1,
        use_batch_norm=True,
    )
    print(utils.get_num_parameters(net))
    outputs = net(inputs, context)
    print(outputs.shape)


if __name__ == "__main__":
    main()
