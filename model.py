import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F

def drop_connect(inputs, training: bool = False, drop_connect_rate: float = 0.):
    if not training:
        return inputs

    keep_prob = 1 - drop_connect_rate
    random_tensor = keep_prob + torch.rand(
        (inputs.size()[0], 1, 1, 1), dtype=inputs.dtype, device=inputs.device)
    random_tensor.floor_()  # binarize
    output = inputs.div(keep_prob) * random_tensor
    return output

class MobileBlock(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, expand_ratio=1, bn_mom=0.99, bn_eps=1e-3, se_ratio=0.25, id_skip=True):
        super().__init__()

        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.stride = stride
        self.id_skip = id_skip

        # Expansion phase
        inp = in_chn
        oup = in_chn * expand_ratio
        if expand_ratio != 1:
            self._expand_conv = nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=bn_mom, eps=bn_eps)

        # Depthwise convolution phase
        p = kernel_size // 2
        if stride == 0.5:
            self._depthwise_conv = nn.Conv2d(in_channels=oup, out_channels=oup, groups=oup, kernel_size=kernel_size, stride=1, padding=p, bias=False)
        else:
            self._depthwise_conv = nn.Conv2d(in_channels=oup, out_channels=oup, groups=oup, kernel_size=kernel_size, stride=stride, padding=p, bias=False)

        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=bn_mom, eps=bn_eps)

        # Squeeze and Excitation layer, if desired
        if se_ratio != 0:
            num_squeezed_channels = max(1, int(in_chn * se_ratio))
            self._se_reduce = nn.Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Pointwise convolution phase
        final_oup = out_chn
        self._project_conv = nn.Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=bn_mom, eps=bn_eps)

        # If upsampling
        if stride == 0.5:
            self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)


        self.relu = nn.ReLU()

    def forward(self, inputs, drop_connect_rate=0.2, concat=None):

        # Expansion and Depthwise Convolution
        x = inputs
        if concat is not None:
            x = torch.cat((concat, x), dim=1)
        if self.expand_ratio != 1:
            x = self._expand_conv(x)
            x = self._bn0(x)
            x = self.relu(x)

        if self.stride == 0.5:
            x = self.upsample(x)
        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self.relu(x)

        # Squeeze and Excitation
        if self.se_ratio != 0:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self.relu(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self.in_chn, self.out_chn
        if self.id_skip and input_filters == output_filters and self.stride == 1:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, drop_connect_rate=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x


class EfficientSeg(nn.Module):
    def __init__(self, num_classes):
        super(EfficientSeg, self).__init__()

        self.inc = MobileBlock(3, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256, repeat=2)
        self.down3 = down(256, 512, kernel_size=5, repeat=2)
        self.down4 = down(512, 512, repeat=3)
        self.up1 = up(1024, 256, kernel_size=5, repeat=3)
        self.up2 = up(512, 128, kernel_size=5, repeat=4)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = MobileBlock(64, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        x = self.outc(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, expand_ratio=1, repeat=1):
        super(down, self).__init__()
        reps = []
        for _ in range(repeat-1):
            reps.append( MobileBlock(in_ch, in_ch, kernel_size=kernel_size, expand_ratio=expand_ratio) )
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            *reps,
            MobileBlock(in_ch, out_ch, kernel_size=kernel_size, expand_ratio=expand_ratio)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, expand_ratio=1, repeat=1):
        super(up, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        reps = []
        for _ in range(repeat-1):
            reps.append( MobileBlock(in_ch, in_ch, kernel_size=kernel_size, expand_ratio=expand_ratio) )
        self.conv = nn.Sequential( *reps, 
            MobileBlock(in_ch, out_ch, kernel_size=kernel_size, expand_ratio=expand_ratio)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

"""
from torchsummary import summary
model = EfficientSeg(33).to( torch.device("cuda:0") )
summary(model, input_size=(3,384,768))
"""

#inp = torch.rand(1,3,256,256).to( torch.device("cuda:0") )
#out = model(inp)

