import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import math
import numpy as np
import scipy.ndimage

device = torch.device("cuda:0")

def drop_connect(inputs, training: bool = False, drop_connect_rate: float = 0.):
    if not training:
        return inputs

    keep_prob = 1 - drop_connect_rate
    random_tensor = keep_prob + torch.rand(
        (inputs.size()[0], 1, 1, 1), dtype=inputs.dtype, device=inputs.device)
    random_tensor.floor_()  # binarize
    output = inputs.div(keep_prob) * random_tensor
    return output


class EdgeDetector(nn.Module):
    def __init__(self, num_classes, gaussian_size=11):
        super().__init__()

        self.num_classes = num_classes
        self.gs = gaussian_size
        self.gp = self.gs // 2
        self.sigma = 1

        n = np.zeros((gaussian_size,gaussian_size))
        n[self.gp,self.gp] = 1
        self.gaussian_kernel = torch.tensor(scipy.ndimage.gaussian_filter(n,sigma=self.gp//2))

        self.gk = self.gaussian_kernel.unsqueeze(0).unsqueeze(0).expand(self.num_classes,-1,-1,-1).float()
        self.gk = self.gk.to(device)

        self.sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(self.num_classes,-1,-1,-1)
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(self.num_classes,-1,-1,-1)
        
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)

    def forward(self, mask):

        mask = mask.long()
        bs, h, w = mask.shape
        y = mask.reshape(bs * h * w, 1)

        y_onehot = torch.FloatTensor(bs * h * w, self.num_classes).to(device)
        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)

        mask_onehot = y_onehot.reshape(bs, h, w, self.num_classes).permute(0, 3, 1, 2)

        noise_reducted = F.conv2d(mask_onehot, self.gk, stride=1, padding=self.gp, groups=self.num_classes)
        edge_x = F.conv2d(noise_reducted, self.sobel_x, stride=1, padding=1, groups=self.num_classes)
        edge_y = F.conv2d(noise_reducted, self.sobel_y, stride=1, padding=1, groups=self.num_classes)
        edge_map = torch.sqrt(torch.square(edge_x) + torch.square(edge_y))
        edge_map /= torch.max(edge_map)
        
        return (edge_map > 0.3).float()

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

    def forward(self, inputs, drop_connect_rate=0.2):

        # Expansion and Depthwise Convolution
        x = inputs

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

def depth_multiplier(x, y):
    return int( math.ceil(x * y) )

def width_multiplier(x, y):
    res = int( round(x * y) )
    if res % 2 == 1:
        res += 1
    return res

class EfficientSeg(nn.Module):
    def __init__(self, num_classes, depth_coeff, width_coeff):
        super(EfficientSeg, self).__init__()

        self.detector = EdgeDetector(num_classes, gaussian_size=11)

        ord_1 = width_multiplier(64, width_coeff)
        ord_2 = ord_1 * 2
        ord_3 = ord_2 * 2
        ord_4 = ord_3 * 2
        ord_5 = ord_4 * 2

        self.inc = MobileBlock(3, ord_1)
        self.down1 = down(ord_1, ord_2, repeat=depth_multiplier(1, depth_coeff))
        self.down2 = down(ord_2, ord_3, repeat=depth_multiplier(2, depth_coeff))
        self.down3 = down(ord_3, ord_4, kernel_size=5, repeat=depth_multiplier(2, depth_coeff))
        self.down4 = down(ord_4, ord_4, repeat=depth_multiplier(3, depth_coeff))
        self.up1 = up(ord_5, ord_3, kernel_size=5, repeat=depth_multiplier(3, depth_coeff))
        self.up2 = up(ord_4, ord_2, kernel_size=5, repeat=depth_multiplier(4, depth_coeff))
        self.up3 = up(ord_3, ord_1, repeat=depth_multiplier(1, depth_coeff))
        self.up4 = up(ord_2, ord_1, repeat=depth_multiplier(1, depth_coeff))
        self.outb = MobileBlock(ord_1, ord_1)
        self.outc = MobileBlock(ord_1, num_classes)
        self.outd = MobileBlock(ord_1, num_classes)

    def forward(self, x, mask=None, give_mid_output=False):
        if self.training:
            edge = self.detector(mask)

        if give_mid_output:
            mid_outputs = []

        x1 = self.inc(x)
        if give_mid_output: mid_outputs.append(x1)
        x2 = self.down1(x1)
        if give_mid_output: mid_outputs.append(x2)
        x3 = self.down2(x2)
        if give_mid_output: mid_outputs.append(x3)
        x4 = self.down3(x3)
        if give_mid_output: mid_outputs.append(x4)
        x5 = self.down4(x4)
        if give_mid_output: mid_outputs.append(x5)
        x = self.up1(x5,x4)
        if give_mid_output: mid_outputs.append(x)
        x = self.up2(x,x3)
        if give_mid_output: mid_outputs.append(x)
        x = self.up3(x,x2)
        if give_mid_output: mid_outputs.append(x)
        x = self.up4(x,x1)
        if give_mid_output: mid_outputs.append(x)
        x = self.outb(x)
        if give_mid_output: mid_outputs.append(x)
        last_x = x
        x = self.outc(x)
        if give_mid_output: mid_outputs.append(x)

        if self.training:
            edge_pred = self.outd(last_x)

        if self.training:
            return x, edge_pred, edge
        elif give_mid_output:
            return mid_outputs
        else:
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
model = EfficientSeg(20, width_coeff=1.0, depth_coeff=1.0).to( torch.device("cuda:0") )
summary(model, input_size=[(3,384,768), (384,768)])
"""

#inp = torch.rand(1,3,256,256).to( torch.device("cuda:0") )
#out = model(inp)

