import torch
from torch import nn
device = torch.device("cuda:0")

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class EfficientSeg(nn.Module):
    def __init__(self, enc_config, dec_config, num_classes, width_coeff):
        super(EfficientSeg, self).__init__()

        block = InvertedResidual

        input_channel = _make_divisible(16 * width_coeff, 8)

        self.enc_layers = nn.ModuleList()
        self.enc_layers.append( conv_3x3_bn(3, input_channel, 2) )

        self.dec_layers = nn.ModuleList()
        self.dec_layers.append( conv_3x3_bn(input_channel, num_classes, 1) )

        self.down_map = []
        last_s = None

        for idx, (k, t, c, use_se, use_hs, s) in enumerate(enc_config):
            output_channel = _make_divisible(c * width_coeff, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            self.enc_layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))

            if idx < len(enc_config) - 1 and enc_config[idx+1][-1] == 2:
                self.dec_layers.insert(0,block(output_channel*2, exp_size, input_channel, k, 1, use_se, use_hs))
            else:
                self.dec_layers.insert(0,block(output_channel, exp_size, input_channel, k, 1, use_se, use_hs))


            input_channel = output_channel
            last_s = s

            if( s == 1 ): self.down_map.append(0)
            else: self.down_map.append(1)

        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.conv2 = conv_1x1_bn(exp_size, input_channel)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)




    def forward(self, x):

        concatputs = []

        for idx, layer in enumerate(self.enc_layers):
            if idx > 0 and self.down_map[idx-1] == 1:
                concatputs.append(x)
            x = layer(x)
        
        x = self.conv(x)
        x = self.conv2(x)

        conc_put_idx = len(concatputs) - 1

        for idx, layer in enumerate(self.dec_layers):
            
            if idx < len(self.dec_layers) - 1:
                if self.down_map[-1-idx] == 1:
                    x = self.upsample(x)
                if self.down_map[-idx] == 1:
                    x = torch.cat((x, concatputs[conc_put_idx]), dim=1)
                    conc_put_idx -= 1

            if idx == len(self.dec_layers) - 1:
                x = self.upsample(x)

            x = layer(x)



        return x



enc_config = [
    # k, t, c, SE, HS, s 
    [3,   1,  16, 0, 0, 1],
    [3,   4,  24, 0, 0, 2],
    [3,   3,  24, 0, 0, 1],
    [5,   3,  40, 1, 0, 2],
    [5,   3,  40, 1, 0, 1],
    [5,   3,  40, 1, 0, 1],
    [3,   6,  80, 0, 1, 2],
    [3, 2.5,  80, 0, 1, 1],
    [3, 2.3,  80, 0, 1, 1],
    [3, 2.3,  80, 0, 1, 1],
    [3,   6, 112, 1, 1, 1],
    [3,   6, 112, 1, 1, 1],
    [5,   6, 160, 1, 1, 2],
    [5,   6, 160, 1, 1, 1],
    [5,   6, 160, 1, 1, 1]
]

#inp = torch.rand(1,3,256,256).to( torch.device("cuda:0") )
#out = model(inp)

