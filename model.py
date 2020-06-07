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
    def __init__(self, num_class, options):
        super().__init__()

        self.initial_channel_size = 64
        self.num_class = num_class
        self.num_repeats = options["num_repeats"]
        self.kernel_sizes = options["kernel_sizes"]

        self.forward_stages = nn.ModuleList([])
        self.forward_stages.append(
            nn.Sequential(
                nn.Conv2d(3, self.initial_channel_size, kernel_size=1, stride=1), # forward
                MobileBlock(self.initial_channel_size, self.initial_channel_size) # forward
            )
        )

        for i, (rep, ker) in enumerate(zip(self.num_repeats, self.kernel_sizes)):
            in_chn = self.initial_channel_size * int(2**i)
            out_chn = self.initial_channel_size * int(2**(i+1))
            seq = [ MobileBlock(in_chn, out_chn, stride=2, expand_ratio=6, kernel_size=ker) ] # down
            for _ in range(rep-1):
                seq.append( MobileBlock(out_chn, out_chn, expand_ratio=6, kernel_size=ker) ) # forward
            self.forward_stages.append( nn.Sequential(*seq) )

        self.backward_stages = nn.ModuleList([])
        self.backward_stages.append(
            MobileBlock(self.initial_channel_size * 16, self.initial_channel_size * 8, stride=0.5) # up
        )

        for i, (rep, ker) in enumerate( list(zip(self.num_repeats[::-1], self.kernel_sizes[::-1]))[1:] ):
            in_chn = self.initial_channel_size * int(2**(3-i))
            out_chn = self.initial_channel_size * int(2**(2-i))
            self.backward_stages.append(
                MobileBlock(in_chn*2,in_chn, expand_ratio=6, kernel_size=ker) # concat and forward
            )
            seq = []
            for _ in range(rep-1):
                seq.append( MobileBlock(in_chn, in_chn, expand_ratio=6, kernel_size=ker) ) # forward
            seq.append( MobileBlock(in_chn, out_chn, stride=0.5, kernel_size=ker) ) # up
            self.backward_stages.append( nn.Sequential( *seq ) )
            
        self.backward_stages.append(
            MobileBlock(self.initial_channel_size * 2, self.initial_channel_size) # last forward
        )
        self.backward_stages.append(
            nn.Conv2d(self.initial_channel_size, self.num_class, kernel_size=1, stride=1) # final
        )


    def forward(self, inputs):

        x = inputs
        outputs = []

        for stage in self.forward_stages:
            x = stage(x)
            outputs.append(x)

        x = self.backward_stages[0](x)
        
        for idx in range(1,len(self.backward_stages)):
            stage = self.backward_stages[idx]
            if idx % 2 == 1:
                x = stage(x, concat=outputs[-2-idx//2])
            else:
                x = stage(x)


        return x


#from torchsummary import summary

#options = { "num_repeats": [1,1,1,1], "kernel_sizes": [3,5,5,3] }
#model = EfficientSeg(33, options=options).to( torch.device("cuda:0") )

#summary(model, input_size=(3,384,768))

#inp = torch.rand(1,3,256,256).to( torch.device("cuda:0") )
#out = model(inp)

