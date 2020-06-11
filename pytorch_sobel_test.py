import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import scipy.ndimage
import matplotlib.pyplot as plt


img = Image.open("test.png").convert("RGB")

to_tensor = transforms.ToTensor()
to_img = transforms.ToPILImage()

img = to_tensor(img).unsqueeze(0)



class EdgeDetector(nn.Module):
    def __init__(self, gaussian_size=11):
        super().__init__()

        self.gs = gaussian_size
        self.gp = self.gs // 2
        self.sigma = 1

        n = np.zeros((gaussian_size,gaussian_size))
        n[self.gp,self.gp] = 1
        self.gaussian_kernel = torch.tensor(scipy.ndimage.gaussian_filter(n,sigma=self.gp//2))

        self.gk = self.gaussian_kernel.unsqueeze(0).unsqueeze(0).expand(3,-1,-1,-1).float()
        self.gk = self.gk

        self.sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(3,-1,-1,-1)
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(3,-1,-1,-1)
        
        self.sobel_x = self.sobel_x
        self.sobel_y = self.sobel_y

    def forward(self, inputs, mask):
        seg_mask = mask != 0        
        noise_reducted = F.conv2d(inputs * seg_mask, self.gk, stride=1, padding=self.gp, groups=3)
        edge_x = F.conv2d(noise_reducted, self.sobel_x, stride=1, padding=1, groups=3)
        edge_y = F.conv2d(noise_reducted, self.sobel_y, stride=1, padding=1, groups=3)

        return torch.sqrt(torch.square(edge_x) + torch.square(edge_y))

ed = EdgeDetector(gaussian_size=7)
output = ed(img).squeeze(0)
print(output.shape)
to_img( output ).save("test_edge.png")

