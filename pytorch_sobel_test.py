import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import scipy.ndimage
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


img = Image.open("test.png").convert("RGB")

to_tensor = transforms.ToTensor()
to_img = transforms.ToPILImage()

img = to_tensor(img).unsqueeze(0)




class EdgeDetector(nn.Module):
    def __init__(self, gaussian_size=11):
        super().__init__()

        self.gs = gaussian_size
        self.gp = self.gs // 2+1

        self.gk = torch.tensor(gaussian_kernel(self.gs)).float().unsqueeze(0).unsqueeze(0).expand(3,-1,-1,-1)
        self.sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(3,-1,-1,-1)
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(3,-1,-1,-1)
        
    def forward(self, inputs):
        noise_reducted = F.conv2d(inputs, self.gk, stride=1, padding=self.gp, groups=3)
        edge_x = F.conv2d(noise_reducted, self.sobel_x, stride=1, padding=1, groups=3)
        edge_y = F.conv2d(noise_reducted, self.sobel_y, stride=1, padding=1, groups=3)

        return torch.sqrt(torch.square(edge_x) + torch.square(edge_y))


ed = EdgeDetector()

to_img( ed(img).squeeze(0)).save("test_edge.png")

