import os, shutil
import matplotlib as plt
import numpy as np
from PIL import Image
from helpers.minicity import MiniCity
import torchvision.transforms.functional as TF
import torch

def train_trans(image, mask):

    # From PIL to Tensor
    image = TF.to_tensor(image)

    # Convert ids to train_ids
    mask = np.array(mask, np.uint8) # PIL Image to numpy array
    mask = torch.from_numpy(mask) # Numpy array to tensor
        
    return image, mask

trainset = MiniCity("./minicity", split='train', transforms=train_trans)

loader = torch.utils.data.DataLoader(trainset,
            batch_size=16, shuffle=True, num_workers=8)


mean_img = torch.zeros(3, 1024, 2048)
cnt = 0

for epoch_step, (inputs, _, _) in enumerate(loader):
    mean_img += torch.sum(inputs, dim=0)
    cnt += inputs.shape[0]

mean_img /= cnt

mean_R, mean_G, mean_B = torch.mean(mean_img, dim=(1,2))

print("Mean:", mean_R, mean_G, mean_B)

std_img = torch.zeros(3, 1024, 2048)
mean_tensor = torch.tensor([mean_R, mean_G, mean_B]).reshape(1,3,1,1)
cnt = 0

for epoch_step, (inputs, _, _) in enumerate(loader):
    std_img += torch.sum((inputs - mean_tensor) ** 2, dim=0)
    cnt += inputs.shape[0]

std_img /= cnt
std_R, std_G, std_B = (torch.mean(std_img, dim=(1,2))) ** 0.5

print("Std:", std_R, std_G, std_B)

class_counts = [0 for _ in range(20)]

for epoch_step, (_, labels, _) in enumerate(loader):
    for idx in range(20):
        class_counts[idx] += int(torch.sum(labels == idx))

scaled_counts = [ x / max(class_counts) for x in class_counts]
coeffs = [ 1 / x for x in scaled_counts]
print(coeffs)
