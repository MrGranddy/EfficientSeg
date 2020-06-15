import os, shutil
import matplotlib as plt
import numpy as np
from PIL import Image
from helpers.minicity import MiniCity
import torchvision.transforms.functional as TF
import torch
from model import EfficientSeg


model_pack = torch.load("baseline_run/best_weights.pth.tar")
model_state_dict = model_pack["model_state_dict"]

batch_size = 1
num_workers = 8
dataset_path = "./minicity"

dataset_mean = [0.2870, 0.3257, 0.2854]
dataset_std = [0.1879, 0.1908, 0.1880]

def transform(image, mask):
    th, tw = 384, 768

    # Resize, 1 for Image.LANCZOS
    image = TF.resize(image, (th, tw), interpolation=1)
    # Resize, 0 for Image.NEAREST
    mask = TF.resize(mask, (th, tw), interpolation=0)

    # From PIL to Tensor
    image = TF.to_tensor(image)
    
    # Normalize
    image = TF.normalize(image, dataset_mean, dataset_std)
    
    # Convert ids to train_ids
    mask = np.array(mask, np.uint8) # PIL Image to numpy array
    mask = torch.from_numpy(mask) # Numpy array to tensor
        
    return image, mask

def transform_test(image):
    th, tw = 384, 768

    # Resize, 1 for Image.LANCZOS
    image = TF.resize(image, (th, tw), interpolation=1)

    # From PIL to Tensor
    image = TF.to_tensor(image)
    
    # Normalize
    image = TF.normalize(image, dataset_mean, dataset_std)

    return image


model = EfficientSeg(20, width_coeff=1.0, depth_coeff=1.0)
model.load_state_dict(model_state_dict)
model = model.eval().to( torch.device("cuda:0") )


trainset = MiniCity(dataset_path, split='train', transforms=transform)
valset = MiniCity(dataset_path, split='val', transforms=transform)
testset = MiniCity(dataset_path, split='test', transforms=transform_test)

dataloaders = {}    
dataloaders['train'] = torch.utils.data.DataLoader(trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers)
dataloaders['val'] = torch.utils.data.DataLoader(valset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers)
dataloaders['test'] = torch.utils.data.DataLoader(testset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers)

dest_dir = "mid_outputs/"
if os.path.isdir(dest_dir):
    shutil.rmtree(dest_dir)
os.makedirs(dest_dir)


key = "train"
sub_dir = dest_dir + key + "/"
os.makedirs(sub_dir)
for epoch_step, (inputs, _, _) in enumerate(dataloaders[key]):
    inputs = inputs.to( torch.device("cuda:0") )
    outputs = model(inputs, give_mid_output=True)
    sample_dir = sub_dir + str(epoch_step) + "/"
    os.makedirs( sample_dir )
    for idx, output in enumerate(outputs):
        out = torch.mean(output.squeeze(0), dim=0, keepdim=True)
        img = TF.to_pil_image(out.cpu()).save(sample_dir + str(idx) + ".png")

key = "val"
sub_dir = dest_dir + key + "/"
os.makedirs(sub_dir)
for epoch_step, (inputs, _, _) in enumerate(dataloaders[key]):
    inputs = inputs.to( torch.device("cuda:0") )
    outputs = model(inputs, give_mid_output=True)
    print(epoch_step)

"""
key = "test"
sub_dir = dest_dir + key + "/"
os.makedirs(sub_dir)
for epoch_step, (inputs, _) in enumerate(dataloaders[key]):
    inputs = inputs.to( torch.device("cuda:0") )
    outputs = model(inputs, give_mid_output=True)
    print(epoch_step)
"""