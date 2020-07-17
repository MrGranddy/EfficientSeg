import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
from torchvision import transforms

import time
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import warnings
import random
from PIL import Image
from Minicity_train import MiniCity_train
from helpers.model import UNet
from helpers.minicity import MiniCity
from helpers.helpers import AverageMeter, ProgressMeter, iouCalc
from model import enc_config
from model import EfficientSeg
import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torchvision.datasets import Cityscapes

import warnings

test_size = [512,1024]
dataset_mean = [0.2870, 0.3257, 0.2854]
dataset_std = [0.1879, 0.1908, 0.1880]
from imgaug import augmenters as iaa

voidClass = 19

# Convert ids to train_ids
id2trainid = np.array([label.train_id for label in Cityscapes.classes if label.train_id >= 0], dtype='uint8')
id2trainid[np.where(id2trainid == 255)] = voidClass

criterion = nn.CrossEntropyLoss(ignore_index=MiniCity.voidClass, weight=torch.from_numpy(np.array([1.0,  # road
																								   1.0,  # sidewalk
																								   1.0,  # building
																								   2.0,  # wall
																								   2.0,  # fence
																								   2.0,  # pole
																								   1.0,  # traffic light
																								   1.0,  # traffic sign
																								   1.0,  # vegetation
																								   1.0,  # terrain
																								   1.0,  # sky
																								   1.0,  # person
																								   2.0,  # rider
																								   1.0,  # car
																								   3.0,  # truck
																								   3.0,  # bus
																								   3.0,  # train
																								   2.0,  # motorcycle
																								   2.0,  # bicycle
																								   2.0]  # void
																								  )).float().cuda())


def test_trans_5deg(image, mask=None):
	# Resize, 1 for Image.LANCZOS
	image = TF.resize(image, test_size, interpolation=1)

	image = np.array(image)


	image_reshaped = np.zeros((image.shape[0]*3,image.shape[1]*3,3))

	image_reshaped[image.shape[0]:image.shape[0]*2, image.shape[1]:image.shape[1]*2,:] = image

	#rotator = iaa.Affine(rotate=0)
	#image = rotator(image=image_reshaped)

	image = image_reshaped

	# From PIL to Tensor
	image = TF.to_tensor(image)#torch.Size([3, 512, 1024])

	# Normalize
	image = TF.normalize(image, dataset_mean, dataset_std)


	mask = TF.resize(mask, test_size, interpolation=0)
	mask = np.array(mask, np.uint8) # PIL Image to numpy array
	mask = torch.from_numpy(mask) # Numpy array to tensor
	return image, mask

def test_trans(image, mask=None):
	# Resize, 1 for Image.LANCZOS
	image = TF.resize(image, test_size, interpolation=1)
	# From PIL to Tensor
	image = TF.to_tensor(image)
	# Normalize
	image = TF.normalize(image, dataset_mean, dataset_std)

	if mask:
		# Resize, 0 for Image.NEAREST
		mask = TF.resize(mask, test_size, interpolation=0)
		mask = np.array(mask, np.uint8) # PIL Image to numpy array
		mask = torch.from_numpy(mask) # Numpy array to tensor
		return image, mask
	else:
		return image

def vislbl(label, mask_colors):
	label = label.cpu()
	# Convert label data to visual representation
	label = np.array(label.numpy())
	if label.shape[-1] == 1:
		label = label[:,:,0]

	# Convert train_ids to colors
	label = mask_colors[label]
	return label

def visim(img):
	img = img.cpu()
	# Convert image data to visual representation
	img *= torch.tensor(dataset_std)[:,None,None]
	img += torch.tensor(dataset_mean)[:,None,None]
	npimg = (img.numpy()*255).astype('uint8')
	if len(npimg.shape) == 3 and npimg.shape[0] == 3:
		npimg = np.transpose(npimg, (1, 2, 0))
	else:
		npimg = npimg[0,:,:]
	return npimg



def validate_epoch(dataloader, model, criterion, epoch, classLabels, validClasses, void=-1, maskColors=None, flip = False,deg=None):
	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	loss_running = AverageMeter('Loss', ':.4e')
	acc_running = AverageMeter('Accuracy', ':.4e')
	iou = iouCalc(classLabels, validClasses, voidClass = void)
	progress = ProgressMeter(
		len(dataloader),
		[batch_time, data_time, loss_running, acc_running],
		prefix="Test, epoch: [{}]".format(epoch))

	# input resolution
	res = test_size[0]*test_size[1]

	all_predictions = torch.zeros((200, 20, test_size[0],test_size[1])).float().cuda()
	all_labels = torch.zeros((200, test_size[0],test_size[1])).long().cuda()

	# Set model in evaluation mode
	model.eval() # TODO ADD PLATO SCHEDULAR INSPECT LOSSES

	all_filepaths = []
	with torch.no_grad():
		end = time.time()
		for epoch_step, (inputs, labels, filepath) in enumerate(dataloader):

			filepath = filepath[0].split('/')[-1]

			data_time.update(time.time()-end)


			inputs = inputs.float().cuda()
			labels = labels.long().cuda()

			if flip:
				idx = [i for i in range(inputs.shape[3] - 1, -1, -1)]
				idx = torch.LongTensor(idx)

				inputs = inputs[:,:,:,idx]

			# forward
			outputs = model(inputs)


			if deg != None:

				preds = outputs.squeeze().permute((1,2,0)).cpu().numpy()

				#rotator = iaa.Affine(rotate=-deg, order=0, cval=19)
				outputs = torch.from_numpy(preds[test_size[0]:test_size[0]*2, test_size[1]:test_size[1]*2,:]).cuda().permute((2,0,1)).unsqueeze(0)


			if flip:

				idx = [i for i in range(1024 - 1, -1, -1)]
				idx = torch.LongTensor(idx)

				outputs = outputs[:,:,:,idx]

			all_predictions[epoch_step,:,:,:] = F.softmax(outputs,1)
			all_labels[epoch_step,:,:] = labels


			preds = torch.argmax(outputs, 1)
			loss = criterion(outputs, labels)

			# Statistics
			bs = inputs.size(0) # current batch size
			loss = loss.item()
			loss_running.update(loss, bs)
			corrects = torch.sum(preds == labels.data)
			nvoid = int((labels==void).sum())
			acc = corrects.double()/(bs*res-nvoid) # correct/(batch_size*resolution-voids)
			acc_running.update(acc, bs)
			# Calculate IoU scores of current batch

			iou.evaluateBatch(preds, labels)

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			# print progress info
			progress.display(epoch_step)

			all_filepaths.append(filepath)

		miou = iou.outputScores()
		print('Accuracy      : {:5.3f}'.format(acc_running.avg))
		print('---------------------')

	return acc_running.avg, loss_running.avg, miou, all_predictions, all_labels, all_filepaths


model = EfficientSeg(enc_config=enc_config, dec_config=None, num_classes=len(MiniCity.validClasses),
					 width_coeff=6.0)

model = model.cuda()



#model_states = ['best_weights0.5701434786223034.pth.tar', 'best_weights0.5701709298202121.pth.tar', 'best_weights0.5705692485382649.pth.tar', 'best_weights0.5706960924935263.pth.tar',
#				'best_weights0.5707014705959753.pth.tar', 'best_weights0.5708345545793978.pth.tar', 'best_weights0.5713351926425956.pth.tar', 'best_weights0.5740528525200369.pth.tar']

model_states = ['best_weights0.5740528525200369.pth.tar']


image_predictions = torch.zeros((200,20,test_size[0],test_size[1])).float()
image_labels = torch.zeros((200,test_size[0],test_size[1])).long()

all_filepaths = []


for i in range(len(model_states)):

	checkpoint = torch.load('/mnt/disk5/EfficientSeg_attrous/baseline_run/' +model_states[0])

	model.load_state_dict(checkpoint['model_state_dict'], strict=True)


	# testset_5deg = MiniCity('./minicity', split='val', transforms=test_trans_5deg)
	#
	# dataloader_test = torch.utils.data.DataLoader(testset_5deg,
	# 											  batch_size=1, shuffle=False,
	# 											  pin_memory=True, num_workers=2)
	#
	#
	# val_acc, val_loss, miou, all_predictions, all_labels = validate_epoch(dataloader_test,
	# 										 model,
	# 										 criterion, i,
	# 										 MiniCity.classLabels,
	# 										 MiniCity.validClasses,
	# 										 void=MiniCity.voidClass,
	# 										 maskColors=MiniCity.mask_colors, flip=False,deg=0)
	#
	# image_predictions += all_predictions.cpu()
	# image_labels = all_labels.cpu()


	testset = MiniCity('./minicity', split='val', transforms=test_trans)

	dataloader_test = torch.utils.data.DataLoader(testset,
												  batch_size=1, shuffle=False,
												  pin_memory=True, num_workers=2)



	val_acc, val_loss, miou, all_predictions, all_labels, all_filepaths = validate_epoch(dataloader_test,
											 model,
											 criterion, i,
											 MiniCity.classLabels,
											 MiniCity.validClasses,
											 void=MiniCity.voidClass,
											 maskColors=MiniCity.mask_colors, flip=True, deg=None)

	image_predictions += all_predictions.cpu()
	image_labels = all_labels.cpu()

	val_acc, val_loss, miou, all_predictions, all_labels, all_filepaths = validate_epoch(dataloader_test,
											 model,
											 criterion, i,
											 MiniCity.classLabels,
											 MiniCity.validClasses,
											 void=MiniCity.voidClass,
											 maskColors=MiniCity.mask_colors, flip=False, deg=None)

	image_predictions += all_predictions.cpu()
	image_labels = all_labels.cpu()




#softmaxed_predictions = F.softmax(image_predictions, 2)

#print(softmaxed_predictions.shape)

#image_predictions = torch.sum(image_predictions,0)

#print(softmaxed_predictions.shape)

preds = torch.argmax(image_predictions, 1)


iou = iouCalc(MiniCity.classLabels, MiniCity.validClasses, voidClass = MiniCity.voidClass)


iou.evaluateBatch(preds, image_labels)

miou = iou.outputScores()

for i in range(preds.shape[0]):

	selected_filepath = all_filepaths[i]
	selected_preds = preds[i,:,:]

	pred_id = MiniCity.trainid2id[selected_preds]
	pred_id = Image.fromarray(pred_id)
	pred_id = pred_id.resize((2048, 1024), resample=Image.NEAREST)
	pred_id.save('results/'+ selected_filepath)

#image_predictions = image_predictions.cpu().numpy()
#image_labels = image_labels.cpu().numpy()

#print(np.sum(image_labels[0,:,:,:]-image_labels[1,:,:,:]))

#np.save('/mnt/disk2/images_labels.npy', {'image_predictions': image_predictions,'image_labels': image_labels[0,:,:,:]})