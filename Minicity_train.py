# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 11:40:51 2019 by Attila Lengyel - attila@lengyel.nl
"""

from PIL import Image
import numpy as np
import os
import torch
import torchvision.transforms.functional as TF
from imgaug import augmenters as iaa
from torchvision.datasets import Cityscapes
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class MiniCity_train(Cityscapes):

	voidClass = 19

	# Convert ids to train_ids
	id2trainid = np.array([label.train_id for label in Cityscapes.classes if label.train_id >= 0], dtype='uint8')
	id2trainid[np.where(id2trainid == 255)] = voidClass

	# Convert train_ids to colors
	mask_colors = [list(label.color) for label in Cityscapes.classes if label.train_id >= 0 and label.train_id <= 19]
	mask_colors.append([0, 0, 0])
	mask_colors = np.array(mask_colors)

	# Convert train_ids to ids
	trainid2id = np.zeros((256), dtype='uint8')
	for label in Cityscapes.classes:
		if label.train_id >= 0 and label.train_id < 255:
			trainid2id[label.train_id] = label.id

	# List of valid class ids
	validClasses = np.unique([label.train_id for label in Cityscapes.classes if label.id >= 0])
	validClasses[np.where(validClasses == 255)] = voidClass
	validClasses = list(validClasses)

	# Create list of class names
	classLabels = [label.name for label in Cityscapes.classes if not (label.ignore_in_eval or label.id < 0)]
	classLabels.append('void')


	def train_trans(self, image, mask, index):

		image_scale_x = self.epoch_image_scales_x[index]
		image_scale_y = self.epoch_image_scales_y[index]

		hue_factor = 0.6
		brightness_factor = 0.6  # was 0.5
		p_flip = 0.5
		p_imgaug = 0.7

		p_jpeg = 0.4
		jpeg_scale = 0, 70  # was 70

		p_pixel_attack = 0.0
		pixel_attack_density = 0.05

		rotation_angle = (np.random.rand()-0.5)*20


		image = TF.resize(image, (self.image_size[0]*image_scale_x, self.image_size[1]*image_scale_y), interpolation=1)
		mask = TF.resize(mask, (self.image_size[0]*image_scale_x, self.image_size[1]*image_scale_y), interpolation=0)


		image = np.array(image)
		mask = np.array(mask)

		image = image[int(self.epoch_image_main_direcs[index,0]):int(self.epoch_image_main_direcs[index,0]+self.train_size[0]), int(self.epoch_image_main_direcs[index,1]):int(self.epoch_image_main_direcs[index,1]+self.train_size[1]), :]
		mask = mask[int(self.epoch_image_main_direcs[index,0]):int(self.epoch_image_main_direcs[index,0]+self.train_size[0]), int(self.epoch_image_main_direcs[index,1]):int(self.epoch_image_main_direcs[index,1]+self.train_size[1])]

		brightness = iaa.MultiplyBrightness((1 - brightness_factor, 1 + brightness_factor))
		hue = iaa.MultiplyHue((1 - hue_factor, 1 + hue_factor))
		jpeg = iaa.JpegCompression(compression=jpeg_scale)
		rotator = iaa.Affine(rotate=rotation_angle)

		if np.random.rand() < p_imgaug:

			img_transforms = iaa.Sequential([hue, rotator])
			image = img_transforms(image=image)

			rotator = iaa.Affine(rotate=rotation_angle, order=0, cval=19)

			mask_transforms = iaa.Sequential([rotator])
			mask = mask_transforms(image=mask)


		if np.random.rand() < p_flip:
			image = np.flip(image, axis=1)
			mask = np.flip(mask, axis=1)

		if np.random.rand() < p_pixel_attack:

			sel_pixels = np.random.choice(np.arange(self.train_size[0]*self.train_size[1]), int((self.train_size[0]*self.train_size[1]*pixel_attack_density)//1))

			rand_pixels = np.random.randint(0,255, (sel_pixels.shape[0],3))

			image = image.reshape((self.train_size[0]*self.train_size[1],3))

			image[sel_pixels] = rand_pixels

			image = image.reshape((self.train_size[0], self.train_size[1], 3))

			mask = mask.reshape((self.train_size[0]*self.train_size[1],1))

			mask[sel_pixels] = 19

			mask = mask.reshape((self.train_size[0],self.train_size[1]))





		image = Image.fromarray(image)
		mask = Image.fromarray(mask)

		# From PIL to Tensor
		image = TF.to_tensor(image)
		# Normalize
		image = TF.normalize(image, self.dataset_mean, self.dataset_std)

		# Convert ids to train_ids
		mask = np.array(mask, np.uint8)
		mask = torch.from_numpy(mask)  # Numpy array to tensor

		return image, mask

	def __init__(self, root, split='train', transform=None, target_transform=None, transforms=None, train_size = (384, 768),class_additions= 0, image_size = (512,1024)):
		super(Cityscapes, self).__init__(root, transforms, transform, target_transform)
		self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
		self.targets_dir = os.path.join(self.root, 'gtFine', split)
		self.split = split
		self.images = []
		self.targets = []
		self.centroids = np.load('classes_center_dict_trainval.npy', allow_pickle=True).item()
		self.train_size = train_size
		self.dataset_mean = [0.2870, 0.3257, 0.2854]
		self.dataset_std = [0.1879, 0.1908, 0.1880]
		self.class_additions = class_additions
		self.image_size = (512,1024)
		self.train_size = (384, 768)

		del self.centroids[11]
		del self.centroids[9]
		del self.centroids[12]

		assert split in ['train', 'val', 'test'], 'Unknown value {} for argument split.'.format(split)

		for file_name in os.listdir(self.images_dir):
			self.images.append(os.path.join(self.images_dir, file_name))
			if split != 'test':
				target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
											 'gtFine_labelIds.png')
				self.targets.append(os.path.join(self.targets_dir, target_name))

		self.epoch_image_names = []
		self.epoch_image_main_direcs = np.zeros((len(self.images)+self.class_additions*len(self.centroids.keys()),2))
		self.epoch_image_scales_x = np.zeros((len(self.images)+self.class_additions*len(self.centroids.keys()),1))
		self.epoch_image_scales_y = np.zeros((len(self.images)+self.class_additions*len(self.centroids.keys()),1))


	def create_an_epoch(self):

		self.epoch_image_names = []
		self.epoch_image_main_direcs = np.zeros((len(self.images)+self.class_additions*len(self.centroids.keys()),2))
		self.epoch_image_scales = np.zeros((len(self.images)+self.class_additions*len(self.centroids.keys()),1))


		for i in range(len(self.images)):

			self.epoch_image_names.append(self.images[i])

			self.epoch_image_scales_x[i] = np.random.uniform(0.75,2)
			self.epoch_image_scales_y[i] = self.epoch_image_scales_x[i]+ np.random.uniform(-1,1)

			if self.epoch_image_scales_y[i] < 0.75:
				self.epoch_image_scales_y[i] = 0.75


			new_image_0 = int(self.image_size[0]*self.epoch_image_scales_x[i])
			new_image_1 = int(self.image_size[1]*self.epoch_image_scales_y[i])

			self.epoch_image_main_direcs[i,0] = np.random.randint(0, new_image_0 - self.train_size[0]+1)
			self.epoch_image_main_direcs[i,1] = np.random.randint(0, new_image_1 - self.train_size[1]+1)

		counter = len(self.images)

		for j in range(len(self.centroids.keys())):

			for tour_id in range(self.class_additions):


				sel_key = list(self.centroids.keys())[j]

				sel_key_images = self.centroids[sel_key]

				sel_key_image_name = np.random.choice(list(sel_key_images.keys()))

				sel_key_images = sel_key_images[sel_key_image_name]

				sel_centroid = np.random.randint(sel_key_images.shape[0])
				sel_centroid = sel_key_images[sel_centroid,:]

				self.epoch_image_scales_x[counter] = np.random.uniform(0.75, 2)
				self.epoch_image_scales_y[counter] = self.epoch_image_scales_x[counter] + np.random.uniform(-1, 1)

				if self.epoch_image_scales_y[counter] < 0.75:
					self.epoch_image_scales_y[counter] = 0.75

				new_image_0 = int(self.image_size[0] * self.epoch_image_scales_x[counter])
				new_image_1 = int(self.image_size[1] * self.epoch_image_scales_y[counter])

				#sel_centroid = sel_centroid * self.epoch_image_scales[counter]

				sel_centroid_x = int(sel_centroid[0]*self.epoch_image_scales_x[counter])
				sel_centroid_y = int(sel_centroid[1]*self.epoch_image_scales_y[counter])

				x_starter = sel_centroid_x - self.train_size[0]
				y_starter = sel_centroid_y - self.train_size[1]

				if x_starter < 0:
					x_starter = 0

				if y_starter < 0:
					y_starter = 0

				x_ender = sel_centroid_x
				y_ender = sel_centroid_y

				if x_ender > new_image_0 - self.train_size[0]:
					x_ender = new_image_0-self.train_size[0]

				if y_ender > new_image_1  - self.train_size[1]:
					y_ender = new_image_1-self.train_size[1]

				self.epoch_image_main_direcs[counter, 0] = np.random.randint(x_starter, x_ender + 1)
				self.epoch_image_main_direcs[counter, 1] = np.random.randint(y_starter, y_ender + 1)

				self.epoch_image_names.append(self.images_dir+ '/'+ sel_key_image_name + '.png')

				counter += 1



	def __getitem__(self, index):
		"""
		Args:
			index (int): Index
		Returns:
			tuple: (image, target)
		"""

		filepath = self.epoch_image_names[index]
		image = Image.open(filepath).convert('RGB')

		target_path = filepath.replace('leftImg8bit','gtFine', 1)

		target_path = target_path.replace('leftImg8bit','gtFine_labelIds', 1)

		target = Image.open(target_path)

		image, target = self.train_trans(image, target, index)

		#print("X:" + str(np.unique(target)))

		target = self.id2trainid[target]

		#print("X:" + str(np.unique(target)))

		return image, target

	def __len__(self):
		return len(self.images)+self.class_additions*len(self.centroids.keys())



# trainset = MiniCity_train("./minicity", split='train', class_additions=20)
# trainset.create_an_epoch()
#
# trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1)
#
#
# for epoch_step, (inputs, labels) in enumerate(trainset_loader):
#
# 	if epoch_step < 200:
# 		continue
#
# 	inputs = (inputs.squeeze().permute((1,2,0)).float().numpy()*255).astype(np.uint8)
# 	labels = labels.long().numpy().squeeze().astype(np.uint8)
#
# 	cv2.imwrite(str(epoch_step) + '.png', inputs)
#
# 	unique_labels = np.unique(labels)
#
# 	sel_label = (epoch_step-200)//20 + 3
#
# 	labels_mask = (labels == sel_label).reshape(inputs.shape[0],inputs.shape[1],1)
# 	labels_mask = np.tile(labels_mask,(1,1,3))
#
#
# 	cv2.imwrite(str(epoch_step)+'-'+ str(sel_label)+'.png', inputs*labels_mask)
#


# def test_trans(image, mask=None):
# 	# Resize, 1 for Image.LANCZOS
# 	image = TF.resize(image, (1024,2048), interpolation=1)
# 	# From PIL to Tensor
# 	image = TF.to_tensor(image)
# 	# Normalize
# 	#image = TF.normalize(image, args.dataset_mean, args.dataset_std)
#
# 	if mask:
# 		# Resize, 0 for Image.NEAREST
# 		mask = TF.resize(mask, (1024,2048), interpolation=0)
# 		mask = np.array(mask, np.uint8) # PIL Image to numpy array
# 		mask = torch.from_numpy(mask) # Numpy array to tensor
# 		return image, mask
# 	else:
# 		return image
#
# from helpers.minicity import MiniCity
#
#
# valset = MiniCity("./minicity", split='val', transforms=test_trans)
#
# dataloaders_val = torch.utils.data.DataLoader(valset,
# 		   batch_size=1, shuffle=False,
# 		   pin_memory=True, num_workers=1)
#
# for epoch_step, (inputs, labels, filepath) in enumerate(dataloaders_val):
#
#
# 	inputs = (inputs.squeeze().permute((1,2,0)).float().numpy()*255).astype(np.uint8)
# 	labels = labels.long().numpy().squeeze().astype(np.uint8)
#
# 	cv2.imwrite(str(epoch_step) + '_ts.png', inputs)
#
# 	# unique_labels = np.unique(labels)
# 	#
# 	# for i in range(len(unique_labels)):
# 	#
# 	# 	sel_label = unique_labels[i]
# 	#
# 	# 	labels_mask = (labels == sel_label).reshape(inputs.shape[0],inputs.shape[1],1)
# 	# 	labels_mask = np.tile(labels_mask,(1,1,3))
# 	#
# 	#
# 	# 	cv2.imwrite(str(epoch_step)+'-'+ str(sel_label)+'_new.png', inputs*labels_mask)
#
# 	if epoch_step == 50:
# 		break
