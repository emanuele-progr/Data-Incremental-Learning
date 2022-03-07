import numpy as np
from sklearn.utils import shuffle
import torch
import torchvision
import os
import sys
from torch.utils.data import random_split, ConcatDataset
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as TorchVisionFunc
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torchvision import models, utils, datasets, transforms
from PIL import Image
from utils import get_seed


DATA_DIR = 'tiny-imagenet-200'  # Original images come in shapes of [3,64,64]

# Define training and validation data paths
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'val')

class TinyImageNet(Dataset):
	def __init__(self, root, train=True, transform=None):
		self.Train = train
		self.root_dir = root
		self.transform = transform
		self.train_dir = os.path.join(self.root_dir, "train")
		self.val_dir = os.path.join(self.root_dir, "val")

		if (self.Train):
			self._create_class_idx_dict_train()
		else:
			self._create_class_idx_dict_val()

		self._make_dataset(self.Train)

		words_file = os.path.join(self.root_dir, "words.txt")
		wnids_file = os.path.join(self.root_dir, "wnids.txt")

		self.set_nids = set()

		with open(wnids_file, 'r') as fo:
			data = fo.readlines()
			for entry in data:
				self.set_nids.add(entry.strip("\n"))

		self.class_to_label = {}
		with open(words_file, 'r') as fo:
			data = fo.readlines()
			for entry in data:
				words = entry.split("\t")
				if words[0] in self.set_nids:
					self.class_to_label[words[0]] = (
						words[1].strip("\n").split(","))[0]

	def _create_class_idx_dict_train(self):
		if sys.version_info >= (3, 5):
			classes = [d.name for d in os.scandir(
				self.train_dir) if d.is_dir()]
		else:
			classes = [d for d in os.listdir(
				self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
		classes = sorted(classes)
		num_images = 0
		for root, dirs, files in os.walk(self.train_dir):
			for f in files:
				if f.endswith(".JPEG"):
					num_images = num_images + 1

		self.len_dataset = num_images

		self.tgt_idx_to_class = {i: classes[i]
			for i in range(len(classes))}
		self.class_to_tgt_idx = {
			classes[i]: i for i in range(len(classes))}

	def _create_class_idx_dict_val(self):
		val_image_dir = os.path.join(self.val_dir, "images")
		if sys.version_info >= (3, 5):
			images = [d.name for d in os.scandir(
				val_image_dir) if d.is_file()]
		else:
			images = [d for d in os.listdir(
				val_image_dir) if os.path.isfile(os.path.join(train_dir, d))]
		val_annotations_file = os.path.join(
			self.val_dir, "val_annotations.txt")
		self.val_img_to_class = {}
		set_of_classes = set()
		with open(val_annotations_file, 'r') as fo:
			entry = fo.readlines()
			for data in entry:
				words = data.split("\t")
				self.val_img_to_class[words[0]] = words[1]
				set_of_classes.add(words[1])

		self.len_dataset = len(list(self.val_img_to_class.keys()))
		classes = sorted(list(set_of_classes))
		# self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
		self.class_to_tgt_idx = {
			classes[i]: i for i in range(len(classes))}
		self.tgt_idx_to_class = {i: classes[i]
			for i in range(len(classes))}

	def _make_dataset(self, Train=True):
		self.images = []
		if Train:
			img_root_dir = self.train_dir
			list_of_dirs = [
				target for target in self.class_to_tgt_idx.keys()]
		else:
			img_root_dir = self.val_dir
			list_of_dirs = ["images"]

		for tgt in list_of_dirs:
			dirs = os.path.join(img_root_dir, tgt)
			if not os.path.isdir(dirs):
				continue

			for root, _, files in sorted(os.walk(dirs)):
				for fname in sorted(files):
					if (fname.endswith(".JPEG")):
						path = os.path.join(root, fname)
						if Train:
							item = (path, self.class_to_tgt_idx[tgt])
						else:
							item = (
								path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
						self.images.append(item)

	def return_label(self, idx):
		return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

	def __len__(self):
		return self.len_dataset

	def __getitem__(self, idx):
		img_path, tgt = self.images[idx]
		with open(img_path, 'rb') as f:
			sample = Image.open(img_path)
			sample = sample.convert('RGB')
		if self.transform is not None:
			sample = self.transform(sample)

		return sample, tgt



def get_split_cifar100_tasks(num_tasks, batch_size):

    datasets = {}

    # convention: tasks starts from 1 not 0 !
    # task_id = 1 (i.e., first task) => start_class = 0, end_class = 4
    cifar_train_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, padding=4,padding_mode="reflect"),torchvision.transforms.ToTensor(),])
    cifar_test_transforms = torchvision.transforms.Compose(
       [torchvision.transforms.ToTensor(), ])
    cifar_train = torchvision.datasets.CIFAR100(
       './data/', train=True, download=True, transform=cifar_train_transforms)
    cifar_test = torchvision.datasets.CIFAR100(
       './data/', train=False, download=True, transform=cifar_test_transforms)

    num_elements_train = len(cifar_train)/num_tasks
    num_elements_test = len(cifar_test)/2


    test_ds, val_ds = random_split(cifar_test, [int(num_elements_test), int(
       num_elements_test)], generator=torch.Generator().manual_seed(get_seed()))

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

    train = cifar_train

    for task_id in range(1, num_tasks+1):

        train_ds, residual = random_split(train, [int(num_elements_train), int(
           (len(train)-num_elements_train))], generator=torch.Generator().manual_seed(get_seed()))
        train_loader = torch.utils.data.DataLoader(
           train_ds, batch_size=batch_size, shuffle=True)
        exemplar_loader = torch.utils.data.DataLoader(
           train_ds, batch_size=batch_size)
        train = residual

        datasets[task_id] = {'train': train_loader, 'test': test_loader,
           'val': val_loader, 'exemplar': exemplar_loader}

    return datasets



def get_split_cifar10_tasks(num_tasks, batch_size):

    datasets = {}

    # convention: tasks starts from 1 not 0 !
    # task_id = 1 (i.e., first task) => start_class = 0, end_class = 4
    cifar_train_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, padding=4,padding_mode="reflect"),torchvision.transforms.ToTensor(),])
    cifar_test_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
    cifar_train = torchvision.datasets.CIFAR10(
       './data/', train=True, download=True, transform=cifar_train_transforms)
    cifar_test = torchvision.datasets.CIFAR10(
       './data/', train=False, download=True, transform=cifar_test_transforms)

    num_elements_train = len(cifar_train)/num_tasks
    num_elements_test = len(cifar_test)/2

    test_ds, val_ds = random_split(cifar_test, [int(num_elements_test), int(
       num_elements_test)], generator=torch.Generator().manual_seed(get_seed()))

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

    train = cifar_train

    for task_id in range(1, num_tasks+1):

        train_ds, residual = random_split(train, [int(num_elements_train), int(
           (len(train)-num_elements_train))], generator=torch.Generator().manual_seed(get_seed()))
        train_loader = torch.utils.data.DataLoader(
           train_ds, batch_size=batch_size, shuffle=True)
        exemplar_loader = torch.utils.data.DataLoader(
           train_ds, batch_size=batch_size)
        train = residual

        datasets[task_id] = {'train': train_loader,
           'test': test_loader, 'val': val_loader, 'exemplar': exemplar_loader}

    return datasets


def get_split_MNIST_tasks(num_tasks, batch_size):

    datasets = {}

    # convention: tasks starts from 1 not 0 !
    # task_id = 1 (i.e., first task) => start_class = 0, end_class = 4

    mnist_train_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomVerticalFlip(), torchvision.transforms.RandomRotation(5),torchvision.transforms.ToTensor(),])
    mnist_test_transforms = torchvision.transforms.Compose(
       [torchvision.transforms.ToTensor(), ])
    mnist_train = torchvision.datasets.MNIST(
       './data/', train=True, download=True, transform=mnist_train_transforms)
    mnist_test = torchvision.datasets.MNIST(
       './data/', train=False, download=True, transform=mnist_test_transforms)

    num_elements_train = len(mnist_train)/num_tasks
    num_elements_test = len(mnist_test)/2

    test_ds, val_ds = random_split(mnist_test, [int(num_elements_test), int(
       num_elements_test)], generator=torch.Generator().manual_seed(get_seed()))

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

    list_item = list(range(len(mnist_train.targets)))

    train = mnist_train

    for task_id in range(1, num_tasks+1):

        train_ds, residual = random_split(train, [int(num_elements_train), int(
           (len(train)-num_elements_train))], generator=torch.Generator().manual_seed(get_seed()))
        train_loader = torch.utils.data.DataLoader(
           train_ds, batch_size=batch_size, shuffle=True)
        exemplar_loader = torch.utils.data.DataLoader(
           train_ds, batch_size=batch_size)
        train = residual

        datasets[task_id] = {'train': train_loader, 'test': test_loader,
           'val': val_loader, 'exemplar': exemplar_loader}

    return datasets


def get_split_cifar100_tasks_with_augment(num_tasks, batch_size):

    datasets = {}

    # convention: tasks starts from 1 not 0 !
    # task_id = 1 (i.e., first task) => start_class = 0, end_class = 4
    cifar_train_transforms_with_aug = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, padding=4,padding_mode="reflect"), torchvision.transforms.ColorJitter(brightness = 0.25, contrast= 0.25, saturation = 0.25, hue = 0.25), torchvision.transforms.ToTensor(),])
    cifar_test_transforms = torchvision.transforms.Compose(
       [torchvision.transforms.ToTensor(), ])
    cifar_train = torchvision.datasets.CIFAR100(
       './data/', train=True, download=True, transform=cifar_train_transforms_with_aug)
    cifar_test = torchvision.datasets.CIFAR100(
       './data/', train=False, download=True, transform=cifar_test_transforms)

    num_elements_train = len(cifar_train)/num_tasks
    num_elements_test = len(cifar_test)/2
    num_aug = 2000

    #test_indices, _ = train_test_split(list(range(len(cifar_test.targets))), train_size = num_elements_test, stratify = cifar_test.targets)
    #test_dataset = torch.utils.data.Subset(cifar_test, test_indices)

    test_ds, val_ds = random_split(cifar_test, [int(num_elements_test), int(
       num_elements_test)], generator=torch.Generator().manual_seed(get_seed()))

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

    list_item = list(range(len(cifar_train.targets)))

    train = cifar_train

    for task_id in range(1, num_tasks+1):

        train_ds, residual = random_split(train, [int(num_elements_train), int(
           (len(train)-num_elements_train))], generator=torch.Generator().manual_seed(get_seed()))
        if task_id == 1:
            aug_ds, aug_residual = random_split(train, [int(num_elements_train), int(
               (len(train)-num_elements_train))], generator=torch.Generator().manual_seed(get_seed()))
            aug_ds, aug_residual = random_split(aug_ds, [int(num_aug), int(
               (len(aug_ds)-num_aug))], generator=torch.Generator().manual_seed(get_seed()))
            train_loader = torch.utils.data.DataLoader(
               train_ds + aug_ds, batch_size=batch_size, shuffle=True)
        else:
            train_loader = torch.utils.data.DataLoader(
               train_ds, batch_size=batch_size, shuffle=True)
        exemplar_loader = torch.utils.data.DataLoader(
           train_ds, batch_size=batch_size)
        train = residual

        datasets[task_id] = {'train': train_loader, 'test': test_loader,
           'val': val_loader, 'exemplar': exemplar_loader}

    return datasets


def get_split_cifar100_tasks_with_exemplars_linear_memory(num_tasks, batch_size):

    datasets = {}

    # convention: tasks starts from 1 not 0 !
    # task_id = 1 (i.e., first task) => start_class = 0, end_class = 4
    cifar_train_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, padding=4,padding_mode="reflect"),torchvision.transforms.ToTensor(),])
    cifar_test_transforms = torchvision.transforms.Compose(
       [torchvision.transforms.ToTensor(), ])
    cifar_train = torchvision.datasets.CIFAR100(
       './data/', train=True, download=True, transform=cifar_train_transforms)
    cifar_test = torchvision.datasets.CIFAR100(
       './data/', train=False, download=True, transform=cifar_test_transforms)

    num_elements_train = len(cifar_train)/num_tasks
    num_elements_test = len(cifar_test)/2

    test_ds, val_ds = random_split(cifar_test, [int(num_elements_test), int(
       num_elements_test)], generator=torch.Generator().manual_seed(get_seed()))

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

    train = cifar_train
    accumulator = None
    exemplar_loader_list = []

    for task_id in range(1, num_tasks+1):

        train_ds, residual = random_split(train, [int(num_elements_train), int(
           (len(train)-num_elements_train))], generator=torch.Generator().manual_seed(get_seed()))
        accumulator = train_ds
        if task_id > 1:
            for exemplars in exemplar_loader_list:
                accumulator += exemplars
        train_loader = torch.utils.data.DataLoader(
           train_ds, batch_size=batch_size, shuffle=True)
        exemplar_loader = torch.utils.data.DataLoader(
           accumulator, batch_size=batch_size)
        train = residual

        datasets[task_id] = {'train': train_loader, 'test': test_loader,
           'val': val_loader, 'exemplar': exemplar_loader}

    return datasets


def organize_validation_data_tiny_ImageNet():

    val_img_dir = os.path.join(VALID_DIR, 'images')
    # Open and read val annotations text file
    fp = open(os.path.join(VALID_DIR, 'val_annotations.txt'), 'r')
    data = fp.readlines()
    # # Create dictionary to store img filename (word 0) and corresponding
    # # label (word 1) for every line in the txt file (as key value pair)
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(val_img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(val_img_dir, img)):
            os.rename(os.path.join(val_img_dir, img),
                      os.path.join(newpath, img))

    return


def get_split_tiny_ImageNet_tasks(num_tasks, batch_size):

    datasets = {}
    imageNet_train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
            torchvision.transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(), ])
    imageNet_test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor(), ])

    imageNet_train = TinyImageNet(DATA_DIR, train=True, transforms=imageNet_train_transforms)
    imageNet_test = TinyImageNet(DATA_DIR, train=False, transforms=imageNet_test_transforms)
    num_elements_train = len(imageNet_train)/num_tasks
    num_elements_test = len(imageNet_test)/2

    #test_indices, _ = train_test_split(list(range(len(cifar_test.targets))), train_size = num_elements_test, stratify = cifar_test.targets)
    #test_dataset = torch.utils.data.Subset(cifar_test, test_indices)

    test_ds, val_ds = random_split(imageNet_test, [int(num_elements_test), int(
       num_elements_test)], generator=torch.Generator().manual_seed(get_seed()))

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

    list_item = list(range(len(imageNet_train.targets)))

    train = imageNet_train

    for task_id in range(1, num_tasks+1):

        train_ds, residual = random_split(train, [int(num_elements_train), int(
           (len(train)-num_elements_train))], generator=torch.Generator().manual_seed(get_seed()))
        train_loader = torch.utils.data.DataLoader(
           train_ds, batch_size=batch_size, shuffle=True)
        exemplar_loader = torch.utils.data.DataLoader(
           train_ds, batch_size=batch_size)
        train = residual

        datasets[task_id] = {'train': train_loader, 'test': test_loader,
           'val': val_loader, 'exemplar': exemplar_loader}

    return datasets



def get_split_cifar100_tasks_joint(num_tasks, batch_size):

    datasets = {}

    # convention: tasks starts from 1 not 0 !
    # task_id = 1 (i.e., first task) => start_class = 0, end_class = 4
    cifar_train_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomCrop(32, padding=4, padding_mode="reflect"),torchvision.transforms.ToTensor(),])
    cifar_test_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), ])
    cifar_train = torchvision.datasets.CIFAR100(
        './data/', train=True, download=True, transform=cifar_train_transforms)
    cifar_test = torchvision.datasets.CIFAR100(
        './data/', train=False, download=True, transform=cifar_test_transforms)

    num_elements_train = len(cifar_train)/num_tasks
    num_elements_test = len(cifar_test)/2

    test_ds, val_ds = random_split(cifar_test, [int(num_elements_test), int(
       num_elements_test)], generator=torch.Generator().manual_seed(get_seed()))

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)
    exemplar_loader = []

    train = cifar_train

    for task_id in range(1, num_tasks+1):

        train_ds, residual = random_split(train, [int(num_elements_train), int(
            (len(train)-num_elements_train))], generator=torch.Generator().manual_seed(get_seed()))
         
        if task_id == 1:
            train_j = train_ds
        else:
            train_j = torch.utils.data.ConcatDataset([train_j, train_ds])
        exemplar_loader = torch.utils.data.DataLoader(
           train_ds, batch_size=batch_size)
        train_loader = torch.utils.data.DataLoader(
           train_j, batch_size=batch_size, shuffle=True)
        train = residual

        datasets[task_id] = {'train': train_loader,
           'test': test_loader, 'val': val_loader, 'exemplar': exemplar_loader}

    return datasets
