import numpy as np
import torch
import torchvision
import aug_lib
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split, ConcatDataset
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as TorchVisionFunc


def get_permuted_mnist(task_id, batch_size):
	"""
	Get the dataset loaders (train and test) for a `single` task of permuted MNIST.
	This function will be called several times for each task.
	
	:param task_id: id of the task [starts from 1]
	:param batch_size:
	:return: a tuple: (train loader, test loader)
	"""
	
	# convention, the first task will be the original MNIST images, and hence no permutation
	if task_id == 1:
		idx_permute = np.array(range(784))
	else:
		idx_permute = torch.from_numpy(np.random.RandomState().permutation(784))
	transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
				torchvision.transforms.Lambda(lambda x: x.view(-1)[idx_permute] ),
				])
	mnist_train = torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transforms)
	train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)
	test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=True, transform=transforms),  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

	return train_loader, test_loader


def get_permuted_mnist_tasks(num_tasks, batch_size):
	"""
	Returns the datasets for sequential tasks of permuted MNIST
	
	:param num_tasks: number of tasks.
	:param batch_size: batch-size for loaders.
	:return: a dictionary where each key is a dictionary itself with train, and test loaders.
	"""
	datasets = {}
	for task_id in range(1, num_tasks+1):
		train_loader, test_loader = get_permuted_mnist(task_id, batch_size)
		datasets[task_id] = {'train': train_loader, 'test': test_loader}
	return datasets


class RotationTransform:
	"""
	Rotation transforms for the images in `Rotation MNIST` dataset.
	"""
	def __init__(self, angle):
		self.angle = angle

	def __call__(self, x):
		return TorchVisionFunc.rotate(x, self.angle, fill=(0,))


def get_rotated_mnist(task_id, batch_size):
	"""
	Returns the dataset for a single task of Rotation MNIST dataset
	:param task_id:
	:param batch_size:
	:return:
	"""
	per_task_rotation = 10
	rotation_degree = (task_id - 1)*per_task_rotation
	rotation_degree -= (np.random.random()*per_task_rotation)

	transforms = torchvision.transforms.Compose([
		RotationTransform(rotation_degree),
		torchvision.transforms.ToTensor(),
		])

	train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transforms), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=True, transform=transforms),  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

	return train_loader, test_loader


def get_rotated_mnist_tasks(num_tasks, batch_size):
	"""
	Returns data loaders for all tasks of rotation MNIST dataset.
	:param num_tasks: number of tasks in the benchmark.
	:param batch_size:
	:return:
	"""
	datasets = {}
	for task_id in range(1, num_tasks+1):
		train_loader, test_loader = get_rotated_mnist(task_id, batch_size)
		datasets[task_id] = {'train': train_loader, 'test': test_loader}
	return datasets


def get_split_cifar100(task_id, batch_size, cifar_train, cifar_test):
	"""
	Returns a single task of split CIFAR-100 dataset
	:param task_id:
	:param batch_size:
	:return:
	"""
	

	start_class = (task_id-1)*5
	end_class = task_id * 5

	targets_train = torch.tensor(cifar_train.targets)
	target_train_idx = ((targets_train >= start_class) & (targets_train < end_class))
	
	targets_test = torch.tensor(cifar_test.targets)
	target_test_idx = ((targets_test >= start_class) & (targets_test < end_class))

	train_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(cifar_train, np.where(target_train_idx==1)[0]), batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(cifar_test, np.where(target_test_idx==1)[0]), batch_size=batch_size)

	return train_loader, test_loader


def get_split_cifar100_tasks(num_tasks, batch_size):
	"""
	Returns data loaders for all tasks of split CIFAR-100
	:param num_tasks:
	:param batch_size:
	:return:
	"""
	datasets = {}
	
	# convention: tasks starts from 1 not 0 !
	# task_id = 1 (i.e., first task) => start_class = 0, end_class = 4
	cifar_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
	cifar_train = torchvision.datasets.CIFAR100('./data/', train=True, download=True, transform=cifar_transforms)
	cifar_test = torchvision.datasets.CIFAR100('./data/', train=False, download=True, transform=cifar_transforms)
	
	for task_id in range(1, num_tasks+1):
		train_loader, test_loader = get_split_cifar100(task_id, batch_size, cifar_train, cifar_test)
		datasets[task_id] = {'train': train_loader, 'test': test_loader}
	return datasets

# if __name__ == "__main__":
# 	dataset = get_split_cifar100(1)

def get_split_cifar100_tasks2(num_tasks, batch_size):

	datasets = {}

	# convention: tasks starts from 1 not 0 !
	# task_id = 1 (i.e., first task) => start_class = 0, end_class = 4
	cifar_train_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),torchvision.transforms.RandomCrop(32,padding=4,padding_mode="reflect"),torchvision.transforms.ToTensor(),])
	cifar_test_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
	cifar_train = torchvision.datasets.CIFAR100('./data/', train=True, download=True, transform=cifar_train_transforms)
	cifar_test = torchvision.datasets.CIFAR100('./data/', train=False, download=True, transform=cifar_test_transforms)

	num_elements_train = len(cifar_train)/num_tasks
	num_elements_test = len(cifar_test)/2


	#test_indices, _ = train_test_split(list(range(len(cifar_test.targets))), train_size = num_elements_test, stratify = cifar_test.targets)
	#test_dataset = torch.utils.data.Subset(cifar_test, test_indices)

	test_ds, val_ds = random_split(cifar_test, [int(num_elements_test), int(num_elements_test)], generator=torch.Generator().manual_seed(42))


	test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
	val_loader  = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

	list_item = list(range(len(cifar_train.targets)))

	train = cifar_train


	
	for task_id in range(1, num_tasks+1):

		train_ds, residual = random_split(train, [int(num_elements_train), int((len(train)-num_elements_train))], generator=torch.Generator().manual_seed(42))
		train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
		exemplar_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)
		train = residual
		'''
		train_indices, cifar_train_indices = train_test_split(list_item, train_size = num_elements_train, stratify = cifar_train.targets)
		train_dataset = torch.utils.data.Subset(cifar_train, train_indices)
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
		list_item = list(set(list_item) - set(cifar_train_indices))
		'''
		datasets[task_id] = {'train': train_loader, 'test': test_loader, 'val': val_loader, 'exemplar': exemplar_loader}
		
	return datasets


def get_split_cifar100_tasks2_with_augment(num_tasks, batch_size):

	datasets = {}

	# convention: tasks starts from 1 not 0 !
	# task_id = 1 (i.e., first task) => start_class = 0, end_class = 4
	cifar_train_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),torchvision.transforms.RandomCrop(32,padding=4,padding_mode="reflect"),torchvision.transforms.ToTensor(),])
	cifar_test_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
	augment = torchvision.transforms.Compose([aug_lib.TrivialAugment(), torchvision.transforms.ToTensor(),])
	cifar_train = torchvision.datasets.CIFAR100('./data/', train=True, download=True, transform=cifar_train_transforms)
	cifar_test = torchvision.datasets.CIFAR100('./data/', train=False, download=True, transform=cifar_test_transforms)
	cifar_train_aug = torchvision.datasets.CIFAR100('./data/', train=True, download=True, transform=augment)

	num_elements_train = len(cifar_train)/num_tasks
	num_elements_test = len(cifar_test)/2
	num_aug = 2000


	#test_indices, _ = train_test_split(list(range(len(cifar_test.targets))), train_size = num_elements_test, stratify = cifar_test.targets)
	#test_dataset = torch.utils.data.Subset(cifar_test, test_indices)

	test_ds, val_ds = random_split(cifar_test, [int(num_elements_test), int(num_elements_test)], generator=torch.Generator().manual_seed(42))


	test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
	val_loader  = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

	list_item = list(range(len(cifar_train.targets)))

	train = cifar_train


	
	for task_id in range(1, num_tasks+1):

		train_ds, residual = random_split(train, [int(num_elements_train), int((len(train)-num_elements_train))], generator=torch.Generator().manual_seed(42))
		if task_id == 1:
			aug_ds, aug_residual = random_split(train, [int(num_elements_train), int((len(train)-num_elements_train))], generator=torch.Generator().manual_seed(42))
			aug_ds, aug_residual = random_split(aug_ds, [int(num_aug), int((len(aug_ds)-num_aug))], generator=torch.Generator().manual_seed(42))
			train_loader = torch.utils.data.DataLoader(train_ds + aug_ds, batch_size=batch_size, shuffle=True)
		else:
			train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
		exemplar_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)
		train = residual
		'''
		train_indices, cifar_train_indices = train_test_split(list_item, train_size = num_elements_train, stratify = cifar_train.targets)
		train_dataset = torch.utils.data.Subset(cifar_train, train_indices)
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
		list_item = list(set(list_item) - set(cifar_train_indices))
		'''
		datasets[task_id] = {'train': train_loader, 'test': test_loader, 'val': val_loader, 'exemplar': exemplar_loader}
		
	return datasets


def get_split_cifar100_tasks2_memory(num_tasks, batch_size):

	datasets = {}

	# convention: tasks starts from 1 not 0 !
	# task_id = 1 (i.e., first task) => start_class = 0, end_class = 4
	cifar_train_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),torchvision.transforms.RandomCrop(32,padding=4,padding_mode="reflect"),torchvision.transforms.ToTensor(),])
	cifar_test_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
	cifar_train = torchvision.datasets.CIFAR100('./data/', train=True, download=True, transform=cifar_train_transforms)
	cifar_test = torchvision.datasets.CIFAR100('./data/', train=False, download=True, transform=cifar_test_transforms)

	num_elements_train = len(cifar_train)/num_tasks
	num_elements_test = len(cifar_test)/2


	#test_indices, _ = train_test_split(list(range(len(cifar_test.targets))), train_size = num_elements_test, stratify = cifar_test.targets)
	#test_dataset = torch.utils.data.Subset(cifar_test, test_indices)

	test_ds, val_ds = random_split(cifar_test, [int(num_elements_test), int(num_elements_test)], generator=torch.Generator().manual_seed(42))


	test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
	val_loader  = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

	list_item = list(range(len(cifar_train.targets)))

	train = cifar_train
	accumulator = None
	exemplar_loader_list = []


	
	for task_id in range(1, num_tasks+1):

		train_ds, residual = random_split(train, [int(num_elements_train), int((len(train)-num_elements_train))], generator=torch.Generator().manual_seed(42))
		accumulator = train_ds
		if task_id > 1:
			for exemplars in exemplar_loader_list:
				accumulator += exemplars
		train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
		exemplar_loader = torch.utils.data.DataLoader(accumulator, batch_size=batch_size)
		train = residual
		'''
		train_indices, cifar_train_indices = train_test_split(list_item, train_size = num_elements_train, stratify = cifar_train.targets)
		train_dataset = torch.utils.data.Subset(cifar_train, train_indices)
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
		list_item = list(set(list_item) - set(cifar_train_indices))
		'''
		datasets[task_id] = {'train': train_loader, 'test': test_loader, 'val': val_loader, 'exemplar': exemplar_loader}
		
	return datasets
'''
def dataset_manipulation():
	datasets = {}

	# convention: tasks starts from 1 not 0 !
	# task_id = 1 (i.e., first task) => start_class = 0, end_class = 4
	cifar_train_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),torchvision.transforms.RandomCrop(32,padding=4,padding_mode="reflect"),torchvision.transforms.ToTensor(),])
	cifar_test_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
	cifar_train = torchvision.datasets.CIFAR100('./data/', train=True, download=True, transform=cifar_train_transforms)
	cifar_test = torchvision.datasets.CIFAR100('./data/', train=False, download=True, transform=cifar_test_transforms)

	num_elements_train = len(cifar_train)/10
	num_elements_test = len(cifar_test)/2


	#test_indices, _ = train_test_split(list(range(len(cifar_test.targets))), train_size = num_elements_test, stratify = cifar_test.targets)
	#test_dataset = torch.utils.data.Subset(cifar_test, test_indices)

	test_ds, val_ds = random_split(cifar_test, [int(num_elements_test), int(num_elements_test)], generator=torch.Generator().manual_seed(42))

	sub_test = torch.utils.data.Subset(test_ds, range(0,1000))



	test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64)
	val_loader  = torch.utils.data.DataLoader(val_ds, batch_size=64)

	sub_test2 = torch.utils.data.Subset(test_loader.dataset, range(0,1000))

	print(len(sub_test2))

	list_item = list(range(len(cifar_train.targets)))

	train = cifar_train

	targets = np.array([])


	for data, target in test_loader:
		arr = target.cpu().detach().numpy()
		targets = np.concatenate([targets, arr])

	global_class_indices = np.column_stack(np.nonzero(targets))
	label = 2
	
'''


	



def get_split_cifar100_tasks_with_random_exemplar(num_tasks, batch_size):

	datasets = {}

	# convention: tasks starts from 1 not 0 !
	# task_id = 1 (i.e., first task) => start_class = 0, end_class = 4
	cifar_train_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),torchvision.transforms.RandomCrop(32,padding=4,padding_mode="reflect"),torchvision.transforms.ToTensor(),])
	cifar_test_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
	cifar_train = torchvision.datasets.CIFAR100('./data/', train=True, download=True, transform=cifar_train_transforms)
	cifar_test = torchvision.datasets.CIFAR100('./data/', train=False, download=True, transform=cifar_test_transforms)

	num_elements_train = len(cifar_train)/num_tasks
	num_elements_test = len(cifar_test)/2


	#test_indices, _ = train_test_split(list(range(len(cifar_test.targets))), train_size = num_elements_test, stratify = cifar_test.targets)
	#test_dataset = torch.utils.data.Subset(cifar_test, test_indices)

	test_ds, val_ds = random_split(cifar_test, [int(num_elements_test), int(num_elements_test)], generator=torch.Generator().manual_seed(42))


	test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
	val_loader  = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

	list_item = list(range(len(cifar_train.targets)))

	train = cifar_train

	memory = 2000
	old_dataset = []
	
	for task_id in range(1, num_tasks+1):

		train_ds, residual = random_split(train, [int(num_elements_train), int((len(train)-num_elements_train))], generator=torch.Generator().manual_seed(42))
		#train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)
		train = residual


		if task_id > 1:
			num_exemplar = int(memory)/(task_id - 1)
			for old_data in old_dataset:
				exemplar, res = random_split(old_data, [round(num_exemplar), int((len(old_data)-round(num_exemplar)))], generator=torch.Generator().manual_seed(42))
				train_ds = torch.utils.data.ConcatDataset([train_ds, exemplar])
				
		train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)
		old_dataset.append(train_ds)
		'''
		train_indices, cifar_train_indices = train_test_split(list_item, train_size = num_elements_train, stratify = cifar_train.targets)
		train_dataset = torch.utils.data.Subset(cifar_train, train_indices)
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
		list_item = list(set(list_item) - set(cifar_train_indices))
		'''
		print(len(train_ds))
		
		datasets[task_id] = {'train': train_loader, 'test': test_loader, 'val': val_loader}

	return datasets


def get_split_cifar100_tasks_with_random_exemplar2(num_tasks, batch_size):

	datasets = {}

	# convention: tasks starts from 1 not 0 !
	# task_id = 1 (i.e., first task) => start_class = 0, end_class = 4
	cifar_train_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),torchvision.transforms.RandomCrop(32,padding=4,padding_mode="reflect"),torchvision.transforms.ToTensor(),])
	cifar_test_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
	cifar_train = torchvision.datasets.CIFAR100('./data/', train=True, download=True, transform=cifar_train_transforms)
	cifar_test = torchvision.datasets.CIFAR100('./data/', train=False, download=True, transform=cifar_test_transforms)

	num_elements_train = len(cifar_train)/num_tasks
	num_elements_test = len(cifar_test)/2


	#test_indices, _ = train_test_split(list(range(len(cifar_test.targets))), train_size = num_elements_test, stratify = cifar_test.targets)
	#test_dataset = torch.utils.data.Subset(cifar_test, test_indices)

	test_ds, val_ds = random_split(cifar_test, [int(num_elements_test), int(num_elements_test)], generator=torch.Generator().manual_seed(42))


	test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
	val_loader  = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

	list_item = list(range(len(cifar_train.targets)))

	train = cifar_train

	memory = 2000
	old_dataset = []
	
	for task_id in range(1, num_tasks+1):

		train_ds, residual = random_split(train, [int(num_elements_train), int((len(train)-num_elements_train))], generator=torch.Generator().manual_seed(42))
		#train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)
		train = residual


		if task_id > 1:
			num_exemplar = int(memory)
			for old_data in old_dataset:
				exemplar, res = random_split(old_data, [round(num_exemplar), int((len(old_data)-round(num_exemplar)))], generator=torch.Generator().manual_seed(42))
				train_ds = torch.utils.data.ConcatDataset([train_ds, exemplar])
				
		train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)
		old_dataset.append(train_ds)
		'''
		train_indices, cifar_train_indices = train_test_split(list_item, train_size = num_elements_train, stratify = cifar_train.targets)
		train_dataset = torch.utils.data.Subset(cifar_train, train_indices)
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
		list_item = list(set(list_item) - set(cifar_train_indices))
		'''
		print(len(train_ds))
		
		datasets[task_id] = {'train': train_loader, 'test': test_loader, 'val': val_loader}

	return datasets

def get_split_cifar100_tasks_joint(num_tasks, batch_size):


    datasets = {}

    # convention: tasks starts from 1 not 0 !
    # task_id = 1 (i.e., first task) => start_class = 0, end_class = 4
    cifar_train_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomCrop(32,padding=4,padding_mode="reflect"),torchvision.transforms.ToTensor(),])
    cifar_test_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),])
    cifar_train = torchvision.datasets.CIFAR100(
        './data/', train=True, download=True, transform=cifar_train_transforms)
    cifar_test = torchvision.datasets.CIFAR100(
        './data/', train=False, download=True, transform=cifar_test_transforms)

    num_elements_train = len(cifar_train)/num_tasks
    num_elements_test = len(cifar_test)/2
    
    test_ds, val_ds = random_split(cifar_test, [int(num_elements_test), int(num_elements_test)], generator=torch.Generator().manual_seed(42))

    # test_indices, _ = train_test_split(list(range(len(cifar_test.targets))), train_size = num_elements_test, stratify = cifar_test.targets)
    # test_dataset = torch.utils.data.Subset(cifar_test, test_indices)

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

    list_item = list(range(len(cifar_train.targets)))

    train = cifar_train

    for task_id in range(1, num_tasks+1):

        train_ds, residual = random_split(train, [int(num_elements_train), int(
            (len(train)-num_elements_train))], generator=torch.Generator().manual_seed(42))
        if task_id == 1:
            train_j = train_ds
        else:
            train_j = torch.utils.data.ConcatDataset([train_j, train_ds])
        train_loader = torch.utils.data.DataLoader(train_j, batch_size=batch_size, shuffle=True)
        train = residual
        '''
        train_indices, cifar_train_indices = train_test_split(
            list_item, train_size = num_elements_train, stratify = cifar_train.targets)
        train_dataset = torch.utils.data.Subset(cifar_train, train_indices)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size)
        list_item = list(set(list_item) - set(cifar_train_indices))
        '''
        datasets[task_id] = {'train': train_loader, 'test': test_loader, 'val': val_loader}

    return datasets

def get_split_cifar10_tasks(num_tasks, batch_size):

	datasets = {}
	
	# convention: tasks starts from 1 not 0 !
	# task_id = 1 (i.e., first task) => start_class = 0, end_class = 4
	cifar_train_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),torchvision.transforms.RandomCrop(32,padding=4,padding_mode="reflect"),torchvision.transforms.ToTensor(),])
	cifar_test_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])	
	cifar_train = torchvision.datasets.CIFAR10('./data/', train=True, download=True, transform=cifar_train_transforms)
	cifar_test = torchvision.datasets.CIFAR10('./data/', train=False, download=True, transform=cifar_test_transforms)

	num_elements_train = len(cifar_train)/num_tasks
	num_elements_test = len(cifar_test)/2


	#test_indices, _ = train_test_split(list(range(len(cifar_test.targets))), train_size = num_elements_test, stratify = cifar_test.targets)
	#test_dataset = torch.utils.data.Subset(cifar_test, test_indices)


	
	test_ds, val_ds = random_split(cifar_test, [int(num_elements_test), int(num_elements_test)], generator=torch.Generator().manual_seed(42))


	test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
	val_loader  = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

	list_item = list(range(len(cifar_train.targets)))

	train = cifar_train


	
	for task_id in range(1, num_tasks+1):

		train_ds, residual = random_split(train, [int(num_elements_train), int((len(train)-num_elements_train))], generator=torch.Generator().manual_seed(42))
		train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)
		train = residual
		'''
		train_indices, cifar_train_indices = train_test_split(list_item, train_size = num_elements_train, stratify = cifar_train.targets)
		train_dataset = torch.utils.data.Subset(cifar_train, train_indices)
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
		list_item = list(set(list_item) - set(cifar_train_indices))
		'''
		datasets[task_id] = {'train': train_loader, 'test': test_loader, 'val': val_loader}
		
	return datasets




