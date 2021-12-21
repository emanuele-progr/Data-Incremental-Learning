import os
import torch
import copy
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from model import MLP, ResNet18, ResNet32
from data_utils import get_permuted_mnist_tasks, get_rotated_mnist_tasks,get_split_cifar100_tasks2_memory, get_split_cifar100_tasks2, get_split_cifar100_tasks, get_split_cifar100_tasks2_with_augment, get_split_cifar100_tasks_with_random_exemplar2, get_split_cifar100_tasks, get_split_cifar100_tasks_with_random_exemplar, get_split_cifar10_tasks, get_split_cifar100_tasks_joint
from utils import data_to_csv, parse_arguments, DEVICE, init_experiment, end_experiment, log_metrics, save_checkpoint,distanceExemplarsSelector, post_train_process_ewc, post_train_process_fd, herdingExemplarsSelector, entropyExemplarsSelector, randomExemplarsSelector, compute_mean_of_exemplars
from sklearn.metrics import confusion_matrix


def train_single_epoch(net, optimizer, loader, criterion, task_id=None):
	"""
	Train the model for a single epoch
	
	:param net:
	:param optimizer:
	:param loader:
	:param criterion:
	:param task_id:
	:return:
	"""
	net = net.to(DEVICE)
	net.train()
	
	for batch_idx, (data, target) in enumerate(loader):
		data = data.to(DEVICE)
		target = target.to(DEVICE)
		optimizer.zero_grad()
		if task_id:
			pred = net(data, task_id)
		else:
			pred = net(data)
		loss = criterion(pred, target)
		loss.backward()
		optimizer.step()
	return net

def train_single_epoch_ewc(net, optimizer, loader, criterion, old_params, fisher, task_id=None):
	"""
	Train the model for a single epoch
	
	:param net:
	:param optimizer:
	:param loader:
	:param criterion:
	:param task_id:
	:return:
	"""
	net = net.to(DEVICE)
	loss_penalty = 0
	
	net.train()
	for batch_idx, (data, target) in enumerate(loader):
		data = data.to(DEVICE)
		target = target.to(DEVICE)
		optimizer.zero_grad()
		if task_id:
			pred= net(data, task_id)
		else:
			pred = net(data)
		if task_id > 1:
			loss_penalty = ewc_penalty(net, fisher, old_params)
		loss = criterion(pred, target) + loss_penalty
		loss.backward()
		optimizer.step()
	print('ewc penalty : {}'.format(loss_penalty))
	return net

def train_single_epoch_fd(net, optimizer, loader, criterion, old_model, task_id=None):
	"""
	Train the model for a single epoch
	
	:param net:
	:param optimizer:
	:param loader:
	:param criterion:
	:param task_id:
	:return:
	"""
	net = net.to(DEVICE)
	loss_penalty = 1
	
	net.train()
	for batch_idx, (data, target) in enumerate(loader):
		data = data.to(DEVICE)
		target = target.to(DEVICE)
		optimizer.zero_grad()
		if task_id:
			pred, feat = net(data, task_id, return_features=True)
		else:
			pred, feat = net(data, return_features=True)
		if task_id > 1:
			pred_old, feat_old = old_model(data, task_id, return_features=True) 
			loss_penalty = feature_distillation_penalty(feat, feat_old)
		loss = criterion(pred, target) + loss_penalty
		loss.backward()
		optimizer.step()
	return net	

def train_single_epoch_iCarl(net, optimizer, loader, criterion, old_model, task_id=None):
	"""
	Train the model for a single epoch
	
	:param net:
	:param optimizer:
	:param loader:
	:param criterion:
	:param task_id:
	:return:
	"""
	net = net.to(DEVICE)
	loss_penalty = 1
	loss_penalty2 = 1
	outputs = []
	outputs_old = []
	
	net.train()
	for batch_idx, (data, target) in enumerate(loader):
		data = data.to(DEVICE)
		target = target.to(DEVICE)
		optimizer.zero_grad()
		if task_id:
			pred, feat = net(data, task_id, return_features=True)
		else:
			pred, feat = net(data, return_features=True)
		if task_id > 1:
			pred_old, feat_old = old_model(data, task_id, return_features=True)
			outputs.append(pred)
			outputs_old.append(pred_old)
			loss_penalty = feature_distillation_penalty(feat, feat_old)
			loss_penalty2 = icarl_penalty(outputs, outputs_old)
			outputs = []
			outputs_old = []
		loss = criterion(pred, target) + loss_penalty2
		loss.backward()
		optimizer.step()
	return net

def classify(features, targets, exemplar_means):
	# expand means to all batch images
	means = torch.stack(exemplar_means)
	means = torch.stack([means] * features.shape[0])
	means = means.transpose(1, 2)
	# expand all features to all classes
	features = features / features.norm(dim=1).view(-1, 1)
	features = features.unsqueeze(2)
	features = features.expand_as(means)
	features = features.to('cpu')
	means = means.to('cpu')
	# get distances for all images to all exemplar class means -- nearest prototype
	dists = (features - means).pow(2).sum(1).squeeze()
	# Task-Aware Multi-Head
	# Task-Agnostic Multi-Head
	pred = dists.argmin(1)
	hits_tag = (pred.to(DEVICE) == targets.to(DEVICE)).float()
	return pred.to('cpu'), hits_tag

def eval_single_epoch(net, loader, criterion, task_id=None):
	"""
	Evaluate the model for single epoch
	
	:param net:
	:param loader:
	:param criterion:
	:param task_id:
	:return:
	"""
	net = net.to(DEVICE)
	net.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in loader:
			data = data.to(DEVICE)
			target = target.to(DEVICE)
			# for cifar head
			if task_id is not None:
				output = net(data, task_id)
			else:
				output = net(data)
			test_loss += criterion(output, target).item() * loader.batch_size
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
	test_loss /= len(loader.dataset)
	correct = correct.to('cpu')
	avg_acc = 100.0 * float(correct.numpy()) / len(loader.dataset)
	return {'accuracy': avg_acc, 'loss': test_loss}

def eval_single_epoch_fd(net, loader, criterion, old_model, task_id=None):
	"""
	Evaluate the model for single epoch
	
	:param net:
	:param loader:
	:param criterion:
	:param task_id:
	:return:
	"""
	net = net.to(DEVICE)
	net.eval()
	test_loss = 0
	loss_penalty = torch.tensor(1.0)
	lwf_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in loader:
			data = data.to(DEVICE)
			target = target.to(DEVICE)
			# for cifar head
			if task_id is not None:
				output, feat = net(data, task_id, return_features = True)
			else:
				output, feat = net(data, return_features = True)
			if task_id > 1:
				pred_old, feat_old = old_model(data, task_id, return_features=True) 
				loss_penalty = feature_distillation_penalty(feat, feat_old)

			test_loss += (criterion(output, target).item() + loss_penalty.item()) * loader.batch_size
			lwf_loss += loss_penalty.item() * loader.batch_size
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
	test_loss /= len(loader.dataset)
	lwf_loss /= len(loader.dataset)
	correct = correct.to('cpu')
	avg_acc = 100.0 * float(correct.numpy()) / len(loader.dataset)
	return {'accuracy': avg_acc, 'loss': test_loss, 'fd_loss': lwf_loss}

def eval_single_epoch_ewc(net, loader, criterion, fisher, old_params, task_id=None):
	"""
	Evaluate the model for single epoch
	
	:param net:
	:param loader:
	:param criterion:
	:param task_id:
	:return:
	"""
	net = net.to(DEVICE)
	net.eval()
	test_loss = 0
	correct = 0
	loss_penalty = 0
	ewc_loss = 0

	if task_id > 1:
		loss_penalty = ewc_penalty(net, fisher, old_params)
	else:
		loss_penalty = torch.tensor(0.0)

	with torch.no_grad():
		for data, target in loader:
			data = data.to(DEVICE)
			target = target.to(DEVICE)
			# for cifar head
			if task_id is not None:
				output = net(data, task_id)
			else:
				output = net(data)
			test_loss += (criterion(output, target).item() + loss_penalty.item()) * loader.batch_size
			ewc_loss += loss_penalty.item() * loader.batch_size
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
	test_loss /= len(loader.dataset)
	ewc_loss /= len(loader.dataset)
	correct = correct.to('cpu')
	avg_acc = 100.0 * float(correct.numpy()) / len(loader.dataset)
	return {'accuracy': avg_acc, 'loss': test_loss, 'ewcloss': ewc_loss}

def eval_single_epoch_iCarl(net, loader, criterion, old_model, exemplar_means, task_id):
	"""
	Evaluate the model for single epoch
	
	:param net:
	:param loader:
	:param criterion:
	:param task_id:
	:return:
	"""
	net = net.to(DEVICE)
	net.eval()
	test_loss = 0
	loss_penalty = torch.tensor(1.0)
	loss_penalty2 = torch.tensor(1.0)
	hits = torch.tensor(1.0)
	lwf_loss = 0
	icarl_loss = 0
	correct = 0
	total_acc = 0
	total_num = 0
	outputs = []
	outputs_old = []
	with torch.no_grad():
		for data, target in loader:
			data = data.to(DEVICE)
			target = target.to(DEVICE)
			# for cifar head
			output, feat = net(data, task_id, return_features = True)
			
			if task_id > 1:
				_, hits = classify(feat, target, exemplar_means)
				pred_old, feat_old = old_model(data, task_id, return_features=True) 
				loss_penalty = feature_distillation_penalty(feat, feat_old)
				outputs.append(output)
				outputs_old.append(pred_old)
				loss_penalty2 = icarl_penalty(outputs, outputs_old)
				outputs = []
				outputs_old = []

			test_loss += (criterion(output, target).item() + loss_penalty2.item()) * loader.batch_size
			lwf_loss += loss_penalty.item() * loader.batch_size
			icarl_loss += loss_penalty.item() * loader.batch_size
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()

			total_acc += hits.sum().item()
			total_num += len(target)
			
	test_loss /= len(loader.dataset)
	lwf_loss /= len(loader.dataset)
	icarl_loss /= len(loader.dataset)

	correct = correct.to('cpu')
	avg_acc = 100.0 * float(correct.numpy()) / len(loader.dataset)

	acc = round((total_acc / total_num) * 100, 2)
	if task_id == 1:
		acc = avg_acc

	return {'accuracy': acc, 'loss': test_loss, 'icarl_loss': icarl_loss}	


def final_eval(net, loader, criterion, task_id=None):
    """
    Evaluate the model for single epoch
    
    :param net:
    :param loader:
    :param criterion:
    :param task_id:
    :return:
    """
    X = []
    Y = []
    net = net.to(DEVICE)
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            # for cifar head
            if task_id is not None:
                output = net(data, task_id)
            else:
                output = net(data)
            test_loss += criterion(output, target).item() * loader.batch_size
            pred = output.data.max(1, keepdim=True)[1]
            Y.append(pred.view_as(target.data).cpu().numpy().tolist())
            X.append(target.data.cpu().numpy().tolist())
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(loader.dataset)
    correct = correct.to('cpu')
    avg_acc = 100.0 * float(correct.numpy()) / len(loader.dataset)
    X = sum(X, [])
    Y = sum(Y, [])
    return {'accuracy': avg_acc, 'loss': test_loss}, X, Y

def final_eval_iCarl(net, loader, criterion, exemplar_means, task_id=None):
    """
    Evaluate the model for single epoch
    
    :param net:
    :param loader:
    :param criterion:
    :param task_id:
    :return:
    """
    X = []
    Y = []
    net = net.to(DEVICE)
    net.eval()
    test_loss = 0
    correct = 0
    total_acc = 0
    total_num = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            # for cifar head
            if task_id is not None:
                output, feat = net(data, task_id, return_features = True)
            else:
                output = net(data)
            pred_icarl, hits = classify(feat, target, exemplar_means)
            total_acc += hits.sum().item()
            total_num += len(target)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            Y.append(pred_icarl.view_as(target.data).cpu().numpy().tolist())
            X.append(target.data.cpu().numpy().tolist())
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(loader.dataset)
    correct = correct.to('cpu')
    avg_acc = 100.0 * float(correct.numpy()) / len(loader.dataset)
    avg_acc = round((total_acc/total_num) * 100, 2)
    X = sum(X, [])
    Y = sum(Y, [])
    return {'accuracy': avg_acc, 'loss': test_loss}, X, Y

def get_benchmark_data_loader(args):
	"""
	Returns the benchmark loader which could be either of these:
	get_split_cifar100_tasks, get_permuted_mnist_tasks, or get_rotated_mnist_tasks
	
	:param args:
	:return: a function which when called, returns all tasks
	"""
	if args.dataset == 'perm-mnist' or args.dataset == 'permuted-mnist':
		return get_permuted_mnist_tasks
	elif args.dataset == 'rot-mnist' or args.dataset == 'rotation-mnist':
		return get_rotated_mnist_tasks
	elif args.dataset == 'cifar-100' or args.dataset == 'cifar100' and args.compute_joint_incremental:
		return get_split_cifar100_tasks_joint
	elif args.dataset == 'cifar-100' or args.dataset == 'cifar100' and args.compute_joint_incremental is False:
		return get_split_cifar100_tasks2_with_augment
		#return get_split_cifar100_tasks_with_random_exemplar2
	elif args.dataset == 'cifar-10' or args.dataset == 'cifar10':
		return get_split_cifar10_tasks
	else:
		raise Exception("Unknown dataset.\n"+
						"The code supports 'perm-mnist, rot-mnist, and cifar-100.")


def get_benchmark_model(args):
	"""
	Return the corresponding PyTorch model for experiment
	:param args:
	:return:
	"""
	if 'mnist' in args.dataset:
		if args.tasks == 20 and args.hiddens < 256:
			print("Warning! the main paper MLP with 256 neurons for experiment with 20 tasks")
		return MLP(args.hiddens, {'dropout': args.dropout}).to(DEVICE)
	elif 'cifar' in args.dataset:
		return ResNet32(config={'dropout': args.dropout}).to(DEVICE)
	else:
		raise Exception("Unknown dataset.\n"+
						"The code supports 'perm-mnist, rot-mnist, and cifar-100.")

def plot_conf_matrix(matrix):

    plt.figure(1, figsize=(9, 6))
 
    plt.title("Confusion Matrix")
    df_cm = pd.DataFrame(matrix, range(len(matrix[0])), range(len(matrix[0])))

    sns.set(font_scale=1.4) # for label size
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 12},  fmt='g') # font size
    s = "matrix"
    print(s)
    plt.savefig("confmatrix", bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def ewc_penalty(model, fisher, older_params):
	lamb = 1000
	loss = 0
	loss_reg = 0
	for n, p in model.named_parameters():
		if n in fisher.keys():

			loss_reg += torch.sum(fisher[n] * (p - older_params[n]).pow(2))/2
	loss += lamb * loss_reg

	return loss

def feature_distillation_penalty(feat, feat_old):

	lamb = 1
	loss = lamb * torch.mean(torch.norm(feat - feat_old, p=2, dim=1))

	return loss

def icarl_penalty(out, out_old):

	lamb = 1
	g = torch.sigmoid(torch.cat(out))
	q_i = torch.sigmoid(torch.cat(out_old))
	loss = lamb * torch.nn.functional.binary_cross_entropy(g, q_i)

	return loss


def make_prediction_vector(X, Y):
	pred_vector = [0] * len(X)
	for i in range(len(X)):
		if X[i] == Y[i]:
			pred_vector[i] = 1

	return pred_vector

def count_common_pred(pred_vector1, pred_vector2):
	count = 0
	for i in range(len(pred_vector2)):
		if pred_vector2[i] == 1:
			count += pred_vector1[i]
	
	return count

def forgetting_metric(current_pred_vector, pred_vector_list, current_task_id):
	num = 0
	den = 0
	for i in reversed(range(current_task_id)):
		if i > 0:
			num += count_common_pred(current_pred_vector, pred_vector_list[i])
			den += sum(pred_vector_list[i])
	
	if current_task_id > 1:
		forgetting = float(num)/den
	else:
		forgetting = 0

	return forgetting
		

'''

def run(args):
	"""
	Run a single run of experiment.
	
	:param args: please see `utils.py` for arguments and options
	"""
	# init experiment
	acc_db, loss_db, hessian_eig_db = init_experiment(args)

	# load benchmarks and model
	print("Loading {} tasks for {}".format(args.tasks, args.dataset))
	tasks = get_benchmark_data_loader(args)(args.tasks, args.batch_size)
	print("loaded all tasks!")
	model = get_benchmark_model(args)

	# criterion
	criterion = nn.CrossEntropyLoss().to(DEVICE)
	time = 0
	
	for current_task_id in range(1, args.tasks+1):
		print("================== TASK {} / {} =================".format(current_task_id, args.tasks))
		train_loader = tasks[current_task_id]['train']
		lr = max(args.lr * args.gamma ** (current_task_id), 0.00005)
		
		for epoch in range(1, args.epochs_per_task+1):
			# 1. train and save
	
			optimizer = torch.optim.SGD(model.parameters(), lr=lr)
			train_single_epoch(model, optimizer, train_loader, criterion, current_task_id)
			time += 1

			model = model.to(DEVICE)
			val_loader = tasks[current_task_id]['val']
			metrics = eval_single_epoch(model, val_loader, criterion, current_task_id)
			acc_db, loss_db = log_metrics(metrics, time, current_task_id, acc_db, loss_db)
			print(acc_db[current_task_id][epoch])
			#if epoch == args.epochs_per_task:
			#	tune.report(val_accuracy= acc_db[current_task_id][epoch])

			# 2. evaluate on all tasks up to now, including the current task
			for prev_task_id in range(1, current_task_id+1):
				# 2.0. only evaluate once a task is finished
				if epoch == args.epochs_per_task:
					model = model.to(DEVICE)
					val_loader = tasks[prev_task_id]['test']
					
					# 2.1. compute accuracy and loss
					metrics = eval_single_epoch(model, val_loader, criterion, prev_task_id)
					acc_db, loss_db = log_metrics(metrics, time, prev_task_id, acc_db, loss_db)
					
					# 2.2. (optional) compute eigenvalues and eigenvectors of Loss Hessian
                    
					#if prev_task_id == current_task_id and args.compute_eigenspectrum:
					#	hessian_eig_db = log_hessian(model, val_loader, time, prev_task_id, hessian_eig_db)

						
					# 2.3. save model parameters
					save_checkpoint(model, time)

	end_experiment(args, acc_db, loss_db, hessian_eig_db)



def run_experiment(args):


	tasks = get_benchmark_data_loader(args)(args.tasks, args.batch_size)
	model = get_benchmark_model(args)
	acc_db, loss_db, hessian_eig_db = init_experiment(args)


	# criterion
	criterion = nn.CrossEntropyLoss().to(DEVICE)
	time = 0
	the_last_loss = 100
	patience = 30
	trigger_times = 0
	check = 0
	ewc = 0
	lwf = 1
	old_model = 0
	pred_vector_list = [[0]]
	exemplars_vector_list = []
	#lr = [0.01, 0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
	lr = [0.001, 0.0001, 0.0001, 0.0001, 0.0001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]
	#lr = [0.001, 0.0001, 0.0001, 0.00001, 0.00001]
	
	fisher = {n: torch.zeros(p.shape).to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)#lr=lr[current_task_id - 1])


	for current_task_id in range(1, args.tasks+1):
		#lr = max(args.lr * args.gamma ** (current_task_id), 0.00005)
		ewc_loss = []
		all_loss = []
		counter = []
		if args.compute_joint_incremental:
			model = get_benchmark_model(args)
		old_params = {n: p.clone().detach() for n,p in model.named_parameters() if p.requires_grad}
		print("================== TASK {} / {} =================".format(current_task_id, args.tasks))
		train_loader = tasks[current_task_id]['train']
		print(len(train_loader.dataset))
		exemplar_loader = tasks[current_task_id]['exemplar']
		
		accumulator = None

		if current_task_id > 1:
			accumulator = train_loader.dataset
			for exemplars in exemplars_vector_list:
				accumulator += exemplars
			train_loader = torch.utils.data.DataLoader(accumulator, batch_size=args.batch_size, shuffle=True)
			print(len(train_loader.dataset))
		if (check == 1) :
			model, optimizer = load_checkpoint(model, optimizer, 'check.pth')	
		
		for epoch in range(1, args.epochs_per_task+1):
			# 1. train and save

			prev_model = get_benchmark_model(args)
			prev_model.load_state_dict(model.state_dict())
			prev_opt = type(optimizer)(prev_model.parameters(), lr=args.lr)
			prev_opt.load_state_dict(optimizer.state_dict())
			if ewc == 1:
				train_single_epoch_ewc(model, optimizer, train_loader, criterion, old_params, fisher, current_task_id)
			elif lwf == 1:
				train_single_epoch_fd(model, optimizer, train_loader, criterion, old_model, current_task_id)
			else:
				train_single_epoch(model, optimizer, train_loader, criterion, current_task_id)
			time += 1
			model = model.to(DEVICE)
			val_loader = tasks[current_task_id]['val']
			if ewc == 1:
				metrics = eval_single_epoch_ewc(model, train_loader, criterion, fisher, old_params, current_task_id)
				if current_task_id > 1:
					ewc_loss.append(metrics['ewcloss'])
					all_loss.append(metrics['loss'])
					counter.append(epoch)
			elif lwf == 1:
				metrics = eval_single_epoch_fd(model, train_loader, criterion, old_model, current_task_id)
			else:
				metrics = eval_single_epoch(model, train_loader, criterion, current_task_id)			


			acc_db, loss_db = log_metrics(metrics, time, current_task_id, acc_db, loss_db)
			

			if loss_db[current_task_id][epoch-1] > the_last_loss:
				if trigger_times == 0:
					backup_model = get_benchmark_model(args)
					backup_model.load_state_dict(prev_model.state_dict())
					backup_opt = type(prev_opt)(backup_model.parameters(), lr=args.lr)
					backup_opt.load_state_dict(prev_opt.state_dict())
				trigger_times += 1
				print('trigger times:', trigger_times)
			else:
				trigger_times = 0
			if trigger_times >= patience:
				print('Early stopping!')
				#tune.report(val_loss = loss_db[current_task_id][epoch])
				model = backup_model.to(DEVICE)
				optimizer = type(backup_opt)(model.parameters(), lr=args.lr)
				optimizer.load_state_dict(backup_opt.state_dict())
				val_loader = tasks[current_task_id]['test']
				
				# 2.1. compute accuracy and loss
				metrics, X, Y = final_eval(model, val_loader, criterion, current_task_id)
				pred_vector = make_prediction_vector(X, Y)
				print(forgetting_metric(pred_vector, pred_vector_list, current_task_id))
				pred_vector_list.append(pred_vector)
				fisher = post_train_process_ewc(train_loader, model, optimizer, current_task_id, fisher)
				old_model = post_train_process_fd(model)
				res = herdingExemplarsSelector(model, exemplar_loader, current_task_id, 20)
				selected_exemplar = torch.utils.data.Subset(exemplar_loader.dataset, res)
				#exemplars_vector_list = []
				exemplars_vector_list.append(selected_exemplar)
				print(len(selected_exemplar))
				matrix = confusion_matrix(X, Y)
				#plot_conf_matrix(matrix)
				acc_db, loss_db = log_metrics(metrics, time, current_task_id, acc_db, loss_db)
				save_checkpoint_Adam(backup_model, backup_opt)
				time = 0
				trigger_times = 0
				the_last_loss = 100
				if current_task_id > 1:
					e_loss = np.array(ewc_loss)
					a_loss = np.array(all_loss)
					epochs = np.array(counter)
					df = pd.DataFrame({"Item Name": epochs, "loss" : a_loss, "ewc_loss" : e_loss})
					string = 'prova{}.csv'.format(current_task_id)
					df.to_csv(string, sep = ';', index = False)
				break

			if loss_db[current_task_id][epoch-1] < the_last_loss:
				the_last_loss = loss_db[current_task_id][epoch-1]
				
			if epoch == args.epochs_per_task:
				#tune.report(val_loss= loss_db[current_task_id][epoch])
				if trigger_times > 0:
					model = backup_model.to(DEVICE)
					optimizer = type(backup_opt)(model.parameters(), lr=args.lr)
					optimizer.load_state_dict(backup_opt.state_dict())
				else:
					model = model.to(DEVICE)
				val_loader = tasks[current_task_id]['test']
					
				# 2.1. compute accuracy and loss
				metrics, X, Y = final_eval(model, val_loader, criterion, current_task_id)
				pred_vector = make_prediction_vector(X, Y)
				print(forgetting_metric(pred_vector, pred_vector_list, current_task_id))
				pred_vector_list.append(pred_vector)
				fisher = post_train_process_ewc(train_loader, model, optimizer, current_task_id, fisher)
				old_model = post_train_process_fd(model)
				res = herdingExemplarsSelector(model, exemplar_loader, current_task_id, 20)
				selected_exemplar = torch.utils.data.Subset(exemplar_loader.dataset, res)
				#exemplars_vector_list = []
				exemplars_vector_list.append(selected_exemplar)
				matrix = confusion_matrix(X, Y)
				#plot_conf_matrix(matrix)
				acc_db, loss_db = log_metrics(metrics, time, current_task_id, acc_db, loss_db)
				save_checkpoint_Adam(model, optimizer)
				time = 0
				trigger_times = 0
				the_last_loss = 100
				if current_task_id > 1:
					e_loss = np.array(ewc_loss)
					a_loss = np.array(all_loss)
					epochs = np.array(counter)
					df = pd.DataFrame({"Item Name": epochs, "loss" : a_loss, "ewc_loss" : e_loss})
					string = 'prova{}.csv'.format(current_task_id)
					df.to_csv(string, sep = ';', index = False)
	return

'''

def run_experiment(args):


	tasks = get_benchmark_data_loader(args)(args.tasks, args.batch_size)
	model = get_benchmark_model(args)
	acc_db, loss_db, hessian_eig_db = init_experiment(args)


	# criterion
	criterion = nn.CrossEntropyLoss().to(DEVICE)
	time = 0
	the_last_loss = 100
	patience = 30
	trigger_times = 0
	check = 0
	ewc = 0
	lwf = 1
	old_model = 0
	pred_vector_list = [[0]]
	exemplars_vector_list = []
	accuracy_results = []
	forgetting_result = []
	task_counter = []
	exemplar_means = []
	#lr = [0.01, 0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
	lr = [0.001, 0.0001, 0.0001, 0.0001, 0.0001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]
	#lr = [0.001, 0.0001, 0.0001, 0.00001, 0.00001]
	
	fisher = {n: torch.zeros(p.shape).to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)#lr=lr[current_task_id - 1])


	for current_task_id in range(1, args.tasks+1):
		#lr = max(args.lr * args.gamma ** (current_task_id), 0.00005)
		ewc_loss = []
		all_loss = []
		counter = []
		if args.compute_joint_incremental:
			model = get_benchmark_model(args)
		old_params = {n: p.clone().detach() for n,p in model.named_parameters() if p.requires_grad}
		print("================== TASK {} / {} =================".format(current_task_id, args.tasks))
		train_loader = tasks[current_task_id]['train']
		print(len(train_loader.dataset))
		exemplars_per_class = 20
		
		
		
		counter = 0

		if current_task_id > 1:
			accumulator = train_loader.dataset
			for exemplars in exemplars_vector_list:

				num = int((exemplars_per_class * 100)/(current_task_id - 1))
				selected_exemplar = torch.utils.data.Subset(exemplar_loader.dataset, exemplars[:num])
				accumulator += selected_exemplar
				if counter == 0:
					exemplar_dataset = selected_exemplar
				else:
					exemplar_dataset += selected_exemplar
				counter += 1 
			train_loader = torch.utils.data.DataLoader(accumulator, batch_size=args.batch_size, shuffle=True)
			print(len(train_loader.dataset))
			exemplar_means = compute_mean_of_exemplars(model,torch.utils.data.DataLoader(exemplar_dataset, batch_size=args.batch_size) , current_task_id)
		if (check == 1) :
			model, optimizer = load_checkpoint(model, optimizer, 'check.pth')	
		exemplar_loader = tasks[current_task_id]['exemplar']
		for epoch in range(1, args.epochs_per_task+1):
			# 1. train and save

			prev_model = get_benchmark_model(args)
			prev_model.load_state_dict(model.state_dict())
			prev_opt = type(optimizer)(prev_model.parameters(), lr=args.lr)
			prev_opt.load_state_dict(optimizer.state_dict())
			if ewc == 1:
				train_single_epoch_ewc(model, optimizer, train_loader, criterion, old_params, fisher, current_task_id)
			elif lwf == 1:
				#train_single_epoch_fd(model, optimizer, train_loader, criterion, old_model, current_task_id)
				train_single_epoch_iCarl(model, optimizer, train_loader, criterion, old_model, current_task_id)
			else:
				train_single_epoch(model, optimizer, train_loader, criterion, current_task_id)
			time += 1
			model = model.to(DEVICE)
			val_loader = tasks[current_task_id]['val']
			if ewc == 1:
				metrics = eval_single_epoch_ewc(model, train_loader, criterion, fisher, old_params, current_task_id)
				if current_task_id > 1:
					ewc_loss.append(metrics['ewcloss'])
					all_loss.append(metrics['loss'])
					counter.append(epoch)
			elif lwf == 1:
				#metrics = eval_single_epoch_fd(model, train_loader, criterion, old_model, current_task_id)
				metrics = eval_single_epoch_iCarl(model, train_loader, criterion, old_model, exemplar_means, current_task_id)
			else:
				metrics = eval_single_epoch(model, train_loader, criterion, current_task_id)			


			acc_db, loss_db = log_metrics(metrics, time, current_task_id, acc_db, loss_db)
			

			if loss_db[current_task_id][epoch-1] > the_last_loss:
				if trigger_times == 0:
					backup_model = get_benchmark_model(args)
					backup_model.load_state_dict(prev_model.state_dict())
					backup_opt = type(prev_opt)(backup_model.parameters(), lr=args.lr)
					backup_opt.load_state_dict(prev_opt.state_dict())
				trigger_times += 1
				print('trigger times:', trigger_times)
			else:
				trigger_times = 0
			if trigger_times >= patience:
				print('Early stopping!')
				#tune.report(val_loss = loss_db[current_task_id][epoch])
				model = backup_model.to(DEVICE)
				optimizer = type(backup_opt)(model.parameters(), lr=args.lr)
				optimizer.load_state_dict(backup_opt.state_dict())
				val_loader = tasks[current_task_id]['test']
				
				# 2.1. compute accuracy and loss
				metrics, X, Y = final_eval(model, val_loader, criterion, current_task_id)
				pred_vector = make_prediction_vector(X, Y)
				print(forgetting_metric(pred_vector, pred_vector_list, current_task_id))
				accuracy_results.append(metrics['accuracy'])
				forgetting_result.append(forgetting_metric(pred_vector, pred_vector_list, current_task_id))
				task_counter.append(current_task_id)
				pred_vector_list.append(pred_vector)
				fisher = post_train_process_ewc(train_loader, model, optimizer, current_task_id, fisher)
				old_model = post_train_process_fd(model)
				res = randomExemplarsSelector(model, exemplar_loader, current_task_id, exemplars_per_class)
				#selected_exemplar = torch.utils.data.Subset(exemplar_loader.dataset, res)
				#exemplars_vector_list = []
				exemplars_vector_list.append(res)
				print(len(selected_exemplar))
				matrix = confusion_matrix(X, Y)
				#plot_conf_matrix(matrix)
				acc_db, loss_db = log_metrics(metrics, time, current_task_id, acc_db, loss_db)
				save_checkpoint_Adam(backup_model, backup_opt)
				time = 0
				trigger_times = 0
				the_last_loss = 100
				if current_task_id > 1:
					e_loss = np.array(ewc_loss)
					a_loss = np.array(all_loss)
					epochs = np.array(counter)
					df = pd.DataFrame({"Item Name": epochs, "loss" : a_loss, "ewc_loss" : e_loss})
					string = 'prova{}.csv'.format(current_task_id)
					df.to_csv(string, sep = ';', index = False)
				break

			if loss_db[current_task_id][epoch-1] < the_last_loss:
				the_last_loss = loss_db[current_task_id][epoch-1]
				
			if epoch == args.epochs_per_task:
				#tune.report(val_loss= loss_db[current_task_id][epoch])
				if trigger_times > 0:
					model = backup_model.to(DEVICE)
					optimizer = type(backup_opt)(model.parameters(), lr=args.lr)
					optimizer.load_state_dict(backup_opt.state_dict())
				else:
					model = model.to(DEVICE)
				val_loader = tasks[current_task_id]['test']
					
				# 2.1. compute accuracy and loss
				metrics, X, Y = final_eval(model, val_loader, criterion, current_task_id)

				pred_vector = make_prediction_vector(X, Y)
				print(forgetting_metric(pred_vector, pred_vector_list, current_task_id))
				accuracy_results.append(metrics['accuracy'])
				forgetting_result.append(forgetting_metric(pred_vector, pred_vector_list, current_task_id))
				task_counter.append(current_task_id)
				pred_vector_list.append(pred_vector)
				fisher = post_train_process_ewc(train_loader, model, optimizer, current_task_id, fisher)
				old_model = post_train_process_fd(model)
				res = randomExemplarsSelector(model, exemplar_loader, current_task_id, exemplars_per_class)
				#selected_exemplar = torch.utils.data.Subset(exemplar_loader.dataset, res)
				#exemplars_vector_list = []
				exemplars_vector_list.append(res)
				matrix = confusion_matrix(X, Y)
				#plot_conf_matrix(matrix)
				acc_db, loss_db = log_metrics(metrics, time, current_task_id, acc_db, loss_db)
				save_checkpoint_Adam(model, optimizer)
				time = 0
				trigger_times = 0
				the_last_loss = 100
				compute_mean_of_exemplars(model, train_loader, current_task_id)
				if current_task_id > 1:
					e_loss = np.array(ewc_loss)
					a_loss = np.array(all_loss)
					epochs = np.array(counter)
					df = pd.DataFrame({"Item Name": epochs, "loss" : a_loss, "ewc_loss" : e_loss})
					string = 'prova{}.csv'.format(current_task_id)
					df.to_csv(string, sep = ';', index = False)
	
	data_to_csv(accuracy_results, forgetting_result, task_counter)
	return

def run_experiment_iCarl(args):


	tasks = get_benchmark_data_loader(args)(args.tasks, args.batch_size)
	model = get_benchmark_model(args)
	acc_db, loss_db, hessian_eig_db = init_experiment(args)


	# criterion
	criterion = nn.CrossEntropyLoss().to(DEVICE)
	time = 0
	the_last_loss = 100
	patience = 30
	trigger_times = 0
	check = 0
	ewc = 0
	lwf = 1
	old_model = 0
	pred_vector_list = [[0]]
	exemplars_vector_list = []
	accuracy_results = []
	forgetting_result = []
	task_counter = []
	exemplar_means = []
	#lr = [0.01, 0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
	lr = [0.001, 0.0001, 0.0001, 0.0001, 0.0001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]
	#lr = [0.001, 0.0001, 0.0001, 0.00001, 0.00001]
	
	fisher = {n: torch.zeros(p.shape).to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)#lr=lr[current_task_id - 1])


	for current_task_id in range(1, args.tasks+1):
		#lr = max(args.lr * args.gamma ** (current_task_id), 0.00005)
		ewc_loss = []
		all_loss = []
		counter = []
		if args.compute_joint_incremental:
			model = get_benchmark_model(args)
		old_params = {n: p.clone().detach() for n,p in model.named_parameters() if p.requires_grad}
		print("================== TASK {} / {} =================".format(current_task_id, args.tasks))
		train_loader = tasks[current_task_id]['train']
		print(len(train_loader.dataset))
		exemplars_per_class = 20
		
		
		
		counter2 = 0

		if current_task_id > 1:
			accumulator = train_loader.dataset
			for exemplars in exemplars_vector_list:

				num = int((exemplars_per_class * 100)/(current_task_id - 1))
				selected_exemplar = torch.utils.data.Subset(exemplar_loader.dataset, exemplars[:num])
				accumulator += selected_exemplar
				if counter2 == 0:
					exemplar_dataset = selected_exemplar
				else:
					exemplar_dataset += selected_exemplar
				counter2 += 1 
			train_loader = torch.utils.data.DataLoader(accumulator, batch_size=args.batch_size, shuffle=True)
			print(len(train_loader.dataset))
			#exemplar_means = compute_mean_of_exemplars(model,torch.utils.data.DataLoader(exemplar_dataset, batch_size=args.batch_size) , current_task_id)
		if (check == 1) :
			model, optimizer = load_checkpoint(model, optimizer, 'check.pth')	
		exemplar_loader = tasks[current_task_id]['exemplar']
		for epoch in range(1, args.epochs_per_task+1):
			# 1. train and save

			prev_model = get_benchmark_model(args)
			prev_model.load_state_dict(model.state_dict())
			prev_opt = type(optimizer)(prev_model.parameters(), lr=args.lr)
			prev_opt.load_state_dict(optimizer.state_dict())
			if ewc == 1:
				train_single_epoch_ewc(model, optimizer, train_loader, criterion, old_params, fisher, current_task_id)
			elif lwf == 1:
				#train_single_epoch_fd(model, optimizer, train_loader, criterion, old_model, current_task_id)
				train_single_epoch_iCarl(model, optimizer, train_loader, criterion, old_model, current_task_id)
			else:
				train_single_epoch(model, optimizer, train_loader, criterion, current_task_id)
			time += 1
			model = model.to(DEVICE)
			val_loader = tasks[current_task_id]['val']
			if ewc == 1:
				metrics = eval_single_epoch_ewc(model, train_loader, criterion, fisher, old_params, current_task_id)
				if current_task_id > 1:
					ewc_loss.append(metrics['ewcloss'])
					all_loss.append(metrics['loss'])
					counter.append(epoch)
			elif lwf == 1:
				#metrics = eval_single_epoch_fd(model, train_loader, criterion, old_model, current_task_id)
				metrics = eval_single_epoch_iCarl(model, train_loader, criterion, old_model, exemplar_means, current_task_id)
			else:
				metrics = eval_single_epoch(model, train_loader, criterion, current_task_id)			


			acc_db, loss_db = log_metrics(metrics, time, current_task_id, acc_db, loss_db)
			

			if loss_db[current_task_id][epoch-1] > the_last_loss:
				if trigger_times == 0:
					backup_model = get_benchmark_model(args)
					backup_model.load_state_dict(prev_model.state_dict())
					backup_opt = type(prev_opt)(backup_model.parameters(), lr=args.lr)
					backup_opt.load_state_dict(prev_opt.state_dict())
				trigger_times += 1
				print('trigger times:', trigger_times)
			else:
				trigger_times = 0
			if trigger_times >= patience:
				print('Early stopping!')
				#tune.report(val_loss = loss_db[current_task_id][epoch])
				model = backup_model.to(DEVICE)
				optimizer = type(backup_opt)(model.parameters(), lr=args.lr)
				optimizer.load_state_dict(backup_opt.state_dict())
				val_loader = tasks[current_task_id]['test']
				
				# 2.1. compute accuracy and loss
				metrics, X, Y = final_eval(model, val_loader, criterion, current_task_id)
				pred_vector = make_prediction_vector(X, Y)
				print(forgetting_metric(pred_vector, pred_vector_list, current_task_id))
				accuracy_results.append(metrics['accuracy'])
				forgetting_result.append(forgetting_metric(pred_vector, pred_vector_list, current_task_id))
				task_counter.append(current_task_id)
				pred_vector_list.append(pred_vector)
				fisher = post_train_process_ewc(train_loader, model, optimizer, current_task_id, fisher)
				old_model = post_train_process_fd(model)
				res = randomExemplarsSelector(model, exemplar_loader, current_task_id, exemplars_per_class)
				#selected_exemplar = torch.utils.data.Subset(exemplar_loader.dataset, res)
				#exemplars_vector_list = []
				exemplars_vector_list.append(res)
				print(len(selected_exemplar))
				matrix = confusion_matrix(X, Y)
				#plot_conf_matrix(matrix)
				acc_db, loss_db = log_metrics(metrics, time, current_task_id, acc_db, loss_db)
				save_checkpoint_Adam(backup_model, backup_opt)
				time = 0
				trigger_times = 0
				the_last_loss = 100
				if current_task_id > 1:
					e_loss = np.array(ewc_loss)
					a_loss = np.array(all_loss)
					epochs = np.array(counter)
					df = pd.DataFrame({"Item Name": epochs, "loss" : a_loss, "ewc_loss" : e_loss})
					string = 'prova{}.csv'.format(current_task_id)
					df.to_csv(string, sep = ';', index = False)
				break

			if loss_db[current_task_id][epoch-1] < the_last_loss:
				the_last_loss = loss_db[current_task_id][epoch-1]
				
			if epoch == args.epochs_per_task:
				#tune.report(val_loss= loss_db[current_task_id][epoch])
				if trigger_times > 0:
					model = backup_model.to(DEVICE)
					optimizer = type(backup_opt)(model.parameters(), lr=args.lr)
					optimizer.load_state_dict(backup_opt.state_dict())
				else:
					model = model.to(DEVICE)
				val_loader = tasks[current_task_id]['test']

				res = herdingExemplarsSelector(model, exemplar_loader, current_task_id, exemplars_per_class)
				exemplars_vector_list.append(res)
				counter3 = 0
				for exemplars in exemplars_vector_list:
					num = int((exemplars_per_class * 100)/(current_task_id))
					selected_exemplar = torch.utils.data.Subset(exemplar_loader.dataset, exemplars[:num])
					if counter3 == 0:
						exemplar_dataset = selected_exemplar
					else:
						exemplar_dataset += selected_exemplar
					counter3 += 1
				exemplar_means = compute_mean_of_exemplars(model, train_loader, current_task_id)	
				# 2.1. compute accuracy and loss
				metrics, X, Y = final_eval_iCarl(model, val_loader, criterion, exemplar_means, current_task_id)

				pred_vector = make_prediction_vector(X, Y)
				print(forgetting_metric(pred_vector, pred_vector_list, current_task_id))
				accuracy_results.append(metrics['accuracy'])
				forgetting_result.append(forgetting_metric(pred_vector, pred_vector_list, current_task_id))
				task_counter.append(current_task_id)
				pred_vector_list.append(pred_vector)
				fisher = post_train_process_ewc(train_loader, model, optimizer, current_task_id, fisher)
				old_model = post_train_process_fd(model)
				#res = randomExemplarsSelector(model, exemplar_loader, current_task_id, exemplars_per_class)
				#selected_exemplar = torch.utils.data.Subset(exemplar_loader.dataset, res)
				#exemplars_vector_list = []
				#exemplars_vector_list.append(res)
				matrix = confusion_matrix(X, Y)
				#plot_conf_matrix(matrix)
				acc_db, loss_db = log_metrics(metrics, time, current_task_id, acc_db, loss_db)
				save_checkpoint_Adam(model, optimizer)
				time = 0
				trigger_times = 0
				the_last_loss = 100
				compute_mean_of_exemplars(model, train_loader, current_task_id)
				if current_task_id > 1:
					e_loss = np.array(ewc_loss)
					a_loss = np.array(all_loss)
					epochs = np.array(counter)
					df = pd.DataFrame({"Item Name": epochs, "loss" : a_loss, "ewc_loss" : e_loss})
					string = 'prova{}.csv'.format(current_task_id)
					df.to_csv(string, sep = ';', index = False)
	
	data_to_csv(accuracy_results, forgetting_result, task_counter)
	return
    
def load_checkpoint(model, optimizer, filename='check.pth'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        #start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        #history = checkpoint['history']
        print("=> loaded checkpoint '{}'")
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer
    
def save_checkpoint_Adam(model, optimizer):
    PATH = './check.pth'
    state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),}
    torch.save(state, PATH)
    
if __name__ == "__main__":
    args = parse_arguments()
    run_experiment(args)
    #analysis = tune.run(tuning, config={"lr" : tune.grid_search([0.001, 0.01, 0.0001])})
    #print("Best config: ", analysis.get_best_config(metric="val_loss"))
    #run(args)


	
