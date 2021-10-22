import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from model import MLP, ResNet18, ResNet32
from data_utils import get_permuted_mnist_tasks, get_rotated_mnist_tasks, get_split_cifar100_tasks2, get_split_cifar100_tasks, get_split_cifar10_tasks, get_split_cifar100_tasks_joint
from utils import parse_arguments, DEVICE, init_experiment, end_experiment, log_metrics, save_checkpoint, post_train_process
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
	lamb = 5000
	net = net.to(DEVICE)
	loss_penalty = 0

	if task_id > 1:

		
		loss_penalty = ewc_penalty(net, fisher, old_params)
		print(loss_penalty)
	
	net.train()
	for batch_idx, (data, target) in enumerate(loader):
		data = data.to(DEVICE)
		target = target.to(DEVICE)
		optimizer.zero_grad()
		if task_id:
			pred = net(data, task_id)
		else:
			pred = net(data)
		loss = criterion(pred, target) + loss_penalty
		print(loss)
		loss.backward(retain_graph = True)
		optimizer.step()
	return net


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
			test_loss += criterion(output, target).item()
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
	test_loss /= len(loader.dataset)
	correct = correct.to('cpu')
	avg_acc = 100.0 * float(correct.numpy()) / len(loader.dataset)
	return {'accuracy': avg_acc, 'loss': test_loss}


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
            test_loss += criterion(output, target).item()
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
		return get_split_cifar100_tasks2
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
	lamb = 5000
	loss = 0
	loss_reg = 0
	for n, p in model.named_parameters():
		if n in fisher.keys():
			loss_reg += torch.sum(fisher[n] * (p - older_params[n]).pow(2))/2
	loss += lamb * loss_reg

	return loss



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
	ewc = 1
	#lr = [0.01, 0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
	lr = [0.001, 0.0001, 0.0001, 0.0001, 0.0001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]
	#lr = [0.001, 0.0001, 0.0001, 0.00001, 0.00001]
	
	fisher = {n: torch.zeros(p.shape).to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}


	for current_task_id in range(1, args.tasks+1):
		#lr = max(args.lr * args.gamma ** (current_task_id), 0.00005)
		if args.compute_joint_incremental:
			model = get_benchmark_model(args)
		old_params = {n: p.clone().detach() for n,p in model.named_parameters() if p.requires_grad}
		print("================== TASK {} / {} =================".format(current_task_id, args.tasks))
		train_loader = tasks[current_task_id]['train']
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)#lr=lr[current_task_id - 1])
		if (check == 1) :
			model, optimizer = load_checkpoint(model, optimizer, 'check.pth')
		
		
		
		
		
		for epoch in range(1, args.epochs_per_task+1):
			# 1. train and save

			prev_model = model
			prev_opt = optimizer
			if ewc == 1:
				s = 'ok'
				print(s)
				train_single_epoch_ewc(model, optimizer, train_loader, criterion, old_params, fisher, current_task_id)
			else:
				train_single_epoch(model, optimizer, train_loader, criterion, current_task_id)
			time += 1
			model = model.to(DEVICE)
			val_loader = tasks[current_task_id]['val']
			metrics = eval_single_epoch(model, val_loader, criterion, current_task_id)
			fisher = post_train_process(train_loader, model, optimizer, current_task_id, fisher)
			acc_db, loss_db = log_metrics(metrics, time, current_task_id, acc_db, loss_db)
			

			if loss_db[current_task_id][epoch-1] > the_last_loss:
				if trigger_times == 0:
					backup_model = prev_model
					backup_opt = prev_opt
				trigger_times += 1
				print('trigger times:', trigger_times)
			else:
				trigger_times = 0
			if trigger_times >= patience:
				print('Early stopping!')
				#tune.report(val_loss = loss_db[current_task_id][epoch])
				model = backup_model.to(DEVICE)
				val_loader = tasks[current_task_id]['test']
				
				# 2.1. compute accuracy and loss
				metrics, X, Y = final_eval(model, val_loader, criterion, current_task_id)
				fisher = post_train_process(train_loader, model, optimizer, current_task_id, fisher)
				matrix = confusion_matrix(X, Y)
				#plot_conf_matrix(matrix)
				acc_db, loss_db = log_metrics(metrics, time, current_task_id, acc_db, loss_db)
				save_checkpoint_Adam(backup_model, backup_opt)
				time = 0
				trigger_times = 0
				the_last_loss = 100
				break


			if loss_db[current_task_id][epoch-1] < the_last_loss:
				the_last_loss = loss_db[current_task_id][epoch-1]
				
			if epoch == args.epochs_per_task:
				#tune.report(val_loss= loss_db[current_task_id][epoch])
				if trigger_times > 0:
					model = backup_model.to(DEVICE)
					s = "ok"
					print(s)
				else:
					model = model.to(DEVICE)
				val_loader = tasks[current_task_id]['test']
					
				# 2.1. compute accuracy and loss
				metrics, X, Y = final_eval(model, val_loader, criterion, current_task_id)
				fisher = post_train_process(train_loader, model, optimizer, current_task_id, fisher)
				matrix = confusion_matrix(X, Y)
				#plot_conf_matrix(matrix)
				acc_db, loss_db = log_metrics(metrics, time, current_task_id, acc_db, loss_db)
				save_checkpoint_Adam(model, optimizer)
				time = 0
				trigger_times = 0
				the_last_loss = 100
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


	
