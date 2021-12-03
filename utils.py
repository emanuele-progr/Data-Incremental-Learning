import uuid
import torch
import copy
import itertools
import argparse
import matplotlib
import random
import numpy as np
import torch.nn as nn
import pandas as pd
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
#from external_libs.hessian_eigenthings import compute_hessian_eigenthings


TRIAL_ID = uuid.uuid4().hex.upper()[0:6]
EXPERIMENT_DIRECTORY = './outputs/{}'.format(TRIAL_ID)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_arguments():
	parser = argparse.ArgumentParser(description='Argument parser')
	parser.add_argument('--tasks', default=5, type=int, help='total number of tasks')
	parser.add_argument('--epochs-per-task', default=1, type=int, help='epochs per task')
	parser.add_argument('--dataset', default='rot-mnist', type=str, help='dataset. options: rot-mnist, perm-mnist, cifar100')
	parser.add_argument('--batch-size', default=64, type=int, help='batch-size')
	parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
	parser.add_argument('--gamma', default=0.0, type=float, help='learning rate decay. Use 1.0 for no decay')
	parser.add_argument('--dropout', default=0.0, type=float, help='dropout probability. Use 0.0 for no dropout')
	parser.add_argument('--hiddens', default=256, type=int, help='num of hidden neurons in each layer of a 2-layer MLP')
	parser.add_argument('--compute-eigenspectrum', default=False, type=bool, help='compute eigenvalues/eigenvectors?')
	parser.add_argument('--seed', default=1234, type=int, help='random seed')
	parser.add_argument('--compute-joint-incremental', default=False, type=bool, help='compute joint incremental?')

	args = parser.parse_args()
	return args


def init_experiment(args):
	print('------------------- Experiment started -----------------')
	print(f"Parameters:\n  seed={args.seed}\n  benchmark={args.dataset}\n  num_tasks={args.tasks}\n  "+
		  f"epochs_per_task={args.epochs_per_task}\n  batch_size={args.batch_size}\n  "+
		  f"learning_rate={args.lr}\n  learning rate decay(gamma)={args.gamma}\n  dropout prob={args.dropout}\n")
	
	# 1. setup seed for reproducibility
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	# 2. create directory to save results
	Path(EXPERIMENT_DIRECTORY).mkdir(parents=True, exist_ok=True)
	print("The results will be saved in {}\n".format(EXPERIMENT_DIRECTORY))
	
	# 3. create data structures to store metrics
	loss_db = {t: [0 for i in range(args.tasks*args.epochs_per_task)] for t in range(1, args.tasks+1)}
	acc_db =  {t: [0 for i in range(args.tasks*args.epochs_per_task)] for t in range(1, args.tasks+1)}
	hessian_eig_db = {}
	return acc_db, loss_db, hessian_eig_db


def end_experiment(args, acc_db, loss_db, hessian_eig_db):
	
	# 1. save all metrics into csv file
	acc_df = pd.DataFrame(acc_db)
	acc_df.to_csv(EXPERIMENT_DIRECTORY+'/accs.csv')
	visualize_result(acc_df, EXPERIMENT_DIRECTORY+'/accs.png')
	
	loss_df = pd.DataFrame(loss_db)
	loss_df.to_csv(EXPERIMENT_DIRECTORY+'/loss.csv')
	visualize_result(loss_df, EXPERIMENT_DIRECTORY+'/loss.png')
	
	hessian_df = pd.DataFrame(hessian_eig_db)
	hessian_df.to_csv(EXPERIMENT_DIRECTORY+'/hessian_eigs.csv')
	
	# 2. calculate average accuracy and forgetting (c.f. ``evaluation`` section in our paper)
	score = np.mean([acc_db[i][-1] for i in acc_db.keys()])
	forget = np.mean([max(acc_db[i])-acc_db[i][-1] for i in range(1, args.tasks)])/100.0
	
	print('average accuracy = {}, forget = {}'.format(score, forget))
	print()
	print('------------------- Experiment ended -----------------')


def log_metrics(metrics, time, task_id, acc_db, loss_db):
	"""
	Log accuracy and loss at different times of training
	"""
	print('epoch {}, task:{}, metrics: {}'.format(time, task_id, metrics))
	# log to db
	acc = metrics['accuracy']
	loss = metrics['loss']
	loss_db[task_id][time-1] = loss
	acc_db[task_id][time-1] = acc
	return acc_db, loss_db


def save_eigenvec(filename, arr):
	"""
	Save eigenvectors to file
	"""
	np.save(filename, arr)

'''
def log_hessian(model, loader, time, task_id, hessian_eig_db):
	"""
	Compute and log Hessian for a specific task
	
	:param model:  The PyTorch Model
	:param loader: Dataloader [to calculate loss and then Hessian]
	:param time: time is a discrete concept regarding epoch. If we have T tasks each with E epoch,
	time will be from 0, to (T x E)-1. E.g., if we have 5 tasks with 5 epochs each, then when we finish
	task 1, time will be 5.
	:param task_id: Task id (to distiniguish between Hessians of different tasks)
	:param hessian_eig_db: (The dictionary to store hessians)
	:return:
	"""
	criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
	use_gpu = True if DEVICE != 'cpu' else False
	est_eigenvals, est_eigenvecs = compute_hessian_eigenthings(
		model,
		loader,
		criterion,
		num_eigenthings=3,
		power_iter_steps=18,
		power_iter_err_threshold=1e-5,
		momentum=0,
		use_gpu=use_gpu,
	)
	key = 'task-{}-epoch-{}'.format(task_id, time-1)
	hessian_eig_db[key] = est_eigenvals
	save_eigenvec(EXPERIMENT_DIRECTORY+"/{}-vec.npy".format(key), est_eigenvecs)
	return hessian_eig_db

'''


def save_checkpoint(model, time):
	"""
	Save checkpoints of model paramters
	:param model: pytorch model
	:param time: int
	"""
	filename = '{directory}/model-{trial}-{time}.pth'.format(directory=EXPERIMENT_DIRECTORY, trial=TRIAL_ID, time=time)
	torch.save(model.cpu().state_dict(), filename)


def visualize_result(df, filename):
	ax = sns.lineplot(data=df,  dashes=False)
	ax.figure.savefig(filename, dpi=250)
	plt.close()


def compute_fisher_matrix_diag(train_loader, model, optimizer, current_task_id, sampling_type='true'):

	fisher = {n: torch.zeros(p.shape).to(DEVICE) for n, p in model.named_parameters() if p.requires_grad}
	n_samples_batches = (len(train_loader.dataset) // train_loader.batch_size)
	criterion = nn.CrossEntropyLoss().to(DEVICE)
	model.eval()
	loss = 0
	for batch_idx, (data, target) in enumerate(train_loader):
		data = data.to(DEVICE)
		target = target.to(DEVICE)
		# for cifar head
		if current_task_id is not None:
			output = model(data, current_task_id)
		else:
			output = model(data)
		#loss = criterion(output, target)
		loss = criterion(output, target)
			
		optimizer.zero_grad()
		loss.backward()
		for n, p in model.named_parameters():
			if n == 'linear.weight' or n == 'linear.bias':
				pass
			elif n == 'bn1.weight' or n == 'bn1.bias':
				pass
			elif 'IC1' in n or 'IC2' in n:
				pass
			else:
				if p.grad is not None:
					fisher[n] += p.grad.pow(2) * len(target)
					
	n_samples = n_samples_batches * train_loader.batch_size
	fisher = {n: (p / n_samples) for n, p in fisher.items()}

	return fisher

def post_train_process_ewc(train_loader, model, optimizer, current_task_id, fisher):

	alpha = 0.5

	current_fisher = compute_fisher_matrix_diag(train_loader, model, optimizer, current_task_id)
		
	for n in fisher.keys():
		fisher[n] = (alpha * fisher[n] + (1 - alpha) * current_fisher[n])

	return fisher

def post_train_process_fd(model):

	model_old = copy.deepcopy(model)
	model_old.eval()
	model_old.freeze_all()
	
	return model_old






def herdingExemplarsSelector(model, loader, task_id, num_exemplars):

	exemplars_per_class = num_exemplars

	extracted_features = []
	extracted_targets = []
	with torch.no_grad():
		model.eval()
		for data, target in loader:
			data = data.to(DEVICE)
			target = target.to('cpu')
			_, feat = model(data, task_id, return_features=True)
			feat = feat / feat.norm(dim=1).view(-1, 1)
			extracted_features.append(feat)
			extracted_targets.extend(target)
	extracted_features = (torch.cat(extracted_features)).cpu()
	extracted_targets = np.array(extracted_targets)
	result = []
	final_result = []

	for curr_cls in np.unique(extracted_targets):
		cls_ind = np.where(extracted_targets == curr_cls)[0]

		cls_feats = extracted_features[cls_ind]
		cls_mu = cls_feats.mean(0)

		selected = []
		selected_feat = []
		for k in range(exemplars_per_class):
			sum_others = torch.zeros(cls_feats.shape[1])
			for j in selected_feat:
				sum_others += j / (k + 1)
			dist_min = np.inf

			for item in cls_ind:
				if item not in selected:
					feat = extracted_features[item]
					dist = torch.norm(cls_mu - feat / (k + 1) - sum_others)
					if dist < dist_min:
						dist_min = dist
						newone = item
						newonefeat = feat
			selected_feat.append(newonefeat)
			selected.append(newone)
		result.append(selected)
	for i in range(exemplars_per_class):
		for cls in np.unique(extracted_targets):
			final_result.append(result[cls][i])

	
	return final_result


def randomExemplarsSelector(model, loader, task_id, num_exemplars):
	exemplars_per_class = num_exemplars
	num_cls = 100
	result = []
	final_result = []
	targets = np.array([])
	for data, target in loader:
		arr = target.cpu().detach().numpy()
		targets = np.concatenate([targets, arr])
	labels = targets
	for curr_cls in range(num_cls):
		cls_ind = np.where(labels == curr_cls)[0]
		result.append(random.sample(list(cls_ind), exemplars_per_class))
	for i in range(exemplars_per_class):
		for cls in range(num_cls):
			final_result.append(result[cls][i])

	
	return final_result


def entropyExemplarsSelector(model, loader, task_id,  num_exemplars):

	extracted_logits = []
	extracted_targets = []
	with torch.no_grad():
		model.eval()
		for images, targets in loader:
			extracted_logits.append(model(images.to(DEVICE), task_id))
			extracted_targets.extend(targets)
		extracted_logits = (torch.cat(extracted_logits)).cpu()
		extracted_targets = np.array(extracted_targets)
		result = []

		for curr_cls in np.unique(extracted_targets):
			cls_ind = np.where(extracted_targets == curr_cls)[0]
			cls_logits = extracted_logits[cls_ind]

			probs = torch.softmax(cls_logits, dim=1)
			log_probs = torch.log(probs)
			minus_entropy = (probs * log_probs).sum(1)
			selected = cls_ind[minus_entropy.sort()[1][:num_exemplars]]
			result.extend(selected)

		return result



def distanceExemplarsSelector(model, loader, task_id, num_exemplars):

	exemplars_per_class = num_exemplars

	extracted_logits = []
	extracted_targets = []
	with torch.no_grad():
		model.eval()
		for images, targets in loader:
			extracted_logits.append(model(images.to(DEVICE), task_id))
			extracted_targets.extend(targets)

		extracted_logits = (torch.cat(extracted_logits)).cpu()
		extracted_targets = np.array(extracted_targets)
		result = []

		for curr_cls in np.unique(extracted_targets):

			cls_ind = np.where(extracted_targets == curr_cls)[0]
			cls_logits = extracted_logits[cls_ind]
			distance = cls_logits[:, curr_cls]
			selected = cls_ind[distance.sort()[1][:exemplars_per_class]]
			result.extend(selected)
		
	return result





