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
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from prettytable import PrettyTable


TRIAL_ID = uuid.uuid4().hex.upper()[0:6]
EXPERIMENT_DIRECTORY = './outputs/{}'.format(TRIAL_ID)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 123


def parse_arguments():
	parser = argparse.ArgumentParser(description='Argument parser')
	parser.add_argument('--tasks', default=5, type=int, help='total number of tasks')
	parser.add_argument('--epochs-per-task', default=1, type=int, help='epochs per task')
	parser.add_argument('--dataset', default='cifar100', type=str, help='dataset. options: mnist, cifar10, cifar100, imagenet')
	parser.add_argument('--batch-size', default=64, type=int, help='batch-size')
	parser.add_argument('--exemplars', default=0, type=int, help='exemplars memory size at each task')
	parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
	parser.add_argument('--gamma', default=0.0, type=float, help='learning rate decay. Use 1.0 for no decay')
	parser.add_argument('--dropout', default=0.0, type=float, help='dropout probability. Use 0.0 for no dropout')
	parser.add_argument('--hiddens', default=256, type=int, help='num of hidden neurons in each layer of a 2-layer MLP')
	parser.add_argument('--compute-eigenspectrum', default=False, type=bool, help='compute eigenvalues/eigenvectors?')
	parser.add_argument('--seed', default=1234, type=int, help='random seed')
	parser.add_argument('--compute-joint-incremental', default=False, type=bool, help='compute joint incremental?')

	args = parser.parse_args()
	return args


def get_seed() :
	return SEED

def init_experiment(args):
	print('------------------- Experiment started -----------------')
	print(f"Parameters:\n  seed={args.seed}\n  benchmark={args.dataset}\n  num_tasks={args.tasks}\n  "+
		  f"epochs_per_task={args.epochs_per_task}\n  batch_size={args.batch_size}\n  "+
		  f"learning_rate={args.lr}\n  learning rate decay(gamma)={args.gamma}\n  dropout prob={args.dropout}\n  " +
		  f"exemplars memory={args.exemplars}\n ")
	
	# 1. setup seed for reproducibility
	torch.manual_seed(args.seed)
	global SEED
	SEED = args.seed
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
	

def data_to_csv( acc_db, forgetting, task_counter, lambda_value =None, alpha_value=None, beta_value=None):

	acc = np.array(acc_db)
	forg = np.array(forgetting)
	#print(task_counter)
	task = np.array(task_counter)
	if lambda_value is not None:
		lamb = np.array(lambda_value)
		alp= np.array(alpha_value)
		beta = np.array(beta_value)

		df = pd.DataFrame({"Task": task, "accuracy": acc, "forgetting": forg , "lambda": lamb, "alpha": alp, "beta": beta})
	else:
		df = pd.DataFrame({"Task": task, "accuracy": acc, "forgetting": forg })
		
	df.to_csv(EXPERIMENT_DIRECTORY + '/RESULTS', sep= ';', index = False)




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


def randomExemplarsSelector(model, loader, task_id, num_exemplars, num_cls):
	exemplars_per_class = num_exemplars
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




def compute_mean_of_exemplars(model, exemplars_loader, task_id):

	extracted_features = []
	extracted_targets = []
	exemplar_means = []

	
	with torch.no_grad():
		model.eval()
		for images, targets in exemplars_loader:
			targets.to('cpu')
			_, feats = model(images.to(DEVICE), task_id, return_features=True)
			# normalize
			extracted_features.append(feats / feats.norm(dim=1).view(-1, 1))
			extracted_targets.extend(targets)
	extracted_features = (torch.cat(extracted_features)).cpu()
	extracted_targets = np.array(extracted_targets)
	for curr_cls in np.unique(extracted_targets):
		# get all indices from current class
		cls_ind = np.where(extracted_targets == curr_cls)[0]
		# get all extracted features for current class
		cls_feats = extracted_features[cls_ind]
		# add the exemplars to the set and normalize
		cls_feats_mean = cls_feats.mean(0) / cls_feats.mean(0).norm()
		exemplar_means.append(cls_feats_mean)

	return exemplar_means


def plot_conf_matrix(matrix):

    plt.figure(1, figsize=(9, 6))
 
    plt.title("Confusion Matrix")
    df_cm = pd.DataFrame(matrix, range(len(matrix[0])), range(len(matrix[0])))
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 12},  fmt='g') 
    plt.savefig("confmatrix", bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()



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

	return 1 - forgetting

def get_PCA_components(features):

    x = StandardScaler().fit_transform(features[0])
    pca = PCA(n_components=2)

    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents, columns = ['principal component 1', 'principal component 2'])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    legend = ['task_1', 'task_5', 'task_10']
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    ax.scatter(principalDf.loc[:, 'principal component 1'], principalDf.loc[:, 'principal component 2'], c='r', s = 50)

    x = StandardScaler().fit_transform(features[4])
    pca = PCA(n_components=2)

    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents, columns = ['principal component 1', 'principal component 2'])

    ax.scatter(principalDf.loc[:, 'principal component 1'], principalDf.loc[:, 'principal component 2'], c='g', s = 50)

    x = StandardScaler().fit_transform(features[9])
    pca = PCA(n_components=2)

    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents, columns = ['principal component 1', 'principal component 2'])

    ax.scatter(principalDf.loc[:, 'principal component 1'], principalDf.loc[:, 'principal component 2'], c='b', s = 50)
    ax.legend(legend)
    ax.grid()
    fig.savefig('prova.png')

def count_parameters(model):
	table = PrettyTable(["Modules", "Parameters"])
	total_params = 0
	for name, parameter in model.named_parameters():
		if not parameter.requires_grad: continue
		param = parameter.numel()
		table.add_row([name, param])
		total_params+=param
	print(table)
	print(f"Total trainable params : {total_params}")
