import os
import torch
import math
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from train import *
from model import MLP, ResNet18, ResNet32, ResNet50
from data_utils import *
from utils import *
from sklearn.metrics import confusion_matrix
TRAIN_CLASSES = 100


def get_benchmark_data_loader(args):
    """
    Returns the benchmark loader which could be either of these:
    get_split_cifar100_tasks, get_split_cifar100_tasks_joint, get_split_MNIST_tasks, get_split_cifar10_tasks,
    get_split_cifar10_tasks_joint, get_split_tiny_ImageNet_tasks, get_split_tiny_ImageNet_tasks_joint

    :param args:
    :return: a function which when called, returns all tasks
    """

    if args.dataset == 'mnist':
        return get_split_MNIST_tasks
    elif args.dataset == 'cifar-100' or args.dataset == 'cifar100' and args.compute_joint_incremental:
        return get_split_cifar100_tasks_joint
    elif args.dataset == 'cifar-100' or args.dataset == 'cifar100' and args.compute_joint_incremental is False:
        return get_split_cifar100_tasks
    elif args.dataset == 'cifar-10' or args.dataset == 'cifar10' and args.compute_joint_incremental:
        return get_split_cifar10_tasks_joint      
    elif args.dataset == 'cifar-10' or args.dataset == 'cifar10' and args.compute_joint_incremental is False:
        return get_split_cifar10_tasks
    elif args.dataset == 'tiny-imagenet' or args.dataset == 'imagenet'and args.compute_joint_incremental is False:
        return get_split_tiny_ImageNet_tasks
    elif args.dataset == 'tiny-imagenet' or args.dataset == 'imagenet'and args.compute_joint_incremental:
        return get_split_tiny_ImageNet_tasks_joint   
    else:
        raise Exception("Unknown dataset.\n" +
                        "The code supports 'mnist, cifar-10, cifar-100 and imagenet.")


def get_benchmark_model(args):
    """
    Return the corresponding PyTorch model for experiment
    :param args:
    :return:
    """
    if args.resnet is 32:
        if 'mnist' in args.dataset:
            TRAIN_CLASSES = 10
            return MLP(256, {'dropout': args.dropout}).to(DEVICE)
        elif 'cifar100' in args.dataset:
            TRAIN_CLASSES = 100
            return ResNet32(config={'dropout': args.dropout}).to(DEVICE)
        elif 'cifar10' in args.dataset:
            TRAIN_CLASSES = 10
            return ResNet32(nclasses=10, config={'dropout': args.dropout}).to(DEVICE)
        elif 'imagenet' in args.dataset:
            TRAIN_CLASSES = 200
            return ResNet32(nclasses=200, config={'dropout': args.dropout}).to(DEVICE)
        else:
            raise Exception("Unknown dataset.\n" +
                            "The code supports 'mnist, cifar10, cifar100 and imagenet.")
    elif args.resnet is 18:
        if 'mnist' in args.dataset:
            TRAIN_CLASSES = 10
            return MLP(256, {'dropout': args.dropout}).to(DEVICE)
        elif 'cifar100' in args.dataset:
            TRAIN_CLASSES = 100
            return ResNet18(config={'dropout': args.dropout}).to(DEVICE)
        elif 'cifar10' in args.dataset:
            TRAIN_CLASSES = 10
            return ResNet18(nclasses=10, config={'dropout': args.dropout}).to(DEVICE)
        elif 'imagenet' in args.dataset:
            TRAIN_CLASSES = 200
            return ResNet18(nclasses=200, config={'dropout': args.dropout}).to(DEVICE)
        else:
            raise Exception("Unknown dataset.\n" +
                            "The code supports 'mnist, cifar10, cifar100 and imagenet.")
    
    elif args.resnet is 50:
        if 'mnist' in args.dataset:
            TRAIN_CLASSES = 10
            return MLP(256, {'dropout': args.dropout}).to(DEVICE)
        elif 'cifar100' in args.dataset:
            TRAIN_CLASSES = 100
            return ResNet50(config={'dropout': args.dropout}).to(DEVICE)
        elif 'cifar10' in args.dataset:
            TRAIN_CLASSES = 10
            return ResNet50(nclasses=10, config={'dropout': args.dropout}).to(DEVICE)
        elif 'imagenet' in args.dataset:
            TRAIN_CLASSES = 200
            return ResNet50(nclasses=200, config={'dropout': args.dropout}).to(DEVICE)
        else:
            raise Exception("Unknown dataset.\n" +
                            "The code supports 'mnist, cifar10, cifar100 and imagenet.")

    else:
        raise Exception("Unknown resnet.\n" +
                            "The code supports 'resnet32 (32), resnet18 (18) and resnet50 (50).") 



def run_experiment(args):

    # organize_validation_data_tiny_ImageNet()
    acc_db, loss_db, hessian_eig_db = init_experiment(args)
    tasks = get_benchmark_data_loader(args)(args.tasks, args.batch_size)
    model = get_benchmark_model(args)

    # display model parameters
    count_parameters(model)
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
    mean_vector = []

    fisher = {n: torch.zeros(p.shape).to(DEVICE)
              for n, p in model.named_parameters() if p.requires_grad}

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for current_task_id in range(1, args.tasks+1):
        ewc_loss = []
        all_loss = []
        counter = []

        if args.reboot:
            if current_task_id > 1:
                with torch.no_grad():
                    # student-teacher head random init
                    stdv = 1. / math.sqrt(model.linear.weight.size(1))
                    model.linear.weight.data.uniform_(-stdv, stdv)
                    model.linear.bias.data.uniform_(-stdv, stdv)


        old_params = {n: p.clone().detach()
                      for n, p in model.named_parameters() if p.requires_grad}
        print("================== TASK {} / {} =================".format(current_task_id, args.tasks))
        train_loader = tasks[current_task_id]['train']
        print("training data in this task: {}".format(len(train_loader.dataset)))

        exemplars_per_class = args.exemplars_per_class
        counter = 0

        if current_task_id > 1:
            accumulator = train_loader.dataset
            for exemplars in exemplars_vector_list:

                num = int((exemplars_per_class * 100)/(current_task_id - 1))
                selected_exemplar = torch.utils.data.Subset(
                    exemplar_loader.dataset, exemplars[:num])
                accumulator += selected_exemplar
                if counter == 0:
                    exemplar_dataset = selected_exemplar
                else:
                    exemplar_dataset += selected_exemplar
                counter += 1
            train_loader = torch.utils.data.DataLoader(
                accumulator, batch_size=args.batch_size, shuffle=True)
            print(len(train_loader.dataset))
            #exemplar_means = compute_mean_of_exemplars(model,torch.utils.data.DataLoader(exemplar_dataset, batch_size=args.batch_size) , current_task_id)
        if (check == 1):
            model, optimizer = load_checkpoint(model, optimizer, 'check.pth')

        exemplar_loader = tasks[current_task_id]['exemplar']
        for epoch in range(1, args.epochs_per_task+1):
            # 1. train and save

            prev_model = get_benchmark_model(args)
            prev_model.load_state_dict(model.state_dict())
            prev_opt = type(optimizer)(prev_model.parameters(), lr=args.lr)
            prev_opt.load_state_dict(optimizer.state_dict())
            if ewc == 1:
                train_single_epoch_ewc(
                    model, optimizer, train_loader, criterion, old_params, fisher, current_task_id)
            elif lwf == 1:
                train_single_epoch_lwf(
                    model, optimizer, train_loader, criterion, old_model, current_task_id)
                #train_single_epoch_iCarl(model, optimizer, train_loader, criterion, old_model, current_task_id)
            else:
                train_single_epoch(
                    model, optimizer, train_loader, criterion, current_task_id)
            time += 1
            model = model.to(DEVICE)
            val_loader = tasks[current_task_id]['val']
            if ewc == 1:
                metrics = eval_single_epoch_ewc(
                    model, val_loader, criterion, fisher, old_params, current_task_id)
                if current_task_id > 1:
                    ewc_loss.append(metrics['ewcloss'])
                    all_loss.append(metrics['loss'])
                    # counter.append(epoch)
            elif lwf == 1:
                metrics = eval_single_epoch_lwf(
                    model, val_loader, criterion, old_model, current_task_id)
                #metrics = eval_single_epoch_iCarl(model, val_loader, criterion, old_model, exemplar_means, current_task_id)
            else:
                metrics = eval_single_epoch(
                    model, val_loader, criterion, current_task_id)

            acc_db, loss_db = log_metrics(
                metrics, time, current_task_id, acc_db, loss_db)

            if loss_db[current_task_id][epoch-1] > the_last_loss:
                if trigger_times == 0:
                    backup_model = get_benchmark_model(args)
                    backup_model.load_state_dict(prev_model.state_dict())
                    backup_opt = type(prev_opt)(
                        backup_model.parameters(), lr=args.lr)
                    backup_opt.load_state_dict(prev_opt.state_dict())
                trigger_times += 1
                print('trigger times:', trigger_times)
            else:
                trigger_times = 0
            if trigger_times >= patience:
                print('Early stopping!')
                model = backup_model.to(DEVICE)
                optimizer = type(backup_opt)(model.parameters(), lr=args.lr)
                optimizer.load_state_dict(backup_opt.state_dict())
                test_loader = tasks[current_task_id]['test']

                # 2.1. compute accuracy and loss
                metrics, X, Y = final_eval(
                    model, test_loader, criterion, current_task_id)
                pred_vector = make_prediction_vector(X, Y)
                print(forgetting_metric(pred_vector,
                      pred_vector_list, current_task_id))
                accuracy_results.append(metrics['accuracy'])
                forgetting_result.append(forgetting_metric(
                    pred_vector, pred_vector_list, current_task_id))
                task_counter.append(current_task_id)
                pred_vector_list.append(pred_vector)
                fisher = post_train_process_ewc(
                    train_loader, model, optimizer, current_task_id, fisher)
                old_model = post_train_process_freeze_model(model)
                res = randomExemplarsSelector(
                    model, exemplar_loader, current_task_id, exemplars_per_class, TRAIN_CLASSES)
                #selected_exemplar = torch.utils.data.Subset(exemplar_loader.dataset, res)
                #exemplars_vector_list = []
                exemplars_vector_list.append(res)
                # print(len(selected_exemplar))
                matrix = confusion_matrix(X, Y)
                # plot_conf_matrix(matrix)
                acc_db, loss_db = log_metrics(
                    metrics, time, current_task_id, acc_db, loss_db)
                save_checkpoint_Adam(backup_model, backup_opt)
                time = 0
                trigger_times = 0
                the_last_loss = 100

                break

            if loss_db[current_task_id][epoch-1] < the_last_loss:
                the_last_loss = loss_db[current_task_id][epoch-1]

            if epoch == args.epochs_per_task:
                if trigger_times > 0:
                    model = get_benchmark_model(args)
                    model.load_state_dict(backup_model.state_dict())
                    model = model.to(DEVICE)
                    optimizer = type(backup_opt)(
                        model.parameters(), lr=args.lr)
                    optimizer.load_state_dict(backup_opt.state_dict())
                else:
                    model = model.to(DEVICE)
                test_loader = tasks[current_task_id]['test']

                # 2.1. compute accuracy and loss
                metrics, X, Y = final_eval(
                    model, test_loader, criterion, current_task_id)

                pred_vector = make_prediction_vector(X, Y)
                print(forgetting_metric(pred_vector,
                      pred_vector_list, current_task_id))
                accuracy_results.append(metrics['accuracy'])
                forgetting_result.append(forgetting_metric(
                    pred_vector, pred_vector_list, current_task_id))
                task_counter.append(current_task_id)
                pred_vector_list.append(pred_vector)
                fisher = post_train_process_ewc(
                    train_loader, model, optimizer, current_task_id, fisher)
                old_model = post_train_process_freeze_model(model)
                res = randomExemplarsSelector(
                    model, exemplar_loader, current_task_id, exemplars_per_class, TRAIN_CLASSES)
                #selected_exemplar = torch.utils.data.Subset(exemplar_loader.dataset, res)
                #exemplars_vector_list = []
                exemplars_vector_list.append(res)
                matrix = confusion_matrix(X, Y)
                # plot_conf_matrix(matrix)
                acc_db, loss_db = log_metrics(
                    metrics, time, current_task_id, acc_db, loss_db)
                save_checkpoint_Adam(model, optimizer)
                time = 0
                trigger_times = 0
                the_last_loss = 100
                mean = compute_mean_of_exemplars(
                    model, train_loader, current_task_id)
                mean_vector.append(torch.stack(mean).numpy())

    # get_PCA_components(mean_vector)
    data_to_csv(accuracy_results, forgetting_result, task_counter)
    return


'''
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
'''


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
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer


def save_checkpoint_Adam(model, optimizer):
    PATH = './check.pth'
    state = {'state_dict': model.state_dict(
    ), 'optimizer': optimizer.state_dict(), }
    torch.save(state, PATH)


config = {
    #"lambda": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    #"lambda": [1, 1, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    "lambda": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    "alpha": [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1 ],
    "beta": [1, 0, 1, 2, 5, 10, 20, 1, 0, 1, 2, 5, 10, 20, 1, 0, 1, 2, 5, 10, 20]

    #"lambda": [1, 1, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    #"alpha": [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    #"beta": [1, 0, 1, 2, 5, 10, 20, 1, 0, 1, 2, 5, 10, 20]
    #"lambda": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    #"alpha": [0, 1, 1, 1, 1, 1, 1],
    #"beta": [1, 0, 1, 2, 5, 10, 20]

    #"lambda": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    #"alpha": [1, 1, 1, 1, 1, 1, 1],
    #"beta": [0, 0, 0, 0, 0, 0, 0]


}


def tuning_on_task2(args):

    # organize_validation_data_tiny_ImageNet()
    acc_db, loss_db, hessian_eig_db = init_experiment(args)
    tasks = get_benchmark_data_loader(args)(args.tasks, args.batch_size)
    model = get_benchmark_model(args)
    # count_parameters(model)

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
    lambda_value = []
    alpha_value = []
    beta_value = []
    pred_vector_list = [[0]]
    exemplars_vector_list = []
    accuracy_results = []
    forgetting_result = []
    task_counter = []
    exemplar_means = []
    mean_vector = []

    fisher = {n: torch.zeros(p.shape).to(DEVICE)
              for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for current_task_id in range(1, 3):
        #lr = max(args.lr * args.gamma ** (current_task_id), 0.00005)
        ewc_loss = []
        all_loss = []
        counter = []
        if args.compute_joint_incremental:
            model = get_benchmark_model(args)

        if args.reboot:
            if current_task_id > 1:
                with torch.no_grad():
                    # student-teacher head random init
                    stdv = 1. / math.sqrt(model.linear.weight.size(1))
                    model.linear.weight.data.uniform_(-stdv, stdv)
                    model.linear.bias.data.uniform_(-stdv, stdv)

        old_params = {n: p.clone().detach()
                      for n, p in model.named_parameters() if p.requires_grad}
        print("================== TASK {} / {} =================".format(current_task_id, args.tasks))
        train_loader = tasks[current_task_id]['train']
        print("training data in this task: {}".format(len(train_loader.dataset)))

        exemplars_per_class = args.exemplars_per_class
        counter = 0

        if current_task_id > 1:
            accumulator = train_loader.dataset
            for exemplars in exemplars_vector_list:

                num = int((exemplars_per_class * 100)/(current_task_id - 1))
                selected_exemplar = torch.utils.data.Subset(
                    exemplar_loader.dataset, exemplars[:num])
                accumulator += selected_exemplar
                if counter == 0:
                    exemplar_dataset = selected_exemplar
                else:
                    exemplar_dataset += selected_exemplar
                counter += 1
            train_loader = torch.utils.data.DataLoader(
                accumulator, batch_size=args.batch_size, shuffle=True)
            print(len(train_loader.dataset))
            #exemplar_means = compute_mean_of_exemplars(model,torch.utils.data.DataLoader(exemplar_dataset, batch_size=args.batch_size) , current_task_id)

        exemplar_loader = tasks[current_task_id]['exemplar']
        if current_task_id == 1:
			# first task, standard training
            for epoch in range(1, args.epochs_per_task+1):
                # 1. train and save

                prev_model = get_benchmark_model(args)
                prev_model.load_state_dict(model.state_dict())
                prev_opt = type(optimizer)(prev_model.parameters(), lr=args.lr)
                prev_opt.load_state_dict(optimizer.state_dict())
                if ewc == 1:
                    train_single_epoch_ewc(
                        model, optimizer, train_loader, criterion, old_params, fisher, current_task_id)
                elif lwf == 1:
                    train_single_epoch_fd(
                        model, optimizer, train_loader, criterion, old_model, current_task_id)
                    #train_single_epoch_iCarl(model, optimizer, train_loader, criterion, old_model, current_task_id)
                else:
                    train_single_epoch(
                        model, optimizer, train_loader, criterion, current_task_id)
                time += 1
                model = model.to(DEVICE)
                val_loader = tasks[current_task_id]['val']
                if ewc == 1:
                    metrics = eval_single_epoch_ewc(
                        model, val_loader, criterion, fisher, old_params, current_task_id)
                    if current_task_id > 1:
                        ewc_loss.append(metrics['ewcloss'])
                        all_loss.append(metrics['loss'])
                        # counter.append(epoch)
                elif lwf == 1:
                    metrics = eval_single_epoch_fd(
                        model, val_loader, criterion, old_model, current_task_id)
                    #metrics = eval_single_epoch_iCarl(model, val_loader, criterion, old_model, exemplar_means, current_task_id)
                else:
                    metrics = eval_single_epoch(
                        model, val_loader, criterion, current_task_id)

                acc_db, loss_db = log_metrics(
                    metrics, time, current_task_id, acc_db, loss_db)

                if loss_db[current_task_id][epoch-1] > the_last_loss:
                    if trigger_times == 0:
                        backup_model = get_benchmark_model(args)
                        backup_model.load_state_dict(prev_model.state_dict())
                        backup_opt = type(prev_opt)(
                            backup_model.parameters(), lr=args.lr)
                        backup_opt.load_state_dict(prev_opt.state_dict())
                    trigger_times += 1
                    print('trigger times:', trigger_times)
                else:
                    trigger_times = 0
                if trigger_times >= patience:
                    print('Early stopping!')
                    model = backup_model.to(DEVICE)
                    optimizer = type(backup_opt)(
                        model.parameters(), lr=args.lr)
                    optimizer.load_state_dict(backup_opt.state_dict())

                    # 2.1. compute accuracy and loss
                    metrics, X, Y = final_eval(
                        model, val_loader, criterion, current_task_id)
                    pred_vector = make_prediction_vector(X, Y)
                    print(forgetting_metric(pred_vector,
                                            pred_vector_list, current_task_id))
                    pred_vector_list.append(pred_vector)
                    fisher = post_train_process_ewc(
                        train_loader, model, optimizer, current_task_id, fisher)
                    old_model = post_train_process_freeze_model(model)
                    res = randomExemplarsSelector(
                        model, exemplar_loader, current_task_id, exemplars_per_class, TRAIN_CLASSES)
                    exemplars_vector_list.append(res)
                    matrix = confusion_matrix(X, Y)
                    # plot_conf_matrix(matrix)
                    acc_db, loss_db = log_metrics(
                        metrics, time, current_task_id, acc_db, loss_db)
                    save_checkpoint_Adam(model, optimizer)
                    time = 0
                    trigger_times = 0
                    the_last_loss = 100

                    break

                if loss_db[current_task_id][epoch-1] < the_last_loss:
                    the_last_loss = loss_db[current_task_id][epoch-1]

                if epoch == args.epochs_per_task:
                    if trigger_times > 0:
                        model = get_benchmark_model(args)
                        model.load_state_dict(backup_model.state_dict())
                        model = model.to(DEVICE)
                        optimizer = type(backup_opt)(
                            model.parameters(), lr=args.lr)
                        optimizer.load_state_dict(backup_opt.state_dict())
                    else:
                        model = model.to(DEVICE)

                    # 2.1. compute accuracy and loss
                    metrics, X, Y = final_eval(
                        model, val_loader, criterion, current_task_id)

                    pred_vector = make_prediction_vector(X, Y)
                    print(forgetting_metric(pred_vector,
                                            pred_vector_list, current_task_id))
                    pred_vector_list.append(pred_vector)
                    fisher = post_train_process_ewc(
                        train_loader, model, optimizer, current_task_id, fisher)
                    old_model = post_train_process_freeze_model(model)
                    res = randomExemplarsSelector(
                        model, exemplar_loader, current_task_id, exemplars_per_class, TRAIN_CLASSES)
                    exemplars_vector_list.append(res)
                    matrix = confusion_matrix(X, Y)
                    # plot_conf_matrix(matrix)
                    acc_db, loss_db = log_metrics(
                        metrics, time, current_task_id, acc_db, loss_db)
					# save model checkpoint on task 1
                    save_checkpoint_Adam(model, optimizer)
                    time = 0
                    trigger_times = 0
                    the_last_loss = 100
                    #mean = compute_mean_of_exemplars(
                        #model, train_loader, current_task_id)
                    #mean_vector.append(torch.stack(mean).numpy())

        if current_task_id > 1:
			# parameters trials on task 2
            for index in range(len(config['lambda'])):
                print('trial n.{}/{}'.format(index+1, len(config['lambda'])))
                print("lambda : {} , alpha : {}, beta : {}".format(config['lambda'][index], config['alpha'][index], config['beta'][index]))
                model, optimizer = load_checkpoint(
                    model, optimizer, 'check.pth')
                for epoch in range(1, args.epochs_per_task+1):

                    prev_model = get_benchmark_model(args)
                    prev_model.load_state_dict(model.state_dict())
                    prev_opt = type(optimizer)(
                        prev_model.parameters(), lr=args.lr)
                    prev_opt.load_state_dict(optimizer.state_dict())
                    if ewc == 1:
                        train_single_epoch_ewc(
                            model, optimizer, train_loader, criterion, old_params, fisher, current_task_id, config, index)
                    elif lwf == 1:
                        train_single_epoch_focal_fd(
                            model, optimizer, train_loader, criterion, old_model, current_task_id, config, index)
                        #train_single_epoch_iCarl(model, optimizer, train_loader, criterion, old_model, current_task_id)
                    else:
                        train_single_epoch(
                            model, optimizer, train_loader, criterion, current_task_id)
                    time += 1
                    model = model.to(DEVICE)
                    val_loader = tasks[current_task_id]['val']
                    if ewc == 1:
                        metrics = eval_single_epoch_ewc(
                            model, val_loader, criterion, fisher, old_params, current_task_id, config, index)

                    elif lwf == 1:
                        metrics = eval_single_epoch_focal_fd(
                            model, val_loader, criterion, old_model, current_task_id, config, index)
                        #metrics = eval_single_epoch_iCarl(model, val_loader, criterion, old_model, exemplar_means, current_task_id)
                    else:
                        metrics = eval_single_epoch(
                            model, val_loader, criterion, current_task_id)

                    acc_db, loss_db = log_metrics(
                        metrics, time, current_task_id, acc_db, loss_db)

                    if loss_db[current_task_id][epoch-1] > the_last_loss:
                        if trigger_times == 0:
                            backup_model = get_benchmark_model(args)
                            backup_model.load_state_dict(
                                prev_model.state_dict())
                            backup_opt = type(prev_opt)(
                                backup_model.parameters(), lr=args.lr)
                            backup_opt.load_state_dict(prev_opt.state_dict())
                        trigger_times += 1
                        print('trigger times:', trigger_times)
                    else:
                        trigger_times = 0
                    if trigger_times >= patience:
                        print('Early stopping!')
                        model = backup_model.to(DEVICE)
                        optimizer = type(backup_opt)(
                            model.parameters(), lr=args.lr)
                        optimizer.load_state_dict(backup_opt.state_dict())

                        # 2.1. compute accuracy and loss
                        metrics, X, Y = final_eval(
                            model, val_loader, criterion, current_task_id)
                        pred_vector = make_prediction_vector(X, Y)
                        print(forgetting_metric(pred_vector,
                                                pred_vector_list, current_task_id))
                        accuracy_results.append(metrics['accuracy'])
                        forgetting_result.append(round(forgetting_metric(
                            pred_vector, pred_vector_list, current_task_id), 2))
                        lambda_value.append(config['lambda'][index])
                        alpha_value.append(config['alpha'][index])
                        beta_value.append(config['beta'][index])

                        task_counter.append(current_task_id)
                        pred_vector_list.append(pred_vector)
                        fisher = post_train_process_ewc(
                            train_loader, model, optimizer, current_task_id, fisher)
                        matrix = confusion_matrix(X, Y)
                        acc_db, loss_db = log_metrics(
                            metrics, time, current_task_id, acc_db, loss_db)
                        time = 0
                        trigger_times = 0
                        the_last_loss = 100
                        break

                    if loss_db[current_task_id][epoch-1] < the_last_loss:
                        the_last_loss = loss_db[current_task_id][epoch-1]

                    if epoch == args.epochs_per_task:
                        if trigger_times > 0:
                            model = get_benchmark_model(args)
                            model.load_state_dict(backup_model.state_dict())
                            model = model.to(DEVICE)
                            optimizer = type(backup_opt)(
                                model.parameters(), lr=args.lr)
                            optimizer.load_state_dict(backup_opt.state_dict())
                        else:
                            model = model.to(DEVICE)

                        # 2.1. compute accuracy and loss
                        metrics, X, Y = final_eval(
                            model, val_loader, criterion, current_task_id)

                        pred_vector = make_prediction_vector(X, Y)
                        print(forgetting_metric(pred_vector,
                                                pred_vector_list, current_task_id))
                        accuracy_results.append(metrics['accuracy'])
                        forgetting_result.append(round(forgetting_metric(
                            pred_vector, pred_vector_list, current_task_id), 2))
                        lambda_value.append(config['lambda'][index])
                        alpha_value.append(config['alpha'][index])
                        beta_value.append(config['beta'][index])						
                        task_counter.append(current_task_id)
                        pred_vector_list.append(pred_vector)
                        matrix = confusion_matrix(X, Y)
                        # plot_conf_matrix(matrix)
                        acc_db, loss_db = log_metrics(
                            metrics, time, current_task_id, acc_db, loss_db)
                        time = 0
                        trigger_times = 0
                        the_last_loss = 100

    
    data_to_csv(accuracy_results, forgetting_result,
                task_counter, lambda_value, alpha_value, beta_value)


if __name__ == "__main__":
    args = parse_arguments()
    if args.grid_search:
        print('grid search on task 2 with config')
        tuning_on_task2(args)
        
    else:
        run_experiment(args)
