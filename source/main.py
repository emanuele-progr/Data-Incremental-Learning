import os
import json
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


def get_benchmark_model(args):
    """
    Return the corresponding PyTorch model for experiment
    :param args:
    :return:
    """
    global TRAIN_CLASSES

    if args.net == 'resnet32':
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
            organize_validation_data_tiny_ImageNet()
            return ResNet32(nclasses=200, config={'dropout': args.dropout}).to(DEVICE)
        else:
            raise Exception("Unknown dataset.\n" +
                            "The code supports 'mnist, cifar10, cifar100 and imagenet.")
    elif args.net == 'resnet18':
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
            organize_validation_data_tiny_ImageNet()
            return ResNet18(nclasses=200, config={'dropout': args.dropout}).to(DEVICE)
        else:
            raise Exception("Unknown dataset.\n" +
                            "The code supports 'mnist, cifar10, cifar100 and imagenet.")
    
    elif args.net == 'resnet50':
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
            organize_validation_data_tiny_ImageNet()
            return ResNet50(nclasses=200, config={'dropout': args.dropout}).to(DEVICE)
        else:
            raise Exception("Unknown dataset.\n" +
                            "The code supports 'mnist, cifar10, cifar100 and imagenet.")

    else:
        raise Exception("Unknown resnet.\n" +
                            "The code supports 'resnet32 (32), resnet18 (18) and resnet50 (50).") 



def run_experiment(args):

    
    acc_db, loss_db = init_experiment(args)
    tasks = get_benchmark_data_loader(args)(args.tasks, args.batch_size)
    model = get_benchmark_model(args)

    # display model parameters
    count_parameters(model)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    time = 0
    the_last_loss = 100
    patience = 30
    trigger_times = 0
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

    for current_task_id in range(1, args.tasks+1):

        lr = max(args.lr * args.gamma ** (current_task_id), 0.00005)
        optimizer = torch.optim.Adam(model.parameters(), lr)

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

        exemplars_per_class = args.exemplars_per_class

        if current_task_id > 1:
            accumulator = train_loader.dataset
            for exemplars in exemplars_vector_list:
                accumulator += selected_exemplar

            train_loader = torch.utils.data.DataLoader(
                accumulator, batch_size=args.batch_size, shuffle=True)
        print("training data in this task: {}".format(len(train_loader.dataset)))

        exemplar_loader = tasks[current_task_id]['exemplar']
        for epoch in range(1, args.epochs_per_task+1):

            prev_model = get_benchmark_model(args)
            prev_model.load_state_dict(model.state_dict())
            prev_opt = type(optimizer)(prev_model.parameters(), lr=lr)
            prev_opt.load_state_dict(optimizer.state_dict())

            train_single_epoch_approach(args.approach, model, optimizer, train_loader, criterion,
                                        old_params, old_model, fisher, current_task_id)

            time += 1
            model = model.to(DEVICE)
            val_loader = tasks[current_task_id]['val']

            metrics = eval_single_epoch_approach(args.approach, model, val_loader, criterion, old_model,
                                       old_params, fisher, exemplar_means, current_task_id)


            acc_db, loss_db = log_metrics(
                metrics, time, current_task_id, acc_db, loss_db)

            if loss_db[current_task_id][epoch-1] > the_last_loss:
                if trigger_times == 0:
                    backup_model = get_benchmark_model(args)
                    backup_model.load_state_dict(prev_model.state_dict())
                    backup_opt = type(prev_opt)(
                        backup_model.parameters(), lr=lr)
                    backup_opt.load_state_dict(prev_opt.state_dict())
                trigger_times += 1
                print('trigger times:', trigger_times)
            else:
                trigger_times = 0
            if trigger_times >= patience:
                print('Early stopping!')
                model = backup_model.to(DEVICE)
                optimizer = type(backup_opt)(model.parameters(), lr=lr)
                optimizer.load_state_dict(backup_opt.state_dict())
                test_loader = tasks[current_task_id]['test']
                
                if args.approach == 'icarl':
                    res = herdingExemplarsSelector(model, exemplar_loader, exemplars_per_class)
                else:
                    res = randomExemplarsSelector(exemplar_loader, exemplars_per_class, TRAIN_CLASSES)
                exemplars_vector_list.append(res)
                for exemplars in exemplars_vector_list:
                    num = int((exemplars_per_class * TRAIN_CLASSES)/(current_task_id))
                    selected_exemplar = torch.utils.data.Subset(exemplar_loader.dataset, exemplars[:num])
                    if current_task_id == 1:
                        exemplar_dataset = selected_exemplar
                    else:
                        exemplar_dataset += selected_exemplar
               
                if args.approach == 'icarl':
                    exemplar_means = compute_mean_of_exemplars(model,torch.utils.data.DataLoader(exemplar_dataset, batch_size=args.batch_size))
                    metrics, X, Y = final_eval_iCarl(model, test_loader, criterion, exemplar_means, current_task_id)
                else:
                    metrics, X, Y = final_eval(model, test_loader, criterion, current_task_id)
                pred_vector = make_prediction_vector(X, Y)
                if current_task_id > 1:
                    print('forgetting : {}'.format(forgetting_metric(pred_vector,
                      pred_vector_list, current_task_id)))
                accuracy_results.append(metrics['accuracy'])
                forgetting_result.append(forgetting_metric(
                    pred_vector, pred_vector_list, current_task_id))
                task_counter.append(current_task_id)
                pred_vector_list.append(pred_vector)
                if args.approach == 'ewc':
                    fisher = post_train_process_ewc(
                    train_loader, model, optimizer, current_task_id, fisher)
                old_model = post_train_process_freeze_model(model)

                #matrix = confusion_matrix(X, Y)
                #plot_conf_matrix(matrix)
                acc_db, loss_db = log_metrics(
                    metrics, time, current_task_id, acc_db, loss_db)
                #save_checkpoint_Adam(backup_model, backup_opt)
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
                        model.parameters(), lr=lr)
                    optimizer.load_state_dict(backup_opt.state_dict())
                else:
                    model = model.to(DEVICE)
                test_loader = tasks[current_task_id]['test']

                if args.approach == 'icarl':
                    res = herdingExemplarsSelector(model, exemplar_loader, exemplars_per_class)
                else:
                    res = randomExemplarsSelector(exemplar_loader, exemplars_per_class, TRAIN_CLASSES)
                exemplars_vector_list.append(res)
                for exemplars in exemplars_vector_list:
                    num = int((exemplars_per_class * TRAIN_CLASSES)/(current_task_id))
                    selected_exemplar = torch.utils.data.Subset(exemplar_loader.dataset, exemplars[:num])
                    if current_task_id == 1:
                        exemplar_dataset = selected_exemplar
                    else:
                        exemplar_dataset += selected_exemplar
                if args.approach == 'icarl':
                    exemplar_means = compute_mean_of_exemplars(model,torch.utils.data.DataLoader(exemplar_dataset, batch_size=args.batch_size))
                    metrics, X, Y = final_eval_iCarl(model, test_loader, criterion, exemplar_means, current_task_id)
                else:
                    metrics, X, Y = final_eval(model, test_loader, criterion, current_task_id)

                pred_vector = make_prediction_vector(X, Y)
                if current_task_id > 1:
                    print('forgetting : {}'.format(forgetting_metric(pred_vector,
                      pred_vector_list, current_task_id)))
                accuracy_results.append(metrics['accuracy'])
                forgetting_result.append(forgetting_metric(
                    pred_vector, pred_vector_list, current_task_id))
                task_counter.append(current_task_id)
                pred_vector_list.append(pred_vector)
                if args.approach == 'ewc':
                    fisher = post_train_process_ewc(
                    train_loader, model, optimizer, current_task_id, fisher)
                old_model = post_train_process_freeze_model(model)

                #matrix = confusion_matrix(X, Y)
                #plot_conf_matrix(matrix)
                acc_db, loss_db = log_metrics(
                    metrics, time, current_task_id, acc_db, loss_db)
                #save_checkpoint_Adam(model, optimizer)
                time = 0
                trigger_times = 0
                the_last_loss = 100

    # get_PCA_components(mean_vector)
    data_to_csv(accuracy_results, forgetting_result, task_counter)
    return



def tuning_on_task2(args):

    with open('grid_search_config.txt') as f:
        data = f.read()
    config = json.loads(data)
    acc_db, loss_db = init_experiment(args)
    tasks = get_benchmark_data_loader(args)(args.tasks, args.batch_size)
    model = get_benchmark_model(args)

    # criterion
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    time = 0
    the_last_loss = 100
    patience = 30
    trigger_times = 0
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

    fisher = {n: torch.zeros(p.shape).to(DEVICE)
              for n, p in model.named_parameters() if p.requires_grad}


    for current_task_id in range(1, 3):
        lr = max(args.lr * args.gamma ** (current_task_id), 0.00005)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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


        exemplars_per_class = args.exemplars_per_class

        if current_task_id == 2:
            accumulator = train_loader.dataset
            for exemplars in exemplars_vector_list:
                accumulator += selected_exemplar
            train_loader = torch.utils.data.DataLoader(
                accumulator, batch_size=args.batch_size, shuffle=True)
        print("training data in this task: {}".format(len(train_loader.dataset)))
            #exemplar_means = compute_mean_of_exemplars(model,torch.utils.data.DataLoader(exemplar_dataset, batch_size=args.batch_size) , current_task_id)

        exemplar_loader = tasks[current_task_id]['exemplar']
        if current_task_id == 1:
			# first task, standard training
            for epoch in range(1, args.epochs_per_task+1):
                # 1. train and save

                prev_model = get_benchmark_model(args)
                prev_model.load_state_dict(model.state_dict())
                prev_opt = type(optimizer)(prev_model.parameters(), lr=lr)
                prev_opt.load_state_dict(optimizer.state_dict())

                train_single_epoch_approach(args.approach, model, optimizer, train_loader, criterion,
                                            old_params, old_model, fisher, current_task_id)
                time += 1
                model = model.to(DEVICE)
                val_loader = tasks[current_task_id]['val']

                metrics = eval_single_epoch_approach(args.approach, model, val_loader, criterion, old_model,
                                        old_params, fisher, exemplar_means, current_task_id)

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
                        model.parameters(), lr=lr)
                    optimizer.load_state_dict(backup_opt.state_dict())
                    if args.approach == 'icarl':
                        res = herdingExemplarsSelector(model, exemplar_loader, exemplars_per_class)
                    else:
                        res = randomExemplarsSelector(exemplar_loader, exemplars_per_class, TRAIN_CLASSES)
                    exemplars_vector_list.append(res)
                    for exemplars in exemplars_vector_list:
                        num = int((exemplars_per_class * TRAIN_CLASSES)/(current_task_id))
                        selected_exemplar = torch.utils.data.Subset(exemplar_loader.dataset, exemplars[:num])
                        exemplar_dataset = selected_exemplar

                    # 2.1. compute accuracy and loss
                    if args.approach == 'icarl':
                        exemplar_means = compute_mean_of_exemplars(model,torch.utils.data.DataLoader(exemplar_dataset, batch_size=args.batch_size))
                        metrics, X, Y = final_eval_iCarl(model, val_loader, criterion, exemplar_means, current_task_id)
                    else:
                        metrics, X, Y = final_eval(model, val_loader, criterion, current_task_id)

                    pred_vector = make_prediction_vector(X, Y)
                    print(forgetting_metric(pred_vector,
                                            pred_vector_list, current_task_id))
                    pred_vector_list.append(pred_vector)
                    if args.approach == 'ewc':
                        fisher = post_train_process_ewc(
                        train_loader, model, optimizer, current_task_id, fisher)
                    old_model = post_train_process_freeze_model(model)
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
                            model.parameters(), lr=lr)
                        optimizer.load_state_dict(backup_opt.state_dict())
                    else:
                        model = model.to(DEVICE)

                    if args.approach == 'icarl':
                        res = herdingExemplarsSelector(model, exemplar_loader, exemplars_per_class)
                    else:
                        res = randomExemplarsSelector(exemplar_loader, exemplars_per_class, TRAIN_CLASSES)
                    exemplars_vector_list.append(res)
                    for exemplars in exemplars_vector_list:
                        num = int((exemplars_per_class * TRAIN_CLASSES)/(current_task_id))
                        selected_exemplar = torch.utils.data.Subset(exemplar_loader.dataset, exemplars[:num])
                        exemplar_dataset = selected_exemplar

                    if args.approach == 'icarl':
                        exemplar_means = compute_mean_of_exemplars(model,torch.utils.data.DataLoader(exemplar_dataset, batch_size=args.batch_size))
                        metrics, X, Y = final_eval_iCarl(model, val_loader, criterion, exemplar_means, current_task_id)
                    else:
                        metrics, X, Y = final_eval(model, val_loader, criterion, current_task_id)                            
                    # 2.1. compute accuracy and loss

                    pred_vector = make_prediction_vector(X, Y)
                    print(forgetting_metric(pred_vector,
                                            pred_vector_list, current_task_id))
                    pred_vector_list.append(pred_vector)
                    if args.approach == 'ewc':
                        fisher = post_train_process_ewc(
                        train_loader, model, optimizer, current_task_id, fisher)
                    old_model = post_train_process_freeze_model(model)
                    matrix = confusion_matrix(X, Y)
                    # plot_conf_matrix(matrix)
                    acc_db, loss_db = log_metrics(
                        metrics, time, current_task_id, acc_db, loss_db)
					# save model checkpoint on task 1
                    save_checkpoint_Adam(model, optimizer)
                    time = 0
                    trigger_times = 0
                    the_last_loss = 100

        if current_task_id > 1:
			# parameters trials on task 2
        
            for index in range(len(config['lambda'])):
                exemplar_dataset_t2 = exemplar_dataset
                exemplars_vector_list_t2 = exemplars_vector_list
                print('trial n.{}/{}'.format(index+1, len(config['lambda'])))
                print("lambda : {} , alpha : {}, beta : {}".format(config['lambda'][index], config['alpha'][index], config['beta'][index]))
                model, optimizer = load_checkpoint(
                    model, optimizer, 'check.pth')
                for epoch in range(1, args.epochs_per_task+1):

                    prev_model = get_benchmark_model(args)
                    prev_model.load_state_dict(model.state_dict())
                    prev_opt = type(optimizer)(
                        prev_model.parameters(), lr=lr)
                    prev_opt.load_state_dict(optimizer.state_dict())

                    train_single_epoch_approach(args.approach, model, optimizer, train_loader, criterion,
                                                old_params, old_model, fisher, current_task_id, config, index)
                    time += 1
                    model = model.to(DEVICE)
                    val_loader = tasks[current_task_id]['val']
                    metrics = eval_single_epoch_approach(args.approach, model, val_loader, criterion, old_model,
                                            old_params, fisher, exemplar_means, current_task_id, config, index)

                    acc_db, loss_db = log_metrics(
                        metrics, time, current_task_id, acc_db, loss_db)

                    if loss_db[current_task_id][epoch-1] > the_last_loss:
                        if trigger_times == 0:
                            backup_model = get_benchmark_model(args)
                            backup_model.load_state_dict(
                                prev_model.state_dict())
                            backup_opt = type(prev_opt)(
                                backup_model.parameters(), lr=lr)
                            backup_opt.load_state_dict(prev_opt.state_dict())
                        trigger_times += 1
                        print('trigger times:', trigger_times)
                    else:
                        trigger_times = 0
                    if trigger_times >= patience:
                        print('Early stopping!')
                        model = backup_model.to(DEVICE)
                        optimizer = type(backup_opt)(
                            model.parameters(), lr=lr)
                        optimizer.load_state_dict(backup_opt.state_dict())
                        if args.approach == 'icarl':
                            res = herdingExemplarsSelector(model, exemplar_loader, exemplars_per_class)
                        else:
                            res = randomExemplarsSelector(exemplar_loader, exemplars_per_class, TRAIN_CLASSES)
                        exemplars_vector_list_t2.append(res)
                        for exemplars in exemplars_vector_list_t2:
                            num = int((exemplars_per_class * TRAIN_CLASSES)/(current_task_id))
                            selected_exemplar = torch.utils.data.Subset(exemplar_loader.dataset, exemplars[:num])
                            exemplar_dataset_t2 += selected_exemplar
                        if args.approach == 'icarl':
                            exemplar_means = compute_mean_of_exemplars(model,torch.utils.data.DataLoader(exemplar_dataset_t2, batch_size=args.batch_size))
                            metrics, X, Y = final_eval_iCarl(model, val_loader, criterion, exemplar_means, current_task_id)
                        else:
                            metrics, X, Y = final_eval(model, val_loader, criterion, current_task_id)  
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
                                model.parameters(), lr=lr)
                            optimizer.load_state_dict(backup_opt.state_dict())
                        else:
                            model = model.to(DEVICE)
                        if args.approach == 'icarl':
                            res = herdingExemplarsSelector(model, exemplar_loader, exemplars_per_class)
                        else:
                            res = randomExemplarsSelector(exemplar_loader, exemplars_per_class, TRAIN_CLASSES)
                        exemplars_vector_list_t2.append(res)
                        for exemplars in exemplars_vector_list_t2:
                            num = int((exemplars_per_class * TRAIN_CLASSES)/(current_task_id))
                            selected_exemplar = torch.utils.data.Subset(exemplar_loader.dataset, exemplars[:num])
                            exemplar_dataset_t2 += selected_exemplar
                        # 2.1. compute accuracy and loss
                        if args.approach == 'icarl':
                            exemplar_means = compute_mean_of_exemplars(model,torch.utils.data.DataLoader(exemplar_dataset_t2, batch_size=args.batch_size))
                            metrics, X, Y = final_eval_iCarl(model, val_loader, criterion, exemplar_means, current_task_id)
                        else:
                            metrics, X, Y = final_eval(model, val_loader, criterion, current_task_id) 

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
