import torch
from utils import DEVICE


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

def train_single_epoch_lwf(net, optimizer, loader, criterion, old_model, task_id=None):
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
	loss_penalty2 = 0
	mask = 0
	
	net.train()
	#old_model = copy.deepcopy(net)
	#old_model.eval()
	#old_model.freeze_all()

	for batch_idx, (data, target) in enumerate(loader):

		#print(list(old_model.parameters())[0])

		data = data.to(DEVICE)
		target = target.to(DEVICE)
		optimizer.zero_grad()
		if task_id:
			pred, feat = net(data, task_id, return_features=True)
		else:
			pred, feat = net(data, return_features=True)
		if task_id > 1 :
			pred_old, feat_old = old_model(data, task_id, return_features=True)
			#prediction_old = pred_old.data.max(1, keepdim=True)[1]
			#print(pred_old.eq(target.data.view_as(pred_old))* pred)
			#print(target * pred)
			#mask = pred_old.eq(target.data.view_as(pred_old)).long()
			loss_penalty = knowledge_distillation_penalty(pred, pred_old)
		#loss_penalty2 = feature_distillation_penalty(feat, feat_old)
			
			
			#loss_penalty = criterion(pred, pred_old)
		loss = criterion(pred, target) + loss_penalty
		print(loss_penalty)
		
		loss.backward()
		#torch.nn.utils.clip_grad_norm_(net.parameters(), 10000)
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
	loss_penalty = torch.tensor(0)
	fd_loss = 0
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
			fd_loss += loss_penalty.item() * loader.batch_size
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
	test_loss /= len(loader.dataset)
	fd_loss /= len(loader.dataset)
	correct = correct.to('cpu')
	avg_acc = 100.0 * float(correct.numpy()) / len(loader.dataset)
	return {'accuracy': avg_acc, 'loss': test_loss, 'fd_loss': fd_loss}

def eval_single_epoch_lwf(net, loader, criterion, old_model, task_id=None):
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
	loss_penalty = torch.tensor(0)
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
				loss_penalty = knowledge_distillation_penalty(output, pred_old)

			test_loss += (criterion(output, target).item() + loss_penalty.item()) * loader.batch_size
			lwf_loss += loss_penalty.item() * loader.batch_size
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
	test_loss /= len(loader.dataset)
	lwf_loss /= len(loader.dataset)
	correct = correct.to('cpu')
	avg_acc = 100.0 * float(correct.numpy()) / len(loader.dataset)
	return {'accuracy': avg_acc, 'loss': test_loss, 'lwf_loss': lwf_loss}

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

def knowledge_distillation_penalty(outputs, outputs_old):

	lamb = 1
	T = 2
	loss = lamb * cross_entropy(outputs, outputs_old, exp = 1.0 / T)

	return loss

def icarl_penalty(out, out_old):

	lamb = 1
	g = torch.sigmoid(torch.cat(out))
	q_i = torch.sigmoid(torch.cat(out_old))
	loss = lamb * torch.nn.functional.binary_cross_entropy(g, q_i)

	return loss


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

def cross_entropy(outputs, targets, exp=1.0, size_average=True, eps=1e-5):
	"""Calculates cross-entropy with temperature scaling"""
	out = torch.nn.functional.softmax(outputs, dim=1)
	tar = torch.nn.functional.softmax(targets, dim=1)
	if exp != 1:
		out = out.pow(exp)
		out = out / out.sum(1).view(-1, 1).expand_as(out)
		tar = tar.pow(exp)
		tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
	out = out + eps / out.size(1)
	out = out / out.sum(1).view(-1, 1).expand_as(out)
	ce = -(tar * out.log()).sum(1)
	
	
	if size_average:
		ce = ce.mean()
	return ce