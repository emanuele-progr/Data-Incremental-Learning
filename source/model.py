import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d
from torch.nn.modules import padding


class MLP(nn.Module):
	"""
	Two layer MLP for MNIST benchmarks.
	"""
	def __init__(self, hiddens, config):
		super(MLP, self).__init__()
		self.W1 = nn.Linear(784, hiddens)
		self.relu = nn.ReLU(inplace=True)
		self.dropout_1 = nn.Dropout(p=config['dropout'])
		self.W2 = nn.Linear(hiddens, hiddens)
		self.dropout_2 = nn.Dropout(p=config['dropout'])
		
		self.W3 = nn.Linear(hiddens, 10)

	def forward(self, x, return_features = False):
		x = x.view(-1, 784)
		out = self.W1(x)
		out = self.relu(out)
		out = self.dropout_1(out)
		out = self.W2(out)
		out = self.relu(out)
		feat = self.dropout_2(out)

		out = self.W3(feat)
		if return_features:
			return out, feat
		else:
			return out

	def freeze_all(self):
		for param in self.parameters():
			param.requires_grad = False


def conv3x3(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1, config={}):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(in_planes, planes, stride)
		self.conv2 = conv3x3(planes, planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion * planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
						  stride=stride, bias=False),
			)
		self.IC1 = nn.Sequential(
			nn.BatchNorm2d(planes),
			nn.Dropout(p=config['dropout'])
			)

		self.IC2 = nn.Sequential(
			nn.BatchNorm2d(planes),
			nn.Dropout(p=config['dropout'])
			)

	def forward(self, x):
		out = self.conv1(x)
		out = self.IC1(out)
		out = relu(out)
		out = self.conv2(out)
		out = self.IC2(out)

		out += self.shortcut(x)
		out = relu(out)
		return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, config={}):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.IC1 = nn.Sequential(
            nn.BatchNorm2d(planes),
            nn.Dropout(p=config['dropout'])
            )

        self.IC2 = nn.Sequential(
            nn.BatchNorm2d(planes),
            nn.Dropout(p=config['dropout'])
            )
        self.IC3 = nn.Sequential(
            nn.BatchNorm2d(self.expansion*planes),
            nn.Dropout(p=config['dropout'])
            )
	

    def forward(self, x):

        out = self.conv1(x)
        out = self.IC1(out)
        out = relu(out)
        out = self.conv2(out)
        out = self.IC2(out)
        out = self.conv3(out)
        out = self.IC3(out)
        out += self.shortcut(x)
        out = relu(out)
        return out

class ResNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes, nf, config={}):
		super(ResNet, self).__init__()
		self.in_planes = nf

		self.conv1 = conv3x3(3, nf * 1)
		self.bn1 = nn.BatchNorm2d(nf * 1)
		self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1, config=config)
		self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2, config=config)
		self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2, config=config)
		self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2, config=config)
		self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

	def _make_layer(self, block, planes, num_blocks, stride, config):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride, config=config))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x, return_features = False):
		bsz = x.size(0)
		out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = avg_pool2d(out, 4)
		features = out.view(out.size(0), -1)
		out = self.linear(features)

		if return_features:
			return out, features
		else:
			return out		

	
	def freeze_all(self):
		for param in self.parameters():
			param.requires_grad = False


class ResNet2(nn.Module):
	def __init__(self, block, num_blocks, num_classes, nf, config={}):
		super(ResNet2, self).__init__()
		self.in_planes = nf

		self.conv1 = conv3x3(3, nf * 1)
		self.bn1 = nn.BatchNorm2d(nf * 1)
		self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1, config=config)
		self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2, config=config)
		self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2, config=config)
		self.linear = nn.Linear(nf * 4 * block.expansion, num_classes)

	def _make_layer(self, block, planes, num_blocks, stride, config):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride, config=config))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x, return_features=False):
		bsz = x.size(0)
		out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = avg_pool2d(out, 8, stride=1)
		features = out.view(out.size(0), -1)
		out = self.linear(features)

		if return_features:
			return out, features
		else:
			return out

	def freeze_all(self):
		for param in self.parameters():
			param.requires_grad = False

def ResNet18(nclasses=100, nf=64, config={}):
	net = ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, config=config)
	return net

def ResNet32(nclasses=100, nf=16, config={}):
    net = ResNet2(BasicBlock, [5, 5, 5], nclasses, nf, config=config)
    return net

def ResNet50(nclasses=100, nf=64, config={}):
	net = ResNet(Bottleneck, [3, 4, 6, 3], nclasses, nf, config=config)
	return net


