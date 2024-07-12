import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict


def sample_gumbel(shape, device, eps=1e-20):
	'''
		The function to sample gumbel random variables

		Parameters
		----------
		shape : tuple/np.array/torch.tensor
			a int tuple characterizing the shape of the tensor, usually is (p,)
			
	'''
	U = torch.rand(shape, device=device)
	return -torch.log(-torch.log(U + eps) + eps)


def sigmoid(x):
	'''
		The function to apply sigmoid activation to each entry of a numpy array

		Parameters
		----------
		x : np.array
			the numpy array
			
	'''
	return 1/(1+np.exp(-np.minimum(np.maximum(x, -30), 30)))


class GumbelGate(torch.nn.Module):
	'''
		A class to implement the gumbel gate for the input (with dimension p)

		...
		Attributes
		----------
		logits : nn.Parameter
			the parameter characeterizing log(pi_i), where pi_i is the probability that the i-th gate is set to be 1.

		Methods
		----------
		__init__()
			Initialize the module

		generate_mask(temperature, shape)
			Implementation to generate a gumbel gate mask

	'''
	def __init__(self, input_dim, init_offset=-3, device='cpu'):
		super(GumbelGate, self).__init__()
		self.logits = torch.nn.Parameter((torch.rand(input_dim, device=device) - 0.5) * 1e-5 + init_offset)
		self.input_dim = input_dim
		self.device = device

	def generate_mask(self, temperatures, shape=None):
		if shape is None:
			shape = (1, self.input_dim)
		gumbel_softmax_sample = self.logits / temperatures[0] \
							+ sample_gumbel(self.logits.shape, self.device) \
							- sample_gumbel(self.logits.shape, self.device)
		mask = torch.sigmoid(gumbel_softmax_sample / temperatures[1])
		return torch.reshape(mask, shape)

	def get_logits_numpy(self):
		return self.logits.detach().cpu().numpy()


class FixedGate(torch.nn.Module):
	def __init__(self, input_dim, mask, device):
		self.mask = torch.tensor(mask, device=device).float()
		self.logits = self.mask * 100 - (1 - self.mask) * 100
		self.input_dim = input_dim
		self.device = device

	def generate_mask(self, temperatures=0, shape=None):
		if shape is None:
			shape = (1, self.input_dim)
		return torch.reshape(self.mask, shape)

	def get_logits_numpy(self):
		return self.logits.detach().cpu().numpy()


class NNModule(torch.nn.Module):
	'''
		A class to implemented fully connected neural network

		...
		Attributes
		----------
		relu_stack: nn.module
			the relu neural network module

		Methods
		----------
		__init__()
			Initialize the module
		forward(x)
			Implementation of forwards pass
	'''
	def __init__(self, input_dim, depth, width, add_bn=False, out_act=None, res_connect=True):
		'''
			Parameters
			----------
			input_dim : int
				input dimension
			depth : int
				the number of hidden layers of neural network, depth = 0  ===>  linear model
			width : int
				the number of hidden units in each layer
			res_connect : bool
				whether or not to use residual connection
		'''
		super(NNModule, self).__init__()

		if depth >= 1:
			if add_bn:
				self.relu_nn = [('linear1', nn.Linear(input_dim, width)), ('batch_norm1', nn.BatchNorm1d(width)), ('relu1', nn.ReLU())]
			else:
				self.relu_nn = [('linear1', nn.Linear(input_dim, width)), ('relu1', nn.ReLU())]
			for i in range(depth - 1):
				self.relu_nn.append(('linear{}'.format(i+2), nn.Linear(width, width)))
				if add_bn:
					self.relu_nn.append(('batch_norm{}'.format(i + 2), nn.BatchNorm1d(width)))
				self.relu_nn.append(('relu{}'.format(i+2), nn.ReLU()))

			self.relu_nn.append(('linear{}'.format(depth+1), nn.Linear(width, 1)))
			self.relu_stack = nn.Sequential(
				OrderedDict(self.relu_nn)
			)

			self.res_connect = res_connect
			if self.res_connect:
				self.linear_res = torch.nn.Linear(in_features=input_dim, out_features=1, bias=False)
		else:
			self.relu_stack = torch.nn.Linear(in_features=input_dim, out_features=1, bias=True)
			self.res_connect = False

		self.x_mean = torch.tensor(np.zeros((1, input_dim))).float()
		self.x_std = torch.tensor(np.ones((1, input_dim))).float()

		self.out_act = out_act

	def standardize(self, train_x):
		self.x_mean = torch.tensor(np.mean(train_x, 0, keepdims=True)).float()
		#print(self.x_mean)
		self.x_std = torch.tensor(np.std(train_x, 0, keepdims=True)).float()
		#print(self.x_std)

	def forward(self, x):
		'''
			Parameters
			----------
			x : torch.tensor
				(n, p) matrix of the input

			Returns
			----------
			out : torch.tensor
				(n, 1) matrix of the prediction
		'''
		x = (x - self.x_mean) / self.x_std
		out = self.relu_stack(x)
		if self.res_connect:
			out = out + self.linear_res(x)
		if self.out_act is not None:
			out = self.out_act(out)
		return out


class FairLinear(torch.nn.Module):
	'''
		A class to implement the linear model

		...
		Attributes
		----------
		linear : nn.module
			the linear model

		Methods
		----------
		__init__()
			Initialize the module

		forward(x, is_traininig=False)
			Implementation of forwards pass

	'''
	def __init__(self, input_dim, num_envs):
		'''
			Parameters
			----------
			input_dim : int
				input dimension
		'''
		super(FairLinear, self).__init__()
		self.g = torch.nn.Linear(in_features=input_dim, out_features=1, bias=True)
		self.num_envs = num_envs
		self.fs = []
		for e in range(num_envs):
			self.fs.append(torch.nn.Linear(in_features=input_dim, out_features=1, bias=True))

	def params_g(self):
		return self.g.parameters()

	def params_f(self):
		paras = []
		for i in range(self.num_envs):
			paras += self.fs[i].parameters()
		return paras

	def forward(self, xs):
		'''
			Parameters
			----------
			x : torch.tensor
				potential (batch_size, p) torch tensor

			Returns
			----------
			y : torch.tensor
				potential (batch_size, 1) torch tensor

		'''
		out_gs, out_fs = [], []
		for e in range(self.num_envs):
			out_gs.append(self.g(xs[e]))
			out_fs.append(self.fs[e](xs[e]))
		return out_gs, out_fs


class FairNN(torch.nn.Module):
	'''
		A class to implement the linear model

		...
		Attributes
		----------
		linear : nn.module
			the linear model

		Methods
		----------
		__init__()
			Initialize the module

		forward(x, is_traininig=False)
			Implementation of forwards pass

	'''
	def __init__(self, input_dim, depth_g, width_g, depth_f, width_f, num_envs, xs, add_bn=False, standardize=False):
		'''
			Parameters
			----------
			input_dim : int
				input dimension
		'''
		super(FairNN, self).__init__()
		self.g = NNModule(input_dim=input_dim, depth=depth_g, width=width_g, add_bn=add_bn)
		#if standardize:
		#	self.g.standardize(np.concatenate(xs, 0))
		self.num_envs = num_envs
		self.fs = []
		for e in range(num_envs):
			fe = NNModule(input_dim=input_dim, depth=depth_f, width=width_f, add_bn=add_bn)
			#if standardize:
			#	fe.standardize(xs[e])
			self.fs.append(fe)


	def params_g(self, log=False):
		if log:
			print(f'FairNN Predictor Module Parameters:')
			for para in self.g.parameters():
				print(f'Parameter Shape = {para.shape}')
		return self.g.parameters()

	def params_f(self, log=False):
		paras = []
		for i in range(self.num_envs):
			if log:
				print(f'FairNN Discriminator ({i}) Module Parameters:')
				for para in self.fs[i].parameters():
					print(f'Parameter Shape = {para.shape}')				
			paras += self.fs[i].parameters()
		return paras

	def forward(self, xs, pred=False):
		'''
			Parameters
			----------
			x : torch.tensor
				potential (batch_size, p) torch tensor

			Returns
			----------
			y : torch.tensor
				potential (batch_size, 1) torch tensor

		'''
		if pred:
			return self.g(xs)
		else:
			out_g = self.g(torch.cat(xs, 0))
			out_fs = []
			for e in range(self.num_envs):
				out_fs.append(self.fs[e](xs[e]))
			out_f = torch.cat(out_fs, 0)
			return out_g, out_f

