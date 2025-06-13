import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from data.utils import MultiEnvDataset
import torch.optim as optim 


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


def print_prob(x):
	'''
		Print vector x up to 2 decimals
	'''
	x = np.squeeze(x)
	out_str = '['
	for i in range(np.shape(x)[0]):
		if i > 0:
			out_str += ' '
		out_str += '%.2f,' % (x[i])
	out_str += ']'
	return out_str


class MixedGumbelGate(torch.nn.Module):
	def __init__(self, input_dim, varmask, init_offset=-3, device='cpu'):
		super(MixedGumbelGate, self).__init__()
		self.logits = torch.nn.Parameter((torch.rand(input_dim, device=device) - 0.5) * 1e-5 + init_offset)
		self.mask = torch.tensor(varmask, device=device).float()
		self.input_dim = input_dim
		self.device = device

	def mylogits(self):
		return self.logits * (1 - self.mask) + 100.0 * self.mask

	def generate_mask(self, temperatures, shape=None):
		if shape is None:
			shape = (1, self.input_dim)
		gumbel_softmax_sample = self.mylogits() / temperatures[0] \
							+ sample_gumbel(self.logits.shape, self.device) \
							- sample_gumbel(self.logits.shape, self.device)
		mask = torch.sigmoid(gumbel_softmax_sample / temperatures[1])
		return torch.reshape(mask, shape)

	def get_logits_numpy(self):
		return self.mylogits().detach().cpu().numpy()


class FairModel(torch.nn.Module):
	def __init__(self, num_envs, dim_x):
		super(FairModel, self).__init__()
		self.num_envs = num_envs
		self.dim_x = dim_x

	def parameters_g(self, log=False):
		raise ValueError('Should implement parameters_g() for FairModel module')

	def parameters_f(self, log=False):
		raise ValueError('Should implement parameters_f() for FairModel module')

	def forward(self, x, pred=False):
		raise ValueError('Should implement foward() for FairModel module')


class ReLUMLP(torch.nn.Module):
	def __init__(self, input_dim, depth, width, out_act=None, res_connect=True):
		super(ReLUMLP, self).__init__()

		if depth >= 1:
			self.relu_nn = [('linear1', nn.Linear(input_dim, width)), ('relu1', nn.ReLU())]
			for i in range(depth - 1):
				self.relu_nn.append(('linear{}'.format(i+2), nn.Linear(width, width)))
				self.relu_nn.append(('relu{}'.format(i+2), nn.ReLU()))

			self.relu_nn.append(('linear{}'.format(depth+1), nn.Linear(width, 1)))
			self.relu_stack = nn.Sequential(OrderedDict(self.relu_nn))

			self.res_connect = res_connect
			if self.res_connect:
				self.linear_res = torch.nn.Linear(in_features=input_dim, out_features=1, bias=False)
		else:
			self.relu_stack = torch.nn.Linear(in_features=input_dim, out_features=1, bias=True)
			self.res_connect = False

		self.out_act = out_act

	def forward(self, x):
		out = self.relu_stack(x)
		if self.res_connect:
			out = out + self.linear_res(x)
		if self.out_act is not None:
			out = self.out_act(out)
		return out


class FairMLP(FairModel):
	def __init__(self, num_envs, dim_x, depth_g, width_g, depth_f, width_f, out_act_g=None, out_act_f=None):
		super(FairMLP, self).__init__(num_envs, dim_x)

		self.g = ReLUMLP(input_dim=dim_x, depth=depth_g, width=width_g, out_act=out_act_g)
		self.num_envs = num_envs
		self.fs = []
		for e in range(num_envs):
			fe = ReLUMLP(input_dim=dim_x, depth=depth_f, width=width_f, out_act=out_act_f)
			self.fs.append(fe)

	def parameters_g(self, log=False):
		if log:
			print(f'FairMLP Predictor Module Parameters:')
			for para in self.g.parameters():
				print(f'Parameter Shape = {para.shape}')
		return self.g.parameters()

	def parameters_f(self, log=False):
		paras = []
		for i in range(self.num_envs):
			if log:
				print(f'FairMLP Discriminator ({i}) Module Parameters:')
				for para in self.fs[i].parameters():
					print(f'Parameter Shape = {para.shape}')				
			paras += self.fs[i].parameters()
		return paras

	def forward(self, xs, pred=False):
		if pred:
			return self.g(xs)
		else:
			out_g = self.g(torch.cat(xs, 0))
			out_fs = []
			for e in range(self.num_envs):
				out_fs.append(self.fs[e](xs[e]))
			out_f = torch.cat(out_fs, 0)
			return out_g, out_f


class FairGumbelAlgo(object):

	def __init__(self, num_envs, dim_x, model, gamma, loss, hyper_params):
		self.num_envs = num_envs
		self.dim_x = dim_x
		self.gamma = gamma
		self.loss = loss
		self.hyper_params = hyper_params
		self.model = model
		assert self.model.num_envs == num_envs
		assert self.model.dim_x == dim_x

	def run_gumbel(self, me_train_data, eval_metric=None, me_valid_data=None, me_test_data=None, varmask=None, save_iter=100, eval_iter=1000, gate_samples=100, device='cpu', log=False):
		# Build multi-environment training dataset that contains 'self.num_envs' number of environments
		features, responses = me_train_data		
		assert len(features) == self.num_envs
		assert len(responses) == self.num_envs

		for feature in features:
			assert isinstance(feature, np.ndarray)
			assert np.shape(feature)[1] == self.dim_x, f'feature dim in data = {np.shape(feature)[1]}, dim_x = {self.dim_x}'

		for response in responses:
			assert np.shape(response)[1] == 1
		dataset = MultiEnvDataset(features, responses)

		# Build multi-environment valid set and test set
		if eval_metric is not None:
			valid_features, valid_responses = me_valid_data
			valid_xs = [torch.tensor(valid_feature).float() for valid_feature in valid_features]
			valid_ys = [valid_y + 0.0 for valid_y in valid_responses]
			test_features, test_responses = me_test_data
			test_xs = [torch.tensor(test_feature).float() for test_feature in test_features]
			test_ys = [test_y + 0.0 for test_y in test_responses]

		# Build Gumbel gate
		if varmask is None:
			varmask = np.zeros((self.dim_x,))
		model_var = MixedGumbelGate(self.dim_x, varmask=varmask, init_offset=self.hyper_params['offset'], device=device)
		optimizer_var = optim.Adam(model_var.parameters(), lr=self.hyper_params['gumbel_lr'])

		# Build optimizzer
		optimizer_g = optim.Adam(self.model.parameters_g(log), lr=self.hyper_params['model_lr'], 
									weight_decay=self.hyper_params['weight_decay_g'])
		optimizer_f = optim.Adam(self.model.parameters_f(log), lr=self.hyper_params['model_lr'], 
									weight_decay=self.hyper_params['weight_decay_f'])


		# Preparing training
		gamma = self.gamma
		niters, giters, diters = self.hyper_params['niters'], self.hyper_params['giters'], self.hyper_params['diters']
		tau = self.hyper_params['init_temp']
		final_temp = self.hyper_params['final_temp']
		anneal_rate, anneal_iter = self.hyper_params['anneal_rate'], self.hyper_params['anneal_iter']
		batch_size = self.hyper_params['batch_size']

		loss_rec = []
		gate_rec = []

		for it in range(niters):
			# anneal the temperature
			if (it + 1) % anneal_iter == 0:
				tau = max(tau * anneal_rate, final_temp)

			# train the discriminator
			for i in range(diters):
				optimizer_var.zero_grad()
				optimizer_f.zero_grad()
				optimizer_g.zero_grad()
				self.model.train()

				xs, ys = dataset.next_batch(batch_size)
				cat_y = torch.cat(ys, 0)
				# detach the gate output because we do not wish the gradient back-propagate through gumbel module
				gate = model_var.generate_mask((1, tau)).detach()
				out_g, out_f = self.model([gate * x for x in xs])

				loss_de = - torch.mean((cat_y - out_g.detach()) * out_f - 0.5 * out_f * out_f)
				loss_de.backward()
				optimizer_f.step()

			# train the generator
			for i in range(giters):
				optimizer_g.zero_grad()
				optimizer_var.zero_grad()
				optimizer_f.zero_grad()
				self.model.train()

				xs, ys = dataset.next_batch(batch_size)
				gate = model_var.generate_mask((1, tau))
				cat_y = torch.cat(ys, 0)
				out_g, out_f = self.model([gate * x for x in xs])

				loss_r = self.loss(out_g, cat_y)
				loss_j = torch.mean((cat_y - out_g) * out_f - 0.5 * out_f * out_f)
				loss = loss_r + gamma * loss_j
				loss.backward()
				optimizer_g.step()
				optimizer_var.step()

			if it % save_iter == 0:
				with torch.no_grad():
					logits = model_var.get_logits_numpy()
					gate_rec.append(sigmoid(logits))


			if (it + 1) % eval_iter == 0:
				self.model.eval()

				if eval_metric is not None:
					valid_loss = []
					for e in range(len(valid_xs)):
						print(len(valid_xs))
						preds = []
						# generate multiple gates and predictions
						for k in range(gate_samples):
							gate = model_var.generate_mask((1, tau))
							pred = self.model(gate * valid_xs[e], pred=True)
							preds.append(pred.detach().cpu().numpy())
						out = sum(preds) / len(preds)
						valid_loss.append(eval_metric(out, valid_ys[e]))

					test_loss = []
					gates = []
					# calculate test loss
					for e in range(len(test_xs)):
						preds = []
						for k in range(gate_samples):
							gate = model_var.generate_mask((1, tau))
							pred = self.model(gate * test_xs[e], pred=True)
							preds.append(pred.detach().cpu().numpy())
							gates.append((gate).detach().cpu().numpy() + 0.0)
						out = sum(preds) / len(preds)
						test_loss.append(eval_metric(out, test_ys[e]))

					loss_rec.append(valid_loss + test_loss)
				else:
					valid_loss, test_loss = [0.0], [0.0]
					loss_rec.append(valid_loss + test_loss)
				if log:
					print(f'iteration ({it}/{niters}), valid_loss = {valid_loss}, test_loss = {test_loss}\n' + 
							f'gate logits = {print_prob(sigmoid(logits))}\n' + 
							f'gate est = {print_prob(sum(gates) / len(gates))}')

		ret = {'gate_rec': np.array(gate_rec),
				'loss_rec': np.array(loss_rec),
				'model': self.model}
		return ret
