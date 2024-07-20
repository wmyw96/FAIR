import numpy as np
import torch

class ClassificationSCM:
	def __init__(self, spur=0.95):
		self.spur = spur

	def sample(self, n):
		x = np.random.normal(0, 1, n)
		y = (np.random.uniform(0, 1, n) <= 1.0/(1+np.exp(-x))) * 1.0
		coin_flip = (np.random.uniform(0, 1, n) <= self.spur) * 1.0
		z = (y * coin_flip + (1 - y) * (1 - coin_flip)) * self.spur
		z = z + np.random.normal(0, 0.3, n)
		xx = np.concatenate([np.reshape(x, (n, 1)), np.reshape(z, (n, 1))], 1)
		yy = np.reshape(y, (n, 1))
		return xx, yy


np.random.seed(376)
torch.manual_seed(376)
n = 100
models = [ClassificationSCM(0.99), ClassificationSCM(0.70)]
x1, y1 = models[0].sample(100)
x2, y2 = models[1].sample(100)

e1y1x = x1[np.squeeze(y1) == 1, :]
e1y0x = x1[np.squeeze(y1) == 0, :]
e2y1x = x2[np.squeeze(y2) == 1, :]
e2y0x = x2[np.squeeze(y2) == 0, :]

import matplotlib.pyplot as plt
figs, axs = plt.subplots(1, 2, figsize=(8, 4))

axs[0].scatter(e1y1x[:, 0], e1y1x[:, 1], color='#6bb392', marker='+')
axs[0].scatter(e1y0x[:, 0], e1y0x[:, 1], color='#ec813b', marker='^')
axs[1].scatter(e2y1x[:, 0], e2y1x[:, 1], color='#6bb392', marker='+')
axs[1].scatter(e2y0x[:, 0], e2y0x[:, 1], color='#ec813b', marker='^')

plt.savefig('saved_results/sample_data.png')


from methods.fair_algo import *


class FairLinearClassification(FairModel):
	def __init__(self, num_envs, dim_x):
		super(FairLinearClassification, self).__init__(num_envs, dim_x)
		self.g = nn.Sequential(nn.Linear(dim_x, 1), nn.Sigmoid())
		self.fs = []
		for e in range(num_envs):
			fe = nn.Linear(dim_x, 1)
			self.fs.append(fe)

	def parameters_g(self, log=False):
		if log:
			print(f'FairLinearClassification Predictor Module Parameters:')
			for para in self.g.parameters():
				print(f'Parameter Shape = {para.shape}')
		return self.g.parameters()

	def parameters_f(self, log=False):
		paras = []
		for i in range(self.num_envs):
			if log:
				print(f'FairLinearClassification Discriminator ({i}) Module Parameters:')
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


def misclass(pred_y, y):
	return 1 - np.mean((pred_y >= 0.5) * y + (pred_y < 0.5) * (1 - y))


def bce_loss(out_prob, cat_y):
	return -0.5 * torch.mean((cat_y * torch.log(out_prob + 1e-9) + (1 - cat_y) * torch.log(1 - out_prob + 1e-9)))


model = FairLinearClassification(2, 2)

xs, ys = [x1, x2], [y1, y2]
x3, y3 = ClassificationSCM(0.5).sample(3000)
x4, y4 = ClassificationSCM(0.05).sample(3000)
valid = [x3], [y3]
test = [x4], [y4]

hyper_params = {
	'gumbel_lr': 1e-3, 'model_lr': 1e-3,
	'weight_decay_g': 0, 'weight_decay_f': 0,
	'niters': 30000, 'diters': 3, 'giters': 1, 'batch_size': 100,
	'init_temp': 5, 'final_temp': 0.1, 'offset': -1, 'anneal_iter': 100, 'anneal_rate': 0.993,
}

algo = FairGumbelAlgo(2, 2, model, 36, bce_loss, hyper_params)
packs = algo.run_gumbel((xs, ys), eval_metric=misclass, me_valid_data=valid, me_test_data=test, eval_iter=3000, log=True)

from utils import print_gate_during_training
print_gate_during_training(2, ([0], [1], []), packs['gate_rec'], 'saved_results/sample_gate.png')
