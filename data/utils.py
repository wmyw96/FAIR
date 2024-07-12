import torch


class MultiEnvDataset:
	def __init__(self, xs, ys):
		self.num_envs = len(xs)
		self.xs = []
		self.ys = []
		self.n = []
		self.pointer = []
		for e in range(self.num_envs):
			self.xs.append(torch.tensor(xs[e]).float())
			self.ys.append(torch.tensor(ys[e]).float())
			self.n.append(xs[e].shape[0])
			self.pointer.append(0)

	def next_batch(self, batch_size):
		xs, ys = [], []
		for e in range(self.num_envs):
			if self.pointer[e] + batch_size > self.n[e]:
				idx = torch.randperm(self.xs[e].shape[0])
				self.xs[e] = self.xs[e][idx]
				self.ys[e] = self.ys[e][idx]
				self.pointer[e] = 0
			l = self.pointer[e]
			r = min(self.n[e], l + batch_size)
			self.pointer[e] += batch_size
			#print(f'batch index: {l} - {r}')
			xs.append(self.xs[e][l:r])
			ys.append(self.ys[e][l:r])
		return xs, ys

