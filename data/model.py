import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def generate_exogeneous_variables(n, p, cov_sqrt, dist):
	u = np.zeros((n, p))
	if dist == 'normal':
		u = np.random.normal(0, 1, (n, p))
		u = np.matmul(u, cov_sqrt)
	elif dist == 'uniform':
		u = np.random.uniform(-1, 1, (n, p))
		u = np.matmul(u, cov_sqrt)
	else:
		raise ValueError(f"Error: exogenous variable distribution {dist} not defined.")
	return u


class StructuralCausalModel1:
	def __init__(self, p, exogenous_cov=None, exogenous_dist='normal'):
		self.p = p
		if exogenous_cov is None:
			exogenous_cov = np.eye(p)
		exogenous_cov = np.diag(np.diag(exogenous_cov))
		self.exogenous_cov = exogenous_cov
		self.exogenous_cov_sqrt = np.sqrt(exogenous_cov)
		self.exogenous_dist = exogenous_dist

	def sample(self, n, split=True):
		u = generate_exogeneous_variables(n, self.p, self.exogenous_cov_sqrt, self.exogenous_dist)
		x = np.copy(u)
		# x4
		x[:, 4] = u[:, 4]
		x[:, 1] = u[:, 1]
		x[:, 2] = np.sin(x[:, 4]) * 1 + u[:, 2]
		x[:, 3] = np.cos(x[:, 4]) * 1 + u[:, 3]
		x[:, 5] = np.sin(x[:, 3] + u[:, 5])
		x[:, 10] = x[:, 1] * 2.5 + x[:, 2] * 1.5 + u[:, 10]
		x[:, 0] = x[:, 1] * 3 + x[:, 2] * 2 + x[:, 3] * (-0.5) + u[:, 0]
		x[:, 6] = 0.8 * x[:, 0] * u[:, 6]
		x[:, 7] = x[:, 3] * 0.5 + x[:, 0] + u[:, 7]
		x[:, 8] = 0.5 * x[:, 7] + x[:, 0] * (-1) + x[:, 10] + u[:, 8]
		x[:, 9] = np.tanh(x[:, 7]) + 0.1 * np.cos(x[:, 8]) + u[:, 9]
		x[:, 11] = 0.4 * (x[:, 7] + x[:, 8]) * u[:, 11]
		if split:
			return x[:, 1:], x[:, :1], x[:, :1] - u[:, :1]
		else:
			return x


class StructuralCausalModel2:
	def __init__(self, p, exogenous_cov=None, exogenous_dist='normal'):
		self.p = p
		if exogenous_cov is None:
			exogenous_cov = np.eye(p)
		exogenous_cov = np.diag(np.diag(exogenous_cov))
		self.exogenous_cov = exogenous_cov
		self.exogenous_cov_sqrt = np.sqrt(exogenous_cov)
		self.exogenous_dist = exogenous_dist

	def sample(self, n, split=True):
		u = generate_exogeneous_variables(n, self.p, self.exogenous_cov_sqrt, self.exogenous_dist)
		x = np.copy(u)
		# print(f'E[x eps] = {np.mean(u[:, 0] * u[:, 1])}')

		# x4
		x[:, 4] = u[:, 4] ** 2 - 1
		x[:, 1] = u[:, 1]
		x[:, 2] = np.sin(x[:, 4]) * 1 + u[:, 2]
		x[:, 3] = np.cos(x[:, 4]) * 1 + u[:, 3]
		x[:, 5] = np.sin(x[:, 3] + u[:, 5])
		x[:, 10] = x[:, 1] * 2.5 + x[:, 2] * 1.5 + u[:, 10]
		x[:, 0] = x[:, 1] * 3 + x[:, 2] * 2 + x[:, 3] * (-0.5) + u[:, 0]
		x[:, 6] = 0.8 * x[:, 0] * u[:, 6]
		x[:, 7] = x[:, 3] * 4 + np.tanh(x[:, 0]) + u[:, 7]
		x[:, 8] = 0.5 * x[:, 7] + x[:, 0] * (-1) + x[:, 10] + u[:, 8]
		x[:, 9] = np.tanh(x[:, 7]) + 0.1 * np.cos(x[:, 8]) + u[:, 9]
		x[:, 11] = 0.4 * (x[:, 7] + x[:, 8]) * u[:, 11]
		if split:
			return x[:, 1:], x[:, :1], x[:, :1] - u[:, :1]
		else:
			return x


class StructuralCausalModelNonlinear1:
	def __init__(self, p, exogenous_cov=None, exogenous_dist='normal'):
		self.p = p
		if exogenous_cov is None:
			exogenous_cov = np.eye(p)
		exogenous_cov = np.diag(np.diag(exogenous_cov))
		self.exogenous_cov = exogenous_cov
		self.exogenous_cov_sqrt = np.sqrt(exogenous_cov)
		self.exogenous_dist = exogenous_dist

	def sample(self, n, split=True):
		u = generate_exogeneous_variables(n, self.p, self.exogenous_cov_sqrt, self.exogenous_dist)
		x = np.copy(u)
		# x4
		x[:, 4] = u[:, 4]
		x[:, 1] = u[:, 1]
		x[:, 2] = np.sin(x[:, 4]) * 1 + u[:, 2]
		x[:, 3] = np.cos(x[:, 4]) * 1 + u[:, 3]
		x[:, 5] = np.sin(x[:, 3] + u[:, 5])
		x[:, 10] = x[:, 1] * 2.5 + x[:, 2] * 1.5 + u[:, 10]
		x[:, 0] = np.cos(x[:, 1]) * 3 + np.sin(x[:, 2]) * 2 + -1.5 * np.cos(np.abs(x[:, 3])) + u[:, 0]
		x[:, 6] = 0.8 * x[:, 0] * u[:, 6]
		x[:, 7] = x[:, 3] * 0.5 + x[:, 0] + u[:, 7]
		x[:, 8] = 0.5 * x[:, 7] + x[:, 0] * (-1) + x[:, 10] + u[:, 8]
		x[:, 9] = np.tanh(x[:, 7]) + 0.1 * np.cos(x[:, 8]) + u[:, 9]
		x[:, 11] = 0.4 * (x[:, 7] + x[:, 8]) * u[:, 11]
		if split:
			return x[:, 1:], x[:, :1], x[:, :1] - u[:, :1]
		else:
			return x


class StructuralCausalModelNonlinear2:
	def __init__(self, p, exogenous_cov=None, exogenous_dist='normal'):
		self.p = p
		if exogenous_cov is None:
			exogenous_cov = np.eye(p)
		exogenous_cov = np.diag(np.diag(exogenous_cov))
		self.exogenous_cov = exogenous_cov
		self.exogenous_cov_sqrt = np.sqrt(exogenous_cov)
		self.exogenous_dist = exogenous_dist

	def sample(self, n, split=True):
		u = generate_exogeneous_variables(n, self.p, self.exogenous_cov_sqrt, self.exogenous_dist)
		x = np.copy(u)
		# print(f'E[x eps] = {np.mean(u[:, 0] * u[:, 1])}')

		# x4
		x[:, 4] = u[:, 4] ** 2 - 1
		x[:, 1] = u[:, 1]
		x[:, 2] = np.sin(x[:, 4]) * 1 + u[:, 2]
		x[:, 3] = np.cos(x[:, 4]) * 1 + u[:, 3]
		x[:, 5] = np.sin(x[:, 3] + u[:, 5])
		x[:, 10] = x[:, 1] * 2.5 + x[:, 2] * 1.5 + u[:, 10]
		x[:, 0] = np.cos(x[:, 1]) * 3 + np.sin(x[:, 2]) * 2 - 1.5 * np.cos(np.abs(x[:, 3]))  + u[:, 0]
		x[:, 6] = 0.8 * x[:, 0] * u[:, 6]
		x[:, 7] = x[:, 3] * 4 + np.tanh(x[:, 0]) + u[:, 7]
		x[:, 8] = 0.5 * x[:, 7] + x[:, 0] * (-1) + x[:, 10] + u[:, 8]
		x[:, 9] = np.tanh(x[:, 7]) + 0.1 * np.cos(x[:, 8]) + u[:, 9]
		x[:, 11] = 0.4 * (x[:, 7] + x[:, 8]) * u[:, 11]
		if split:
			return x[:, 1:], x[:, :1], x[:, :1] - u[:, :1]
		else:
			return x

class AdditiveStructuralCausalModel:
	'''
		Consider the simple SCM that the strctural assignments admits additive form, that is 

			x_j <- sum_{k in pa(j)} coeff_{j,k} f_{j,k} (x_k) + coeff_{j,j} u_j
	
	'''
	def __init__(self, num_vars, coefficients_matrix, assignments_matrix, y_index, randtype='gaussian'):
		self.num_vars = num_vars
		self.y_index = y_index
		self.coefficients = coefficients_matrix
		self.assignments = assignments_matrix
		self.randtype = randtype

	def sample(self, n, split=True):
		z = np.zeros((n, self.num_vars))
		if self.randtype == 'gaussian':
			u = np.random.normal(0, 1, (n * self.num_vars))
		else:
			# use unit variance
			u = np.random.uniform(-np.sqrt(1.5), np.sqrt(1.5), (n * self.num_vars))
		u = np.reshape(u, (n, self.num_vars))
		for i in range(self.num_vars):
			z[:, i] = u[:, i] * self.coefficients[i, i]
			for j in range(i):
				function_id = self.assignments[i, j]
				pre_factor = self.coefficients[i, j]
				if function_id == 1:
					# linear function
					z[:, i] += pre_factor * (z[:, j])
				elif function_id == 2:
					# sin function
					z[:, i] += pre_factor * np.sin(z[:, j])
				elif function_id == 3:
					z[:, i] += pre_factor * 1 / (1 + np.exp(-z[:, j]))
				elif function_id == 4:
					z[:, i] += pre_factor * np.cos(z[:, j])
				elif function_id == 5:
					z[:, i] += pre_factor * np.sin(np.pi * z[:, j])
		if split:
			x = np.concatenate([z[:, :self.y_index], z[:, (self.y_index+1):]], 1)
			y = z[:, self.y_index:(self.y_index+1)]
			y_gt = z[:, self.y_index:(self.y_index+1)] - \
					self.coefficients[self.y_index, self.y_index] * u[:, self.y_index:(self.y_index+1)]
			return (x, y, y_gt)
		else:
			return z


	def visualize(self, of_set):
		G = nx.DiGraph()

		for i in range(self.num_vars - 1):
			G.add_node(f"{i+1}")

		node_colors = ['#9acdc4'] * (self.num_vars - 1) + ['white']
		G.add_node("Y")

		for i in range(self.num_vars):
			for j in range(i):
				if i < self.y_index:
					if self.assignments[i, j] > 0:
						G.add_edge(f"{j+1}", f"{i+1}")
				elif i == self.y_index:
					if self.assignments[i, j] > 0:
						G.add_edge(f"{j+1}", "Y")
						node_colors[j] = '#05348b'
				elif i > self.y_index:
					if self.assignments[i, j] > 0:
						if j == self.y_index:
							G.add_edge("Y", f"{i}")
							node_colors[i-1] = '#ae1908'
						else:
							pre_idx = j
							if j < self.y_index:
								pre_idx += 1
							G.add_edge(f"{pre_idx}", f"{i}")

		for i in range(self.num_vars):
			idx = i
			if i > self.y_index:
				idx -= 1
			if node_colors[idx] == '#9acdc4' and i in of_set:
				node_colors[idx] = '#ec813b'

		pos = nx.spring_layout(G)  # Choose a layout algorithm
		nx.draw(G, pos=pos, with_labels=True, arrows=True,
				node_color=node_colors)
		plt.show()




def ident(x):
	return x


def tcos(x):
	return 2 * np.cos(x)


def relu(x):
	return np.maximum(0, x)


def sigmoid(x):
	return 1/(1 + np.exp(-x))

def idx_to_func(idx):
	if idx == 1:
		return np.sin
	elif idx == 2:
		return np.sin
	elif idx == 3:
		return np.tanh
	elif idx == 4:
		return relu
	elif idx == 5:
		return ident


def generate_random_weight(x, rgg):
	while True:
		a = np.random.uniform(-rgg, rgg)
		if np.abs(a - x) > 3:
			return a


class CharysanSCM:
	def __init__(self, num_parents, num_children, func_parent=None, coeff_parent=None, func_children=None, coeff_children=None, num_other=0, other_assc=None, randtype='guassian'):
		self.num_parents = num_parents
		self.num_children = num_children
		self.randtype = randtype
		if func_parent is not None:
			self.func_parent = func_parent
			self.coeff_parent = coeff_parent
		else:
			self.func_parent = []
			self.coeff_parent = []
			for i in range(num_parents):
				self.func_parent.append(np.random.randint(5) + 1)
			for i in range(num_parents):
				self.coeff_parent.append(np.random.uniform(-1.5, 1.5))

		self.func_children = []
		self.coeff_children = []
		self.noise_level = []
		offspring = [i + num_parents for i in range(num_children)]
		if coeff_children is None:
			for i in range(num_children):
				fid = np.random.randint(2) * 2 + 3
				self.func_children.append([fid, fid])
				self.coeff_children.append([np.random.uniform(-1.5, 1.5), np.random.uniform(-1.5, 1.5)])
				self.noise_level.append(np.random.uniform(1, 1.5))
		else:
			for i in range(num_children):
				fid = 8 - func_children[i][0]
				self.func_children.append([fid, fid])
				self.coeff_children.append([generate_random_weight(coeff_children[i][0], 5), generate_random_weight(coeff_children[i][1], 5)])
				self.noise_level.append(np.random.uniform(1, 1.5))

		self.num_other = num_other
		self.other_assc = []
		for i in range(num_other):
			nvar = np.random.randint(5)
			comp = []
			#mystr = 'var %d: ' % (i + num_parents + num_children)
			if other_assc is None:
				for k in range(nvar):
					idx = np.random.randint(num_parents + num_children)
					#mystr += ' %d ' % (idx)
					coeff = np.random.uniform(-1, 1)
					func = np.random.randint(3) + 1
					if idx >= num_parents:
						offspring.append(i + num_children + num_parents)
					comp.append((idx, func, coeff))
				noise_level = np.random.uniform(2, 2.5)
				self.other_assc.append((nvar, comp, noise_level))

				#print(mystr)
			else:
				nvar, comp0, noise_level0 = other_assc[i]
				for k in range(nvar):
					idx, func, coeff = comp0[k]
					coeff = np.random.uniform(-1, 1)
					func = np.random.randint(3) + 1
					comp.append((idx, func, coeff))
				noise_level = np.random.uniform(2.5, 3)
				self.other_assc.append((nvar, comp, noise_level))

		self.offspring = list(set(offspring))
		self.hcm = 0


	def sample(self, n, split=True):
		num_vars = self.num_parents + self.num_children + self.num_other + 1
		z = np.zeros((n, num_vars))
		u1 = np.reshape(np.random.uniform(-1.5, 1.5, (n * (num_vars - 1))), (n, (num_vars - 1)))
		u2 = np.reshape(np.random.normal(0, 1, (n * 1)), (n, 1))
		u = np.concatenate([u1, u2], 1)

		for i in range(self.num_parents):
			z[:, i] = u[:, i]
			func = idx_to_func(self.func_parent[i])
			z[:, num_vars - 1] += func(u[:, i])

		if self.hcm == 1:
			z[:, num_vars - 1] = z[:, 0] * (z[:, 1] ** 3) + 2 * np.log(1 + np.exp(np.tanh(z[:, 2])) + np.exp(z[:, 3])) + np.sin(z[:, 4])
			print('Use HCM mode 1')

		z[:, num_vars - 1] += u[:, num_vars - 1]

		for i in range(self.num_children):
			z[:, i + self.num_parents] = u[:, i + self.num_parents] * self.noise_level[i]
			func = idx_to_func(self.func_children[i][0])
			func2 = idx_to_func(self.func_children[i][1])
			z[:, i + self.num_parents] += self.coeff_children[i][0] * np.tanh(z[:, num_vars - 1]) #+ self.coeff_children[i][1] * np.tanh(u[:, num_vars - 1])

		base = self.num_parents + self.num_children
		for i in range(self.num_other):
			#print('other variable')
			nvar, comp, noise_level = self.other_assc[i]
			z[:, i + base] = u[:, i + base] * noise_level
			for j in range(nvar):
				idx, func_id, coeff = comp[j]
				func = idx_to_func(func_id)
				z[:, i + base] += coeff * func(z[:, idx])

		if split:
			x = z[:, :num_vars-1]
			y = z[:, num_vars-1:]
			yt = y - u[:, num_vars-1:]
			return x, y, yt
		else:
			return z


	def visualize(self):
		G = nx.DiGraph()

		num_vars = self.num_parents + self.num_children + self.num_other

		for i in range(num_vars):
			G.add_node(f"{i+1}")

		G.add_node("Y")

		for i in range(self.num_parents):
			G.add_edge(f"{i+1}", "Y")
		for i in range(self.num_children):
			G.add_edge("Y", f"{i+1+self.num_parents}")

		base = self.num_parents + self.num_children + 1
		node_colors = ['#05348b'] * self.num_parents + ['#ae1908'] * self.num_children

		for i in range(self.num_other):
			nvar, comp, _ = self.other_assc[i]
			color = '#9acdc4'
			for j in range(nvar):
				idx, _, __ = comp[j]
				G.add_edge(f"{idx + 1}", f"{i + base}")
				if idx >= self.num_parents:
					color = '#ec813b'
			node_colors.append(color)
		node_colors.append('white')

		pos = nx.spring_layout(G)  # Choose a layout algorithm
		nx.draw(G, pos=pos, with_labels=True, arrows=True,
				node_color=node_colors)
		plt.show()


def generate_nonlinear_SCM(num_envs, nparent, nchild, nother=0):
	models = []
	e0 = CharysanSCM(nparent, nchild, num_other=nother)
	models.append(e0)
	for i in range(num_envs - 1):
		models.append(CharysanSCM(nparent, nchild, e0.func_parent, e0.coeff_parent, e0.func_children, e0.coeff_children, num_other=nother, other_assc=e0.other_assc))
	parent_set = [i for i in range(nparent)]
	children_set = [(i + nparent) for i in range(nchild)]
	return models, parent_set, children_set, e0.offspring


def random_assignment_matrix(num_vars, ratio, function_id_max, coefficient_max, degree_max, reference_g=None):
	function_matrix = np.zeros((num_vars, num_vars), dtype=np.int)
	coefficient_matrix = np.zeros((num_vars, num_vars), dtype=np.float)
	if reference_g is None:
		for i in range(num_vars):
			cnt = 0
			idx = np.random.permutation(i)
			for j in range(i):
				if np.random.uniform(0, 1) < ratio:
					function_matrix[i, idx[j]] = np.random.randint(function_id_max) + 1
					coefficient_matrix[i, idx[j]] = np.random.uniform(-coefficient_max, coefficient_max)
					cnt += 1
				if cnt >= degree_max:
					break
			coefficient_matrix[i, i] = np.abs(np.random.uniform(-coefficient_max, coefficient_max)) + 0.5
	else:
		for i in range(num_vars):
			for j in range(i):
				if reference_g[i, j] > 0:
					function_matrix[i, j] = np.random.randint(function_id_max) + 1
					coefficient_matrix[i, j] = np.random.uniform(-coefficient_max, coefficient_max)
			coefficient_matrix[i, i] = np.abs(np.random.uniform(-coefficient_max, coefficient_max)) + 0.5

	return function_matrix, coefficient_matrix


def generate_random_SCM(num_vars, y_index=None, min_child=0, min_parent=0, num_envs=2, nonlinear_id=5, law='linear', same_var=True):
	if y_index is None:
		y_index = np.random.randint(num_vars - 1) + 1

	models = []
	func_mat0, coeff_mat0 = random_assignment_matrix(num_vars, 0.4, nonlinear_id, 1, 4)

	num_child = np.sum(func_mat0 > 0, 0)[y_index]
	if num_child < min_child:
		remain_child = min_child - num_child
		idx = np.random.permutation(num_vars-y_index-1)
		for i in range(num_vars-y_index-1):
			if func_mat0[idx[i]+y_index+1, y_index] == 0:
				remain_child -= 1
				func_mat0[idx[i]+y_index+1, y_index] = np.random.randint(nonlinear_id) + 1
			if remain_child == 0:
				break

	if num_child > min_child + 1:
		remain_child = num_child - min_child - 1
		idx = np.random.permutation(num_vars-y_index-1)
		for i in range(num_vars-y_index-1):
			if func_mat0[idx[i]+y_index+1, y_index] > 0:
				remain_child -= 1
				func_mat0[idx[i]+y_index+1, y_index] = 0
			if remain_child == 0:
				break

	num_parent = np.sum(func_mat0 > 0, 1)[y_index]
	if num_parent < min_parent:
		remain_parent = min_parent - num_parent
		for i in range(y_index):
			if func_mat0[y_index, i] == 0:
				remain_parent -= 1
				func_mat0[y_index, i] = np.random.randint(nonlinear_id - 1) + 1
			if remain_parent == 0:
				break

	if law == 'linear':
		func_mat0[y_index, :] = np.minimum(func_mat0[y_index, :], 1)
		ratio = 1.0
	else:
		func_mat0[y_index, :] = np.minimum(func_mat0[y_index, :], nonlinear_id - 1)
		ratio = 1.0

	parent_set = []
	# enforce large signal
	for i in range(y_index):
		if func_mat0[y_index, i] > 0:
			coeff_mat0[y_index, i] = (np.abs(np.random.uniform(0, 0.5)) + 1) * (2*np.random.randint(2)-1)
			parent_set.append(i)
		if func_mat0[y_index, i] >= 2:
			coeff_mat0[y_index, i]  *= 2
	coeff_mat0[y_index, y_index] = 1

	# enforce large bias
	child_set = []
	for i in range(y_index+1, num_vars):
		if func_mat0[i, y_index] > 0:
			coeff_mat0[i, y_index] = (np.abs(np.random.uniform(0, 1)) + 0.5) * (2*np.random.randint(2)-1)
			child_set.append(i)

	for i in range(num_envs):
		func_mat, coeff_mat = random_assignment_matrix(num_vars, 0.4, nonlinear_id, 1.5, 4, func_mat0)
		func_mat[y_index, :] = func_mat0[y_index, :]
		if same_var:
			coeff_mat[y_index, :] = coeff_mat0[y_index, :]
		else:
			coeff_mat[y_index, :y_index] = coeff_mat0[y_index, :y_index]
		for child in child_set:
			while True:
				coeff_mat[child, y_index] = (np.abs(np.random.uniform(0, 1)) + 0.5) * (2*np.random.randint(2)-1)
				#print(coeff_mat0[child, y_index])
				if np.abs(coeff_mat[child, y_index] - coeff_mat0[child, y_index]) > 0.5 and np.abs(coeff_mat[child, y_index] + coeff_mat0[child, y_index]) > 0.5:
					break
		model = AdditiveStructuralCausalModel(num_vars, coeff_mat, func_mat, y_index)
		models.append(model)
	true = coeff_mat0[y_index, :]
	true[y_index] = 0.0

	offspring_set = []
	child_set = []
	for i in range(y_index+1, num_vars):
		if func_mat[i, y_index] > 0:
			offspring_set.append(i)
			child_set.append(i)
		else:
			for j in range(y_index+1, i):
				if func_mat[i, j] > 0 and j in offspring_set:
					offspring_set.append(i)
	offspring_set = list(set(offspring_set))
	#print(f'function assignment = {func_mat0[y_index, :y_index]}')
	return models, func_mat0[y_index, :-1], coeff_mat0[y_index, :-1], parent_set, child_set, offspring_set


def SCM_ex1():
	num_vars = 3
	y_index = 1
	func_mat = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.int)
	coeff1 = np.array([[1, 0, 0], [2, 1, 0], [1, -0.3, 1]], dtype=np.float)
	coeff2 = np.array([[1.5, 0, 0], [2, 1, 0], [0.3, 0.7, 0.3]], dtype=np.float)
	models = [
		AdditiveStructuralCausalModel(num_vars, coeff1, func_mat, y_index),
		AdditiveStructuralCausalModel(num_vars, coeff2, func_mat, y_index)
	]
	return models, np.array([2, 0], dtype=np.float)


def sample_from_SCM(models, n, index=0, shuffle=False):
	xs, ys, yts = [], [], []
	for i in range(len(models)):
		x, y, yt = models[i].sample(n)
		if shuffle:
			xl = x[:, :index]
			xr = x[:, index:]
			arr = np.arange(n)
			np.random.shuffle(arr)
			xr = xr[arr, :]
			x = np.concatenate([xl, xr], 1)
		
		xs.append(x)
		ys.append(y)
		yts.append(yt)
	return xs, ys, yts


class ClassificationSCM:
	def __init__(self, beta_t=1, spur=0.95):
		self.beta = beta_t
		self.spur = spur

	def sample(self, n):
		x = np.random.normal(0, 1, n)
		y = (np.random.uniform(0, 1, n) <= sigmoid(self.beta * x)) * 1.0
		coin_flip = (np.random.uniform(0, 1, n) <= self.spur) * 1.0
		z = (y * coin_flip + (1 - y) * (1 - coin_flip)) * self.spur
		z = z + np.random.normal(0, 0.3, n)
		xx = np.concatenate([np.reshape(x, (n, 1)), np.reshape(z, (n, 1))], 1)
		yy = np.reshape(y, (n, 1))
		return xx, yy



def SCM_class(signal, s1, s2):
	return [ClassificationSCM(signal, s1), ClassificationSCM(signal, s2)]

