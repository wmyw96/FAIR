from data.model import generate_random_SCM, sample_from_SCM, generate_nonlinear_SCM
from methods.tools import *


def get_linear_SCM(num_vars, num_envs, y_index, min_child, min_parent, nonlinear_id, bias_greater_than=0.0, same_var=True, log=False):
	while True:
		models, true_func, true_coeff, parent_set, child_set, offspring_set = \
			generate_random_SCM(num_vars=num_vars, num_envs=num_envs, y_index=y_index, min_child=min_child, 
								min_parent=min_parent, nonlinear_id=nonlinear_id, same_var=same_var)
		xs, ys, yts = sample_from_SCM(models, 100000)

		beta_ls = pooled_least_squares(xs, ys)
		bias = np.sum(np.square(beta_ls - true_coeff))
		if bias > bias_greater_than:
			if log:
				print(f'Generate linear SCM: bias of PLS = {bias}')
			return models, true_coeff, parent_set, child_set, offspring_set


def get_SCM(num_vars, num_envs, y_index, min_child, min_parent, nonlinear_id, bias_greater_than=0.0, log=False):
	while True:
		models, true_func, true_coeff, parent_set, child_set, offspring_set = \
			generate_random_SCM(num_vars=num_vars, num_envs=num_envs, y_index=y_index, min_child=min_child, 
								min_parent=min_parent, nonlinear_id=nonlinear_id, law='nonlinear')
		xs, ys, yts = sample_from_SCM(models, 100000)

		beta_ls = pooled_least_squares(xs, ys)
		bias = np.sum(np.square(beta_ls - true_coeff))
		if bias > bias_greater_than:
			if log:
				print(f'Generate SCM: bias of PLS = {bias}')
			return models, true_coeff, parent_set, child_set, offspring_set


def get_nonlinear_SCM(num_envs, nchild, nparent, dim_x, bias_greater_than=0.5, log=False):
	while True:
		models, parent_set, child_set, offspring_set = \
			generate_nonlinear_SCM(num_envs, nparent, nchild, dim_x - nchild - nparent)
		xs, ys, yts = sample_from_SCM(models, 100000)
		beta_ls = pooled_least_squares(xs, ys)
		beta = [least_squares(xs[e], ys[e]) for e in range(num_envs)]
		hetero = sum([np.sum(np.square(beta[e] - beta_ls)) for e in range(num_envs)]) / num_envs

		if hetero > bias_greater_than:
			if log:
				print(models[0].func_parent, models[0].coeff_parent)
			return models, parent_set, child_set, offspring_set



valid_hex = '0123456789ABCDEF'.__contains__
def cleanhex(data):
	return ''.join(filter(valid_hex, data.upper()))

def fore_fromhex(text, hexcode):
	"""print in a hex defined color"""
	hexint = int(cleanhex(hexcode), 16)
	return "\x1B[38;2;{};{};{}m{}\x1B[0m".format(hexint>>16, hexint>>8&0xFF, hexint&0xFF, text)

def print_vector(vec, color):
	print_str = "["
	for i in range(np.shape(vec)[0]):
		if i > 0:
			print_str += ','
		print_str += fore_fromhex(vec[i], color[i])
	print_str += ']'
	print(print_str)


def broadcast_beta_vector(beta_restricted, var_inds, p):
	beta_broadcast = np.zeros(p)
	if len(var_inds) == 1:
		beta_broadcast[var_inds[0]] = beta_restricted
		return beta_broadcast
	for i, ind in enumerate(var_inds):
		beta_broadcast[ind] = beta_restricted[i]
	return beta_broadcast


def print_gate_during_training(dim_x, graph_sets, gate, tofile=None):
	parent_set, child_set, offspring_set = graph_sets
	import matplotlib.pyplot as plt
	from matplotlib import rc
	from numpy import genfromtxt

	plt.rcParams["font.family"] = "Times New Roman"
	plt.rc('font', size=20)
	rc('text', usetex=True)


	color_tuple = [
		'#ae1908',  # red
		'#ec813b',  # orange
		'#05348b',  # dark blue
		'#9acdc4',  # pain blue
	]

	rsp = []
	for i in range(dim_x):
		if i in parent_set:
			rsp.append(2)
		elif i in child_set:
			rsp.append(0)
		elif i in offspring_set:
			rsp.append(1)
		else:
			rsp.append(3)
	rsp = np.array(rsp)
	color_rsp = [color_tuple[i] for i in rsp]


	plt.figure(figsize=(6, 4))
	ax1 = plt.subplot(1, 1, 1)

	it_display = np.shape(gate)[0]
	it_arr = np.arange(it_display)

	for i in range(dim_x):
		ax1.plot(it_arr * 100, gate[:it_display, i], color=color_tuple[rsp[i]])

	ax1.set_xlabel(r'Iterations')
	ax1.set_ylabel('sigmoid(logits)')
	plt.ylim(0, 1)

	if tofile is None:
		plt.show()
	else:
		plt.savefig(tofile, bbox_inches='tight')
