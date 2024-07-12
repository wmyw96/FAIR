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