from data.model import *
from methods.brute_force import brute_force, pooled_least_squares, support_set
from methods.predessors import *
import numpy as np

##############################################
#
#              Utility Functions
#
##############################################


def broadcast(beta_restricted, var_inds, p):
	beta_broadcast = np.zeros(p)
	if len(var_inds) == 1:
		beta_broadcast[var_inds[0]] = beta_restricted
		return beta_broadcast
	for i, ind in enumerate(var_inds):
		beta_broadcast[ind] = beta_restricted[i]
	return beta_broadcast


dim_x = 12


def mydist(cov, beta):
	x = np.reshape(beta, (np.shape(cov)[0], 1))
	return float(np.matmul(x.T, np.matmul(cov, x)))


##############################################
#
#                Methods
#
##############################################


def oracle_irm(x_list, y_list, true_para):
	data_list = []
	for i in range(len(x_list)):
		data_list.append((torch.tensor(x_list[i]).float(), torch.tensor(y_list[i]).float()))

	error_min = 1e9
	beta = 0
	for reg in [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]:
		model = InvariantRiskMinimization(data_list, args={'n_iterations': 10000, 'lr': 1e-3, 'verbose': False, 'reg': reg})
		cand = np.squeeze(model.solution().detach().numpy())
		cov_x = sum([np.matmul(x.T, x) / np.shape(x)[0] for x in x_list]) / len(x_list)
		error = mydist(cov_x, cand - true_para)
		if error < error_min:
			error_min = error
			beta = cand

	return beta


def erm(x_list, y_list, true_para=None):
	return pooled_least_squares(x_list, y_list)


def oracle_icp(x_list, y_list, true_para):
	data_list = []
	for i in range(len(x_list)):
		data_list.append((torch.tensor(x_list[i]).float(), torch.tensor(y_list[i]).float()))

	error_min = 1e9
	beta = 0
	for alpha in [0.9, 0.95, 0.99, 0.995]:
		model = InvariantCausalPrediction(data_list, args={'alpha': alpha, "verbose": False})
		cand = np.squeeze(model.solution().numpy())
		cov_x = sum([np.matmul(x.T, x) / np.shape(x)[0] for x in x_list]) / len(x_list)
		error = mydist(cov_x, cand - true_para)
		if error < error_min:
			error_min = error
			beta = cand

	return beta


def oracle_anchor(x_list, y_list, true_para):
	xs, ys, anchors = [], [], []
	for i in range(len(x_list)):
		xs.append(x_list[i])
		ys.append(y_list[i])
		onehot = np.zeros(len(x_list)-1)
		if i + 1 < len(x_list):
			onehot[i] = 1
		anchors.append([onehot] * np.shape(x_list[i])[0])
	
	X, y, A = np.concatenate(xs, 0), np.squeeze(np.concatenate(ys, 0)), np.concatenate(anchors, 0)
	error_min = 1e9
	beta = 0

	for reg in [0, 1, 2, 4, 8, 10, 15, 20, 30, 40, 60, 80, 90, 100, 150, 200, 500, 1000, 5000, 10000]:
		model = AnchorRegression(lamb=reg)
		model.fit(X, y, A)
		cand = np.squeeze(model.coef_)
		cov_x = sum([np.matmul(x.T, x) / np.shape(x)[0] for x in x_list]) / len(x_list)
		error = mydist(cov_x, cand - true_para)
		if error < error_min:
			error_min = error
			beta = cand

	return beta


def causal_dantzig(x_list, y_list, true_para):
	n0 = np.shape(x_list[0])[0]
	n1 = np.shape(x_list[1])[0]
	z = np.matmul(x_list[0].T, y_list[0]) / n0 - np.matmul(x_list[1].T, y_list[1]) / n1
	g = np.matmul(x_list[0].T, x_list[0]) / n0 - np.matmul(x_list[1].T, x_list[1]) / n1
	return np.squeeze(np.matmul(np.linalg.inv(g), z))

def eills(x_list, y_list, true_para=None):
	return brute_force(x_list, y_list, 36, loss_type='eills')

def fair(x_list, y_list, true_para=None):
	return brute_force(x_list, y_list, 36, loss_type='fair')

def lse_s_star(x_list, y_list, true_para=None):
	var_set = []
	for i in range(np.shape(true_para)[0]):
		if (np.abs(true_para[i]) > 1e-9):
			var_set.append(i)
	return broadcast(pooled_least_squares([x[:, var_set] for x in x_list], y_list), var_set, np.shape(true_para)[0])


def lse_gc(x_list, y_list, true_para=None):
	var_set = [0, 1, 2, 3, 4, 5, 9, 10, 11]
	return broadcast(pooled_least_squares([x[:, var_set] for x in x_list], y_list), var_set, dim_x)


def eills_refit(x_list, y_list, true_para=None):
	eills_sol = brute_force(x_list, y_list, 20, loss_type='eills')
	var_set = []
	for i in range(np.shape(eills_sol)[0]):
		if np.abs(eills_sol[i]) > 1e-9:
			var_set.append(i)
	return broadcast(pooled_least_squares([x[:, var_set] for x in x_list], y_list), var_set, dim_x)

def lse_s_rd(x_list, y_list, true_para=None):
	xs = []
	index = np.shape(x_list[0])[1] // 2
	for i in range(len(x_list)):
		x = x_list[i]
		xl = x[:, :index]
		xr = x[:, index:]
		arr = np.arange(np.shape(x)[0])
		np.random.shuffle(arr)
		xr = xr[arr, :]
		x = np.concatenate([xl, xr], 1)
		xs.append(x)
	return pooled_least_squares(xs, y_list)
