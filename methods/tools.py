import numpy as np

def least_squares(X, y):
	cov_x = np.matmul(X.T, X)
	cov_xy = np.matmul(X.T, y)
	return np.squeeze(np.dot(np.linalg.inv(cov_x), cov_xy))


def pooled_least_squares(xs, ys, var_set=None):
	if var_set is None:
		return least_squares(np.concatenate(xs, 0), np.concatenate(ys, 0))
	else:
		dim_x = np.shape(xs[0])[1]
		return broadcast(pooled_least_squares([x[:, var_set] for x in xs], ys), var_set, dim_x)


def broadcast(beta_restricted, var_inds, p):
	beta_broadcast = np.zeros(p)
	if len(var_inds) == 1:
		beta_broadcast[var_inds[0]] = beta_restricted
		return beta_broadcast
	for i, ind in enumerate(var_inds):
		beta_broadcast[ind] = beta_restricted[i]
	return beta_broadcast

