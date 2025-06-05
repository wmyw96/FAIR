from data.model import *
from methods.brute_force import brute_force
from methods.fair_gumbel import fairnn_sgd_gumbel_uni, fairnn_sgd_gumbel_refit
from methods.tools import pooled_least_squares
from utils import *
import numpy as np
import os
import argparse
import time

def broadcast(beta_restricted, var_inds, p):
	beta_broadcast = np.zeros(p)
	if len(var_inds) == 1:
		beta_broadcast[var_inds[0]] = beta_restricted
		return beta_broadcast
	for i, ind in enumerate(var_inds):
		beta_broadcast[ind] = beta_restricted[i]
	return beta_broadcast

parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="random seed", type=int, default=1234)
parser.add_argument("--n", help="number of samples", type=int, default=1000)
parser.add_argument("--batch_size", help="batch size", type=int, default=36)
parser.add_argument("--num_envs", help="number of environments", type=int, default=2)
parser.add_argument("--dim_x", help="number of explanatory vars", type=int, default=60)
parser.add_argument("--niters", help="number of iterations", type=int, default=50000)
parser.add_argument("--lr", help="learning rate", type=float, default=1e-3)
parser.add_argument("--min_child", help="min number of children", type=int, default=5)
parser.add_argument("--min_parent", help="min number of parents", type=int, default=5)
parser.add_argument("--lsbias", help="least square bias", type=float, default=0.5)
parser.add_argument('--init_temp', help='initial temperature', type=float, default=5)
parser.add_argument('--final_temp', help='final temperature', type=float, default=0.1)
parser.add_argument('--gamma', help='hyper parameter gamma', type=float, default=36)
parser.add_argument("--record_dir", help="record directory", type=str, default="logs/")
parser.add_argument("--log", help="show log", type=bool, default=True)
parser.add_argument("--mode", help="test mode", type=int, default=1)
parser.add_argument("--diter", help="iter of discminator", type=int, default=2)
parser.add_argument("--giter", help="iter of predictor", type=int, default=1)
parser.add_argument("--temp_iter", help="iter to attain final temp", type=int, default=50000)
parser.add_argument("--threshold", help="truncation threshold", type=float, default=0.9)
parser.add_argument("--riter", help="iter of refitting", type=int, default=10000)
parser.add_argument("--offset", help="init offset", type=float, default=-3)

args = parser.parse_args()

np.random.seed(args.seed)

TEST_MODE = args.mode

# Set data generating process
if TEST_MODE == 1:
	dim_x = 2
	models, true_coeff = SCM_ex1()
	parent_set, child_set, offspring_set = [0], [1], [1]
elif TEST_MODE == 2:
	dim_x = 12
	models, true_coeff = [StructuralCausalModel1(13), StructuralCausalModel2(13)], np.array([3, 2, -0.5] + [0] * (13 - 4))
	parent_set, child_set, offspring_set = [0, 1, 2], [6, 7], [6, 7, 8]
elif TEST_MODE == 3:
	dim_x = args.dim_x
	models, true_coeff, parent_set, child_set, offspring_set = \
	get_linear_SCM(num_vars=dim_x + 1, num_envs=args.num_envs, y_index=dim_x // 2, 
					min_child=args.min_child, min_parent=args.min_parent, nonlinear_id=5, 
					bias_greater_than=args.lsbias, log=args.log)
elif TEST_MODE == 4:
	dim_x = 26
	models, parent_set, child_set, offspring_set = \
		get_nonlinear_SCM(num_envs=2, nparent=5, nchild=4, dim_x=dim_x, bias_greater_than=args.lsbias, log=args.log)
	models[0].visualize()
elif TEST_MODE == 5:
	dim_x = 26
	models, parent_set, child_set, offspring_set = \
		get_nonlinear_SCM(num_envs=2, nparent=5, nchild=4, dim_x=dim_x, bias_greater_than=args.lsbias, log=args.log)
	models[0].hcm = 1
	models[1].hcm = 1
	models[0].visualize()

# set saving dir
exp_name = f"n{args.n}_nenvs{args.num_envs}_dimx{dim_x}_niters{args.niters}_mch_{args.min_child}_mpa{args.min_parent}_lr{args.lr}"
exp_name += f"_lsbias{args.lsbias}_itemp{args.init_temp}_ftemp{args.final_temp}_gamma{args.gamma}_bz{args.batch_size}_seed{args.seed}"

# generate data
print(parent_set, child_set, offspring_set)
xs, ys, yts = sample_from_SCM(models, args.n)

for j in range(dim_x):
	print('j = %d, x_min0 = %.2f, x_max0 = %.2f, x_min1 = %.2f, x_max1 = %.2f,' % (j, np.min(xs[0][:, j]), np.max(xs[0][:, j]), np.min(xs[1][:, j]), np.max(xs[1][:, j])))

# generate valid & test data
xvs, yvs, yvts = sample_from_SCM(models, args.n // 7 * 3)
xts, yts, ytts = sample_from_SCM(models, args.n)

valid_x, valid_y = np.concatenate(xvs, 0), np.concatenate(yvs, 0)
test_x, test_y = np.concatenate(xts, 0), np.concatenate(ytts, 0)

eval_data = (valid_x, valid_y, test_x, test_y)


# Report ERM estimation performance
mask3 = np.ones((dim_x, ))
packs3 = fairnn_sgd_gumbel_refit(xs, ys, mask3, eval_data, learning_rate=args.lr, niters=args.riter, 
								batch_size=args.batch_size, log=False)
eval_loss3 = packs3['loss_rec']
print('ERM Test Error: {}'.format(eval_loss3[np.argmin(eval_loss3[:, 0]), 1]))

# Report Oracle estimation performance
mask4 = np.zeros((dim_x, ))
for i in parent_set:
	mask4[i] = 1.0
packs4 = fairnn_sgd_gumbel_refit(xs, ys, mask4, eval_data, learning_rate=args.lr, niters=args.riter, 
								batch_size=args.batch_size, log=False)
eval_loss4 = packs4['loss_rec']
print('Oracle Test Error: {}'.format(eval_loss4[np.argmin(eval_loss4[:, 0]), 1]))

# FAIR Gumbel algorithm
niters = args.niters
iter_save = 100
packs = fairnn_sgd_gumbel_uni(xs, ys, eval_data=eval_data, hyper_gamma=args.gamma, learning_rate=args.lr, niters_d=args.diter, 
							niters_g=args.giter, niters=niters, batch_size=args.batch_size, init_temp=args.init_temp, offset=args.offset,
							final_temp=args.final_temp, temp_iter=args.temp_iter, iter_save=iter_save, log=args.log)

mask = (packs['gate_rec'][-1] > args.threshold) * 1.0
print(mask)

# Report estimation performance
packs2 = fairnn_sgd_gumbel_refit(xs, ys, mask, eval_data, learning_rate=args.lr, niters=args.riter, 
								batch_size=args.batch_size, log=False)
eval_loss = packs2['loss_rec']
print('Refit Test Error: {}'.format(eval_loss[np.argmin(eval_loss[:, 0]), 1]))

# visualize gate during training

loss_rec = packs['loss_rec']

import matplotlib.pyplot as plt
from matplotlib import rc
from numpy import genfromtxt

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=20)
rc('text', usetex=True)

gate = packs['gate_rec']

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

print_vector(packs['gate_rec'][-1,:], color_rsp)

plt.figure(figsize=(8, 8))
ax1 = plt.subplot(1, 1, 1)

it_display = niters // iter_save
it_arr = np.arange(it_display)

for i in range(dim_x):
	ax1.plot(it_arr, gate[:it_display, i], color=color_tuple[rsp[i]])

ax1.set_xlabel('iters (100)')
ax1.set_ylabel('sigmoid(logits)')

plt.show()
