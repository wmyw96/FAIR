from data.model import *
from methods.brute_force import brute_force
from methods.fair_gumbel import *
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
parser.add_argument("--diter", help="test mode", type=int, default=2)

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
	models[0].visualize(offspring_set)


# set saving dir
exp_name = f"n{args.n}_nenvs{args.num_envs}_dimx{dim_x}_niters{args.niters}_mch_{args.min_child}_mpa{args.min_parent}_lr{args.lr}"
exp_name += f"_lsbias{args.lsbias}_itemp{args.init_temp}_ftemp{args.final_temp}_gamma{args.gamma}_bz{args.batch_size}_seed{args.seed}"

# generate data
print(parent_set, child_set)
xs, ys, yts = sample_from_SCM(models, args.n)

# FAIR Gumbel algorithm
niters = args.niters
iter_save = 100
packs = fair_ll_sgd_gumbel_uni(xs, ys, hyper_gamma=args.gamma, learning_rate=args.lr, niters_d=args.diter,
							niters=niters, batch_size=args.batch_size, init_temp=args.init_temp,
							final_temp=args.final_temp, iter_save=iter_save, log=args.log)

beta = packs['weight']
mask = packs['gate_rec'][-1] > 0.9

# Refit using LS
full_var = (np.arange(dim_x))
var_set = full_var[mask].tolist()
print(var_set)
beta3 = broadcast(pooled_least_squares([x[:, var_set] for x in xs], ys), var_set, dim_x)


if args.log:
	print(f'True coefficient {true_coeff},')
	print(f'Selected variables {np.where(mask)}')
	print(f'FAIR SGD L2 error = {np.sum(np.square(beta - true_coeff))}')
	print(f'FAIR SGD (LS Refitted) L2 error = {np.sum(np.square(beta3 - true_coeff))}')


# visualize gate during training
loss_rec = packs['loss_rec']

import matplotlib.pyplot as plt
from matplotlib import rc
from numpy import genfromtxt

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=20)
rc('text', usetex=True)

gate = packs['gate_rec']
para = packs['weight_rec']
print(np.shape(para))

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
print_vector(beta, color_rsp)

plt.figure(figsize=(8, 8))
ax1 = plt.subplot(1, 1, 1)

it_display = niters // 100
it_arr = np.arange(it_display)

for i in range(dim_x):
	ax1.plot(it_arr, gate[:it_display, i], color=color_tuple[rsp[i]])
ax1.set_xlabel('iters (100)')
ax1.set_ylabel('sigmoid(logits)')

plt.show()
