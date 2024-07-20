from data.model import *
from methods.fair_algo import *
from methods.tools import pooled_least_squares
from utils import *
import numpy as np
import os
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="random seed", type=int, default=1234)
parser.add_argument("--n", help="number of samples", type=int, default=1000)
parser.add_argument("--num_envs", help="number of environments", type=int, default=2)
parser.add_argument("--dim_x", help="number of explanatory vars", type=int, default=70)
parser.add_argument("--min_child", help="min number of children", type=int, default=5)
parser.add_argument("--min_parent", help="min number of parents", type=int, default=5)
parser.add_argument("--lsbias", help="least square bias", type=float, default=0.5)

parser.add_argument("--batch_size", help="batch size", type=int, default=64)
parser.add_argument("--lr", help="learning rate", type=float, default=1e-3)
parser.add_argument("--niters", help="number of iterations", type=int, default=50000)
parser.add_argument("--diters", help="iter of discminator", type=int, default=2)
parser.add_argument("--giters", help="iter of predictor", type=int, default=1)

parser.add_argument('--gamma', help='hyper parameter gamma', type=float, default=36)

parser.add_argument('--init_temp', help='initial temperature', type=float, default=5)
parser.add_argument('--final_temp', help='final temperature', type=float, default=0.1)
parser.add_argument("--offset", help="init offset", type=float, default=-3)

parser.add_argument("--log", help="show log", type=bool, default=True)
parser.add_argument("--datamodel", help="data generating process mode", type=int, default=1)
parser.add_argument("--fairmodel", help="fair model mode", type=int, default=1)


args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)


# Set data generating process
if args.datamodel == 1:
	dim_x = 2
	models, true_coeff = SCM_ex1()
	parent_set, child_set, offspring_set = [0], [1], [1]
elif args.datamodel == 2:
	dim_x = 12
	models, true_coeff = [StructuralCausalModel1(13), StructuralCausalModel2(13)], np.array([3, 2, -0.5] + [0] * (13 - 4))
	parent_set, child_set, offspring_set = [0, 1, 2], [6, 7], [6, 7, 8]
elif args.datamodel == 3:
	dim_x = args.dim_x
	models, true_coeff, parent_set, child_set, offspring_set = \
		get_linear_SCM(num_vars=dim_x + 1, num_envs=args.num_envs, y_index=dim_x // 2, 
						min_child=args.min_child, min_parent=args.min_parent, nonlinear_id=5, 
						bias_greater_than=args.lsbias, log=args.log)
	models[0].visualize(offspring_set)
elif args.datamodel == 4:
	dim_x = 26
	models, parent_set, child_set, offspring_set = \
		get_nonlinear_SCM(num_envs=2, nparent=5, nchild=4, dim_x=dim_x, bias_greater_than=args.lsbias, log=args.log)
	models[0].visualize()
elif args.datamodel == 5:
	dim_x = 26
	models, parent_set, child_set, offspring_set = \
		get_nonlinear_SCM(num_envs=2, nparent=5, nchild=4, dim_x=dim_x, bias_greater_than=args.lsbias, log=args.log)
	models[0].hcm = 1
	models[1].hcm = 1
	models[0].visualize()


# generate data
graph_sets = (parent_set, child_set, offspring_set)
xs, ys, yts = sample_from_SCM(models, args.n)

for j in range(dim_x):
	print('x%d, x_min0 = %.2f, x_max0 = %.2f, x_min1 = %.2f, x_max1 = %.2f,' % (j, np.min(xs[0][:, j]), np.max(xs[0][:, j]), np.min(xs[1][:, j]), np.max(xs[1][:, j])))

# generate valid & test data
xvs, yvs, yvts = sample_from_SCM(models, args.n // 7 * 3)
xts, yts, ytts = sample_from_SCM(models, args.n)

valid, test = (xvs, yvs), (xts, ytts)

# Set hyper-parameters

hyper_params = {
	'gumbel_lr': args.lr,
	'model_lr': args.lr,
	'weight_decay_g': 5e-4, 
	'weight_decay_f': 5e-4,
	'niters': args.niters,
	'diters': args.diters,
	'giters': args.giters,
	'batch_size': args.batch_size,
	'gamma': args.gamma,
	'init_temp': args.init_temp,
	'final_temp': args.final_temp,
	'offset': args.offset,
	'anneal_iter': 100,
	'anneal_rate': 0.993,
}

# Build FAIR Model
if args.fairmodel == 1:
	model = FairMLP(args.num_envs, dim_x, 0, 0, 0, 0)
else:
	model = FairMLP(2, dim_x, 1, 128, 2, 196)

def np_mse(x, y):
	return np.mean(np.square(x - y))

algo = FairGumbelAlgo(args.num_envs, dim_x, model, args.gamma, torch.nn.MSELoss(), hyper_params)
packs = algo.run_gumbel((xs, ys), eval_metric=np_mse, me_valid_data=valid, me_test_data=test, eval_iter=args.niters//20, log=True)
print_gate_during_training(dim_x, graph_sets, packs['gate_rec'])

