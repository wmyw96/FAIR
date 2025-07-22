from data.model import *
from methods.tools import pooled_least_squares
from methods.fair_algo import *
from methods.fair_gumbel import *
from utils import *
import numpy as np
import os
import argparse
import time
import torch

parser = argparse.ArgumentParser()

# common setup
parser.add_argument("--seed", help="random seed", type=int, default=9)
parser.add_argument("--n", help="number of samples", type=int, default=2000)
parser.add_argument("--setup", help="setup: (linear) or (nonlinear)", type=str, default='linear')


args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)


# Set data generating process and model
if args.setup == 'linear':
	dim_x = 70
	models, true_coeff, parent_set, child_set, offspring_set = \
		get_linear_SCM(num_vars=dim_x + 1, num_envs=2, y_index=dim_x // 2, 
						min_child=5, min_parent=5, nonlinear_id=5, 
						bias_greater_than=0.5, log=True)
	models[0].visualize(offspring_set, f'saved_results/ex-{args.setup}-SCM.pdf')

	# linear model with 50k iterations
	niters = 50000
	fair_model = FairMLP(2, dim_x, 0, 0, 0, 0)


if args.setup == 'nonlinear':
	dim_x = 26
	models, parent_set, child_set, offspring_set = \
		get_nonlinear_SCM(num_envs=2, nparent=5, nchild=4, dim_x=dim_x, bias_greater_than=0.5, log=True)
	models[0].visualize(f'saved_results/ex-{args.setup}-SCM.pdf')

	# nonlinear model with 70k iterations
	niters = 70000
	fair_model = FairMLP(2, dim_x, 1, 128, 2, 196)


# Generate data
graph_sets = (parent_set, child_set, offspring_set)
xs, ys, yts = sample_from_SCM(models, args.n)

# Generate valid & test data
xvs, yvs, yvts = sample_from_SCM(models, args.n // 7 * 3)
xts, yts, ytts = sample_from_SCM(models, args.n)

valid, test = (xvs, yvs), (xts, ytts)

# Set hyper-parameters

from copy import deepcopy
hyper_params = deepcopy(aos_default_hyper_params)
hyper_params['niters'] = niters

# Set losses
def np_mse(x, y):
	return np.mean(np.square(x - y))

def torch_mse(y_hat, y):
	return 0.5 * torch.mean((y_hat - y) ** 2)


algo = FairGumbelAlgo(2, dim_x, fair_model, 36, torch_mse, hyper_params)

packs = algo.run_gumbel((xs, ys), eval_metric=np_mse, me_valid_data=valid, me_test_data=test, eval_iter=niters//10, log=True)
print_gate_during_training(dim_x, graph_sets, packs['gate_rec'], f'saved_results/ex-{args.setup}-n{args.n}-seed{args.seed}-logits.pdf')

if args.setup == 'linear':
	beta = np.reshape(fair_model.g.relu_stack.weight.detach().cpu().numpy(), (-1)) * packs['gate_rec'][-1, :]
	print(f'Estimated beta = {beta}')
else:
	print(packs['loss_rec'])