from data.model import *
from methods.brute_force import brute_force, pooled_least_squares, support_set
from methods.predessors import *
from methods.fair_algo import *
from utils import *
import numpy as np
import time
from methods.demo_wrapper import *
from utils import get_linear_SCM, get_SCM, get_nonlinear_SCM
import argparse
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("--exp_id", help="mode", type=int, default=1)
parser.add_argument("--ntrial", help="repeat", type=int, default=50)
args = parser.parse_args()

def np_mse(x, y):
	return np.mean(np.square(x - y))

def torch_mse(y_hat, y):
	return 0.5 * torch.mean((y_hat - y) ** 2)


if args.exp_id == 1:
	candidate_n = [200, 500, 1000, 2000, 5000]

	num_repeats = args.ntrial

	methods = [None, lse_s_star, lse_s_rd, erm]

	result = np.zeros((len(candidate_n), num_repeats, len(methods) + 2, 70))

	for (ni, n) in enumerate(candidate_n):
		for t in range(num_repeats):
			start_time = time.time()
			np.random.seed(t)
			torch.manual_seed(t)

			#generate random graph with 71 nodes
			models, true_coeff, parent_set, child_set, offspring_set = \
				get_linear_SCM(num_vars=71, num_envs=2, y_index=35, 
								min_child=5, min_parent=5, nonlinear_id=5, 
								bias_greater_than=0.5, same_var=False, log=False)
			
			result[ni, t, 0, :] = true_coeff

			# generate data
			xs, ys = [], []
			for i in range(2):
				x, y, _ = models[i].sample(n)
				xs.append(x)
				ys.append(y)

			for mid, method in enumerate(methods):
				if mid == 0:
					# run fair algorithm
					fair_model = FairMLP(2, 70, 0, 0, 0, 0)
					hyper_params = aos_default_hyper_params
					algo = FairGumbelAlgo(2, 70, fair_model, 36, torch_mse, hyper_params)
					packs = algo.run_gumbel((xs, ys), eval_metric=None, eval_iter=hyper_params['niters']//10)

					# get parameter and variable selection
					beta = beta = np.reshape(fair_model.g.relu_stack.weight.detach().cpu().numpy(), (-1)) * packs['gate_rec'][-1, :]
					mask = packs['gate_rec'][-1, :] > 0.7

					# Refit using LS
					full_var = (np.arange(70))
					var_set = full_var[mask].tolist()
					beta3 = broadcast(pooled_least_squares([x[:, var_set] for x in xs], ys), var_set, 70)

					result[ni, t, mid + 1, :] = beta
					result[ni, t, len(methods) + 1, :] = beta3
				else:
					beta = method(xs, ys, true_coeff)

					result[ni, t, mid + 1, :] = beta
				
				print(f'method {mid}, l2 error = {np.sum(np.square(true_coeff - beta))}')
			print(f'method {len(methods)}, l2 error = {np.sum(np.square(true_coeff - result[ni, t, len(methods) + 1, :]))}')
			end_time = time.time()
			print(f'Running Case: n = {n}, t = {t}, secs = {end_time - start_time}s')

	np.save('saved_results/unit_test_1.npy', result)


if args.exp_id == 2:
	candidate_n = [50, 100, 300, 500, 800, 1000]

	num_repeats = args.ntrial

	methods = [
		eills,
		fair,
		None,
		lse_s_star,
		oracle_irm,
		oracle_anchor,
		oracle_icp,
		erm
	]

	names = ['EILLS', 'FAIR-BF', 'FAIR-GB', 'Oracle', 'IRM', 'Anchor', 'ICP', 'LSE']

	result = np.zeros((len(candidate_n), num_repeats, len(methods) + 2, 15))

	for (ni, n) in enumerate(candidate_n):
		for t in range(num_repeats):
			start_time = time.time()
			np.random.seed(t)
			torch.manual_seed(t)

			#generate random graph with 15 nodes
			models, true_coeff, parent_set, child_set, offspring_set = \
				get_linear_SCM(num_vars=16, num_envs=2, y_index=8, 
								min_child=5, min_parent=5, nonlinear_id=5, 
								bias_greater_than=0.5, same_var=True, log=False)
			
			result[ni, t, 0, :] = true_coeff

			# generate data
			xs, ys = [], []
			for i in range(2):
				x, y, _ = models[i].sample(n)
				xs.append(x)
				ys.append(y)

			for mid, method in enumerate(methods):
				if mid == 2:
					# run fair algorithm
					fair_model = FairMLP(2, 15, 0, 0, 0, 0)
					hyper_params = aos_default_hyper_params
					algo = FairGumbelAlgo(2, 15, fair_model, 36, torch_mse, hyper_params)
					packs = algo.run_gumbel((xs, ys), eval_metric=None, eval_iter=hyper_params['niters']//10)

					# get parameter and variable selection
					beta = beta = np.reshape(fair_model.g.relu_stack.weight.detach().cpu().numpy(), (-1)) * packs['gate_rec'][-1, :]
					mask = packs['gate_rec'][-1, :] > 0.9

					# Refit using LS
					full_var = (np.arange(15))
					var_set = full_var[mask].tolist()
					beta3 = broadcast(pooled_least_squares([x[:, var_set] for x in xs], ys), var_set, 15)

					result[ni, t, mid + 1, :] = beta
					result[ni, t, len(methods) + 1, :] = beta3
				else:
					beta = method(xs, ys, true_coeff)

					# restore the estimated coeffs
					result[ni, t, mid + 1, :] = beta
				print(f'method ({names[mid]}), l2 error = {np.sum(np.square(true_coeff - beta))}')

			print(f'method (FAIR-refit), l2 error = {np.sum(np.square(true_coeff - result[ni, t, len(methods) + 1, :]))}')
			end_time = time.time()
			print(f'Running Case: n = {n}, t = {t}, secs = {end_time - start_time}s')


	np.save('saved_results/unit_test_2.npy', result)


if args.exp_id == 3:
	candidate_n = [1000, 2000, 4000, 8000]

	num_repeats = args.ntrial

	np.random.seed(0)

	result = np.zeros((len(candidate_n), num_repeats, 5))

	for (ni, n) in enumerate(candidate_n):
		for t in range(num_repeats):
			start_time = time.time()
			np.random.seed(t)
			torch.manual_seed(t)

			print(f'================ Test ID = {t}, n = {n} ================')
			dim_x = 26
			models, parent_set, child_set, offspring_set = \
				get_nonlinear_SCM(num_envs=2, nparent=5, nchild=4, dim_x=dim_x, bias_greater_than=0.5, log=False)
			#print(f'number of child = {len(child_set)}')

			xs, ys, yts = sample_from_SCM(models, n)
			# generate valid & test data
			xvs, yvs, yvts = sample_from_SCM(models, n // 7 * 3)
			xts, yts, ytts = sample_from_SCM(models, 30000)

			valid_x, valid_y = np.concatenate(xvs, 0), np.concatenate(yvs, 0)
			test_x, test_y = np.concatenate(xts, 0), np.concatenate(ytts, 0)

			eval_data = (valid_x, valid_y, test_x, test_y)

			# common hyper-parameter
			batch_size, lr, riter = 64, 1e-3, 10000

			# Report Oracle estimation performance
			mask4 = np.zeros((dim_x, ))
			for i in parent_set:
				mask4[i] = 1.0
			packs4 = nn_least_squares_refit(xs, ys, mask4, eval_data, learning_rate=lr, niters=riter, 
											depth_g=2, width_g=128, batch_size=batch_size, log=False)
			eval_loss4 = packs4['loss_rec']
			loss4 = eval_loss4[np.argmin(eval_loss4[:, 0]), 1]
			print(f'Oracle Test Error: {loss4}')
			result[ni, t, 0] = loss4

			# Report ERM estimation performance
			mask3 = np.ones((dim_x, ))
			packs3 = nn_least_squares_refit(xs, ys, mask3, eval_data, learning_rate=lr, niters=riter, 
											depth_g=2, width_g=128, batch_size=batch_size, log=False)
			eval_loss3 = packs3['loss_rec']
			loss3 = eval_loss3[np.argmin(eval_loss3[:, 0]), 1]
			print(f'ERM Test Error: {loss3}')
			result[ni, t, 1] = loss3

			# Report FAIR-Gumbel performance

			niters = 70000
			threshold = 0.9
			if n <= 2000:
				threshold = 0.6


			valid = ((valid_x,), (valid_y,))
			test = ((test_x,), (test_y,))
			fair_model = FairMLP(2, 26, 1, 128, 2, 196)
			hyper_params = deepcopy(aos_default_hyper_params)
			hyper_params['niters'] = niters
			algo = FairGumbelAlgo(2, 26, fair_model, 36, torch_mse, hyper_params)
			packs1 = algo.run_gumbel((xs, ys), eval_metric=np_mse, me_valid_data=valid, me_test_data=test, eval_iter=niters//10, log=False)
			
			mask = (packs1['gate_rec'][-1] > threshold) * 1.0

			eval_loss1 = packs1['loss_rec']
			loss1 = eval_loss1[-1, 1]
			print(f'FAIR Test Error: {loss1}')
			result[ni, t, 2] = loss1

			# FAIR-Gumbel Refit performance
			packs2 = nn_least_squares_refit(xs, ys, mask, eval_data, learning_rate=lr, niters=riter, 
								depth_g=2, width_g=128, batch_size=batch_size, log=False)
			eval_loss2 = packs2['loss_rec']
			loss2 = eval_loss2[np.argmin(eval_loss2[:, 0]), 1]
			print('Refit Test Error: {}'.format(loss2))
			result[ni, t, 3] = loss2

			print(f'FAIR mask = {mask}')
			print(print_prob(packs1['gate_rec'][-1]))

			end_time = time.time()
			print(f'Running Case: n = {n}, t = {t}, secs = {end_time - start_time}s\n')

	np.save('saved_results/unit_test_3.npy', result)



if args.exp_id == 4:
	candidate_n = [1000, 2000, 4000, 8000]

	num_repeats = args.ntrial

	np.random.seed(0)

	result = np.zeros((len(candidate_n), num_repeats, 5))

	for (ni, n) in enumerate(candidate_n):
		for t in range(num_repeats):
			start_time = time.time()
			np.random.seed(t)
			torch.manual_seed(t)

			print(f'================ Test ID = {t}, n = {n} ================')
			dim_x = 26
			models, parent_set, child_set, offspring_set = \
				get_nonlinear_SCM(num_envs=2, nparent=5, nchild=4, dim_x=dim_x, bias_greater_than=0.5, log=False)
			models[0].hcm = 1
			models[1].hcm = 1
			print(f'number of child = {len(child_set)}')

			xs, ys, yts = sample_from_SCM(models, n)
			# generate valid & test data
			xvs, yvs, yvts = sample_from_SCM(models, n // 7 * 3)
			xts, yts, ytts = sample_from_SCM(models, 30000)

			valid_x, valid_y = np.concatenate(xvs, 0), np.concatenate(yvs, 0)
			test_x, test_y = np.concatenate(xts, 0), np.concatenate(ytts, 0)

			eval_data = (valid_x, valid_y, test_x, test_y)

			# common hyper-parameter
			batch_size, lr, riter = 64, 1e-3, 10000

			# Report Oracle estimation performance
			mask4 = np.zeros((dim_x, ))
			for i in parent_set:
				mask4[i] = 1.0
			packs4 = nn_least_squares_refit(xs, ys, mask4, eval_data, learning_rate=lr, niters=riter, 
											depth_g=2, width_g=128, batch_size=batch_size, log=False)
			eval_loss4 = packs4['loss_rec']
			loss4 = eval_loss4[np.argmin(eval_loss4[:, 0]), 1]
			print(f'Oracle Test Error: {loss4}')
			result[ni, t, 0] = loss4

			# Report ERM estimation performance
			mask3 = np.ones((dim_x, ))
			packs3 = nn_least_squares_refit(xs, ys, mask3, eval_data, learning_rate=lr, niters=riter, 
											depth_g=2, width_g=128, batch_size=batch_size, log=False)
			eval_loss3 = packs3['loss_rec']
			loss3 = eval_loss3[np.argmin(eval_loss3[:, 0]), 1]
			print(f'ERM Test Error: {loss3}')
			result[ni, t, 1] = loss3

			# Report FAIR-Gumbel performance

			niters = 80000
			threshold = 0.9
			if n <= 2000:
				threshold = 0.6

			valid = ((valid_x,), (valid_y,))
			test = ((test_x,), (test_y,))
			fair_model = FairMLP(2, 26, 1, 128, 2, 196)
			hyper_params = deepcopy(aos_default_hyper_params)
			hyper_params['niters'] = niters
			algo = FairGumbelAlgo(2, 26, fair_model, 36, torch_mse, hyper_params)
			packs1 = algo.run_gumbel((xs, ys), eval_metric=np_mse, me_valid_data=valid, me_test_data=test, eval_iter=niters//10, log=False)
			
			mask = (packs1['gate_rec'][-1] > threshold) * 1.0

			eval_loss1 = packs1['loss_rec']
			loss1 = eval_loss1[-1, 1]
			print(f'FAIR Test Error: {loss1}')
			result[ni, t, 2] = loss1

			# FAIR-Gumbel Refit performance
			packs2 = nn_least_squares_refit(xs, ys, mask, eval_data, learning_rate=lr, niters=riter, 
								depth_g=2, width_g=128, batch_size=batch_size, log=False)
			eval_loss2 = packs2['loss_rec']
			loss2 = eval_loss2[np.argmin(eval_loss2[:, 0]), 1]
			print('Refit Test Error: {}'.format(loss2))
			result[ni, t, 3] = loss2

			print(f'FAIR mask = {mask}')
			print(print_prob(packs1['gate_rec'][-1]))

			end_time = time.time()
			print(f'Running Case: n = {n}, t = {t}, secs = {end_time - start_time}s\n')

	np.save('saved_results/unit_test_4.npy', result)

