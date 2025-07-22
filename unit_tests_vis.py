import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from numpy import genfromtxt
from data.model import *
from methods.demo_wrapper import *
import argparse
from utils import *

plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', size=16)
rc('text', usetex=True)

parser = argparse.ArgumentParser()
parser.add_argument("--exp_id", help="exp id", type=int, default=1)
parser.add_argument("--ntrial", help="repeat", type=int, default=50)
args = parser.parse_args()

color_tuple = [
	'#ae1908',  # red
	'#ec813b',  # orange
	'#05348b',  # dark blue
	'#9acdc4',  # pain blue
	'#6bb392',  # green
	'#e5a84b',   # yellow
]

if args.exp_id == 1:
	results = np.load('saved_results/unit_test_1.npy')

	num_n = results.shape[0]
	num_sml = results.shape[1]

	vec_n = [200, 500, 1000, 2000, 5000]
	method_name = ["FAIR-GB", "FAIR-RF", "Oracle", r"Semi-Oracle", "Pool-LS"]
	method_idx = [0, 4, 1, 2, 3]

	lines = [
		'solid',
		'solid',
		'dashed',
		'dashed',
		'dashed'
	]

	markers = [
		'D',
		's',
		'^',
		'v',
		'x',
	]

	colors = [
		'#9acdc4',
		'#05348b',
		'#ae1908',
		'#ec813b',
		'#e5a84b',
	]

	fig = plt.figure(figsize=(6, 6))
	ax1 = fig.add_subplot(111)
	plt.subplots_adjust(top=0.98, bottom=0.12, left=0.17, right=0.98)
	ax1.set_ylabel(r"$\|\hat{\beta} - \beta^\star\|_2^2$", fontsize=22)

	for (j, mid) in enumerate(method_idx):
		metric = []
		for i in range(len(vec_n)):
			measures = []
			for k in range(num_sml):
				error = np.sum(np.square(results[i, k, mid+1, :] - results[i, k, 0, :]))
				if error > 0.2 and mid != 3:
					print(f'method = {mid}, n = {vec_n[i]}, seed = {k}, error = {error}')
				measures.append(np.sum(np.square(results[i, k, mid+1, :] - results[i, k, 0, :])))
			metric.append(np.median(measures))
		ax1.plot(vec_n, metric, linestyle=lines[j], marker=markers[j], label=method_name[j], color=colors[j])
	ax1.set_yscale("log")
	ax1.set_xscale("log")
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	ax1.set_xlabel('$n$', fontsize=22)

	ax1.legend(loc='best')
	plt.show()



if args.exp_id == 2:
	results = np.load('saved_results/unit_test_2.npy')

	num_n = results.shape[0]
	num_sml = results.shape[1]

	vec_n = [50, 100, 300, 500, 800, 1000]
	method_name = ['EILLS', "FAIR-BF", r"FAIR-RF", "Oracle", r"IRM", r"Anchor", r"ICP", r"Pool-LS"]
	method_idx = [0, 1, 8, 3, 4, 5, 6, 7]

	lines = [
		'solid',
		'solid',
		'solid',
		'dashed',
		'dotted',
		'dotted',
		'dotted',
		'dashed',
	]

	markers = [
		'>',
		's',
		'D',
		'*',
		'+',
		'o',
		'v',
		'x',
	]

	colors = [
		'#9acdc4',
		'#05348b',
		'#05348b',
		'#ae1908',
		'#ec813b',
		'#e5a84b',
		'#008000',
		'gray',
	]

	fig = plt.figure(figsize=(6, 6))
	ax1 = fig.add_subplot(111)
	plt.subplots_adjust(top=0.98, bottom=0.12, left=0.17, right=0.98)
	ax1.set_ylabel(r"$\|\hat{\beta} - \beta^\star\|_2^2$", fontsize=22)

	for (j, mid) in enumerate(method_idx):
		metric = []
		for i in range(len(vec_n)):
			measures = []
			for k in range(num_sml):
				error = np.sum(np.square(results[i, k, mid+1, :] - results[i, k, 0, :]))
				measures.append(error)
			if i > 0:
				metric.append(np.median(measures))
		ax1.plot(np.array(vec_n)[1:], metric, linestyle=lines[j], marker=markers[j], label=method_name[j], color=colors[j])

	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	ax1.set_xlabel('$n$', fontsize=22)
	ax1.set_yscale("log")
	ax1.set_xscale("log")

	ax1.legend(loc='upper left')
	plt.show()


if args.exp_id == 3:
	results = np.load('saved_results/unit_test_3.npy')
	num_n = results.shape[0]
	num_sml = results.shape[1]

	vec_n = [1000, 2000, 4000, 8000]
	method_name = ["FAIR-GB", "FAIR-RF", "Oracle", "Pool-LS"]
	method_idx = [2, 3, 0, 1]

	lines = [
		'solid',
		'solid',
		'dashed',
		'dashed'
	]

	markers = [
		'D',
		's',
		'*',
		'x',
	]

	colors = [
		'#9acdc4',
		'#05348b',
		'#ae1908',
		'#e5a84b',
	]

	fig = plt.figure(figsize=(6, 6))
	ax1 = fig.add_subplot(111)
	plt.subplots_adjust(top=0.98, bottom=0.12, left=0.17, right=0.98)
	ax1.set_ylabel(r"$\widehat{\mathtt{MSE}}$", fontsize=22)

	for (j, mid) in enumerate(method_idx):
		metric = []
		for i in range(len(vec_n)):
			measures = []
			for k in range(num_sml):
				error = results[i, k, mid]
				if error > 0.2 and mid != 3:
					print(f'method = {mid}, n = {vec_n[i]}, seed = {k}, error = {error}')
				measures.append(error)
			
			metric += [np.median(measures)]
		ax1.plot(vec_n, metric, linestyle=lines[j], marker=markers[j], label=method_name[j], color=colors[j])

	plt.xticks(ticks=[1000, 2000, 4000, 8000], fontsize=20)
	ax1.set_xlabel('$n$', fontsize=22)
	#ax1.set_yscale("log")
	#ax1.set_xscale("log")
	ax1.set_yticks(np.array([1e-2, 1e-1, 0.2, 0.4, 0.5]))

	ax1.legend(loc='best')
	plt.show()



if args.exp_id == 4:
	results = np.load('saved_results/unit_test_4.npy')
	num_n = results.shape[0]
	num_sml = results.shape[1]

	#vec_n = [1000, 2000, 3000, 5000, 10000]
	vec_n = [1000, 2000, 4000, 8000]

	method_name = ["FAIR-GB", "FAIR-RF", "Oracle", "Pool-LS"]
	method_idx = [2, 3, 0, 1]

	lines = [
		'solid',
		'solid',
		'dashed',
		'dashed'
	]

	markers = [
		'D',
		's',
		'*',
		'x',
	]

	colors = [
		'#9acdc4',
		'#05348b',
		'#ae1908',
		'#e5a84b',
	]

	fig = plt.figure(figsize=(6, 6))
	ax1 = fig.add_subplot(111)
	plt.subplots_adjust(top=0.98, bottom=0.12, left=0.17, right=0.98)
	ax1.set_ylabel(r"$\widehat{\mathtt{MSE}}$", fontsize=22)

	for (j, mid) in enumerate(method_idx):
		metric = []
		for i in range(len(vec_n)):
			measures = []
			for k in range(num_sml):

				error = results[i, k, mid]
				if error > 0.2 and mid != 3:
					print(f'method = {mid}, n = {vec_n[i]}, seed = {k}, error = {error}')
				measures.append(error)
			
			metric += [np.median(measures)]
		#if mid == 3:
		#	metric[0] = 0.2635995550394437
		ax1.plot(np.array(vec_n), np.array(metric), linestyle=lines[j], marker=markers[j], label=method_name[j], color=colors[j])

	#plt.xticks(ticks=[1000, 3000, 5000], fontsize=20)
	ax1.set_xlabel('$n$', fontsize=22)
	#ax1.set_yscale("log")
	#ax1.set_xscale("log")
	ax1.set_yticks(np.array([4e-2, 1e-1, 0.5, 1.0]))

	ax1.legend(loc='best')
	plt.show()

