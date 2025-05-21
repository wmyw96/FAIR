from methods.fair_gumbel import *
import numpy as np
import torch
import torch.nn.modules as nn
import argparse
import time
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso,LogisticRegression
from methods.dro.run_dro import run_dro
from methods.irm import run_irm
import sys
import csv

from methods.fair_algo import *
import logging 

print(torch.cuda.device_count())
class FairLinearClassification(FairModel):
	def __init__(self, num_envs, dim_x):
		super(FairLinearClassification, self).__init__(num_envs, dim_x)
		self.g = nn.Sequential(nn.Linear(dim_x, 1), nn.Sigmoid())
		self.fs = []
		for e in range(num_envs):
			fe = nn.Linear(dim_x, 1)
			self.fs.append(fe)

	def parameters_g(self, log=False):
		if log:
			print(f'FairLinearClassification Predictor Module Parameters:')
			for para in self.g.parameters():
				print(f'Parameter Shape = {para.shape}')
		return self.g.parameters()

	def parameters_f(self, log=False):
		paras = []
		for i in range(self.num_envs):
			if log:
				print(f'FairLinearClassification Discriminator ({i}) Module Parameters:')
				for para in self.fs[i].parameters():
					print(f'Parameter Shape = {para.shape}')				
			paras += self.fs[i].parameters()
		return paras

	def forward(self, xs, pred=False):
		if pred:
			return self.g(xs)
		else:
			out_g = self.g(torch.cat(xs, 0))
			out_fs = []
			for e in range(self.num_envs):
				out_fs.append(self.fs[e](xs[e]))
			out_f = torch.cat(out_fs, 0)
			return out_g, out_f

def acc(traindata,testdata):
    n = len(traindata)
    num = 0
    for i in range(n):
          if(abs(traindata[i]-testdata[i])<0.5):
             num=num+1
    print(num/n)

def bce_loss(out_prob, cat_y):
	return -0.5 * torch.mean((cat_y * torch.log(out_prob + 1e-9) + (1 - cat_y) * torch.log(1 - out_prob + 1e-9)))

def misclass(pred_y, y):
	return 1 - np.mean((pred_y >= 0.5) * y + (pred_y < 0.5) * (1 - y))
    
def main():
    parser = argparse.ArgumentParser()

    # Settings
    parser.add_argument( '--method', choices=['LASSO','FAIR','GroupDRO','IRM'], default='FAIR')
    parser.add_argument( '--nstart', type=int, default=10)
    parser.add_argument( '--sample', type=int, default=30000)
    args = parser.parse_args()   
    mode = args.method
    nstart = args.nstart
    sample = args.sample
    r_water_list=[0.95,0.75,0.5]
    r_land_list=[0.9,0.7,0.5]
    x=[]
    y=[]
    z=[]
    pca = PCA(n_components=500)
    #sys.stdout=open('log.txt','w')
    for i in range(len(r_water_list)):
        r_water=r_water_list[i]
        r_land=r_land_list[i]
        x.append(np.load(f'./res/train/rwater_{r_water}_rland_{r_land}_x.npy'))
        y.append(np.load(f'./res/train/rwater_{r_water}_rland_{r_land}_y.npy').reshape(-1,1))
        z.append(np.load(f'./res/train/rwater_{r_water}_rland_{r_land}_z.npy'))
    pincipal_componets = pca.fit_transform(x[2])
    w = pca.components_
    x[0] = np.dot(x[0]-np.mean(x[0],axis=0),w.T)
    x[1] = np.dot(x[1]-np.mean(x[1],axis=0),w.T)
    x[2] = pincipal_componets
    x_test=np.load(f'./res/test/rwater_0.02_rland_0.02_x.npy')
    y_test=np.load(f'./res/test/rwater_0.02_rland_0.02_y.npy').reshape(-1,1)
    z_test=np.load(f'./res/test/rwater_0.02_rland_0.02_z.npy')
    x_test= np.dot(x_test-np.mean(x_test,axis=0),w.T)

    x_cat = np.concatenate((x[0],x[1]))
    y_cat = np.concatenate((y[0],y[1]))
	


    if(mode=='LASSO'):
    #baseline LASSO regression
        for i in range(nstart):
            l75=Lasso(alpha=0.001,max_iter=1000)
            l75.fit(x[1],y[1])

            l_star=Lasso(alpha=0.001,max_iter=1000)
            l_env=Lasso(alpha=0.001)
            l_star.fit(x[2],y[2])
            l_env.fit(x_cat,y_cat)

            y0=l_star.predict(x_test)
            print("accuracy of env*:")
            acc(y0,y_test)
            y0=l75.predict(x_test)
            print("accuracy of env2:")
            acc(y0,y_test)
            y0=l_env.predict(x_test)
            print("accuracy of env1+env2:")
            acc(y0,y_test)
    elif(mode=='FAIR'):
        hyper_params= {
            'gumbel_lr': 1e-3, 'model_lr': 1e-2, 'batch_size': 4000, 'niters': 20000,
            'weight_decay_g': 0, 'weight_decay_f': 0,
            'diters': 3, 'giters': 1, 
            'init_temp': 5, 'final_temp': 0.1, 'offset': -1, 'anneal_iter': 100, 'anneal_rate': 0.993,
            }
        eval_freq=1000
        res=np.zeros((nstart,hyper_params['niters']//eval_freq))
        for i in range(nstart):
            print("restart:",i)
            random_row=np.random.choice(x[0].shape[0],size=sample,replace=False)
            valid=[ torch.from_numpy(x_test).to(torch.float32).cuda()],[torch.from_numpy(y_test).view(-1,1).cuda()]	
            test=[ torch.from_numpy(x_test).to(torch.float32).cuda()],[torch.from_numpy(y_test).view(-1,1).cuda()]
            model = FairLinearClassification(2, 500)
            algo = FairGumbelAlgo(num_envs=2, dim_x=500, model=model, gamma=100, loss=bce_loss, hyper_params=hyper_params)
            packs = algo.run_gumbel(([x[0][random_row],x[1][random_row]],
            [y[0][random_row],y[1][random_row]]), 
            eval_metric=misclass, me_valid_data=valid, me_test_data=test, eval_iter=eval_freq, log=True)
            res[i]=packs['loss_rec']
        with open('waterbird_testfreq.csv','w',newline='') as f:
            writer=csv.writer(f)
            writer.writerows(res)
    elif(mode=='GroupDRO'):
        for i in range(nstart):
            run_dro()
    elif(mode=='IRM'):
        for i in range(nstart):
                print("restart:",i)
                random_row=np.random.choice(x[0].shape[0],size=sample,replace=False)
                run_irm(envs=[{'images': torch.from_numpy(x[0][random_row]).to(torch.float32).cuda(),'labels': torch.from_numpy(y[0][random_row]).view(-1,1).cuda()},
                {'images': torch.from_numpy(x[1][random_row]).to(torch.float32).cuda(),'labels': torch.from_numpy(y[1][random_row]).view(-1,1).cuda()},
                {'images': torch.from_numpy(x_test).to(torch.float32).cuda(),'labels': torch.from_numpy(y_test).view(-1,1).cuda()}])
    
if __name__ == "__main__":
    main()
