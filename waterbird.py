from methods.fair_gumbel_one import *
from methods.tools import pooled_least_squares
import numpy as np
import os
import argparse
import time
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso,LogisticRegression
from sympy import GramSchmidt,Matrix

def acc(traindata,testdata):
    n = len(traindata)
    num = 0
    for i in range(n):
          if(abs(traindata[i]-testdata[i])<0.5):
             num=num+1
    print(num/n)
    
r_water_list=[0.95,0.75,0.5]
r_land_list=[0.9,0.7,0.5]
x=[]
y=[]
z=[]
pca = PCA(n_components=500)
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

#baseline LASSO regression
l75=Lasso(alpha=0.001,max_iter=1000)
l75.fit(x[1],y[1])

l_star=Lasso(alpha=0.001,max_iter=1000)
l_env=Lasso(alpha=0.001)
l_star.fit(x[2],y[2])
l_env.fit(x_cat,y_cat)

y0=l75.predict(x_test)
acc(y0,y_test)

packs = fair_ll_classification_sgd_gumbel_uni(x[0:2], y[0:2],(x_test,y_test),hyper_gamma=200, learning_rate=1e-2,niters=2000,log=True,)
