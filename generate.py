import os
import numpy as np
import random
import pandas as pd
from PIL import Image
from tqdm import tqdm
from dataset_utils import crop_and_resize, combine_and_mask
import torch
import torch.nn as nn
from torchvision import models,transforms
import random
import matplotlib.pyplot as plt

class ft_net(nn.Module):

    def __init__(self):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft

    def forward(self, x):
        if True: # draw features or not
            x = self.model.conv1(x)

            x = self.model.bn1(x)

            x = self.model.relu(x)

            x = self.model.maxpool(x)

            x = self.model.layer1(x)

            x = self.model.layer2(x)

            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)

            x = x.view(x.size(0), -1)
           # x = self.model.fc(x)
            

        return x

def split(l,rate):
    n_total=len(l)
    offset=int(n_total*rate)
    random.shuffle(l)
    train=l[:offset]
    test=l[offset:]
    return train,test

#generate spilt result and image 
def generate(r_water,r_land,n_sample_train,n_sample_test,splitrate=0.7,model_name='resnet50',cub_dir = './CUB',places_dir = './data_large'):
    #r_water: 水鸟在水环境的概率
    #r_land: 陆鸟在陆环境的概率
    #model_name: resnet50, resnet34, wideresnet50
    
    

    target_places = [
        ['bamboo_forest', 'forest/broadleaf'],  # Land backgrounds
        ['ocean', 'lake/natural']]              # Water backgrounds
    ######################################################################################

    images_path = os.path.join(cub_dir, 'images.txt')

    df = pd.read_csv(
        images_path,
        sep=" ",
        header=None,
        names=['img_id', 'img_filename'],
        index_col='img_id')

    ### Set up labels of waterbirds vs. landbirds
    # We consider water birds = seabirds and waterfowl.
    water_birds_list = [
        'Albatross', # Seabirds
        'Auklet',
        'Cormorant',
        'Frigatebird',
        'Fulmar',
        'Gull',
        'Jaeger',
        'Kittiwake',
        'Pelican',
        'Puffin',
        'Tern',
        'Gadwall', # Waterfowl
        'Grebe',
        'Mallard',
        'Merganser',
        'Guillemot',
        'Pacific_Loon'
    ]

    wb=[]
    lb=[]
    for species_name in df['img_filename']:
        iswater=0
        species_name_new=species_name.split('/')[0].split('.')[1].lower()
        for water_bird in water_birds_list:
            if water_bird.lower() in species_name_new:
                wb.append(species_name)
                iswater=1
        if iswater==0:
            lb.append(species_name)
    train_lb,test_lb=split(lb,splitrate)
    train_wb,test_wb=split(wb,splitrate)
    with open('trainlb.txt','w') as f:
          for i in train_lb:
                f.write(i +'\n')
    with open('testlb.txt','w') as f:
          for i in test_lb:
                f.write(i +'\n')
    with open('trainwb.txt','w') as f:
          for i in train_wb:
                f.write(i +'\n')
    with open('testwb.txt','w') as f:
          for i in test_wb:
                f.write(i +'\n')
    
    if model_name=='resnet50':
        resnet=ft_net()
    elif model_name=='resnet34':
        resnet=models.resnet34(pretrained=True)
    elif model_name=='wideresnet50':
        resnet=models.wide_resnet50_2(pretrained=True)
    save_freq=50
    images=torch.zeros(save_freq,3,280,280)
    x=np.zeros((n_sample_train,2048))
    y=np.zeros(n_sample_train)
    z=np.zeros(n_sample_train)
    transform=transforms.Compose([transforms.ToTensor()])
    for i in range(n_sample_train):
        # Load bird image and segmentation
        iswater = np.random.binomial(1,0.5,1)
        if iswater[0]==1:
            bird_name=random.choice(train_wb)
        else:
            bird_name=random.choice(train_lb)
        img_path = os.path.join(cub_dir, 'images', bird_name)
        seg_path = os.path.join(cub_dir, 'segmentations', bird_name.replace('.jpg','.png'))
        img_np = np.asarray(Image.open(img_path).convert('RGB'))
        seg_np = np.asarray(Image.open(seg_path).convert('RGB')) / 255

        # Load place background
        # Skip front /
        #print(df.loc[i])
        if iswater[0]==1:
            background= np.random.binomial(1,r_water,1)
        else:
            background= np.random.binomial(1,1-r_land,1)
        background_path=random.choice(target_places[background[0]])
        num_background=str(random.randint(0,4999)+1).rjust(8,'0')+".jpg"

        place_path = os.path.join(places_dir, background_path[0],background_path,num_background)
        place = Image.open(place_path).convert('RGB')

        img_black = Image.fromarray(np.around(img_np * seg_np).astype(np.uint8))
        combined_img = combine_and_mask(place, seg_np, img_black)
        combined_img = transform(combined_img.resize((280,280)))
        plt.imshow(combined_img)
        plt.axis('off') 
        plt.show()
        #y[i]=iswater[0]  #0: land 1: water
        #z[i]=background[0] #0: land 1:water
        #images[i]=combined_img
        images[i%save_freq]=combined_img
        y[i]=iswater[0]
        z[i]=background[0]
        if i%save_freq==(save_freq-1):
            feature=resnet(images)
            x[i-save_freq+1:i+1]=feature.detach().numpy()
        print("iter:",i)
    #file.close()
    np.save(f'./res/train/rwater_{r_water}_rland_{r_land}_x.npy',x)
    np.save(f'./res/train/rwater_{r_water}_rland_{r_land}_y.npy',y)
    np.save(f'./res/train/rwater_{r_water}_rland_{r_land}_z.npy',z)

    images=torch.zeros(save_freq,3,280,280)
    x=np.zeros((n_sample_test,2048))
    y=np.zeros(n_sample_test)
    z=np.zeros(n_sample_test)
    for i in range(n_sample_test):
        # Load bird image and segmentation
        iswater = np.random.binomial(1,0.5,1)
        if iswater[0]==1:
            bird_name=random.choice(test_wb)
        else:
            bird_name=random.choice(test_lb)
        img_path = os.path.join(cub_dir, 'images', bird_name)
        seg_path = os.path.join(cub_dir, 'segmentations', bird_name.replace('.jpg','.png'))
        img_np = np.asarray(Image.open(img_path).convert('RGB'))
        seg_np = np.asarray(Image.open(seg_path).convert('RGB')) / 255

        # Load place background
        # Skip front /
        #print(df.loc[i])
        if iswater[0]==1:
            background= np.random.binomial(1,0.5,1)
        else:
            background= np.random.binomial(1,0.5,1)
        background_path=random.choice(target_places[background[0]])
        num_background=str(random.randint(0,4999)+1).rjust(8,'0')+".jpg"

        place_path = os.path.join(places_dir, background_path[0],background_path,num_background)
        place = Image.open(place_path).convert('RGB')

        img_black = Image.fromarray(np.around(img_np * seg_np).astype(np.uint8))
        combined_img = combine_and_mask(place, seg_np, img_black)
        combined_img = transform(combined_img.resize((280,280)))
        images[i%save_freq]=combined_img
        y[i]=iswater[0]
        z[i]=background[0]
        if i%save_freq==(save_freq-1):
            feature=resnet(images)
            x[i-save_freq+1:i+1]=feature.detach().numpy()
        print("iter:",i)
    #file.close()
    np.save(f'./res/test/rwater_{r_water}_rland_{r_land}_x.npy',x)
    np.save(f'./res/test/rwater_{r_water}_rland_{r_land}_y.npy',y)
    np.save(f'./res/test/rwater_{r_water}_rland_{r_land}_z.npy',z)

    #x=resnet(images)
    return x,y,z


def generate_train(r_water,r_land,n_sample_train,model_name='resnet50',cub_dir = './CUB',places_dir = './data_large'):
    #r_water: 水鸟在水环境的概率
    #r_land: 陆鸟在陆环境的概率
    #model_name: resnet50, resnet34, wideresnet50
    
    

    target_places = [
        ['bamboo_forest', 'forest/broadleaf'],  # Land backgrounds
        ['ocean', 'lake/natural']]              # Water backgrounds
    ######################################################################################

    images_path = os.path.join(cub_dir, 'images.txt')

    df = pd.read_csv(
        images_path,
        sep=" ",
        header=None,
        names=['img_id', 'img_filename'],
        index_col='img_id')
    
    train_wb=[]
    train_lb=[]
    with open('trainlb.txt','r') as f:
        for line in f:
            line=line.strip('\n')#删除换行符
            train_lb.append(line)

    with open('trainwb.txt','r') as f:
          for line in f:
            line=line.strip('\n')#删除换行符
            train_wb.append(line)
    
    if model_name=='resnet50':
        resnet=ft_net()
    elif model_name=='resnet34':
        resnet=models.resnet34(pretrained=True)
    elif model_name=='wideresnet50':
        resnet=models.wide_resnet50_2(pretrained=True)
    save_freq=50
    images=torch.zeros(save_freq,3,280,280)
    x=np.zeros((n_sample_train,2048))
    y=np.zeros(n_sample_train)
    z=np.zeros(n_sample_train)
    transform=transforms.Compose([transforms.ToTensor()])
    for i in range(n_sample_train):
        # Load bird image and segmentation
        iswater = np.random.binomial(1,0.5,1)
        if iswater[0]==1:
            bird_name=random.choice(train_wb)
        else:
            bird_name=random.choice(train_lb)
        img_path = os.path.join(cub_dir, 'images', bird_name)
        seg_path = os.path.join(cub_dir, 'segmentations', bird_name.replace('.jpg','.png'))
        img_np = np.asarray(Image.open(img_path).convert('RGB'))
        seg_np = np.asarray(Image.open(seg_path).convert('RGB')) / 255

        # Load place background
        # Skip front /
        #print(df.loc[i])
        if iswater[0]==1:
            background= np.random.binomial(1,r_water,1)
        else:
            background= np.random.binomial(1,1-r_land,1)
        background_path=random.choice(target_places[background[0]])
        num_background=str(random.randint(0,4999)+1).rjust(8,'0')+".jpg"

        place_path = os.path.join(places_dir, background_path[0],background_path,num_background)
        place = Image.open(place_path).convert('RGB')

        img_black = Image.fromarray(np.around(img_np * seg_np).astype(np.uint8))
        combined_img = combine_and_mask(place, seg_np, img_black)
        combined_img = transform(combined_img.resize((280,280)))
        images[i%save_freq]=combined_img
        y[i]=iswater[0]
        z[i]=background[0]
        if i%save_freq==(save_freq-1):
            feature=resnet(images)
            x[i-save_freq+1:i+1]=feature.detach().numpy()
        print("iter:",i)
    #file.close()
    np.save(f'./res/train/rwater_{r_water}_rland_{r_land}_x.npy',x)
    np.save(f'./res/train/rwater_{r_water}_rland_{r_land}_y.npy',y)
    np.save(f'./res/train/rwater_{r_water}_rland_{r_land}_z.npy',z)
    return x,y,z

def generate_test(r_water,r_land,n_sample_train,model_name='resnet50',cub_dir = './CUB',places_dir = './data_large'):
    #r_water: 水鸟在水环境的概率
    #r_land: 陆鸟在陆环境的概率
    #model_name: resnet50, resnet34, wideresnet50
    
    

    target_places = [
        ['bamboo_forest', 'forest/broadleaf'],  # Land backgrounds
        ['ocean', 'lake/natural']]              # Water backgrounds
    ######################################################################################

    images_path = os.path.join(cub_dir, 'images.txt')

    df = pd.read_csv(
        images_path,
        sep=" ",
        header=None,
        names=['img_id', 'img_filename'],
        index_col='img_id')
    
    train_wb=[]
    train_lb=[]
    with open('testlb.txt','r') as f:
        for line in f:
            line=line.strip('\n')#删除换行符
            train_lb.append(line)

    with open('testwb.txt','r') as f:
          for line in f:
            line=line.strip('\n')#删除换行符
            train_wb.append(line)
    
    if model_name=='resnet50':
        resnet=ft_net()
    elif model_name=='resnet34':
        resnet=models.resnet34(pretrained=True)
    elif model_name=='wideresnet50':
        resnet=models.wide_resnet50_2(pretrained=True)
    save_freq=50
    images=torch.zeros(save_freq,3,280,280)
    x=np.zeros((n_sample_train,2048))
    y=np.zeros(n_sample_train)
    z=np.zeros(n_sample_train)
    transform=transforms.Compose([transforms.ToTensor()])
    for i in range(n_sample_train):
        # Load bird image and segmentation
        iswater = np.random.binomial(1,0.5,1)
        if iswater[0]==1:
            bird_name=random.choice(train_wb)
        else:
            bird_name=random.choice(train_lb)
        img_path = os.path.join(cub_dir, 'images', bird_name)
        seg_path = os.path.join(cub_dir, 'segmentations', bird_name.replace('.jpg','.png'))
        img_np = np.asarray(Image.open(img_path).convert('RGB'))
        seg_np = np.asarray(Image.open(seg_path).convert('RGB')) / 255

        # Load place background
        # Skip front /
        #print(df.loc[i])
        if iswater[0]==1:
            background= np.random.binomial(1,r_water,1)
        else:
            background= np.random.binomial(1,1-r_land,1)
        background_path=random.choice(target_places[background[0]])
        num_background=str(random.randint(0,4999)+1).rjust(8,'0')+".jpg"

        place_path = os.path.join(places_dir, background_path[0],background_path,num_background)
        place = Image.open(place_path).convert('RGB')

        img_black = Image.fromarray(np.around(img_np * seg_np).astype(np.uint8))
        combined_img = combine_and_mask(place, seg_np, img_black)
        combined_img = transform(combined_img.resize((280,280)))
        images[i%save_freq]=combined_img
        y[i]=iswater[0]
        z[i]=background[0]
        if i%save_freq==(save_freq-1):
            feature=resnet(images)
            x[i-save_freq+1:i+1]=feature.detach().numpy()
        print("iter:",i)
    #file.close()
    np.save(f'./res/test/rwater_{r_water}_rland_{r_land}_x.npy',x)
    np.save(f'./res/test/rwater_{r_water}_rland_{r_land}_y.npy',y)
    np.save(f'./res/test/rwater_{r_water}_rland_{r_land}_z.npy',z)

    #x=resnet(images)
    return x,y,z
mode='train'
rwater=0.75
rland=0.7
if mode=='generate':
    x0,y0,z0=generate_train(rwater,rland,30000)
elif mode=='train':
    x0,y0,z0=generate_train(rwater,rland,30000)
elif mode=='test':
    x0,y0,z0=generate_train(rwater,rland,30000)    
print(x0,y0,z0)