# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 13:22:48 2020

@author: Javed
"""


#importing the libraries

import numpy as np
import pandas as pd
import torch
##Getting the data

movies=pd.read_csv("ml-1m/movies.dat",sep='::',header=None,engine='python',encoding='latin-1')
users=pd.read_csv("ml-1m/users.dat",sep='::',header=None,engine='python',encoding='latin-1')
ratings=pd.read_csv("ml-1m/ratings.dat",sep='::',header=None,engine='python',encoding='latin-1')


##Getting the training set and testset

training_data=pd.read_csv("ml-100k/u1.base",delimiter='\t')
training_data=np.array(training_data,dtype='int')

test_data=pd.read_csv("ml-100k/u1.test",delimiter='\t')
test_data=np.array(test_data,dtype='int')


##Getting the number of movies and users
nusers=int(max(np.max(training_data[:,0]),np.max(test_data[:,0])))
nmovies=int(max(np.max(training_data[:,1]),np.max(test_data[:,1])))


##Preparing the data

def convert(data):
    new_data=[]
    for user in range(1,nusers+1):
        id_movies= data[data[:,0]==user][:,1]
        id_ratings= data[data[:,0]==user][:,2]
        ratings=np.zeros(nmovies)
        ratings[id_movies-1]=id_ratings
        new_data.append(list(ratings))
    return new_data

training_data=convert(training_data)
test_data=convert(test_data)

## Converting the test and train data to tensor

training_data=torch.FloatTensor(training_data)
test_data=torch.FloatTensor(test_data)


## Converting the test and train data to binary rating

training_data[training_data==0]=-1
training_data[training_data==1]=0
training_data[training_data==2]=0
training_data[training_data>=3]=1

test_data[test_data==0]=-1
test_data[test_data==1]=0
test_data[test_data==2]=0
test_data[test_data>=3]=1


## Creating Architechture of the RBM
class RBM():
    def __init__(self,nh,nv):
        self.W=torch.randn(nh,nv)
        self.a=torch.randn(1,nh)
        self.b=torch.randn(1,nv)
        
    def sample_h(self,x):
        wx=torch.mm(x,self.W.t())    #x is the visible node of dim ncxnv and W is weight with dim nhxnv
        activation=wx+self.a.expand_as(wx)
        p_h_given_v=torch.sigmoid(activation)
        return p_h_given_v,torch.bernoulli(p_h_given_v)
    
    def sample_v(self,y):
        wy=torch.mm(y,self.W)        #y is the hidden  nodes of dim ncxnh and W is weight with dim nhxnv
        activation=wy+self.b.expand_as(wy)
        p_v_given_h=torch.sigmoid(activation)
        return p_v_given_h,torch.bernoulli(p_v_given_h)
    
    def train(self, v0, vk, ph0, phk):
        self.W+=(torch.mm(v0.t(), ph0)-torch.mm(vk.t(), phk)).t()
        self.b+=torch.sum((v0-vk),0)
        self.a+=torch.sum((ph0-phk), 0)


nv=len(training_data[0])
nh=100
batch_size=100
rbm=RBM(nh,nv)


#traing the RBM

nb_epoch=10
for i in range(1,nb_epoch+1):
    training_loss=0
    s=0
    
    for user_id in range(0,nusers-batch_size,batch_size):
        vk=training_data[user_id:user_id+batch_size]
        v0=training_data[user_id:user_id+batch_size]
        ph0,_=rbm.sample_h(vk)
        
        for k in range(10):
            _,hk=rbm.sample_h(vk)
            _,vk=rbm.sample_v(hk)
            vk[v0<0]=v0[v0<0]
            
        phk,_=rbm.sample_h(vk)
        rbm.train(v0,vk,ph0,phk)
        training_loss+=torch.mean(torch.abs(v0[v0>0]-vk[v0>0]))
        s+=1
    print("epoch: "+str(i)+" training loss = "+ str(training_loss/s))
        
