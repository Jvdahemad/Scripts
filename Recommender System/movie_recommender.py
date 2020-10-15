# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 10:35:17 2020

@author: Javed
"""

"""
A simple restricted boltzman Machine to make binary classification on whether a
user will like a movie or not based on his previous ratings.

The Data is taken from gouplens.org 

Restricted Boltzman Machine is a undirected graphical model. It is 'Restricted'
in a sense that unlike Boltzman Machine it is bipartite i.e. the hidden and visible
nodes are connected to each other but not among themselves. This makes the visible/hidden
nodes independent from each other which in turn siplify the calculations.

The architecture of this RBM is simple and contains just 1 layer of FFNN. We try
to calulate the hidden nodes based on some random weights and use the same weights
to reconstruct the visible nodes. It is an energy based model and the optimization 
is carried out by attaining a state of low energy. 

The energy in our model will be defined by the weights of our model. The goal is 
to use Contrastive Divergence to minimize the negative of log liklihood function.

Our model will attain a steady state the weights can reconstruct the visible nodes
to exactly match our observed data.

The loss can be calculated based on the euclidian distance between the initial and
final states.
"""

import numpy as np
import pandas as pd
import scipy.stats as st

## Schemas of our movie database which later can be used to identify movie/user details

movies=pd.read_csv("ml-1m/movies.dat",sep='::',header=None,engine='python',encoding='latin-1')
users=pd.read_csv("ml-1m/users.dat",sep='::',header=None,engine='python',encoding='latin-1')
ratings=pd.read_csv("ml-1m/ratings.dat",sep='::',header=None,engine='python',encoding='latin-1')


## Reading test and train data:

#training_set=pd.read_csv("ml-100k/u1.base",header=None,delimiter= '\t')
#test_set=pd.read_csv("ml-100k/u1.test",header=None,delimiter= '\t')


training_set=pd.read_csv("ml-1m/training_set.csv",delimiter= '\t')
test_set=pd.read_csv("ml-1m/test_set.csv",delimiter= '\t')

"""
## Let's see the data, Our data has 4 columns, the last one is the time stamp 
and irrelavant here.
Col 1 has user ID
Col 2 has the movie ID
and Col 3 is the ratings given by the user to the movie

"""

training_set=training_set.values
test_set=test_set.values

def data_prep(dataset):  ## To sepeate the different columns in our dataset
    data=list(map(lambda x:x[0].split(","),dataset))
    data=np.array(data)
    data=data.astype(int)
    
    return data


def convert_data(data): ## Changing it to required format(cross tabulation).
    new_data=[]
    for i in range(1,nuser+1):
        movies_id=data[data[:,0]==i][:,1]
        rating_id=data[data[:,0]==i][:,2]
        ratings=np.zeros(nmovies)
        
        ratings[movies_id-1]=rating_id
            
        new_data.append(list(ratings))
        
    return np.array(new_data)



training_set=data_prep(training_set)
test_set=data_prep(test_set)
## We need data in a clean tabular form, so let's make our data set.

nuser=max(np.max(training_set[:,0]),np.max(test_set[:,0]))
nmovies=max(np.max(training_set[:,1]),np.max(test_set[:,1]))

print('# of Users: {} \n # of Movies: {}'.format(nuser,nmovies))
      

training_set=convert_data(training_set)
test_set=convert_data(test_set)

## Coverting the dataset into binary values for given ratings if the user likes or dislikes the movie

training_set[training_set==0]=-1
training_set[training_set==1]=0
training_set[training_set==2]=0
training_set[training_set>2]=1 ##Rating above 2 means the user likes the movie
    
test_set[test_set==0]=-1
test_set[test_set==1]=0
test_set[test_set==2]=0
test_set[test_set>2]=1






class RBM():
    
    def __init__(self,m,n):
        self.W=np.random.rand(m,n) ##Weights defining the connection from visible to hidden state nodes
        self.a=np.random.rand(1,m) ## Bias term for hidden nodes
        self.b=np.random.rand(1,n) ##Bias term for visible nodes
        
    def sigmoid(self,x):        ## Gives us a probalities of the p_h_given_v
        return 1/(1+np.exp(-x))
        
    def sample_h(self,v):
        wv=np.matmul(v,self.W.transpose())+self.a
        p_h_given_v=self.sigmoid(wv)
        
        return p_h_given_v, st.bernoulli.rvs(p_h_given_v)   ##Sampling 0 and 1s based on p_h_given_v
    
    def sample_v(self,h):
        wh=np.matmul(h,self.W)+self.b
        p_v_given_h=self.sigmoid(wh)
        
        return p_v_given_h, st.bernoulli.rvs(p_v_given_h)   ##Sampling 0 and 1s based on p_h_given_v
    
    def cont_div(self,v0,ph0,vk,phk):### updating the weights based on distance(energy) between final and initial state
        self.W=self.W + (np.matmul(ph0.transpose(),v0)-np.matmul(phk.transpose(),vk))
        self.a=self.a +  np.sum(ph0-phk,axis=0) 
        self.b=self.b +  np.sum((v0-vk),axis=0)  
        
        
        
        
n=training_set.shape[1]
batch_size=2**8 ##Setting up mini-batches
m=2**10 ##Number of hidden units

rbm=RBM(m,n)

nb=int(np.round(training_set.shape[0]/batch_size)+1)
### Training

epoch=10

for ep in range(1,epoch+1):
    
    training_loss=0
    s=0
    for i in range(1,nb+1):
        vk=training_set[(i-1)*batch_size:batch_size*i]
        ph0,hk=rbm.sample_h(vk)
        v0=training_set[(i-1)*batch_size:batch_size*i]
        
        for k in range(5):
            
            _,hk=rbm.sample_h(vk)
            _,vk=rbm.sample_v(hk)
            vk[v0<0]=v0[v0<0]
        phk,_=rbm.sample_h(vk)
        rbm.cont_div(v0,ph0,vk,phk)
        if len(vk[v0>=0])>0:
            training_loss=training_loss+np.mean(np.abs(vk[v0>=0]-v0[v0>=0]))
            s=s+1
    print("Training Error after {} epoch is {}".format(ep,training_loss/s)) ##0.2291808093968085
    


## Lets check how the trained weight perform in unseen dev set.

test_loss=0
s2=0
nb_test=int(np.round(test_set.shape[0]/batch_size)+1)
final_system=np.ones(test_set.shape)*-1
for i in range(1,nb_test+1):
 
    
    v0=test_set[(i-1)*batch_size:batch_size*i]
    
    _,hk=rbm.sample_h(v0)
    
    _,vk=rbm.sample_v(hk)

    #vk[v0<0]=v0[v0<0]
    final_system[(i-1)*batch_size:batch_size*i]=vk
    if (len(v0[v0>=0])>0):  ## To make sure we don't take into account the batches without any user reviews.

        loss=np.mean(np.abs(vk[v0>=0]-v0[v0>=0]))
        test_loss=test_loss+loss
        s2=s2+1
print("Test Loss {} ".format(test_loss/s2)) 




user_id=2101
Past_record=test_set[user_id-1]
print("Movies that user {} liked".format(user_id))

already_liked=list(np.where(Past_record==1)[0]+1) ##adding one as the 0th index corresponds to User ID:1
already_disliked=list(np.where(Past_record==0)[0]+1)


user_detail=final_system[user_id-1]
movies_liked=np.where(user_detail==1)
ind=list(movies_liked[0]+1)

for elem in (already_disliked+already_liked):
    try:
        ind.remove(elem)
    except:
        print("Movie_ID {} not present in our model's prediction".format(elem))
        pass

print("List of movies that we can recommend to user {} are".format(user_id))
movies[movies[0].isin(ind)]
