
import numpy as np

import random
# For Experience Replay

import os

import torch
# torch can handle dynamic graph

import torch.nn as nn
# Neural Network Library

import torch.nn.functional as F
# Functional Module

import torch.optim as optim
# For Optimizers for SGD AND ALL

import torch.autograd as autograd
# for dynamic auto gradient (also graphs)

from torch.autograd import Variable
# For BackPropogation


# Creating the Neural Network
class Network(nn.Module):
    
    def __init__(self, num_input , num_actions):
        super(Network , self).__init__()
        self.input_size = num_input
        self.num_actions = num_actions
        self.layer1 = nn.Linear(self.input_size, 30);
        self.layer2 = nn.Linear(30, num_actions)
        
    def forward(self , state):
        x = F.relu(self.layer1(state))
        q_values= self.layer2(x)
        
        return q_values
        
    
# Now Experience Replay
 # MDP consists of series of states  so we need to break the corelation of subsequent states 
 # like driving on a straight road the car will not learn something useful
 # like learning curve so insteadof learning from subsequent states
 # we will learn of a batch of normalised states to break the co-relations

class ReplayMemory(object):
    def __init__(self , capacity=10000):
        self.capacity = capacity
        self.memory = []
        
    def push(self , event):
        # Event = [ state(t) , action[state(t)] , state(t+1) , reward ]
        self.memory.append(event)
        if len(self.memory)>self.capacity:
            self.memory.pop(0)
    
    def sample(self , batch_size):
        # after obtaining the random samples we just mapped the elements 
        # of svery sample to there indexes like a =  [[1,2,3], [4,5,6]]
        # now if i do a zip of [1,2,3] and [4,5,6] i.e zip([1,2,3],[4,5,6])
        # shortcut zip(*a) (* works as a spread operator i think) 
        # so the result will be [1,4],[2,5],[3,6] in a zip object
        samples = random.sample(self.memory, batch_size)
        batch_samples = zip(*samples)
        # Now We need autograd (pytorch tensor) i.e tensor , gradient
        # k = map(lambda x : x**2 , [1,2,3,4,5])
        # print(list(k)) => [1, 4, 9, 16, 25]
        # Note that each of them is a pytorch tensor like states and all
        # map(lambda x : Variable(torch.cat(x , 0)) , batch_samples)
        return map(self.mapfun , batch_samples)
    
    def mapfun(self , x):
        return Variable(torch.cat(x , 0))
    

    
# Implementing Deep Q Learning Process


class DQN():
    def __init__(self , num_input , num_actions , gamma):
        self.gamma = gamma
        self.rewards =[]
        self.model = Network(num_input, num_actions)
        self.memory = ReplayMemory(capacity=10000)
        self.optimizer = optim.Adagrad(self.model.parameters() , lr=0.001 )
        ls = [0]*num_input
        self.lastState = torch.tensor(ls).unsqueeze(0)
        # Added one extra dimension for batch calculation in network
        self.lastAction = 0
        self.lastReward = 0
        
        
    def select_action(self , state):
        # Without autograd for faster computation
        with torch.no_grad():
            q_values = self.model(state)
            probabilities = F.softmax(q_values * 10) # kind of like temperature variable but not exact
            action = probabilities.multinomial(1)
            
            return action.data[0,0]
        
    def learn(self , batch_states , batch_next_states , batch_rewards , batch_actions):
        # These are the samples created by sample function
        
        outputs =self.model(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
        # Read about gather . it simply selects the values
        next_outputs = self.model(batch_next_states).detach().max(1)[0]
        target = batch_rewards + self.gamma*next_outputs
        temporalDifferenceLoss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        temporalDifferenceLoss.backward()
        self.optimizer.step()
        
    def update(self , reward , new_signal):
        new_state = torch.tensor(new_signal , dtype=torch.float32).unsqueeze(0)
        self.memory.push((self.lastState , new_state , torch.tensor(self.lastReward).unsqueeze(0) , torch.LongTensor([int(self.lastAction )])))
        action = self.select_action(new_state)
        if len(self.memory.memory)>100:
            batch_state , batch_next_state , batch_rewards , batch_actions = self.memory.sample(100)
            self.learn(batch_state , batch_next_state , batch_rewards , batch_actions)
        self.lastAction = action
        self.lastState = new_state
        self.lastReward = reward
        self.rewards.append(reward)
        if len(self.rewards)>1000:
            self.rewards.pop(0)
        return action
    
    def score(self):
        return  sum(self.rewards)/(len(self.rewards) + 1)
    
    def save(self):
        torch.save({ 'stateDict' : self.model.state_dict() ,
                     'optimizer' : self.optimizer.state_dict()
                    } , 'lastBrain.pth') 
    def load(self):
        if os.path.isfile('lastBrain.pth'):
            print('=> Loading Checkpoint ...')
            checkpoint = torch.load('lastBrain.pth')
            self.model.load_state_dict(checkpoint['stateDict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> Model Loaded')
        else:
            print('No Checkpoint found ... ')
            
        
        
        
            
            
        
        
        
    
    
    
    
    
            
    








































