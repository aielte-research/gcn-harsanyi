from torch_geometric.datasets import TUDataset
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchsummary import summary
import torch.nn.functional as F
from torch_geometric.transforms import TargetIndegree
import time
from torch_geometric.data import Data, DataLoader
import numpy.linalg as LA
import networkx as nx
from torch_geometric.utils import to_networkx

from model import Cheb
from training_loop import training_loop

device = torch.cuda.current_device()

dataset=TUDataset('~/data', 'PROTEINS', transform=TargetIndegree()) #saves normalized degree as edge_attr

dataset = dataset.shuffle()
train_dataset = dataset[150:]
test_dataset = dataset[:150]
train_loader=DataLoader(train_dataset,batch_size=100)
test_loader=DataLoader(test_dataset,batch_size=150)

print(f'Number of graphs: {len(dataset)}')
print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')
print(f'Number of classes: {dataset.num_classes}')
print(f'Number of node features: {dataset.num_node_features}')


MODEL_Cheb=Cheb(3,30,30).to(device)
OPTIMIZER =  torch.optim.Adam(MODEL_Cheb.parameters(), lr=0.003)   
EPOCHS = 50
Cheb_history = training_loop(EPOCHS, MODEL_Cheb, OPTIMIZER, device,train_loader,test_loader)

#MODEL_ARMA=ARMA(3,K=3,N=1, hidden=30).to(device)
#OPTIMIZER =  torch.optim.Adam(MODEL_ARMA.parameters(), lr=0.003)   
#EPOCHS = 50
#ARMA_history = training_loop(EPOCHS, MODEL_ARMA, OPTIMIZER, device,train_loader,test_loader)

#MODEL_Spline=Spline(3,hidden=30,kernel=10).to(device)
#OPTIMIZER =  torch.optim.Adam(MODEL_Spline.parameters(), lr=0.03)   
#EPOCHS = 50
#Spline_history = training_loop(EPOCHS, MODEL_Spline, OPTIMIZER, device,train_loader,test_loader)

