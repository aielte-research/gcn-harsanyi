from torch_geometric.nn import ARMAConv
from torch_geometric.nn import ChebConv
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import SplineConv

class ARMA(torch.nn.Module):
    def __init__(self,in_channel,hidden,K,N):
        super().__init__()
        self.conv1 = ARMAConv(in_channel,hidden,num_stacks=K, num_layers=N)
        self.bn1 = BatchNorm(hidden)
        self.dense1 = nn.Linear(hidden,2)
        self.relu = nn.ReLU()
        self.hidden=hidden
        self.history= None
        
    def forward(self,data):
        edge_index=data.edge_index
        h=data.x
        h = self.conv1(h, edge_index)
        h = self.bn1(h)
        h = self.relu(h)
        h = global_mean_pool(h, data.batch)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.dense1(h)
        return h


class Cheb(nn.Module):
    def __init__(self, in_channel,hidden,K):
        super().__init__()
        self.conv1 = ChebConv(in_channel,hidden,K)
        self.bn1 = BatchNorm(hidden)
        self.dense1 = nn.Linear(hidden,2)
        self.relu = nn.ReLU()
        self.hidden=hidden
        self.history= None
        
    def forward(self,data):
        edge_index=data.edge_index
        h=data.x
        h = self.conv1(h, edge_index)
        h = self.bn1(h)
        h = self.relu(h)
        h = global_mean_pool(h, data.batch)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.dense1(h)
        return h

class Spline(torch.nn.Module):
    def __init__(self,in_channel,hidden,kernel):
        super().__init__()
        self.conv1 = SplineConv(in_channel, hidden, dim=1, kernel_size=kernel)
        self.bn1 = BatchNorm(hidden)
        self.dense1 = torch.nn.Linear(hidden,2)

    def forward(self, data):
        h = F.relu(self.bn1(self.conv1(data.x, data.edge_index,data.edge_attr)))
        h = global_mean_pool(h, data.batch)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.dense1(h)
        return h