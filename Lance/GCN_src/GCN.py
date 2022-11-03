import os
import sys
main_Path   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));
data_Path   = os.path.join(main_Path, "data");
sys.path.append(data_Path);

# Now import the data extractor.
from data_extractor import LigandGraphIMDataset;
import torch
import torch.nn.functional as F
# from data.data_extractor import LigandGraphIMDataset
from torch_geometric.loader import DataLoader
import numpy as np
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn import GCNConv, global_add_pool, global_max_pool, GATConv, TopKPooling
import time

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(in_channels=19,out_channels=19)
        self.conv2 = GCNConv(in_channels=19,out_channels=19)
        self.conv3 = GCNConv(in_channels=19,out_channels=19)

        self.topK = TopKPooling(19,0.5)
        # self.conv4 = GCNConv(in_channels=19,out_channels=19)

        # self.gatConv1 = GATConv(in_channels=19,out_channels=19)
        # self.gatConv2 = GATConv(in_channels=19,out_channels=19)
        # self.gatConv3 = GATConv(in_channels=19,out_channels=19)
        
        # self.global_add_pool = global_add_pool
        self.global_max_pool = global_max_pool

        self.lin1 = torch.nn.Linear(in_features=57,out_features=32)
        self.lin2 = torch.nn.Linear(in_features=32,out_features=10)
        self.lin3 = torch.nn.Linear(in_features=10,out_features=1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x1 = global_add_pool(x,batch)

        clustered_graph = self.topK(x, edge_index, batch=batch)
        x, edge_index, batch = clustered_graph[0], clustered_graph[1], clustered_graph[3]
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x2 = global_add_pool(x,batch)

        clustered_graph = self.topK(x, edge_index, batch=batch)
        x, edge_index, batch = clustered_graph[0], clustered_graph[1], clustered_graph[3]

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x3 = global_add_pool(x,batch)

        x = torch.cat((x1,x2,x3),1)

        x = self.lin1(x)
        x = F.relu(x)

        x = self.lin2(x)
        x = F.relu(x)

        x = self.lin3(x)

        return torch.sigmoid(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data = LigandGraphIMDataset(root='../data/train_dataset', hdf5_file_name='postera_protease2_pos_neg_train.hdf5')

train_data.data.x = train_data.data.x.to(device)
train_data.data.edge_index = train_data.data.edge_index.to(device)
train_data.data.y = train_data.data.y.to(device)

# train_data = train_data.to(device)
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loader = DataLoader(train_data, batch_size=32, shuffle=True)
criterion = torch.nn.BCELoss()

model.train()
for epoch in range(150):
    total_loss = 0
    for data in loader:
        # data = data.to(device)
        optimizer.zero_grad()
        out = model(data).to(device)
        # out = model(data)
        out = torch.squeeze(out,1)
        # y = torch.as_tensor(data.y).to(device).float()
        # y = torch.as_tensor(data.y).float()
        loss = criterion(out,data.y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('Epoch:', epoch, 'Total Loss:', total_loss)

test_data = LigandGraphIMDataset(root='../data/test_dataset', hdf5_file_name='postera_protease2_pos_neg_test.hdf5')

test_data.data.x.to(device)
test_data.data.edge_index.to(device)
test_data.data.y.to(device)

test_loader = DataLoader(test_data, batch_size=1)
model.eval()
total = 0
correct = 0
with torch.no_grad():
    for data in test_loader:
        # data = data.to(device)
        pred = model(data).to(device)
        # print('Predict:', np.round(pred.item()), 'Label:', data.y)
        if np.round(pred.item()) == data.y:
            correct += 1
        total += 1

print(f'Accuracy: {correct/total}')
        