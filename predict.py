import pickle
import torch
import torch.nn.functional as F
import pandas as pd
from dataset import HW3Dataset

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, heads):
        super().__init__()
        self.conv1 = GATConv(dataset.num_features,int(hidden_channels/2))
        self.conv2 = GATConv(int(hidden_channels/2),hidden_channels)
        self.conv3 = GATConv(hidden_channels,dataset.num_classes)
        # adding batch norm layers to generalize the data
        self.bn1 = torch.nn.BatchNorm1d(int(hidden_channels/2))
        self.bn2 = torch.nn.BatchNorm1d(int(hidden_channels))



    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.tanh(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)

        x = F.tanh(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index)
        return x



if __name__ == '__main__':
    dataset = HW3Dataset(root='data/hw3/')
    data = dataset[0]
    with open('final_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    out = loaded_model(data.x, data.edge_index)
    # getting the predictions
    predictions = out.argmax(dim=1)
    # getting the index
    ids=[i for i in range(len(predictions))]
    data_dict = {'idx':ids,'prediction':predictions}
    # saving and making a file for each index its prediction
    df_preidcted = pd.DataFrame(data=data_dict)
    df_preidcted.to_csv('prediction.csv',index=False)
