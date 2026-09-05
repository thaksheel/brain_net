import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU


class GIN(torch.nn.Module):
    def __init__(self, in_f, hid_c, out_f, dropout: float = 0.5):
        super().__init__()
        def mlp(in_dim, out_dim):
            return Sequential(Linear(in_dim, out_dim), ReLU(), Linear(out_dim, out_dim))
        self.dropout = dropout

        self.conv1 = GINConv(mlp(in_f, hid_c))
        self.bn1 = BatchNorm1d(hid_c)

        self.conv2 = GINConv(mlp(hid_c, hid_c))
        self.bn2 = BatchNorm1d(hid_c)

        self.conv3 = GINConv(mlp(hid_c, hid_c))
        self.bn3 = BatchNorm1d(hid_c)

        self.conv4 = GINConv(mlp(hid_c, hid_c))
        self.bn4 = BatchNorm1d(hid_c)

        self.linear = Linear(hid_c, out_f)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        batch_index = batch.batch
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x)

        # sum pooling (best for GIN)
        x = global_add_pool(x, batch_index)

        return self.linear(x)
