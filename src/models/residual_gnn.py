import torch
from torch import nn
from torch_geometric.nn import aggr
from torch.nn import ModuleList
from torch_geometric.nn import global_mean_pool

from ..config import Params


class ResidualGNNs(torch.nn.Module):
    def __init__(
        self,
        args: Params,
        GNN: torch.nn.Module,
        model_name: str,
    ):
        super().__init__()
        self.convs = ModuleList()
        self.args = args
        self.model_name = model_name
        self.aggr = aggr.MeanAggregation()
        self.hidden_channels = args.hid_dim
        self.hidden = args.hid_dim // 2
        num_features = args.num_features
        if self.model_name == "ChebConv":
            if args.num_layers > 0:
                self.convs.append(GNN(num_features, self.hidden_channels, K=5))
                for i in range(0, args.num_layers - 1):
                    self.convs.append(
                        GNN(self.hidden_channels, self.hidden_channels, K=5)
                    )
        else:
            if args.num_layers > 0:
                self.convs.append(GNN(num_features, self.hidden_channels))
                for i in range(0, args.num_layers - 1):
                    self.convs.append(GNN(self.hidden_channels, self.hidden_channels))
        input_dim1 = int(
            ((num_features * num_features) / 2)
            - (num_features / 2)
            + (self.hidden_channels * args.num_layers)
        )
        input_dim = int(((num_features * num_features) / 2) - (num_features / 2))
        self.bn = nn.BatchNorm1d(input_dim)
        self.bnh = nn.BatchNorm1d(self.hidden_channels * args.num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim1, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.ReLU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(self.hidden, self.hidden // 2),
            nn.BatchNorm1d(self.hidden // 2),
            nn.ReLU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(self.hidden // 2, self.hidden // 2),
            nn.BatchNorm1d(self.hidden // 2),
            nn.ReLU(),
            nn.Dropout(self.args.dropout),
            nn.Linear((self.hidden // 2), args.num_classes),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = [x]
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]
        h = []
        for i, xx in enumerate(xs):
            if i == 0:
                xx = xx.reshape(data.num_graphs, x.shape[1], -1)
                x = torch.stack(
                    [
                        t.triu().flatten()[t.triu().flatten().nonzero(as_tuple=True)]
                        for t in xx
                    ]
                )
                x = self.bn(x)
            else:
                xx = self.aggr(xx, batch)
                h.append(xx)
        h = torch.cat(h, dim=1)
        h = self.bnh(h)
        x = torch.cat((x, h), dim=1)
        x = self.mlp(x)
        softmax = torch.nn.LogSoftmax(dim=1)
        return softmax(x)
