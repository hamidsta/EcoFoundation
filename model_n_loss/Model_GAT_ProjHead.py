from pytorch_metric_learning import losses
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool
import torch
from torch.nn import Linear
import torch.nn.functional as F



class Graph_MERFISH(torch.nn.Module):
    def __init__(self, num_features_exp, hidden_channels,projection_dim):
        super(Graph_MERFISH, self).__init__()

        # Attention GAT Conv Layers
        per_head_hidden_channels = hidden_channels // 5
        self.conv1_exp = GATConv(num_features_exp, per_head_hidden_channels, heads=5)
        self.conv2_exp = GATConv(per_head_hidden_channels * 5, per_head_hidden_channels, heads=5)

        # Batch norm layers
        self.bn1 = torch.nn.LayerNorm(hidden_channels)
        self.bn2 = torch.nn.LayerNorm(hidden_channels)
        self.dropout = torch.nn.Dropout(0.5)  # Add dropout for regularization

        # Latent space
        self.merge = Linear(hidden_channels, hidden_channels)
        torch.nn.init.xavier_uniform_(self.merge.weight.data)
        
        # Projection head (to project the embeddings into a new  low dimensional space)
        self.projection_head = Linear(hidden_channels, projection_dim)
        torch.nn.init.xavier_uniform_(self.projection_head.weight.data)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, data):
        exp, edge_index = data.x, data.edge_index

        # GATConv layers require edge_index to be long type
        x_exp = exp
        edge_index = edge_index.long()

        x_exp, attention_weights_1 = self.conv1_exp(x_exp, edge_index, return_attention_weights=True)
        x_exp = F.leaky_relu(x_exp)
        x_exp = self.dropout(self.bn1(x_exp))

        x_exp, attention_weights_2 = self.conv2_exp(x_exp, edge_index, return_attention_weights=True)
        x_exp = F.leaky_relu(x_exp)
        #x_exp = self.dropout(self.bn2(x_exp))

        x = self.merge(x_exp)
        x = F.leaky_relu(x)  # return nodes level embeddings 
        # x = global_mean_pool(x, data.batch)  
        
        

        return  x, attention_weights_1, attention_weights_2