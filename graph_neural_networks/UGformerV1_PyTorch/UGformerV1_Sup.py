import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class UGformerV1(nn.Module):

    def __init__(self, feature_dim_size, ff_hidden_size, num_classes,
                 num_self_att_layers, dropout, num_GNN_layers):
        super(UGformerV1, self).__init__()
        self.feature_dim_size = feature_dim_size
        self.ff_hidden_size = ff_hidden_size
        self.num_classes = num_classes
        self.num_self_att_layers = num_self_att_layers #Each layer consists of a number of self-attention layers
        self.num_GNN_layers = num_GNN_layers
        #
        self.ugformer_layers = torch.nn.ModuleList()
        for _ in range(self.num_GNN_layers): # nhead is set to 1 as the size of input feature vectors is odd
            encoder_layers = TransformerEncoderLayer(d_model=self.feature_dim_size, nhead=1, dim_feedforward=self.ff_hidden_size, dropout=0.5)
            self.ugformer_layers.append(TransformerEncoder(encoder_layers, self.num_self_att_layers))
        # Linear function
        self.predictions = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        # self.predictions.append(nn.Linear(feature_dim_size, num_classes)) # For including feature vectors to predict graph labels???
        for _ in range(self.num_GNN_layers):
            self.predictions.append(nn.Linear(self.feature_dim_size, self.num_classes))
            self.dropouts.append(nn.Dropout(dropout))

    def forward(self, input_x, graph_pool, X_concat):
        prediction_scores = 0
        input_Tr = F.embedding(input_x, X_concat)
        for layer_idx in range(self.num_GNN_layers):
            output_Tr = self.ugformer_layers[layer_idx](input_Tr)[0]
            #new input for next layer
            input_Tr = F.embedding(input_x, output_Tr)
            #sum pooling
            graph_embeddings = torch.spmm(graph_pool, output_Tr)
            graph_embeddings = self.dropouts[layer_idx](graph_embeddings)
            # Produce the final scores
            prediction_scores += self.predictions[layer_idx](graph_embeddings)

        return prediction_scores
    
    def get_activation(self, input_x, graph_pool, X_concat):
        activation = 0
        input_Tr = F.embedding(input_x, X_concat)
        for layer_idx in range(self.num_GNN_layers):
            output_Tr = self.ugformer_layers[layer_idx](input_Tr)[0]
            #new input for next layer
            input_Tr = F.embedding(input_x, output_Tr)
            #sum pooling
            graph_embeddings = torch.spmm(graph_pool, output_Tr)
            activation += graph_embeddings

        return activation

def label_smoothing(true_labels: torch.Tensor, classes: int, smoothing=0.1):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist