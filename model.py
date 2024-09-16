import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment




class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, p_drop=0.0, hidden_dim=None, residual=False):
        super(MLP, self).__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        layer2_dim = hidden_dim
        if residual:
            layer2_dim = hidden_dim + input_dim

        self.residual = residual
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(layer2_dim, output_dim)
        self.dropout1 = nn.Dropout(p=p_drop)
        self.dropout2 = nn.Dropout(p=p_drop)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout1(out)
        if self.residual:
            out = self.layer2(torch.cat([out, x], dim=-1))
        else:
            out = self.layer2(out)

        out = self.dropout2(out)
        return out





class Trajectory_Decoder(nn.Module):
    def __init__(self):
        super(Trajectory_Decoder, self).__init__()

        hidden_size = 128

        self.endpoint_predictor = MLP(hidden_size , 20*2, residual=True)
        self.get_trajectory = MLP(hidden_size + 2, 19*2, residual=True)
        self.endpoint_refiner = MLP(hidden_size + 2, 2, residual=True)
        self.get_prob = MLP(hidden_size + 2, 1, residual=True)

    def forward(self, agent_features):
  
        N = agent_features.shape[0]
        M = agent_features.shape[1]
        D = agent_features.shape[2]


        endpoints = self.endpoint_predictor(agent_features).view(N, M, 20, 2)

        # prediction_features.shape = (N, M, 6, 128)
        agent_features = agent_features.unsqueeze(dim=2).expand(N, M, 20, D)
        # meta_info_input.shape = (N, M, 6, 128 + 5)
        # agent_features = torch.cat([agent_features, meta_info_tensor_k], dim=-1)

        # offsets.shape = (N, M, 6, 2)
        offsets = self.endpoint_refiner(torch.cat([agent_features, endpoints.detach()], dim=-1))
        endpoints += offsets

        # agent_features.shape = (N, M, 6, 128 + 5 + 2)
        agent_features = torch.cat([agent_features, endpoints.detach()], dim=-1)

        predictions = self.get_trajectory(agent_features).view(N, M, 20, 19, 2)
        logits = self.get_prob(agent_features).view(N, M, 20)

        predictions = torch.cat([predictions, endpoints.unsqueeze(dim=-2)], dim=-2)

        assert predictions.shape == (N, M, 20, 20, 2)

        return predictions, logits
    


class ADAPT(nn.Module):
    def __init__(self):
        super(ADAPT, self).__init__()
        self.input_mlp = MLP(4, 128, p_drop=0.0, hidden_dim=None, residual=False)
        self.encoder = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.decoder = Trajectory_Decoder()

    def forward(self,x):
        input_emb = self.input_mlp(x)
        agent_features = self.encoder(input_emb)
        trajs, probs = self.decoder(agent_features)
        return trajs, probs