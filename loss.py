import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class Trajectory_Loss(nn.Module):
    def __init__(self, args):
        super(Trajectory_Loss, self).__init__()

    def forward(self, prediction, log_prob, valid, label):
        device = prediction.device
        loss_ = 0

        M = prediction.shape[0]
        # norms.shape = (M_c, 6)
        norms = torch.norm(prediction[:, :, -1] - label[:, -1].unsqueeze(dim=1), dim=-1)
        # best_ids.shape = (M_c)
        best_ids = torch.argmin(norms, dim=-1)

        # === L_reg ===
        # l_reg.shape = (M_c, T_f, 2)
        l_reg = F.smooth_l1_loss(prediction[torch.arange(M, device=device), best_ids], label, reduction="none")
        l_reg = (l_reg*valid).sum()/(valid.sum()*2)
        loss_ += l_reg
        # === === ===

        # === L_cls ===
        loss_ += F.nll_loss(log_prob, best_ids)
        # === === ===

        # === L_end ===
        loss_ += F.smooth_l1_loss(prediction[torch.arange(M, device=device), best_ids, -1], label[:, -1], reduction="mean")
        # === === ===

        return loss_
    


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.l1_loss = F.smooth_l1_loss
        self.cls_loss = F.cross_entropy
    def forward(self, predicted_trajs, ground_truth_trajs, predicted_probs):
        M = ground_truth_trajs.shape[2]
        N = predicted_trajs.shape[2]
        gt_probabilities = np.zeros(N)

        cost_matrix = torch.zeros(M, N)
        # Compute the cost matrix
        for i in range(M):
            for j in range(N):
                cost_matrix[i, j] = self.l1_loss(ground_truth_trajs[0, 0, i], predicted_trajs[0, 0, j]).mean()
        
        # Convert cost matrix to numpy for linear_sum_assignment
        cost_matrix_np = cost_matrix.detach().numpy()

        # Solve the assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix_np)

        final_loss = 0
        # Compute the final loss
        for i in range(len(row_ind)):
            final_loss += self.l1_loss(ground_truth_trajs[0, 0, row_ind[i]], predicted_trajs[0, 0, col_ind[i]]).mean()
        
        final_loss /= len(row_ind)

        # probability loss
        for i in range(len(gt_probabilities)):
            if i in col_ind:
                gt_probabilities[i] = 1

            else:
                gt_probabilities[i] = 0
        
        gt_probabilities_tensor = torch.tensor(gt_probabilities)

        final_loss += F.cross_entropy(gt_probabilities_tensor.view(1,1,N) , predicted_probs)

        return final_loss
