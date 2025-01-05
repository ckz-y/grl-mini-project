import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """Implements a single graph convolutional layer.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): _Size of each output sample.
        aug_adj_type (str): Type of augmented adjacency matrix to use.
            1. Symmetric Normalized Adjacency Matrix with Self-Loop: $(D + I)^{-1/2} (A + I) (D + I)^{-1/2}$
            2. Adjacency Matrix: $A$
            3. Degree Matrix: $D$
            4. Random Walk Adjacency Matrix: $D^{-1} A$

    $$(D + I)^{-1/2} (A + I) (D + I)^{-1/2}$$
    """

    def __init__(self, in_channels: int, out_channels: int, aug_adj_type: str):

        super(GCNLayer, self).__init__()

        self.weight = nn.Parameter(torch.rand((in_channels, out_channels)) / 4 - 0.125)
        self.aug_adj_type = aug_adj_type

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor):

        degree_matrix = torch.diag(torch.sum(adj_matrix, axis=1))
        num_nodes = len(adj_matrix)

        if self.aug_adj_type == "symmetric":
            deg_diag = torch.sum(adj_matrix, axis=1)
            deg_diag_self_loop = deg_diag + 1
            deg_diag_self_loop_inv_sq = torch.pow(deg_diag_self_loop, -0.5)
            deg_mat_self_loop_inv_sq = torch.diag(deg_diag_self_loop_inv_sq)

            aug_adj_matrix = (
                deg_mat_self_loop_inv_sq
                @ (adj_matrix + torch.eye(num_nodes))
                @ deg_mat_self_loop_inv_sq
            )
        elif self.aug_adj_type == "adjacency":
            aug_adj_matrix = adj_matrix
        elif self.aug_adj_type == "degree":
            aug_adj_matrix = degree_matrix
        elif self.aug_adj_type == "random walk":
            deg_diag = torch.sum(adj_matrix, axis=1)
            deg_diag_inv = torch.pow(deg_diag, -1)
            deg_mat_inv = torch.diag(deg_diag_inv)
            aug_adj_matrix = deg_mat_inv @ adj_matrix
        else:
            raise ValueError("Received invalid augmented adjacency matrix type.")

        x = F.relu(aug_adj_matrix @ x @ self.weight)

        return x

class GCN(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        aug_adj_type: str,
        dropout: float = 0.3,
    ):
        super(GCN, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout

        if num_layers >= 2:
            self.conv_layers.append(
                GCNLayer(in_channels, hidden_channels, aug_adj_type)
            )

            for i in range(num_layers - 2):
                self.conv_layers.append(
                    GCNLayer(hidden_channels, hidden_channels, aug_adj_type)
                )

            self.final_conv_layer = GCNLayer(
                hidden_channels, hidden_channels, aug_adj_type
            )
        elif num_layers == 1:
            self.final_conv_layer = GCNLayer(in_channels, hidden_channels, aug_adj_type)
        else:  # num_layers == 0, single feed-forward network
            self.weight = nn.Parameter(
                torch.rand((in_channels, hidden_channels)) / 4 - 0.125
            )

        self.output_layer = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        if self.num_layers >= 1:
            for layer in self.conv_layers:
                x = layer(x, adj_matrix)
                x = F.dropout(x, p=self.dropout)

            x = self.final_conv_layer(x, adj_matrix)

        else:  # num_layers == 0, single feed-forward network
            x = F.relu(x @ self.weight)

        x = self.output_layer(x)
        x = F.softmax(x, dim=1)

        return x