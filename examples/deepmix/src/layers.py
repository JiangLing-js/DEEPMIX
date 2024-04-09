from torch import nn
import torch.nn.functional as F
import torch
from activations import activation_layer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


class DNN(nn.Module):
    '''MLP的全连接层
    '''
    def __init__(self, input_dim, hidden_units, activation='relu', dropout_rate=0, use_bn=False, init_std=1e-4, dice_dim=3):
        super(DNN, self).__init__()
        assert isinstance(hidden_units, (tuple, list)) and len(hidden_units) > 0, 'hidden_unit support non_empty list/tuple inputs'
        self.dropout = nn.Dropout(dropout_rate)
        hidden_units = [input_dim] + list(hidden_units)

        layers = []
        for i in range(len(hidden_units)-1):
            # Linear
            layers.append(nn.Linear(hidden_units[i], hidden_units[i+1]))
            
            # BatchNorm
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_units[i+1]))

            # Activation
            layers.append(activation_layer(activation, hidden_units[i + 1], dice_dim))

            # Dropout
            layers.append(self.dropout)

        self.layers = nn.Sequential(*layers)

        for name, tensor in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

    def forward(self, inputs):
        # inputs: [btz, ..., input_dim]
        return self.layers(inputs)  # [btz, ..., hidden_units[-1]]


class PredictionLayer(nn.Module):
    def __init__(self, out_dim=1, use_bias=True, logit_transform=None, **kwargs):
        super(PredictionLayer, self).__init__()
        self.logit_transform = logit_transform
        if use_bias:
            self.bias = nn.Parameter(torch.zeros((out_dim,)))
        
    def forward(self, X):
        output =  X
        if hasattr(self, 'bias'):
            output += self.bias
        if self.logit_transform == 'sigmoid':
            output = torch.sigmoid(output)
        elif self.logit_transform == 'softmax':
            output = torch.softmax(output, dim=-1)
        return output


class SequencePoolingLayer(nn.Module):
    """seq输入转Pooling，支持多种pooling方式
    """
    def __init__(self, mode='mean', support_masking=False):
        super(SequencePoolingLayer, self).__init__()
        assert mode in {'sum', 'mean', 'max'}, 'parameter mode should in [sum, mean, max]'
        self.mode = mode
        self.support_masking = support_masking
    
    def forward(self, seq_value_len_list):
        # seq_value_len_list: [btz, seq_len, hdsz], [btz, seq_len]/[btz,1]
        seq_input, seq_len = seq_value_len_list

        if self.support_masking:  # 传入的是mask
            mask = seq_len.float()
            user_behavior_len = torch.sum(mask, dim=-1, keepdim=True)  # [btz, 1]
            mask = mask.unsqueeze(2)  # [btz, seq_len, 1]
        else:  # 传入的是behavior长度
            user_behavior_len = seq_len
            mask = torch.arange(0, seq_input.shape[1]) < user_behavior_len.unsqueeze(-1)
            mask = torch.transpose(mask, 1, 2)  # [btz, seq_len, 1]
        
        mask = torch.repeat_interleave(mask, seq_input.shape[-1], dim=2)  # [btz, seq_len, hdsz]
        mask = (1 - mask).bool()
        
        if self.mode == 'max':
            seq_input = torch.masked_fill(seq_input, mask, 1e-8)
            return torch.max(seq_input, dim=1, keepdim=True)  # [btz, 1, hdsz]
        elif self.mode == 'sum':
            seq_input = torch.masked_fill(seq_input, mask, 0)
            return torch.sum(seq_input, dim=1, keepdim=True)  # [btz, 1, hdsz]
        elif self.mode == 'mean':
            seq_input = torch.masked_fill(seq_input, mask, 0)
            seq_sum = torch.sum(seq_input, dim=1, keepdim=True)
            return seq_sum / (user_behavior_len.unsqueeze(-1) + 1e-8)


class CrossNetMix(nn.Module):
    """The Cross Network part of DCN-Mix model, which improves DCN-M by:
      1 add MOE to learn feature interactions in different subspaces
      2 add nonlinear transformations in low-dimensional space
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **in_features** : Positive integer, dimensionality of input features.
        - **low_rank** : Positive integer, dimensionality of low-rank sapce.
        - **num_experts** : Positive integer, number of experts.
        - **layer_num**: Positive integer, the cross layer number
        - **device**: str, e.g. ``"cpu"`` or ``"cuda:0"``
      References
        - [Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020.](https://arxiv.org/abs/2008.13535)
    """

    def __init__(self, in_features, low_rank=32, num_experts=4, layer_num=2):
        super(CrossNetMix, self).__init__()
        self.layer_num = layer_num
        self.num_experts = num_experts

        # U: (in_features, low_rank)
        self.U_list = nn.Parameter(torch.Tensor(self.layer_num, num_experts, in_features, low_rank))
        # V: (in_features, low_rank)
        self.V_list = nn.Parameter(torch.Tensor(self.layer_num, num_experts, in_features, low_rank))
        # C: (low_rank, low_rank)
        self.C_list = nn.Parameter(torch.Tensor(self.layer_num, num_experts, low_rank, low_rank))
        self.gating = nn.ModuleList([nn.Linear(in_features, 1, bias=False) for i in range(self.num_experts)])

        self.bias = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))

        init_para_list = [self.U_list, self.V_list, self.C_list]
        for para in init_para_list:
            for i in range(self.layer_num):
                nn.init.xavier_normal_(para[i])

        for i in range(len(self.bias)):
            nn.init.zeros_(self.bias[i])


    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)  # (bs, in_features, 1)
        x_l = x_0
        for i in range(self.layer_num):
            output_of_experts = []
            gating_score_of_experts = []
            for expert_id in range(self.num_experts):
                # (1) G(x_l)
                # compute the gating score by x_l
                gating_score_of_experts.append(self.gating[expert_id](x_l.squeeze(2)))

                # (2) E(x_l)
                # project the input x_l to $\mathbb{R}^{r}$
                v_x = torch.matmul(self.V_list[i][expert_id].t(), x_l)  # (bs, low_rank, 1)

                # nonlinear activation in low rank space
                v_x = torch.tanh(v_x)
                v_x = torch.matmul(self.C_list[i][expert_id], v_x)
                v_x = torch.tanh(v_x)

                # project back to $\mathbb{R}^{d}$
                uv_x = torch.matmul(self.U_list[i][expert_id], v_x)  # (bs, in_features, 1)

                dot_ = uv_x + self.bias[i]
                dot_ = x_0 * dot_  # Hadamard-product

                output_of_experts.append(dot_.squeeze(2))

            # (3) mixture of low-rank experts
            output_of_experts = torch.stack(output_of_experts, 2)  # (bs, in_features, num_experts)
            gating_score_of_experts = torch.stack(gating_score_of_experts, 1)  # (bs, num_experts, 1)
            moe_out = torch.matmul(output_of_experts, gating_score_of_experts.softmax(1))
            x_l = moe_out + x_l  # (bs, in_features, 1)

        x_l = x_l.squeeze()  # (bs, in_features)
        return x_l
