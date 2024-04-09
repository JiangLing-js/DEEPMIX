import torch
from torch import nn
from inputs import build_input_features, DenseFeat
from inputs import combined_dnn_input, create_embedding_matrix, split_columns, input_from_feature_columns
from layers import DNN, PredictionLayer, CrossNetMix

from torch4keras.model import BaseModel as BM


class BaseModel(BM):
    '''之前是在rec4torch内部实现的，之后单独为Trainer做了一个包torch4keras
       这里是继承torch4keras的BaseModel作为Trainer，并在其基础上加了res_loss和aux_loss
    '''
    def train_step(self, train_X, train_y):
        output, loss, loss_detail = super().train_step(train_X, train_y)
        # 由于前面可能使用了梯度累积，因此这里恢复一下
        loss = loss * self.grad_accumulation_steps if self.grad_accumulation_steps > 1 else loss

        # l1正则和l2正则
        reg_loss = self.get_regularization_loss()
        loss = loss + reg_loss + self.aux_loss

        # 梯度累积
        loss = loss * self.grad_accumulation_steps if self.grad_accumulation_steps > 1 else loss
        return output, loss, loss_detail


class Linear(nn.Module):
    """浅层线性全连接，也就是Wide&Cross的Wide部分
    步骤：
    1. Sparse特征分别过embedding, 得到多个[btz, 1, 1]
    2. VarLenSparse过embeddingg+pooling后，得到多个[btz, 1, 1]
    3. Dense特征直接取用, 得到多个[btz, dense_len]
    4. Sparse和VarLenSparse进行cat得到[btz, 1, featnum]，再sum_pooling得到[btz, 1]的输出
    5. Dense特征过[dense_len, 1]的全连接得到[btz, 1]的输出
    6. 两者求和得到最后输出
    
    参数：
    feature_columns: 各个特征的[SparseFeat, VarlenSparseFeat, DenseFeat, ...]的列表
    feature_index: 每个特征在输入tensor X中的列的起止
    """
    def __init__(self, feature_columns, feature_index, init_std=1e-4, out_dim=1, **kwargs):
        super(Linear, self).__init__()
        self.feature_index = feature_index
        self.out_dim = out_dim
        self.feature_columns = feature_columns
        self.sparse_feature_columns, self.dense_feature_columns, self.varlen_sparse_feature_columns = split_columns(feature_columns)
        
        # 特征embdding字典，{feat_name: nn.Embedding()}
        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, out_dim, sparse=False)  # out_dim=1表示线性
        
        if len(self.dense_feature_columns) > 0:
            self.weight = nn.Parameter(torch.Tensor(sum(fc.dimension for fc in self.dense_feature_columns), out_dim))
            nn.init.normal_(self.weight, mean=0, std=init_std)

    def forward(self, X, sparse_feat_refine_weight=None):
        sparse_embedding_list, dense_value_list = input_from_feature_columns(X, self.feature_columns, self.feature_index, self.embedding_dict)

        linear_logit = torch.zeros([X.shape[0], self.out_dim], device=X.device)
        if len(sparse_embedding_list) > 0:
            sparse_embedding_cat = torch.cat(sparse_embedding_list, dim=-1)  # [btz, 1, feat_cnt]
            if sparse_feat_refine_weight is not None:  # 加权
                sparse_embedding_cat = sparse_embedding_cat * sparse_feat_refine_weight.unsqueeze(1)
            sparse_feat_logit = torch.sum(sparse_embedding_cat, dim=-1)
            linear_logit += sparse_feat_logit
        if len(dense_value_list) > 0:
            dense_value_logit = torch.cat(dense_value_list, dim=-1).float().matmul(self.weight)
            linear_logit += dense_value_logit
        
        return linear_logit


class RecBase(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, l2_reg_linear=1e-5, l2_reg_embedding=1e-5,
                 init_std=1e-4, out_dim=1, **kwargs):
        super(RecBase, self).__init__()
        self.dnn_feature_columns = dnn_feature_columns
        self.aux_loss = 0  # 目前只看到dien里面使用

        # feat_name到col_idx的映射, eg: {'age':(0,1),...}
        self.feature_index = build_input_features(linear_feature_columns + dnn_feature_columns)

        # 为SparseFeat和VarLenSparseFeat特征创建embedding
        self.embedding_dict = create_embedding_matrix(dnn_feature_columns, init_std, sparse=False)
        self.linear_model = Linear(linear_feature_columns, self.feature_index, out_dim=out_dim, **kwargs)

        # l1和l2正则
        self.regularization_weight = []
        self.add_regularization_weight(self.embedding_dict.parameters(), l2=l2_reg_embedding)
        self.add_regularization_weight(self.linear_model.parameters(), l2=l2_reg_linear)

        # 输出层
        self.out = PredictionLayer(out_dim,  **kwargs)

    def compute_input_dim(self, feature_columns, feature_names=[('sparse', 'var_sparse', 'dense')], feature_group=False):
        '''计算输入维度和，Sparse/VarlenSparse的embedding_dim + Dense的dimesion
        '''
        def get_dim(feat):
            if isinstance(feat, DenseFeat):
                return feat.dimension
            elif feature_group:
                return 1
            else:
                return feat.embedding_dim

        feature_col_groups = split_columns(feature_columns, feature_names)
        input_dim = 0
        for feature_col in feature_col_groups:
            if isinstance(feature_col, list):
                for feat in feature_col:
                    input_dim += get_dim(feat)
            else:
                input_dim += get_dim(feature_col)
                    
        return input_dim

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        """记录需要正则的参数项
        """
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self):
        """计算正则损失
        """
        total_reg_loss = 0
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

    def add_auxiliary_loss(self, aux_loss, alpha):
        self.aux_loss = aux_loss * alpha

    @property
    def embedding_size(self, ):
        feature_columns = self.dnn_feature_columns
        sparse_feature_columns = split_columns(feature_columns, ['sparse', 'var_sparse'])
        embedding_size_set = set([feat.embedding_dim for feat in sparse_feature_columns])
        if len(embedding_size_set) > 1:
            raise ValueError("embedding_dim of SparseFeat and VarlenSparseFeat must be same in this model!")
        return list(embedding_size_set)[0]


class DeepMIX(RecBase):
    """DeepCrossing的实现
    和Wide&Deep相比，去掉Wide部分，DNN部分换成残差网络，模型结构简单
    [1] [ACM 2016] Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features (https://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf)
    """
    def __init__(self, linear_feature_columns, dnn_feature_columns, cross_num=2, dnn_hidden_units=(512, 256),
                 l2_reg_linear=1e-3, l2_reg_embedding=1e-3, l2_reg_dnn=1e-4, init_std=1e-4, low_rank=32, num_experts=4,
                 dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, out_dim=1, **kwargs):
        super(DeepMIX, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                           l2_reg_embedding=l2_reg_embedding, init_std=init_std, out_dim=out_dim, **kwargs)
        del self.linear_model
        assert len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0

        input_dim = self.compute_input_dim(dnn_feature_columns)
        self.dnn = DNN(input_dim, dnn_hidden_units, activation=dnn_activation,
                                   dropout_rate=dnn_dropout, use_bn=dnn_use_bn, init_std=init_std)
        self.cross_num = cross_num
        self.dnn_hidden_units = dnn_hidden_units
        # RE 136 CL 152
        self.out_dim = out_dim
        if self.out_dim == 18:
            Linear_in = 288
        else:
            Linear_in = 264
        self.dnn_linear = nn.Linear(Linear_in, out_dim, bias=False)
        self.crossnet = CrossNetMix(in_features=self.compute_input_dim(dnn_feature_columns),
                                    low_rank=low_rank, num_experts=num_experts,
                                    layer_num=cross_num)
        self.add_regularization_weight(filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
        self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

    def forward(self, X):
        # logit = self.dnn_linear(X)
        # 离散变量过embedding，连续变量保留原值
        sparse_embedding_list, dense_value_list = input_from_feature_columns(X, self.dnn_feature_columns, self.feature_index, self.embedding_dict)

        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)  # [btz, sparse_feat_cnt*emb_size+dense_feat_cnt]
        # if self.out_dim == 18:
        #     cross_out = self.crossnet(dnn_input)
        #     logit = self.dnn_linear(cross_out)
        #     return self.out(logit)
        if len(self.dnn_hidden_units) > 0 and self.cross_num > 0:  # Deep & Cross
            deep_out = self.dnn(dnn_input)
            # print(deep_out.shape)
            cross_out = self.crossnet(dnn_input)
            # print(cross_out.shape)
            stack_out = torch.cat((cross_out, deep_out), dim=-1)
            logit = self.dnn_linear(stack_out)
        elif len(self.dnn_hidden_units) > 0:  # Only Deep
            deep_out = self.dnn(dnn_input)
            logit = self.dnn_linear(deep_out)
        elif self.cross_num > 0:  # Only Cross
            cross_out = self.crossnet(dnn_input)
            logit = self.dnn_linear(cross_out)
        else:  # Error
            pass

        y_pred = self.out(logit)

        return y_pred

