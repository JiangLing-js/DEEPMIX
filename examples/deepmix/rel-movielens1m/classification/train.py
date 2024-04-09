# DeepMIX for classification task
# macro_f1: 0.3108, micro_f1: 0.3498
# Runtime: 14.709771156311035s
# Cost: N/A
# Description: Paper Reproduction.

import pandas as pd
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import time
import sys
sys.path.append("../../src")
from inputs import SparseFeat, VarLenSparseFeat, build_input_array, TensorDataset
from models import DeepMIX
from snippets import sequence_padding, seed_everything
from callbacks import Evaluator


device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed_everything(42)


def get_data(features):
    def split(x):
        key_ans = x.split('|')
        for key in key_ans:
            if key not in key2index:
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))

    data1 = pd.read_csv("../../../../rllm/datasets/rel-movielens1m/classification/movies/train.csv")
    data2 = pd.read_csv("../../../../rllm/datasets/rel-movielens1m/classification/movies/test.csv")
    # 检查并填充空值
    data1["Plot"] = features[3107:3107+388].cpu().numpy()
    # 检查并填充空值

    data2["Plot"] = features[:3107].cpu().numpy()
    data = pd.concat([data1, data2], axis=0)
    # print(data1["Plot"])
    # print(data2["Plot"])
    # data['rating'] = data['rating'] - 1
    sparse_features = ["Plot"]

    # print(data["Plot"])
    # Discrete variable coding
    # for feat in sparse_features:
    #     lbe = LabelEncoder()
    #     data[feat] = model.encode(data[feat])

    # Sequential feature processing
    key2index = {}
    genres_list = list(map(split, data['Genre'].values))
    # Guaranteed maximum length in 6 columns without affecting the data
    max_genres = 0
    index = 0
    for i in range(len(genres_list)):
        if len(genres_list[i]) > max_genres:
            max_genres = len(genres_list[i])
            index = i
    if max_genres < 6:
        pad = [0 for i in range(6-max_genres)]
        genres_list[index].extend(pad)
    genres_list = sequence_padding(genres_list)

    data['Genre'] = genres_list.tolist()
    print()

    # Discrete and Sequential Feature Processing
    # For sparse features, they are converted to dense vectors by embedding technique.
    # For dense numerical features, connect them to the input tensor of the fully connected layer.
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=16) for feat in sparse_features]
    varlen_feature_columns = [VarLenSparseFeat(SparseFeat('Genre', vocabulary_size=len(
        key2index) + 1, embedding_dim=16), maxlen=genres_list.shape[-1], pooling='mean')]
    # Generate feature columns
    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

    # Generate training samples
    train_X, train_y = build_input_array(data, linear_feature_columns+dnn_feature_columns, target='Genre')
    test_X = torch.tensor(train_X[389:])
    train_X = torch.tensor(train_X[:389])
    floated = [i for i in train_y]
    for i in range(len(train_y)):
        for j in range(len(train_y[i])):
            floated[i][j] = float(train_y[i][j])
    updated_floated = [[0]*18 for i in range(len(floated))]
    for i in range(len(floated)):
        for j in floated[i]:
            if j != 0:
                updated_floated[i][int(j-1)] = float(1)
    train_y = torch.tensor(updated_floated[:389])
    test_y = torch.tensor(updated_floated[389:])
    return train_X, train_y, test_X, test_y, linear_feature_columns, dnn_feature_columns


def evaluate_data(model, dataloader):
    y_preds = []
    for X, y in dataloader:
        y_prob = model.predict(X).cpu().numpy()
        y = y.cpu().numpy()
        y_preds = (y_prob > 0).astype(int)
    return f1_score(y, y_preds, average='micro', zero_division=1), f1_score(y, y_preds, average='macro', zero_division=1)


class MyEvaluator(Evaluator):
    def evaluate(self):
        micro, macro = evaluate_data(self.model, train_dataloader)
        return {'micro_f1': micro, 'macro_f1': macro}

import sys

sys.path.append("../../../../rllm/dataloader/")
from load_data import load_data

if __name__ == "__main__":
    # Load Dataset
    data, adj, features, labels, idx_train, idx_val, idx_test = load_data('movielens-classification', device=device)
    train_y = labels.cpu()[idx_train.cpu()]
    val_y = labels.cpu()[idx_val.cpu()]
    test_y = labels.cpu()[idx_test.cpu()]
    train_X, train_y, test_X, test_y, linear_feature_columns, dnn_feature_columns = get_data(features)
    # train_X = features[idx_train].cpu()
    # train_y = labels[idx_train].cpu()
    # test_X = features[389:].cpu()
    # test_y = labels[389:].cpu()
    # print(linear_feature_columns.shape)
    # print(linear_feature_columns)
    # exit(0)
    train_dataloader = DataLoader(TensorDataset(train_X, train_y, device=device), batch_size=64, shuffle=True)
    test_dataloader = DataLoader(TensorDataset(test_X, test_y, device=device), batch_size=3108, shuffle=True)

    # Model Definition
    # model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, out_dim=18)
    # test_model = DeepFM(linear_feature_columns, dnn_feature_columns, out_dim=18)
    model = DeepMIX(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, out_dim=18)
    test_model = DeepMIX(linear_feature_columns, dnn_feature_columns, out_dim=18)
    model.to(device)

    model.compile(
        # loss=nn.CrossEntropyLoss(),
        loss=nn.MSELoss(),
        # optimizer=optim.Adam(model.parameters(), lr=1e-2)
        optimizer=optim.Adam(model.parameters(), lr=0.001)
    )

    # Evaluator definition
    evaluator1 = MyEvaluator(monitor='macro_f1')
    evaluator2 = MyEvaluator(monitor='micro_f1')

    start_time = time.time()
    # train
    model.fit(train_dataloader, epochs=100, steps_per_epoch=None, callbacks=[evaluator1, evaluator2])
    end_time = time.time()
    # test
    micro, macro = evaluate_data(model, test_dataloader)
    print()
    print("time: ", end_time - start_time)
    print('test_micro_f1: ', micro)
    print('test_macro_f1: ', macro)
