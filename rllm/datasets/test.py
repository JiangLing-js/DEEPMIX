from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embeddings = model.encode(sentences)
print(embeddings)

import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
# 读取CSV文件

train_data = pd.read_csv("E:/information security/rllm/rllm/datasets/rel-movielens1m/classification/movies/train.csv")
test_data = pd.read_csv("E:/information security/rllm/rllm/datasets/rel-movielens1m/classification/movies/test.csv")
validation_data = pd.read_csv("E:/information security/rllm/rllm/datasets/rel-movielens1m/classification/movies/validation.csv")
train_data['Title'].fillna("Unknown", inplace=True)
test_data['Title'].fillna("Unknown", inplace=True)
validation_data['Title'].fillna("Unknown", inplace=True)
print(len(train_data), len(test_data), len(validation_data))
# 提取标题和剧情信息
titles = test_data['Title'].tolist() + train_data['Title'].tolist() + validation_data['Title'].tolist()
plots = test_data['Plot'].tolist() + train_data['Plot'].tolist() + validation_data['Plot'].tolist()

print(len(titles))
print(len(plots))
# 如果是空值就用title代替
for i in range(len(plots)):
    if pd.isnull(plots[i]):
        plots[i] = titles[i]

# for plot in plots:
#     if isinstance(plot, float):
#         print("浮点数：", plot)
#     else:
#         continue
# exit(0)
# model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

# sentences = ["This is an example sentence", "Each sentence is converted"]
# sentences2 = ["This is an example sentence", "Each sentence is converted"]
# e1 = model.encode(sentences)
# e2 = model.encode(sentences2)
# e3 = np.concatenate((e1, e2), axis=1)
# print(e1)
# print(e3)
# exit(0)
title_embeddings = model.encode(titles)
np.save("embeddings2.npy", title_embeddings)
# print(title_embeddings)
plot_embeddings = model.encode(plots)

# 合并标题和剧情向量
all_embeddings = np.concatenate((title_embeddings, plot_embeddings), axis=1)

# 保存向量到embeddings2.npy文件
np.save("embeddings2.npy", all_embeddings)
