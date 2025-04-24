import os
import uuid
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--port", type=int, default=1238, help="Port number for the local server")
argparser.add_argument("--cuda_device", type=str, default='0', help="Cuda devices to use. Default is 0")
argparser.add_argument("--static_folder", type=str, default='static', help="Folder to store static files")
argparser.add_argument('--checkpoint_folder', type=str, default='./Checkpoints/', help='The folder to store the checkpoint')
argparser.add_argument('--checkpoint_name', type=str, default='checkpoint.LInK', help='The name of the checkpoint file')
argparser.add_argument('--data_folder', type=str, default='./Data/', help='The folder to store the data')
argparser.add_argument('--embedding_folder', type=str, default='./Embeddings/', help='The folder to store the embeddings')
args = argparser.parse_args()


import numpy as np

emb  = np.load(os.path.join(args.embedding_folder, 'embeddings_subset.npy'))
As = np.load(os.path.join(args.data_folder, 'connectivity_subset.npy'))
x0s = np.load(os.path.join(args.data_folder, 'x0_subset.npy'))
node_types = np.load(os.path.join(args.data_folder, 'node_types_subset.npy'))
curves = np.load(os.path.join(args.data_folder, 'target_curves_subset.npy'))
graphs = np.load(os.path.join(args.data_folder, 'graphs_subset.npy'), allow_pickle=True)
alpha = np.load('./TestData/alphabet.npy', allow_pickle=True) * 10

# # 查看 emb 数组的数据结构
# print("emb 数据结构：")
# print("形状：", emb.shape)
# print("数据类型：", emb.dtype)
# print("维度数：", emb.ndim)
# print("部分数据内容：")
# print(emb)
# print("内存占用字节数：", emb.nbytes)
# print()
#
# # 查看 As 数组的数据结构
# print("As 数据结构：")
# print("形状：", As.shape)
# print("数据类型：", As.dtype)
# print("维度数：", As.ndim)
# print("部分数据内容：")
# print(As)
# print("内存占用字节数：", As.nbytes)
# print()
#
# # 查看 x0s 数组的数据结构
# print("x0s 数据结构：")
# print("形状：", x0s.shape)
# print("数据类型：", x0s.dtype)
# print("维度数：", x0s.ndim)
# print("部分数据内容：")
# print(x0s)
# print("内存占用字节数：", x0s.nbytes)
# print()
#
# # 查看 node_types 数组的数据结构
# print("node_types 数据结构：")
# print("形状：", node_types.shape)
# print("数据类型：", node_types.dtype)
# print("维度数：", node_types.ndim)
# print("部分数据内容：")
# print(node_types)
# print("内存占用字节数：", node_types.nbytes)
# print()
#
# # 查看 curves 数组的数据结构
# print("curves 数据结构：")
# print("形状：", curves.shape)
# print("数据类型：", curves.dtype)
# print("维度数：", curves.ndim)
# print("部分数据内容：")
# print(curves)
# print("内存占用字节数：", curves.nbytes)
#
# 查看 graphs 数组的数据结构
# print("graphs 数据结构：")
# print("形状：", graphs.shape)
# print("数据类型：", graphs.dtype)
# print("维度数：", graphs.ndim)
# print("部分数据内容：")
# print(graphs)
# print("内存占用字节数：", graphs.nbytes)

# 查看 alpha 数组的数据结构
print("alpha 数据结构：")
print("形状：", alpha.shape)
print("数据类型：", alpha.dtype)
print("维度数：", alpha.ndim)
print("部分数据内容：")
print(alpha)
print("内存占用字节数：", alpha.nbytes)