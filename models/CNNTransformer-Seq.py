# coding: UTF-8
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# class Config(object):
#     """配置参数"""
#     def __init__(self, args,dataset, embedding):
#         self.model_name = 'CNNTransformer-Seq'
#         self.train_path = dataset + '/data/train.txt'                                # 训练集
#         self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
#         self.test_path = dataset + '/data/test.txt'                                  # 测试集
#         self.class_list = [x.strip() for x in open(
#             dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
#
#
#         self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
#         self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
#         self.log_path = dataset + '/log/' + self.model_name
#         self.embedding_pretrained = torch.tensor(
#             np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
#             if embedding != 'random' else None                                       # 预训练词向量
#         self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')   # 设备
#
#         self.dropout = 0.5                                              # 随机失活
#         self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
#         self.num_classes = len(self.class_list)                         # 类别数
#         self.n_vocab = 0                                                # 词表大小，在运行时赋值
#         self.num_epochs = 200                                            # epoch数
#         self.batch_size = 128                                           # mini-batch大小
#         self.pad_size = 512                                             # 每句话处理成的长度(短填长切)
#         self.learning_rate = 5e-4                                       # 学习率
#         self.embed = self.embedding_pretrained.size(1)\
#             if self.embedding_pretrained is not None else 300           # 字向量维度
#
#         self.embedding = embedding
#         self.word_save_path = "/data/embedding/fasttest-word.ckpt"
#         self.word2idx_path = "/data/embedding/fasttest-word2idx.ckpt"
#
#
#         self.num_layers=4 # 编码器数量 [1,2,3,4,5,6,7,8,9]
#         self.num_head = 3 #[1,2,3,4,5,6]
#
#         # CNN
#         self.filter_sizes = (1,2,3,4)# 卷积核尺寸
#         # (1,2),(1,2,3),(2,3,4),(3,4,5),(1,2, 3, 4),(1,3,5,7),(2,4,6,8),
#         #                              (1,2,3,4,5),(1,2,3,5,7),(1,3,4,5,7),(1,3,5,7,9),(1,2,3,4,5,6),(1,2,3,4,5,6,7)
#         # (1),(3),(5),(7),
#         self.num_filters = 128                                          # 卷积核数量(channels数)

class Config(object):
    """配置参数"""
    def __init__(self, args, dataset, embedding):
        self.model_name = 'CNNTransformer-Seq'
        path="dataset/"

        manual_dataset_path=path+dataset + '-' + args.dataset if hasattr(args, 'dataset') and args.dataset else path+dataset + '-' + 'DFS-Selected60'
        default_dataset_path=path+dataset + '-' + 'DFS'

        self.train_path = manual_dataset_path+ '/data/train.txt' if hasattr(args, 'dataset') and args.dataset else default_dataset_path+ '/data/train.txt'
        self.dev_path = manual_dataset_path+ '/data/dev.txt' if hasattr(args, 'dataset') and args.dataset else default_dataset_path+ '/data/dev.txt'
        self.test_path = manual_dataset_path+ '/data/test.txt' if hasattr(args, 'dataset') and args.dataset else default_dataset_path+ '/data/test.txt'
        self.class_list = [x.strip() for x in open(
            manual_dataset_path+ '/data/class.txt', encoding='utf-8').readlines()] if hasattr(args, 'dataset') and args.dataset else [x.strip() for x in open(
            default_dataset_path+ '/data/class.txt', encoding='utf-8').readlines()]

        self.vocab_path = manual_dataset_path + '/data/vocab.pkl' if hasattr(args, 'dataset') and args.dataset else default_dataset_path + '/data/vocab.pkl'
        self.save_path = manual_dataset_path  + '/saved_dict/' + self.model_name + '.ckpt' if hasattr(args, 'dataset') and args.dataset else default_dataset_path + '/saved_dict/' + self.model_name + '.ckpt'
        self.log_path = manual_dataset_path + '/log/' + self.model_name if hasattr(args, 'dataset') and args.dataset else default_dataset_path + '/log/' + self.model_name

        self.embedding_pretrained = args.embedding_pretrained if hasattr(args, 'embedding_pretrained') and args.embedding_pretrained else torch.tensor(
            np.load(default_dataset_path + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None


        self.dropout = args.dropout if hasattr(args, 'dropout') and args.dropout else 0.5
        self.require_improvement = args.require_improvement if hasattr(args, 'require_improvement') and args.require_improvement else 1000
        self.num_classes = args.num_classes if hasattr(args, 'num_classes') and args.num_classes else len(self.class_list)
        self.n_vocab = args.n_vocab if hasattr(args, 'n_vocab') and args.n_vocab else 0
        self.num_epochs = args.num_epochs if hasattr(args, 'num_epochs') and args.num_epochs else 200
        self.batch_size = args.batch_size if hasattr(args, 'batch_size') and args.batch_size else 128
        self.pad_size = args.pad_size if hasattr(args, 'pad_size') and args.pad_size else 850
        self.learning_rate = args.learning_rate if hasattr(args, 'learning_rate') and args.learning_rate else 5e-4
        self.embed = args.embed if hasattr(args, 'embed') and args.embed else self.embedding_pretrained.size(1) if self.embedding_pretrained is not None else 300

        self.embedding = args.embedding if hasattr(args, 'embedding') and args.embedding else embedding
        self.word_save_path = args.word_save_path if hasattr(args, 'word_save_path') and args.word_save_path else "/data/embedding/fasttest-word.ckpt"
        self.word2idx_path = args.word2idx_path if hasattr(args, 'word2idx_path') and args.word2idx_path else "/data/embedding/fasttest-word2idx.ckpt"
        self.num_layers = int(args.num_layers) if hasattr(args, 'num_layers') and args.num_layers else 4
        self.num_head = int(args.num_head) if hasattr(args, 'num_head') and args.num_head else 4

        self.filter_sizes = args.filter_sizes if hasattr(args, 'filter_sizes') and args.filter_sizes else (1,2,3,4)
        self.num_filters = int(args.num_filters) if hasattr(args, 'num_filters') and args.num_filters else 128

        self.device = torch.device('cuda:'+args.device) if hasattr(args, 'device') and args.device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        # 自己去除的全连接层
        # self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)
        self.config = config

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        # 自己去除的全连接层
        # out = self.fc(out)
        return out

    # def forward(self, x):
    #     out = self.embedding(x[0])
    #     out = out.unsqueeze(1)
    #     # apply convolution and pooling for each filter size
    #     conv_results = []
    #     for i, conv in enumerate(self.convs):
    #         conv_result = self.conv_and_pool(out, conv)
    #         conv_results.append(conv_result.view(-1, 1, self.config.num_filters))
    #     # concatenate the results and reshape them into a matrix
    #     out = torch.cat(conv_results, dim=1).view(-1, len(self.convs), self.config.num_filters)
    #     # out = self.dropout(out)
    #     return out

class TransformerModel(nn.Module):
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.num_layers = config.num_layers
        self.num_heads = config.num_head
        self.hidden_dim = config.num_filters * len(config.filter_sizes)
        # self.hidden_dim = config.num_filters

        self.dropout = config.dropout

        # self.pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout)

        encoder_layers = TransformerEncoderLayer(self.hidden_dim, self.num_heads, self.hidden_dim * 2, self.dropout)
        # encoder_layers = TransformerEncoderLayer(128, self.num_heads, 1024, self.dropout)

        self.transformer_encoder = TransformerEncoder(encoder_layers, self.num_layers)

    def forward(self, x):
        # x: (batch_size, seq_len, hidden_dim)
        # x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

#结合CNN和Transformer的模型
class Model(nn.Module):
    def __init__(self,config ):
        # transformer 参数 input_dim,hidden_dim,kernel_sizes,num_heads,num_layers,output_dim,dropout
        super(Model, self).__init__()

        self.text_cnn = TextCNN(config)
        # input_dim, hidden_dim, kernel_sizes
        self.transformer_model = TransformerModel(config)
        # self.transformer_encoder_layer = TransformerEncoderLayer(config.num_filters * len(config.filter_sizes), config.num_head, config.num_filters * 2, config.dropout)
        # self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, config.num_layers)
        # self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)
        self.fc1 = nn.Linear(config.num_filters * len(config.filter_sizes),
                             config.num_filters * len(config.filter_sizes))
        self.fc2 = nn.Linear(config.num_filters * len(config.filter_sizes),config.num_classes)
        self.dropout = nn.Dropout(config.dropout)
        self.config=config

    def forward(self, x):
        x = self.text_cnn(x)
        x = x.unsqueeze(0)  # add batch dimension 一开始cnn输出一个1维向量时使用
        x = self.transformer_model(x)
        x = x.squeeze(0)  # remove batch dimension
        # x = self.fc(x)
        out = self.fc1(x)
        # out = x.view(-1,self.config.num_filters * len(self.config.filter_sizes))
        # out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
        return x



# class Model(nn.Module):
#     def __init__(self, config):
#         super(Model, self).__init__()
#         if config.embedding_pretrained is not None:
#             self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
#         else:
#             self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
#         self.convs = nn.ModuleList(
#             [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
#         self.dropout = nn.Dropout(config.dropout)
#         self.encoder_layer = TransformerEncoderLayer(d_model=config.num_filters, nhead=config.num_head)
#         self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=config.num_layers)
#         self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)
#         self.config = config
#
#     def conv_and_pool(self, x, conv):
#         x = F.relu(conv(x)).squeeze(3)
#         x = F.max_pool1d(x, x.size(2)).squeeze(2)
#         return x
#
#     def forward(self, x):
#         out = self.embedding(x[0])
#         out = out.unsqueeze(1)
#         # apply convolution and pooling for each filter size
#         conv_results = []
#         for i, conv in enumerate(self.convs):
#             conv_result = self.conv_and_pool(out, conv)
#             conv_results.append(conv_result.view(-1, 1, self.config.num_filters))
#         # concatenate the results and reshape them into a matrix
#         out = torch.cat(conv_results, dim=1).view(-1, len(self.convs), self.config.num_filters)
#         # pass the output through the Transformer Encoder
#         out = self.transformer_encoder(out)
#         # reshape the output into a 2D matrix
#         out = out.view(-1, self.config.num_filters * len(self.config.filter_sizes))
#         out = self.dropout(out)
#         out = self.fc(out)
#         return out
