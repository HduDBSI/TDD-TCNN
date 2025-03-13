# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class Config(object):

    """配置参数"""
    def __init__(self, args,dataset, embedding):
        self.model_name = 'TextCNN'
        # path = "/root/autodl-tmp/Chinese-Text-Classification-Pytorch/"
        path = "/data/td_detection/"
        # path = "/data/td_severity/"

        manual_dataset_path = path + dataset + '-' + args.dataset if hasattr(args,
                                                                             'dataset') and args.dataset else path + dataset + '-' + 'DFS'
        # default_dataset_path = path + dataset + '-' + 'DFS'
        default_dataset_path = path + dataset + '-' + 'Severity'


        self.train_path = manual_dataset_path + '/data/train.txt' if hasattr(args,
                                                                             'dataset') and args.dataset else default_dataset_path + '/data/train.txt'
        self.dev_path = manual_dataset_path + '/data/dev.txt' if hasattr(args,
                                                                         'dataset') and args.dataset else default_dataset_path + '/data/dev.txt'
        self.test_path = manual_dataset_path + '/data/test.txt' if hasattr(args,
                                                                           'dataset') and args.dataset else default_dataset_path + '/data/test.txt'
        self.class_list = [x.strip() for x in open(
            manual_dataset_path + '/data/class.txt',encoding='utf-8').readlines()] if hasattr(args,
                                                                                              'dataset') and args.dataset else [
            x.strip() for x in open(
                default_dataset_path + '/data/class.txt',encoding='utf-8').readlines()]

        self.vocab_path = manual_dataset_path + '/data/vocab.pkl' if hasattr(args,
                                                                             'dataset') and args.dataset else default_dataset_path + '/data/vocab.pkl'
        self.save_path = manual_dataset_path + '/saved_dict/' + self.model_name + '.ckpt' if hasattr(args,
                                                                                                     'dataset') and args.dataset else default_dataset_path + '/saved_dict/' + self.model_name + '.ckpt'
        self.log_path = manual_dataset_path + '/log/' + self.model_name if hasattr(args,
                                                                                   'dataset') and args.dataset else default_dataset_path + '/log/' + self.model_name

        self.embedding_pretrained = args.embedding_pretrained if hasattr(args,
                                                                         'embedding_pretrained') and args.embedding_pretrained else torch.tensor(
            np.load(default_dataset_path + '/data/' + embedding)["embeddings"].astype('float32')) \
            if embedding != 'random' else None

        self.dropout = args.dropout if hasattr(args,'dropout') and args.dropout else 0.5
        self.require_improvement = args.require_improvement if hasattr(args,
                                                                       'require_improvement') and args.require_improvement else 1000
        self.num_classes = args.num_classes if hasattr(args,'num_classes') and args.num_classes else len(
            self.class_list)
        self.n_vocab = args.n_vocab if hasattr(args,'n_vocab') and args.n_vocab else 0
        self.num_epochs = args.num_epochs if hasattr(args,'num_epochs') and args.num_epochs else 200
        self.batch_size = args.batch_size if hasattr(args,'batch_size') and args.batch_size else 128
        self.pad_size = args.pad_size if hasattr(args,'pad_size') and args.pad_size else 850
        self.learning_rate = args.learning_rate if hasattr(args,'learning_rate') and args.learning_rate else 0.01
        self.embed = args.embed if hasattr(args,'embed') and args.embed else self.embedding_pretrained.size(
            1) if self.embedding_pretrained is not None else 300

        self.embedding = args.embedding if hasattr(args,'embedding') and args.embedding else embedding
        self.word_save_path = args.word_save_path if hasattr(args,
                                                             'word_save_path') and args.word_save_path else "/data/embedding/fasttest-word.ckpt"
        self.word2idx_path = args.word2idx_path if hasattr(args,
                                                           'word2idx_path') and args.word2idx_path else "/data/embedding/fasttest-word2idx.ckpt"

        self.filter_sizes = args.filter_sizes if hasattr(args,'filter_sizes') and args.filter_sizes else (1,2,3,4)
        self.num_filters = args.num_filters if hasattr(args,'num_filters') and args.num_filters else 128
        self.device = torch.device('cuda:'+args.device) if hasattr(args, 'device') and args.device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        # out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
