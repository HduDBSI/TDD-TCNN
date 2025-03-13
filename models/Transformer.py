import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class Config(object):

    """配置参数"""
    def __init__(self, args, dataset, embedding):
        self.model_name = 'Transformer'
        path = "/data/td_detection/"
        # path="/root/autodl-tmp/Chinese-Text-Classification-Pytorch/"
        manual_dataset_path = path + dataset + '-' + args.dataset if hasattr(args,
                                                                             'dataset') and args.dataset else path + dataset + '-' + 'DFS'
        default_dataset_path = path + dataset + '-' + 'DFS'

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
        self.pad_size = args.pad_size if hasattr(args,'pad_size') and args.pad_size else 512
        self.learning_rate = args.learning_rate if hasattr(args,'learning_rate') and args.learning_rate else 5e-4
        self.embed = args.embed if hasattr(args,'embed') and args.embed else self.embedding_pretrained.size(
            1) if self.embedding_pretrained is not None else 300

        self.embedding = args.embedding if hasattr(args,'embedding') and args.embedding else embedding
        self.word_save_path = args.word_save_path if hasattr(args,
                                                             'word_save_path') and args.word_save_path else "/data/embedding/fasttest-word.ckpt"
        self.word2idx_path = args.word2idx_path if hasattr(args,
                                                           'word2idx_path') and args.word2idx_path else "/data/embedding/fasttest-word2idx.ckpt"
        self.num_encoder = args.num_encoder if hasattr(args,'num_encoder') and args.num_encoder else 4
        self.num_head = args.num_head if hasattr(args,'num_head') and args.num_head else 3

        self.dim_model = args.dim_model if hasattr(args,'dim_model') and args.dim_model else 300
        self.dim_feedforward = args.dim_feedforward if hasattr(args,'dim_feedforward') and args.dim_feedforward else 1024

        self.device = torch.device('cuda:' + args.device) if hasattr(args,'device') and args.device else torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')


'''Attention Is All You Need'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        self.postion_embedding = Positional_Encoding(config.embed, config.pad_size, config.dropout, config.device)
        # self.encoder = Encoder(config.dim_model, config.num_head, config.dim_feedforward, config.dropout)
        r"""
        Args:
            dim_model: the number of expected features in the input (required).
            num_head: the number of heads in the multiheadattention models (required).
            dim_feedforward: the dimension of the feedforward network model (default=2048).
            dropout: the dropout value (default=0.1).
        """
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            # Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
            for _ in range(config.num_encoder)])

        self.fc1 = nn.Linear(config.pad_size * config.dim_model, config.num_classes)
        # self.fc2 = nn.Linear(config.last_hidden, config.num_classes)
        # self.fc1 = nn.Linear(config.dim_model, config.num_classes)

    def forward(self, x):
        out = self.embedding(x[0])
        out = self.postion_embedding(out)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        # out = torch.mean(out, 1)
        out = self.fc1(out)
        return out


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out
