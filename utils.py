# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import pandas as pd


MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号



def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            # 统计词频
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        # 筛选掉出现频率没有达到min_freq的token，如果词典数量过多则取前MAX_VOCAB_SIZE个
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        # 对token编号，从0开始
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def merge_files(file_paths, output_path):
    """合并多个文件到一个临时文件"""
    with open(output_path, 'w', encoding='UTF-8') as outfile:
        for file_path in file_paths:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='UTF-8') as infile:
                    for line in infile:
                        outfile.write(line)
            else:
                print(f"Warning: {file_path} does not exist, skipping...")

def merge_train_files(train_paths, output_path):
    """合并多个训练文件到一个临时文件（保持向后兼容）"""
    merge_files(train_paths, output_path)


def build_dataset(config, ues_word=True,use_config_pad=True, train_paths=None, dev_paths=None, test_paths=None):
    """
    构建数据集
    train_paths: 如果提供多个训练文件路径，将合并它们用于跨项目训练
    dev_paths: 如果提供多个验证文件路径，将合并它们用于跨项目训练
    test_paths: 如果提供多个测试文件路径，将合并它们用于跨项目训练
    """
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    # if os.path.exists(config.vocab_path):
    #     vocab = pkl.load(open(config.vocab_path, 'rb'))
    # else:
    #     vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
    #     pkl.dump(vocab, open(config.vocab_path, 'wb'))
    # print(f"Vocab size: {len(vocab)}")

    # 如果提供了多个训练路径，合并它们
    actual_train_path = config.train_path
    temp_train_path = None
    if train_paths and len(train_paths) > 0:
        if len(train_paths) > 1:
            import tempfile
            temp_train_path = tempfile.mktemp(suffix='.txt', prefix='merged_train_')
            merge_files(train_paths, temp_train_path)
            actual_train_path = temp_train_path
            print(f"Merged {len(train_paths)} training files into temporary file: {temp_train_path}")
        else:
            # 只有一个训练路径，直接使用
            actual_train_path = train_paths[0]
    
    # 如果提供了多个验证路径，合并它们
    actual_dev_path = config.dev_path
    temp_dev_path = None
    if dev_paths and len(dev_paths) > 0:
        if len(dev_paths) > 1:
            import tempfile
            temp_dev_path = tempfile.mktemp(suffix='.txt', prefix='merged_dev_')
            merge_files(dev_paths, temp_dev_path)
            actual_dev_path = temp_dev_path
            print(f"Merged {len(dev_paths)} dev files into temporary file: {temp_dev_path}")
        else:
            actual_dev_path = dev_paths[0]
    
    # 如果提供了多个测试路径，合并它们
    actual_test_path = config.test_path
    temp_test_path = None
    if test_paths and len(test_paths) > 0:
        if len(test_paths) > 1:
            import tempfile
            temp_test_path = tempfile.mktemp(suffix='.txt', prefix='merged_test_')
            merge_files(test_paths, temp_test_path)
            actual_test_path = temp_test_path
            print(f"Merged {len(test_paths)} test files into temporary file: {temp_test_path}")
        else:
            actual_test_path = test_paths[0]

    if config.embedding=='random':
        # 自己构建词表
        vocab = build_vocab(actual_train_path,tokenizer=tokenizer,max_size=MAX_VOCAB_SIZE,min_freq=1)
        pkl.dump(vocab,open(config.vocab_path,'wb'))
    elif config.embedding in ['cbow', 'skipgram', 'fasttext', 'codebert']:
        # 使用训练好的embedding词表
        if hasattr(config, 'embedding_path') and config.embedding_path:
            # 从embedding路径推断词表路径
            base_path = config.embedding_path.replace('.npy', '')
            vocab_path = base_path + '_vocab.pkl'
            word2idx_path = base_path + '_word2idx.pkl'
            if os.path.exists(vocab_path):
                vocab = pkl.load(open(vocab_path, 'rb'))
                print(f"加载{config.embedding}词表从: {vocab_path}")
                print(f"Vocab size: {len(vocab)}")
                # 更新config中的词表路径
                if hasattr(config, 'word_save_path'):
                    config.word_save_path = vocab_path
                if hasattr(config, 'word2idx_path'):
                    config.word2idx_path = word2idx_path
            else:
                print(f"警告: {vocab_path} 不存在，使用随机词表")
                vocab = build_vocab(actual_train_path,tokenizer=tokenizer,max_size=MAX_VOCAB_SIZE,min_freq=1)
                pkl.dump(vocab,open(config.vocab_path,'wb'))
        elif config.embedding == 'codebert':
            # CodeBERT默认路径
            codebert_vocab_path = os.path.join('embeddings', 'codebert_embedding_vocab.pkl')
            if os.path.exists(codebert_vocab_path):
                vocab = pkl.load(open(codebert_vocab_path, 'rb'))
                print(f"加载CodeBERT词表从: {codebert_vocab_path}")
                print(f"Vocab size: {len(vocab)}")
            else:
                print(f"警告: {codebert_vocab_path} 不存在，使用随机词表")
                vocab = build_vocab(actual_train_path,tokenizer=tokenizer,max_size=MAX_VOCAB_SIZE,min_freq=1)
                pkl.dump(vocab,open(config.vocab_path,'wb'))
        else:
            # 尝试从默认位置加载embedding词表
            default_embedding_paths = [
                os.path.join('embeddings', f'{config.embedding}_embedding_vocab.pkl'),
            ]
            vocab_loaded = False
            for vocab_path in default_embedding_paths:
                if os.path.exists(vocab_path):
                    vocab = pkl.load(open(vocab_path, 'rb'))
                    print(f"加载{config.embedding}词表从默认路径: {vocab_path}")
                    print(f"Vocab size: {len(vocab)}")
                    vocab_loaded = True
                    # 更新config中的词表路径
                    if hasattr(config, 'word_save_path'):
                        config.word_save_path = vocab_path
                    break
            
            if not vocab_loaded:
                # 如果找不到预训练词表，构建新词表
                print(f"警告: 未找到{config.embedding}的预训练词表，从训练数据构建新词表")
                vocab = build_vocab(actual_train_path,tokenizer=tokenizer,max_size=MAX_VOCAB_SIZE,min_freq=1)
                pkl.dump(vocab,open(config.vocab_path,'wb'))
    else:
        # 使用预训练的词表
        vocab = pd.read_pickle(config.word_save_path)
        print(f"Vocab size: {len(vocab)}")

    def get_pad_size(path):
        pad_size=0
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                # 一句话对应的id集合
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if seq_len>pad_size:
                    pad_size=seq_len
        return pad_size

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                # 一句话对应的id集合
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size

                if config.embedding == 'random':
                # 手动构建词表时自己映射word和id
                    for word in token:
                        words_line.append(vocab.get(word, vocab.get(UNK)))
                else:
                    # 使用预训练词表的word2idx
                    word2idx = pd.read_pickle(config.word2idx_path)
                    for word in token:
                      words_line.append(word2idx.get(word, word2idx.get(UNK)))

                contents.append((words_line, int(label), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]

    # 不使用默认配置中的pad_size
    if use_config_pad == False:
        pad_size = get_pad_size(actual_train_path)
        config.pad_size=pad_size
    train = load_dataset(actual_train_path, config.pad_size)
    dev = load_dataset(actual_dev_path, config.pad_size)
    test = load_dataset(actual_test_path, config.pad_size)
    # print(pad_size)
    
    # 清理临时文件
    temp_files = [temp_train_path, temp_dev_path, temp_test_path]
    for temp_file in temp_files:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
    
    return vocab, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = max(1, len(batches) // batch_size)
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = "THUCNews/data/train.txt"
    vocab_dir = "THUCNews/data/vocab.pkl"
    pretrain_dir = "THUCNews/data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "THUCNews/data/embedding_SougouNews"

    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))



    # # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
    # tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
    # word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
    # pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
