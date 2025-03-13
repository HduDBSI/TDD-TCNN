import gensim
import numpy as np
import pandas as pd
import torch
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


# # 预训练的词向量文件路径
# vec_path = "/data/lsc/embedding/wiki-news-300d-1M.vec"
# # 转换成的bin文件名称
embed_path = "/data/lsc/embedding/wiki-news-300d-1M.bin"
# # 加载词向量文件
# wv_from_text = gensim.models.KeyedVectors.load_word2vec_format(vec_path, binary=False,limit=20000)
#
# # 如果每次都用上面的方法加载，速度非常慢，可以将词向量文件保存成bin文件，以后就加载bin文件，速度会变快
# wv_from_text.init_sims(replace=True)
# wv_from_text.save(vec_path.replace(".vec", ".bin"))



wv_from_text = gensim.models.KeyedVectors.load(embed_path, mmap='r')
# 获取所有词
vocab = wv_from_text.vocab
# 词表中加入UNK和PAD
vocab.update({UNK: len(vocab), PAD: len(vocab) + 1})

# 获取所有向量
word_embedding = wv_from_text.syn0
word_embedding=np.r_[word_embedding,[np.zeros(300)],[np.zeros(300)]]

# 将向量和词保存下来
word_embed_save_path = "/data/lsc/embedding/fasttest-emebed.ckpt"
word_save_path = "/data/lsc/embedding/fasttest-word.ckpt"
np.save(word_embed_save_path, word_embedding)
pd.to_pickle(vocab, word_save_path)

# 加载保存的向量和词
weight_numpy = np.load(file="/data/lsc/embedding/fasttest-emebed.ckpt.npy")
vocab = pd.read_pickle(word_save_path)
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for idx, word in enumerate(vocab)}
pd.to_pickle(word2idx, "/data/lsc/embedding/fasttest-word2idx.ckpt")
pd.to_pickle(idx2word, "/data/lsc/embedding/fasttext-idx2word.ckpt")

# 加载
embedding =torch.nn.Embedding.from_pretrained(torch.FloatTensor(weight_numpy))
print("ok")
