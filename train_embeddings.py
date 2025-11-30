# coding: UTF-8
"""
训练不同的embedding策略用于消融实验：
1. CBOW (Word2Vec)
2. Skip-gram (Word2Vec)
3. FastText
4. CodeBERT
"""
import os
import pickle as pkl
import numpy as np
import pandas as pd
from gensim.models import Word2Vec, FastText
from tqdm import tqdm
import argparse
import torch

UNK, PAD = '<UNK>', '<PAD>'


def merge_train_files(dataset_dir='dataset', dataset_suffix='-DFS', output_file='merged_train.txt'):
    """
    合并所有项目的训练文件
    
    Args:
        dataset_dir: 数据集根目录
        dataset_suffix: 数据集后缀（如 '-DFS', '-BFS' 等）
        output_file: 输出文件名
    """
    dataset_list = ['Ant', 'ArgoUML', 'Columba', 'Hibernate', 'JEdit', 
                    'JFreeChart', 'JMeter', 'JRuby', 'SQuirrel']
    
    print(f"合并训练文件到: {output_file}")
    merged_count = 0
    
    with open(output_file, 'w', encoding='UTF-8') as outfile:
        for dataset in dataset_list:
            train_file = os.path.join(dataset_dir, f"{dataset}{dataset_suffix}", 'data', 'train.txt')
            if os.path.exists(train_file):
                with open(train_file, 'r', encoding='UTF-8') as infile:
                    for line in infile:
                        outfile.write(line)
                        merged_count += 1
                print(f"  ✓ 已合并: {train_file}")
            else:
                print(f"  ✗ 警告: {train_file} 不存在，跳过")
    
    print(f"\n总共合并了 {merged_count} 行数据")
    return output_file


def load_corpus(file_path):
    """从训练文件中加载语料库"""
    sentences = []
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f, desc="Loading corpus"):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            # 以空格分割，word-level
            words = content.split(' ')
            if len(words) > 0:
                sentences.append(words)
    return sentences


def train_word2vec(sentences, output_path, embedding_type='cbow', size=300, window=5, negative=5, min_count=1):
    """
    训练Word2Vec模型 (CBOW or Skip-gram)
    
    Args:
        sentences: 训练语料
        output_path: 输出路径（不含扩展名）
        embedding_type: 'cbow' 或 'skipgram'
        size: 词向量维度
        window: 窗口大小
        negative: 负采样数量
        min_count: 最小词频
    """
    print(f"\n训练 {embedding_type.upper()} embedding...")
    print(f"参数: size={size}, window={window}, negative={negative}, min_count={min_count}")
    
    sg = 0 if embedding_type.lower() == 'cbow' else 1
    
    # 兼容新旧版本的gensim参数
    try:
        # 新版本gensim (4.0+) 使用 vector_size
        model = Word2Vec(
            sentences=sentences,
            vector_size=size,
            window=window,
            sg=sg,  # 0=CBOW, 1=Skip-gram
            negative=negative,
            min_count=min_count,
            workers=4,
            epochs=5
        )
    except TypeError:
        # 旧版本gensim (3.x) 使用 size 和 iter
        model = Word2Vec(
            sentences=sentences,
            size=size,
            window=window,
            sg=sg,  # 0=CBOW, 1=Skip-gram
            negative=negative,
            min_count=min_count,
            workers=4,
            iter=5
        )
    
    # 保存模型
    model.save(f"{output_path}.model")
    print(f"模型已保存到: {output_path}.model")
    
    # 保存词向量和词表
    # 兼容新旧版本的gensim
    try:
        # 新版本gensim (4.0+)
        vocab = dict(model.wv.key_to_index)
        word_embedding = model.wv.vectors
    except AttributeError:
        # 旧版本gensim (3.x)
        vocab = dict(model.wv.vocab)
        word_embedding = model.wv.vectors
    
    vocab.update({UNK: len(vocab), PAD: len(vocab) + 1})
    
    # 添加UNK和PAD的向量（零向量）
    unk_pad_vectors = np.zeros((2, size), dtype=np.float32)
    word_embedding = np.vstack([word_embedding, unk_pad_vectors])
    
    # 保存
    np.save(f"{output_path}.npy", word_embedding)
    pkl.dump(vocab, open(f"{output_path}_vocab.pkl", 'wb'))
    
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    pkl.dump(word2idx, open(f"{output_path}_word2idx.pkl", 'wb'))
    pkl.dump(idx2word, open(f"{output_path}_idx2word.pkl", 'wb'))
    
    print(f"Embedding矩阵已保存到: {output_path}.npy")
    print(f"词表已保存到: {output_path}_vocab.pkl")
    print(f"词向量维度: {word_embedding.shape}")
    print(f"词汇表大小: {len(vocab)}")


def train_fasttext(sentences, output_path, size=300, window=5, negative=5, min_count=1):
    """
    训练FastText模型
    
    Args:
        sentences: 训练语料
        output_path: 输出路径（不含扩展名）
        size: 词向量维度
        window: 窗口大小
        negative: 负采样数量
        min_count: 最小词频
    """
    print(f"\n训练 FastText embedding...")
    print(f"参数: size={size}, window={window}, negative={negative}, min_count={min_count}")
    
    # 兼容新旧版本的gensim参数
    try:
        # 新版本gensim (4.0+) 使用 vector_size
        model = FastText(
            sentences=sentences,
            vector_size=size,
            window=window,
            sg=0,  # FastText默认使用CBOW
            negative=negative,
            min_count=min_count,
            workers=4,
            epochs=5
        )
    except TypeError:
        # 旧版本gensim (3.x) 使用 size 和 iter
        model = FastText(
            sentences=sentences,
            size=size,
            window=window,
            sg=0,  # FastText默认使用CBOW
            negative=negative,
            min_count=min_count,
            workers=4,
            iter=5
        )
    
    # 保存模型
    model.save(f"{output_path}.model")
    print(f"模型已保存到: {output_path}.model")
    
    # 保存词向量和词表
    # 兼容新旧版本的gensim
    try:
        # 新版本gensim (4.0+)
        vocab = dict(model.wv.key_to_index)
        word_embedding = model.wv.vectors
    except AttributeError:
        # 旧版本gensim (3.x)
        vocab = dict(model.wv.vocab)
        word_embedding = model.wv.vectors
    
    vocab.update({UNK: len(vocab), PAD: len(vocab) + 1})
    
    # 添加UNK和PAD的向量（零向量）
    unk_pad_vectors = np.zeros((2, size), dtype=np.float32)
    word_embedding = np.vstack([word_embedding, unk_pad_vectors])
    
    # 保存
    np.save(f"{output_path}.npy", word_embedding)
    pkl.dump(vocab, open(f"{output_path}_vocab.pkl", 'wb'))
    
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    pkl.dump(word2idx, open(f"{output_path}_word2idx.pkl", 'wb'))
    pkl.dump(idx2word, open(f"{output_path}_idx2word.pkl", 'wb'))
    
    print(f"Embedding矩阵已保存到: {output_path}.npy")
    print(f"词表已保存到: {output_path}_vocab.pkl")
    print(f"词向量维度: {word_embedding.shape}")
    print(f"词汇表大小: {len(vocab)}")


def extract_codebert_embeddings(train_file, output_path, model_name='microsoft/codebert-base', device='cuda:0', batch_size=32):
    """
    使用CodeBERT提取embedding
    
    Args:
        train_file: 训练文件路径
        output_path: 输出路径（不含扩展名）
        model_name: CodeBERT模型名称
        device: 设备
        batch_size: 批处理大小
    """
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        print("错误: 需要安装transformers库: pip install transformers")
        return
    
    print(f"\n提取 CodeBERT embedding...")
    print(f"模型: {model_name}")
    
    # 加载tokenizer和模型
    print("加载CodeBERT模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device_obj = torch.device(device if torch.cuda.is_available() and 'cuda' in device else 'cpu')
    model.to(device_obj)
    model.eval()
    
    # 获取CodeBERT的embedding维度
    codebert_embed_dim = model.config.hidden_size  # 通常是768
    print(f"CodeBERT embedding维度: {codebert_embed_dim}")
    
    # 构建词汇表（从训练文件中）
    print("构建词汇表...")
    vocab_set = set()
    with open(train_file, 'r', encoding='UTF-8') as f:
        for line in tqdm(f, desc="Loading corpus"):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            words = content.split(' ')
            vocab_set.update(words)
    
    vocab_list = sorted(list(vocab_set))
    print(f"词汇表大小: {len(vocab_list)}")
    
    # 批量提取embedding以提高效率
    print("提取词embedding（批量处理）...")
    word_embeddings = {}
    
    # 将词汇表分批处理
    for i in tqdm(range(0, len(vocab_list), batch_size), desc="Processing batches"):
        batch_words = vocab_list[i:i+batch_size]
        batch_texts = [' '.join([word]) for word in batch_words]  # 每个词作为单独的文本
        
        # 批量编码
        encoded = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        input_ids = encoded['input_ids'].to(device_obj)
        attention_mask = encoded['attention_mask'].to(device_obj)
        
        # 批量获取embedding
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # 使用[CLS] token的embedding（每个序列的第一个token）
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # 存储每个词的embedding
        for j, word in enumerate(batch_words):
            word_embeddings[word] = batch_embeddings[j]
    
    # 构建词汇表映射
    vocab = {word: idx for idx, word in enumerate(vocab_list)}
    vocab.update({UNK: len(vocab), PAD: len(vocab) + 1})
    
    # 创建embedding矩阵
    embedding_matrix = np.zeros((len(vocab), codebert_embed_dim), dtype=np.float32)
    
    # UNK和PAD使用零向量
    unk_idx = vocab[UNK]
    pad_idx = vocab[PAD]
    embedding_matrix[unk_idx] = np.zeros(codebert_embed_dim, dtype=np.float32)
    embedding_matrix[pad_idx] = np.zeros(codebert_embed_dim, dtype=np.float32)
    
    # 填充其他词的embedding
    for word, idx in vocab.items():
        if word in word_embeddings:
            embedding_matrix[idx] = word_embeddings[word]
        elif word not in [UNK, PAD]:
            # 如果词不在word_embeddings中，使用UNK的embedding
            embedding_matrix[idx] = embedding_matrix[unk_idx]
    
    # 保存
    np.save(f"{output_path}.npy", embedding_matrix)
    pkl.dump(vocab, open(f"{output_path}_vocab.pkl", 'wb'))
    
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    pkl.dump(word2idx, open(f"{output_path}_word2idx.pkl", 'wb'))
    pkl.dump(idx2word, open(f"{output_path}_idx2word.pkl", 'wb'))
    
    print(f"Embedding矩阵已保存到: {output_path}.npy")
    print(f"词表已保存到: {output_path}_vocab.pkl")
    print(f"词向量维度: {embedding_matrix.shape}")
    print(f"词汇表大小: {len(vocab)}")


def main():
    parser = argparse.ArgumentParser(description='训练不同的embedding策略')
    parser.add_argument('--train_file', type=str, default=None, help='训练文件路径（如果不存在会自动合并所有项目的训练文件）')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--embedding_type', type=str, choices=['cbow', 'skipgram', 'fasttext', 'codebert', 'all'], 
                       default='all', help='要训练的embedding类型')
    parser.add_argument('--codebert_model', type=str, default='microsoft/codebert-base', help='CodeBERT模型名称')
    parser.add_argument('--codebert_device', type=str, default='cuda:0', help='CodeBERT使用的设备')
    parser.add_argument('--size', type=int, default=300, help='词向量维度')
    parser.add_argument('--window', type=int, default=5, help='窗口大小')
    parser.add_argument('--negative', type=int, default=5, help='负采样数量')
    parser.add_argument('--min_count', type=int, default=1, help='最小词频')
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='数据集根目录')
    parser.add_argument('--dataset_suffix', type=str, default='-DFS', help='数据集后缀（如 -DFS, -BFS）')
    parser.add_argument('--auto_merge', action='store_true', default=False, help='自动合并所有项目的训练文件')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 确定训练文件
    train_file = args.train_file
    if train_file is None or not os.path.exists(train_file):
        # 自动合并所有项目的训练文件
        merged_file = os.path.join(args.output_dir, 'merged_train.txt')
        if not os.path.exists(merged_file) or args.auto_merge:
            print("="*80)
            print("自动合并所有项目的训练文件...")
            print("="*80)
            train_file = merge_train_files(args.dataset_dir, args.dataset_suffix, merged_file)
        else:
            print(f"使用已存在的合并文件: {merged_file}")
            train_file = merged_file
    else:
        train_file = args.train_file
    
    # 检查文件是否存在
    if not os.path.exists(train_file):
        print(f"错误: 训练文件不存在: {train_file}")
        print("\n请使用以下方法之一：")
        print("1. 手动创建合并的训练文件")
        print("2. 使用 --auto_merge 参数自动合并")
        print("3. 使用 --train_file 指定存在的训练文件路径")
        return
    
    # 加载语料库
    print("="*80)
    print("加载训练语料...")
    print("="*80)
    sentences = load_corpus(train_file)
    print(f"加载了 {len(sentences)} 个句子")
    
    # 训练不同的embedding
    if args.embedding_type in ['cbow', 'all']:
        output_path = os.path.join(args.output_dir, 'cbow_embedding')
        train_word2vec(sentences, output_path, 'cbow', args.size, args.window, args.negative, args.min_count)
    
    if args.embedding_type in ['skipgram', 'all']:
        output_path = os.path.join(args.output_dir, 'skipgram_embedding')
        train_word2vec(sentences, output_path, 'skipgram', args.size, args.window, args.negative, args.min_count)
    
    if args.embedding_type in ['fasttext', 'all']:
        output_path = os.path.join(args.output_dir, 'fasttext_embedding')
        train_fasttext(sentences, output_path, args.size, args.window, args.negative, args.min_count)
    
    if args.embedding_type in ['codebert', 'all']:
        output_path = os.path.join(args.output_dir, 'codebert_embedding')
        extract_codebert_embeddings(train_file, output_path, args.codebert_model, args.codebert_device)
    
    print("\n所有embedding训练完成！")


if __name__ == '__main__':
    main()

