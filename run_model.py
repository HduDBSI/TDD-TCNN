# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
# from train_eval_balanced import train, init_network

from importlib import import_module
import pandas as pd
import argparse
# import wandb
import random
import torch, gc

gc.collect()
torch.cuda.empty_cache()


parser = argparse.ArgumentParser(description='TD Classification')
# parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
# parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
# parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--dataset', type=str,help='DFS,BFS,DFS-Cross,DFS-Selected20')
parser.add_argument('--num_layers', type=int)
parser.add_argument('--num_head', type=int)
parser.add_argument('--num_filters', type=int)
parser.add_argument('--device', type=str)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--use_config_padsize', type=bool)


args = parser.parse_args()


if __name__ == '__main__':
    # dataset = 'THUCNews'  # 数据集
    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    # embedding = 'embedding_SougouNews.npz'
    # embedding = 'fasttest'
    embedding = 'random'
    # if args.embedding == 'random':
    #     embedding = 'random'
    # model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    # Transformer-Pytorch
    model_name ='CNNTransformer-Seq-TC' # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer CNNTransformer CNNTransformer-Seq
    # 根据模型指定数据读取方式
    # if model_name == 'FastText':
    #     from utils_fasttext import build_dataset, build_iterator, get_time_dif
    #     embedding = 'random'
    # else:
    #     from utils import build_dataset, build_iterator, get_time_dif

    # 不使用fasttext作为对比方法，所以直接引入utils中的数据读取代码
    from utils import build_dataset,build_iterator,get_time_dif

    x = import_module('models.' + model_name)

    dataset_list = ['Ant','ArgoUML','Columba','Hibernate','JEdit','JFreeChart','JMeter','JRuby','SQuirrel']
    dataset_len = len(dataset_list)

    # # 批量调整参数
    # adjust_para='pad_size'
    # # =================================================================================================
    # para_list= [1550,1600,1650,1700,1750,1800,1850,1900,1950,2000]
    # for para in para_list:
    #     res_table = []
    #     total_pr = 0
    #     total_rc = 0
    #     total_f1 = 0
    #     print(para)
    #     file_name = model_name + "-" + adjust_para+"-" + str(para) + ".csv"# str(time.time())
    #     for dataset in dataset_list:
    #         print(dataset)
    #         # 指定数据集名称和词嵌入方式
    #         config = x.Config(args,dataset,embedding)
    #         setattr(config,adjust_para,para)
    #         np.random.seed(1)
    #         torch.manual_seed(1)
    #         torch.cuda.manual_seed_all(1)
    #         torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    #         start_time = time.time()
    #         print("Loading data...")
    #         # 读取数据
    #         # 默认使用config中的padsize
    #         use_config_padsize = args.use_config_padsize if hasattr(args,
    #                                                                 'use_config_padsize') and args.use_config_padsize else True
    #         vocab,train_data,dev_data,test_data = build_dataset(config,True,use_config_padsize)
    #         train_iter = build_iterator(train_data,config)
    #         dev_iter = build_iterator(dev_data,config)
    #         test_iter = build_iterator(test_data,config)
    #         time_dif = get_time_dif(start_time)
    #         print("Time usage:",time_dif)
    #         # train
    #         config.n_vocab = len(vocab)
    #         model = x.Model(config).to(config.device)
    #
    #         if model_name != 'Transformer':
    #             init_network(model)
    #         print(model.parameters)
    #         # 使用一个项目的训练集数据进行训练并得到测试集上的结果
    #         final_config,pr,rc,f1=train(config,model,train_iter,dev_iter,test_iter)
    #         print("======================================测试集指标:precision:{:.4f}, recall:{:.4f}, f1:{:.4f}======================================".format(pr,rc,f1))
    #         # wandb.log({"Precision": pr,"Recall": rc,"F1-score": f1})
    #
    #         total_pr += pr
    #         total_rc += rc
    #         total_f1 += f1
    #
    #         row = [dataset,round(pr,4),round(rc,4),round(f1,4)]
    #         res_table.append(row)
    #     res_row = ["average",round(total_pr / dataset_len,4),round(total_rc / dataset_len,4),round(total_f1 / dataset_len,4)]
    #     print(
    #         "======================================平均指标:precision:{:.4f}, recall:{:.4f}, f1:{:.4f}======================================".format(
    #             round(total_pr / dataset_len,4),round(total_rc / dataset_len,4),round(total_f1 / dataset_len,4)))
    #     res_table.append(res_row)
    #     # 参上也记录到表格中
    #     config_dict = final_config.__dict__
    #     print(config_dict)
    #     # parameters = config_dict.items()
    #     parameters='\n'.join(('%s:%s' % item for item in config_dict.items()))
    #     para_row = [parameters,'','','']
    #     res_table.append(para_row)
    #     res_df = pd.DataFrame(res_table,columns=['project','precision','recall','f1'])
    #     res_df.to_csv(file_name,index=None,encoding="utf_8_sig")



    # 根据命令行的参数训练
    res_table = []
    total_pr = 0
    total_rc = 0
    total_f1 = 0
    file_name = model_name + "-" + str(time.time())+".csv"
    for dataset in dataset_list:
        print(dataset)
        # 指定数据集名称和词嵌入方式
        config = x.Config(args,dataset ,embedding)
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样
        start_time = time.time()
        print("Loading data...")
        # 读取数据
        # 默认使用config中的padsize
        use_config_padsize = args.use_config_padsize if hasattr(args, 'use_config_padsize') and args.use_config_padsize else True
        vocab,train_data,dev_data,test_data = build_dataset(config,True,use_config_padsize)
        train_iter = build_iterator(train_data,config)
        dev_iter = build_iterator(dev_data,config)
        test_iter = build_iterator(test_data,config)
        time_dif = get_time_dif(start_time)
        print("Time usage:",time_dif)
        # train
        config.n_vocab = len(vocab)
        model = x.Model(config).to(config.device)

        if model_name != 'Transformer':
            init_network(model)
        print(model.parameters)
        # 使用一个项目的训练集数据进行训练并得到测试集上的结果
        final_config,pr,rc,f1 = train(config,model,train_iter,dev_iter,test_iter)
        print(
            "======================================测试集指标:precision:{:.4f}, recall:{:.4f}, f1:{:.4f}======================================".format(
                pr,rc,f1))
        # wandb.log({"Precision": pr,"Recall": rc,"F1-score": f1})

        total_pr += pr
        total_rc += rc
        total_f1 += f1

        row = [dataset,round(pr,4),round(rc,4),round(f1,4)]
        res_table.append(row)
    res_row = ["average",round(total_pr / dataset_len,4),round(total_rc / dataset_len,4),
               round(total_f1 / dataset_len,4)]
    print(
        "======================================平均指标:precision:{:.4f}, recall:{:.4f}, f1:{:.4f}======================================".format(
            round(total_pr / dataset_len,4),round(total_rc / dataset_len,4),round(total_f1 / dataset_len,4)))
    res_table.append(res_row)
    # 参上也记录到表格中
    config_dict = final_config.__dict__
    print(config_dict)
    parameters = '\n'.join(('%s:%s' % item for item in config_dict.items()))
    para_row = [parameters,'','','']
    res_table.append(para_row)
    res_df = pd.DataFrame(res_table,columns=['project','precision','recall','f1'])
    res_df.to_csv(file_name,index=None,encoding="utf_8_sig")

