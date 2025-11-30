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
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
# parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
# parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--dataset', type=str,help='DFS,BFS,DFS-Cross,DFS-Selected20')
parser.add_argument('--num_layers', type=int)
parser.add_argument('--num_head', type=int)
parser.add_argument('--num_filters', type=int)
parser.add_argument('--device', type=str)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--use_max_padsize',action="store_true",default=False)
parser.add_argument('--cross_project_only',action="store_true",default=False,help='只运行跨项目评估和LOO-CV评估，跳过分项目评估')
parser.add_argument('--loo_cv_only',action="store_true",default=False,help='只运行LOO-CV评估，跳过其他所有评估')
parser.add_argument('--within_project_only',action="store_true",default=False,help='只运行分项目评估，跳过跨项目评估和LOO-CV评估')
parser.add_argument('--embedding_strategy',type=str,default='random',choices=['random','cbow','skipgram','fasttext','codebert'],help='Embedding策略: random, cbow, skipgram, fasttext, codebert')
parser.add_argument('--embedding_path',type=str,default=None,help='预训练embedding路径（用于cbow/skipgram/fasttext）')




args = parser.parse_args()


if __name__ == '__main__':

    # 根据embedding策略设置embedding参数
    embedding_strategy = args.embedding_strategy
    if embedding_strategy == 'random':
        embedding = 'random'
    elif embedding_strategy in ['cbow', 'skipgram', 'fasttext']:
        # 这些embedding需要预训练，通过embedding_path指定
        embedding = embedding_strategy
    elif embedding_strategy == 'codebert':
        # CodeBERT embedding
        embedding = 'codebert'
        if args.embedding_path:
            print(f"使用CodeBERT embedding从: {args.embedding_path}")
        else:
            print("警告: 未指定CodeBERT embedding路径，将尝试从默认位置加载")
    else:
        embedding = 'random'

    model_name = args.model  # 使用命令行参数传入的模型名称

    import os
    from utils import build_dataset,build_iterator,get_time_dif

    x = import_module('models.' + model_name)

    dataset_list = ['Ant','ArgoUML','Columba','Hibernate','JEdit','JFreeChart','JMeter','JRuby','SQuirrel']
    dataset_len = len(dataset_list)

    # 根据命令行的参数训练 - 分项目评估
    if args.within_project_only or (not args.cross_project_only and not args.loo_cv_only):
        res_table = []
        total_pr = 0
        total_rc = 0
        total_f1 = 0
        file_name = model_name + f"-within-project-{embedding_strategy}-" + str(time.time())+".csv"
        print("="*80)
        print("开始分项目评估 (Within-Project Evaluation)")
        print(f"使用Embedding策略: {embedding_strategy}")
        print("="*80)
        for dataset in dataset_list:
            print(dataset)
            # 指定数据集名称和词嵌入方式
            # 如果是cbow/skipgram/fasttext/codebert，需要设置embedding_path
            if embedding_strategy in ['cbow', 'skipgram', 'fasttext', 'codebert']:
                if args.embedding_path:
                    # 为每个数据集设置对应的embedding路径
                    dataset_embedding_path = args.embedding_path.replace('_embedding', f'_{dataset}_embedding')
                    if not os.path.exists(dataset_embedding_path + '.npy'):
                        dataset_embedding_path = args.embedding_path  # 如果数据集特定路径不存在，使用通用路径
                    setattr(args, 'embedding_path', dataset_embedding_path + '.npy')
                else:
                    # 尝试使用默认路径
                    default_embedding_path = os.path.join('embeddings', f'{embedding_strategy}_embedding.npy')
                    if os.path.exists(default_embedding_path):
                        setattr(args, 'embedding_path', default_embedding_path)
                        print(f"使用默认embedding路径: {default_embedding_path}")
                    else:
                        print(f"警告: 未指定embedding_path且默认路径 {default_embedding_path} 不存在")
                        print(f"将从训练数据构建词表，但embedding将使用random")
            config = x.Config(args, dataset, embedding)
            np.random.seed(1)
            torch.manual_seed(1)
            torch.cuda.manual_seed_all(1)
            torch.backends.cudnn.deterministic = True  # 保证每次结果一样
            start_time = time.time()
            print("Loading data...")
            # 读取数据
            use_config_padsize = not args.use_max_padsize
            print("args.use_max_padsize:",args.use_max_padsize)
            print("use_config_padsize",use_config_padsize)
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
        # 参数也记录到表格中
        config_dict = final_config.__dict__
        print(config_dict)
        parameters = '\n'.join(('%s:%s' % item for item in config_dict.items()))
        para_row = [parameters,'','','']
        res_table.append(para_row)
        res_df = pd.DataFrame(res_table,columns=['project','precision','recall','f1'])
        res_df.to_csv(file_name,index=None,encoding="utf_8_sig")
        print(f"分项目评估结果已保存到: {file_name}")

    # 跨项目评估 (Cross-Project Evaluation) - 合并所有项目
    if not args.loo_cv_only and not args.within_project_only:
        print("\n" + "="*80)
        print("开始跨项目评估 (Cross-Project Evaluation)")
        print("策略: 合并所有9个项目的训练集、验证集、测试集进行训练和评估")
        print("="*80)
        print("合并所有9个项目的 train.txt, dev.txt, test.txt 进行训练和评估")
        print(f"项目列表: {', '.join(dataset_list)}")
        
        # 收集所有9个项目的 train、dev、test 路径
        train_paths = []
        dev_paths = []
        test_paths = []
        
        for dataset in dataset_list:
            dataset_config = x.Config(args, dataset, embedding)
            train_paths.append(dataset_config.train_path)
            dev_paths.append(dataset_config.dev_path)
            test_paths.append(dataset_config.test_path)
        
        print(f"合并 {len(train_paths)} 个训练文件")
        print(f"合并 {len(dev_paths)} 个验证文件")
        print(f"合并 {len(test_paths)} 个测试文件")
        
        # 使用第一个项目的配置作为基础配置
        base_config = x.Config(args, dataset_list[0], embedding)
        
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True
        
        start_time = time.time()
        print("Loading cross-project data...")
        use_config_padsize = not args.use_max_padsize
        
        # 使用合并的所有数据集进行训练和评估
        vocab, train_data, dev_data, test_data = build_dataset(
            base_config, True, use_config_padsize, 
            train_paths=train_paths, 
            dev_paths=dev_paths, 
            test_paths=test_paths
        )
        train_iter = build_iterator(train_data, base_config)
        dev_iter = build_iterator(dev_data, base_config)
        test_iter = build_iterator(test_data, base_config)
        
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)
        
        # 训练模型
        base_config.n_vocab = len(vocab)
        base_config.save_path = base_config.save_path.replace('.ckpt', '_cross_all_projects.ckpt')
        model = x.Model(base_config).to(base_config.device)
        
        if model_name != 'Transformer':
            init_network(model)
        
        # 使用合并后的所有数据进行训练和评估
        final_cross_config, pr, rc, f1 = train(base_config, model, train_iter, dev_iter, test_iter)
        print(
            "======================================跨项目测试集指标:precision:{:.4f}, recall:{:.4f}, f1:{:.4f}======================================".format(
                pr, rc, f1))
        
        # 保存跨项目评估结果
        cross_res_table = [["all_projects_merged", round(pr, 4), round(rc, 4), round(f1, 4)]]
        cross_file_name = model_name + f"-cross-project-all-merged-{embedding_strategy}-" + str(time.time())+".csv"
        cross_res_df = pd.DataFrame(cross_res_table, columns=['test_project','precision','recall','f1'])
        cross_res_df.to_csv(cross_file_name, index=None, encoding="utf_8_sig")
        print(f"跨项目评估结果已保存到: {cross_file_name}")

    # Leave-One-Out Cross-Validation (LOO-CV) 评估
    if not args.within_project_only:
        print("\n" + "="*80)
        print("开始 Leave-One-Out Cross-Validation (LOO-CV) 评估")
        print("策略: 8个项目作为训练集，剩余1个项目作为严格未见的测试集")
        print("="*80)
        loo_cv_res_table = []
        loo_cv_total_pr = 0
        loo_cv_total_rc = 0
        loo_cv_total_f1 = 0
        
        # 对于每个项目作为测试项目（Leave-One-Out）
        for test_dataset in dataset_list:
            print(f"\n{'='*80}")
            print(f"测试项目 (Unseen): {test_dataset}")
            # 获取其他8个项目的训练集路径
            train_datasets = [d for d in dataset_list if d != test_dataset]
            print(f"训练项目 (8个): {', '.join(train_datasets)}")
            
            # 创建测试项目的配置（用于测试集路径）
            test_config = x.Config(args, test_dataset, embedding)
            
            # 收集所有训练项目的训练集路径
            train_paths_loo = []
            for train_dataset in train_datasets:
                train_config = x.Config(args, train_dataset, embedding)
                train_paths_loo.append(train_config.train_path)
            
            # 设置训练配置（使用第一个训练项目的配置作为基础，但测试集使用测试项目的）
            train_config = x.Config(args, train_datasets[0], embedding)
            # 将测试路径设置为测试项目的测试集（用于最终评估）
            train_config.test_path = test_config.test_path
            # 验证集使用第一个训练项目的验证集（用于模型选择）
            # 注意：在LOO-CV评估中，验证集仍使用训练项目的，只有测试集使用测试项目的
            
            np.random.seed(1)
            torch.manual_seed(1)
            torch.cuda.manual_seed_all(1)
            torch.backends.cudnn.deterministic = True
            
            start_time = time.time()
            print("Loading LOO-CV data...")
            use_config_padsize = not args.use_max_padsize
            
            # 使用合并的训练集进行训练
            vocab, train_data, dev_data, test_data = build_dataset(
                train_config, True, use_config_padsize, train_paths=train_paths_loo
            )
            train_iter = build_iterator(train_data, train_config)
            dev_iter = build_iterator(dev_data, train_config)
            test_iter = build_iterator(test_data, train_config)
            
            time_dif = get_time_dif(start_time)
            print("Time usage:", time_dif)
            
            # 训练模型
            train_config.n_vocab = len(vocab)
            # 使用测试项目的保存路径，避免覆盖
            train_config.save_path = test_config.save_path.replace('.ckpt', f'_loo_cv_{test_dataset}.ckpt')
            model = x.Model(train_config).to(train_config.device)
            
            if model_name != 'Transformer':
                init_network(model)
            
            # 使用其他8个项目的训练集训练，在测试项目的测试集上评估
            final_loo_config, pr, rc, f1 = train(train_config, model, train_iter, dev_iter, test_iter)
            print(
                "======================================Unseen测试集指标:precision:{:.4f}, recall:{:.4f}, f1:{:.4f}======================================".format(
                    pr, rc, f1))
            
            loo_cv_total_pr += pr
            loo_cv_total_rc += rc
            loo_cv_total_f1 += f1
            
            row = [test_dataset, round(pr, 4), round(rc, 4), round(f1, 4)]
            loo_cv_res_table.append(row)
        
        # 计算LOO-CV评估的平均值
        loo_cv_res_row = ["average", round(loo_cv_total_pr / dataset_len, 4), 
                         round(loo_cv_total_rc / dataset_len, 4),
                         round(loo_cv_total_f1 / dataset_len, 4)]
        print(
            "\n" + "="*80)
        print(
            "======================================LOO-CV平均指标:precision:{:.4f}, recall:{:.4f}, f1:{:.4f}======================================".format(
                round(loo_cv_total_pr / dataset_len, 4), round(loo_cv_total_rc / dataset_len, 4), 
                round(loo_cv_total_f1 / dataset_len, 4)))
        print("="*80)
        loo_cv_res_table.append(loo_cv_res_row)
        
        # 保存LOO-CV评估结果
        loo_cv_file_name = model_name + f"-loo-cv-{embedding_strategy}-" + str(time.time())+".csv"
        loo_cv_res_df = pd.DataFrame(loo_cv_res_table, columns=['unseen_test_project','precision','recall','f1'])
        loo_cv_res_df.to_csv(loo_cv_file_name, index=None, encoding="utf_8_sig")
        print(f"LOO-CV评估结果已保存到: {loo_cv_file_name}")
