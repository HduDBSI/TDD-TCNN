# Technical Debt Detection based on Transformer with Convolutional Neural Network (TDD-TCNN)
Source code and data for our paper "Detecting Technical Debt from Method-level Code Snippets". We propose a TDD-TCNN, a novel deep-learning approach for method-level TD detection through source code and comments from multiple perspectives. 

## Directory Structure
```
│  run_batch_fs.py          # Train model with different feature selection percentages
│  run_batch_padsize.py     # Train model with different padding sizes
│  run_cmd.py               # Train model using command-line arguments
│  run_model.py             # Demo script
│  test_pretrain.py         
│  train_eval.py            # Core logic for training, testing, and evaluation
│  train_eval_balanced.py   # Not used currently
│  train_embeddings.py      # Train embedding model
│  utils.py                 # Code for dataset processing
│  utils_fasttext.py        
│  loss.py                  # Loss computation
│  THUCNews/                # dataset for embedding
│  dataset/                 # dataset for training
└─ models/                  # Different model architectures
       ├── CNNTransformer-Seq-TC.py
       ├── CNNTransformer-Seq.py
       ├── TextCNN.py
       ├── TextRNN_Att.py
       ├── Transformer.py
       ├── chatgpt.py
       ├── GraphcodeBert.py
```

## Dataset Source Information
The dataset consists of the following project versions (as listed in the paper). Note that these are based on official source release archives (stable releases), not specific git commit snapshots.

- **Ant 1.7**  
  https://mvnrepository.com/artifact/org.apache.ant/ant/1.7.1

- **ArgoUML 0.34**  
  https://github.com/JavaQualitasCorpus/argouml-0.34

- **Columba 1.4**  
  https://sourceforge.net/projects/columba/files/Columba/1.4/columba-1.4-src.zip/download

- **Hibernate 3.3.2 GA**  
  https://mvnrepository.com/artifact/org.hibernate/hibernate/3.3.2.GA

- **JEdit 4.2**  
  https://sourceforge.net/projects/jedit/files/jedit/4.4.2/

- **JFreeChart 1.0.19**  
  https://sourceforge.net/projects/jfreechart/files/1.%20JFreeChart/1.0.19/

- **JMeter 2.1**  
  https://archive.apache.org/dist/jakarta/jmeter/source/jakarta-jmeter-2.1_src.tgz

- **JRuby 1.4**  
  https://www.jruby.org/files/downloads/1.4.0/index.html

- **SQuirrel 3.3**  
  https://sourceforge.net/projects/squirrel-sql/files/1-stable/3.3.0/squirrel-sql-3.3.0-install.jar/download


## Reproduction Instructions

### RQ-Baseline
```bash
nohup python -u run_cmd.py --model CNNTransformer-Seq --dataset DFS-Selected60 --device 0 > TDD-TCNN.output 2>&1 &
nohup python -u run_cmd.py --model TextCNN --dataset DFS-Selected60 --device 0 > CNN.output 2>&1 &
nohup python -u run_cmd.py --model Transformer --dataset DFS-Selected60 --device 0 > Transformer.output 2>&1 &
nohup python -u chatgpt.py --api_key sk-xxx --input_file dataset/td-dataset.csv --output_file chatgpt_td_results.csv --model gpt-4o --delay 1.0 > ChatGPT.output 2>&1 &
nohup python -u GraphcodeBert.py --input_file ../dataset/td-dataset.csv --save_dir saved_models/graphcodebert --batch_size 32 --eval_batch_size 8 --epochs 100 --lr 2e-5 > GraphCodeBERT.output 2>&1 &

```
### Evaluation Modes (Optional Parameters)
When using:
```bash
python -u run_cmd.py ...
```

You may append any of the following evaluation flags:
```bash
--within_project_only      #Evaluate within each project only
--cross_project_only       #Evaluate on all projects combined
--loo_cv_only              #Leave-One-Out cross-project evaluation
```
Default Behavior:
If none of the above arguments is provided, the script will perform all evaluation modes.

### RQ-Feature Fusion
```bash
nohup python -u run_cmd.py --model CNNTransformer-Seq --dataset DFS-Selected60-filter --device 0 > TDD-TCNN-full.output 2>&1 &
nohup python -u run_cmd.py --model CNNTransformer-Seq --dataset DFS-Selected60-filter-code --device 0 > TDD-TCNN-code.output 2>&1 & 
nohup python -u run_cmd.py --model CNNTransformer-Seq --dataset DFS-Selected60-filter-comment --device 0 > TDD-TCNN-comment.output 2>&1 & 
```

### RQ-Embedding 
```bash
nohup python -u train_embeddings.py --output_dir embeddings/ --embedding_type all > train_embeddings.output 2>&1 &
nohup python -u run_cmd.py --model CNNTransformer-Seq --dataset DFS-Selected60 
    --embedding_strategy CBOW --within_project_only --device 0 > embedding_random.output 2>&1 &
```

### RQ-DFS & BFS
```bash
nohup python -u run_cmd.py --model CNNTransformer-Seq --dataset DFS --device 0 > TDD-TCNN-DFS.output 2>&1 &
nohup python -u run_cmd.py --model CNNTransformer-Seq --dataset BFS --device 0 > TDD-TCNN-BFS.output 2>&1 &
```

### RQ-PadSize
```bash
nohup python -u run_batch_padsize.py --model CNNTransformer-Seq --dataset DFS-Selected60 --device 0 > TDD-TCNN-PadSize.output 2>&1 &
nohup python -u run_cmd.py --model CNNTransformer-Seq --dataset DFS-Selected60 --use_max_padsize --device 0 > TDD-TCNN-MaxPad.output 2>&1 &
```

### RQ-Feature Selection
```bash
nohup python -u run_batch_fs.py --model CNNTransformer-Seq --device 0 > TDD-TCNN-FS.output 2>&1 &
```

### RQ-Order of Local and Global Feature Extraction
```bash
nohup python -u run_cmd.py --model CNNTransformer-Seq --dataset DFS-Selected60 --device 0 > TDD-TCNN-LG.output 2>&1 & 
nohup python -u run_cmd.py --model CNNTransformer-Seq-TC --dataset DFS-Selected60 --device 0 > TDD-TCNN-GL.output 2>&1 &
```

## Notes
- **Datasets**: The dataset files should be placed in the appropriate directory before running the scripts.
- **Logging**: Each experiment logs its output to a separate file for tracking results.
- **GPU Support**: The `--device` flag is used to specify the GPU ID.

For further details, please refer to the corresponding paper.

