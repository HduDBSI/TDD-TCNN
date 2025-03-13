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
│  utils.py                 # Code for dataset processing
│  utils_fasttext.py        
│  loss.py                  # Loss computation
│  THUCNews/                # dataset for processing
│  dataset/                 # dataset for training
└─models/                   # Different model architectures
       ├── CNNTransformer-Seq-TC.py
       ├── CNNTransformer-Seq.py
       ├── TextCNN.py
       ├── TextRNN_Att.py
       ├── Transformer.py
```

## Reproduction Instructions

### RQ-Baseline
```bash
nohup python -u run_cmd.py --model CNNTransformer-Seq --dataset DFS-Selected60 --device 0 > TDD-TCNN.output 2>&1 &
nohup python -u run_cmd.py --model TextCNN --dataset DFS --device 0 > CNN.output 2>&1 &
nohup python -u run_cmd.py --model Transformer --dataset DFS --device 0 > Transformer.output 2>&1 &
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

