# VARMAT

**[AAAI 26] Vulnerability-Aware Robust Multimodal Adversarial Training**

Junrui Zhang, Xinyu Zhao, Jie Peng, Chenjie Wang, Jianmin Ji, and Tianlong Chen

## Note
  1. Due to limited time, the code has not been fully cleaned and documented. This repository contains the exact version of the code submitted with the paper. 
  2. We plan to release a more polished and well-documented version in a future update.

## Datasets
 We use the datasets from the official Multimodal Benchmark: MultiBench(https://arxiv.org/abs/2107.07502)
 
 (1) CMU-MOSEI (2) UR-FUNNY (3) AVMNIST

## Experiments
  1. Train & Test: python private_test_scripts/perceivers/{dataset}_train.py
  2. The training is controlled via environment variables in {dataset}_train.py. Main hyperparameters:

    os.environ['Mode'] = 'Attack'         # Set to 'Attack' for adversarial training; '' for clean training 
    os.environ['AT_Methods'] = 'VARMAT'   # 'VARMAT' for our method, or 'Avg' for baseline
    os.environ['adv_iter'] = '10'         # Number of adversarial attack iterations I
    os.environ['eps'] = '0.01'            # Attack strength λ
    os.environ['temperature'] = '0.5'     # Temperature parameter used in softmax/logits smoothing. Not used by VARMAT.
    os.environ['method'] = 'FGSM-RS'      # Attack method: one of {'FGSM-RS', 'FGSM-EP', 'FGSM-MEP', 'FGSM-PCO'}
    During validation and testing, eps is multiplied by 20 (i.e., 0.2) to follow the experimental setup in the paper.
    
## Checkpoints
  1. ckpt/clean: Models trained on clean samples (without adversarial defense).
  2. ckpt/at_ckpt/{Methods}/{Dataset_reg}: Models trained with adversarial training, including both VARMAT and Avg

## Acknowledgments
1. We thank the developers of [HighMMT](https://github.com/pliang279/HighMMT), [MultiBench](https://github.com/pliang279/MultiBench), for their public code release.
2. Meanwhile, I would like to thank my girlfriend, Zhai, for supporting me when I faced challenges during this work.