# Weight Distillation (WD)

## Requirements

pytorch >= 1.0, python >= 3.6.0, cuda >= 9.2

## How to Reproduce

Take the NIST12 Chinese-English (Zh-En) for Example.

To reproduce the experiments, please run:

    # train the baseline models
    sh train_zh2en.sh
    # translate to produce the pseudo data
    sh translate_kd.sh
    # train the wd phase1 models
    sh train_wd_zh2en_kd_phase1.sh
    # convert wd phase1 models to wd phase2 models
    python3 fair_wd_weight_to_weight.py -model model_wd_phase1.pt
    # train the wd phase2 models
    sh train_wd_zh2en_kd_phase2.sh
    # translate and score on the test and valid sets
    sh translate_zh2en.sh

## Implementations

The code files that implements WD are located in:

`fairseq/models/wd_transformer.py`