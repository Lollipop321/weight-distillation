#!/usr/bin/bash
set -e

model_root_dir=output

# set task
task=zh2en

# set tag
model_dir_tag=zh2en_baseline

# set device
gpu=0
cpu=

# data set
who=train

if [ $task == "zh2en" ]; then
        data_dir=nist12.filtered.zh-en
        ensemble=
        batch_size=64
        beam=6
        length_penalty=1.0
        src_lang=zh
        tgt_lang=en
        sacrebleu_set=
else
        echo "unknown task=$task"
        exit
fi

model_dir=$model_root_dir/$task/$model_dir_tag

checkpoint=checkpoint_best.pt

if [ -n "$ensemble" ]; then
        if [ ! -e "$model_dir/last$ensemble.ensemble.pt" ]; then
                PYTHONPATH=`pwd` python3 scripts/average_checkpoints.py --inputs $model_dir --output $model_dir/last$ensemble.ensemble.pt --num-epoch-checkpoints $ensemble
        fi
        checkpoint=last$ensemble.ensemble.pt
fi

output=$model_dir/translation.log

if [ -n "$cpu" ]; then
        use_cpu=--cpu
fi

export CUDA_VISIBLE_DEVICES=$gpu

python3 generate.py \
../data-bin/$data_dir \
--path $4 \
--gen-subset $who \
--batch-size $batch_size \
--beam $beam \
--lenpen $length_penalty \
--quiet \
--output $model_dir/hypo.txt | tee $output
python3 rerank.py $model_dir/hypo.txt $model_dir/hypo.sorted