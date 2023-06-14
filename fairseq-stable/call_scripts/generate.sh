source $HOME/.bashrc 
conda activate gu_ctc


databin_dir=../data/wmt14.de-en.dist.bin
ck_dir=wmt14.de-en.alg1.3.step_100k.dropout_0.1
python fairseq_cli/generate.py ${databin_dir} \
    --task translation_lev \
    --path checkpoints/$ck_dir/average-best-model.pt \
    --gen-subset test \
    --axe-eps  --iter-decode-collapse-repetition --force-eps-zero \
    --left-pad-source False --left-pad-target False \
    --iter-decode-max-iter 0 --beam 1 \
    --remove-bpe --batch-size 200 \
    --inference-rate 2 
