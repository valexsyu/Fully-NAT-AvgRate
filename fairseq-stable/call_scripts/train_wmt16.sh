source $HOME/.bashrc 
conda activate gu_ctc_test


databin_dir=../data/wmt14.de-en.dist.bin
model_dir=checkpoints/qq
expname=qq
# src-upsample 3=>8192
# src-upsample 4=>4096
# --max-tokens 4096 --update-freq 1 
# --dropout 0.3 
python train.py ${databin_dir} \
    --fp16 \
    --left-pad-source False --left-pad-target False \
    --arch cmlm_transformer_ctc --task translation_lev \
    --noise 'full_mask' --valid-noise 'full_mask' \
    --dynamic-upsample --src-upsample 3 \
    --decoder-learned-pos --encoder-learned-pos \
    --apply-bert-init --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-06 \
    --clip-norm 2.4 --dropout 0.3 --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 --warmup-updates 10000 --lr 0.0005 --min-lr 1e-09 \
    --criterion nat_loss --predict-target 'all' --loss-type 'ctc' \
    --axe-eps --force-eps-zero \
    --label-smoothing 0.1 --weight-decay 0.01 \
    --max-tokens 4096 --update-freq 1 \
    --max-update 100000 --save-dir ${model_dir} \
    --no-epoch-checkpoints  --keep-best-checkpoints 5 \
    --seed 2 --log-interval 100 --no-progress-bar \
    --eval-bleu --eval-bleu-args '{"iter_decode_max_iter":0,"iter_decode_collapse_repetition":true}' \
    --eval-bleu-detok 'space' \
    --eval-tokenized-bleu --eval-bleu-remove-bpe '@@ ' --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --tensorboard-logdir ${model_dir}/tensorboard/${expname} 