python3 run.py --name llama2_loradash_r128 \
  --model meta-llama/Llama-2-7b-hf \
  -p loradash \
  --train \
  --lr 2e-5 \
  --wd 0.00 \
  --batch-size 2 \
  --accum-step 64 \
  --rank 128 \
  --epochs 1 \
  --mix-precision bf16 \
  --mix-precision tf32 \
  --train-dataset metamath \
  --split [:100000]
