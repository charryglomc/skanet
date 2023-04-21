MODEL=skanet_small 
python3 validate.py your_imagenet_path --model $MODEL \
  --checkpoint your_pretained_model_path -b 128 \
  --amp --num-classes 1000  --input-size 3 256 256 \
  > nohup_skanet_small_eval.out  2>&1   &

