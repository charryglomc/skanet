MODEL=skanet_small 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup bash distributed_train.sh 8 your_imagenet_path   --model $MODEL -b 128 -vb 128 --input-size  3 256 256 \
	  --amp --train-interpolation "bicubic" \
	  --epochs  300  --lr 1e-3 \ 
	  --crop-pct  0.9 --model-ema \
	  > nohup_skanet_small.out  2>&1   &
	 
