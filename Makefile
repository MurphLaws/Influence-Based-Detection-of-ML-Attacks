prepare_data:
	python -m data.load_prepare --dataset mnist --subset_ratio 0.1 --seed 0 --save_loaded_data_folder data/.tmp --save_new_data_folder data/clean
	python -m data.load_prepare --dataset fmnist --subset_ratio 0.1 --seed 0 --save_loaded_data_folder data/.tmp --save_new_data_folder data/clean
	python -m data.load_prepare --dataset cifar10 --subset_ratio 0.1 --seed 0 --save_loaded_data_folder data/.tmp --save_new_data_folder data/clean

# Make sure that each dataset has been stored in the appropriate format by running make prepare_data
# If you have not created the model checkpoints for each dataset, leave the "model_ckpt_fp" argument unfilled and
# the model will be trained. The next time that you call the function, you can pass the checkpoint weights (.pt file)
run_evasion_attacks:
	python -m attack_generation.adversarials.run_attacks --data_name mnist --train_data_fp data/clean/mnist/subset_id0_r0.1/train.pt --test_data_fp data/clean/mnist/subset_id0_r0.1/test.pt --model_conf_fp configs/resnet/resnet_mnist.json --savedir data/dirty/mnist/subset_id0_r0.1 --seed 0 --model_ckpt_fp results/resnet20/mnist/clean/ckpts/checkpoint-7.pt --device cpu
	python -m attack_generation.adversarials.run_attacks --data_name fmnist --train_data_fp data/clean/fmnist/subset_id0_r0.1/train.pt --test_data_fp data/clean/fmnist/subset_id0_r0.1/test.pt --model_conf_fp configs/resnet/resnet_fmnist.json --savedir data/dirty/fmnist/subset_id0_r0.1 --seed 0 --device cpu
	python -m attack_generation.adversarials.run_attacks --data_name cifar10 --train_data_fp data/clean/cifar10/subset_id0_r0.1/train.pt --test_data_fp data/clean/cifar10/subset_id0_r0.1/test.pt --model_conf_fp configs/resnet/resnet_fmnist.json --savedir data/dirty/cifar10/subset_id0_r0.1 --seed 0 --device cpu