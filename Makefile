prepare_data:
	python -m data.load_prepare --dataset mnist --subset_ratio 1 --seed 0 --save_loaded_data_folder data/.tmp --save_new_data_folder data/clean
	python -m data.load_prepare --dataset fmnist --subset_ratio 1 --seed 0 --save_loaded_data_folder data/.tmp --save_new_data_folder data/clean
	python -m data.load_prepare --dataset cifar10 --subset_ratio 1 --seed 0 --save_loaded_data_folder data/.tmp --save_new_data_folder data/clean

# Make sure that each dataset has been stored in the appropriate format by running make prepare_data
# If you have not created the model checkpoints for each dataset, leave the "model_ckpt_fp" argument unfilled and
# the model will be trained. The next time that you call the function, you can pass the checkpoint weights (.pt file)
run_evasion_attacks:
	python -m attack_generation.adversarials.run_attacks --data_name fmnist --train_data_fp data/clean/fmnist/subset_id0_r0.1/train.pt --test_data_fp data/clean/fmnist/subset_id0_r0.1/test.pt --model_conf_fp configs/resnet/resnet_fmnist.json --dir_suffix subset_id0_r0.1 --seed 0 --device cpu
	python -m attack_generation.adversarials.run_attacks --data_name mnist --train_data_fp data/clean/mnist/subset_id0_r0.1/train.pt --test_data_fp data/clean/mnist/subset_id0_r0.1/test.pt --model_conf_fp configs/resnet/resnet_mnist.json --dir_suffix subset_id0_r0.1 --seed 0 --device cpu #--model_ckpt_fp results/resnet20/mnist/clean/ckpts/checkpoint-7.pt
	python -m attack_generation.adversarials.run_attacks --data_name cifar10 --train_data_fp data/clean/cifar10/subset_id0_r0.1/train.pt --test_data_fp data/clean/cifar10/subset_id0_r0.1/test.pt --model_conf_fp configs/resnet/resnet_cifar10.json --dir_suffix subset_id0_r0.1 --seed 0 --device cpu


adv_influence:
	python -m test_adversarials_influence --attack $(ATTACK) --data_name mnist --model_name resnet20 --inf_fn_name tracin --subset_id subset_id0_r0.1 --model_conf configs/resnet/resnet_mnist.json --inf_fn_conf configs/resnet/tracin_resnet.json --device cpu
	python -m test_adversarials_influence --attack $(ATTACK) --data_name fmnist --model_name resnet20 --inf_fn_name tracin --subset_id subset_id0_r0.1 --model_conf configs/resnet/resnet_fmnist.json --inf_fn_conf configs/resnet/tracin_resnet.json --device cpu
	python -m test_adversarials_influence --attack $(ATTACK) --data_name cifar10 --model_name resnet20 --inf_fn_name tracin --subset_id subset_id0_r0.1 --model_conf configs/resnet/resnet_cifar10.json --inf_fn_conf configs/resnet/tracin_resnet.json --device cpu
