

prepare_data:
	python -m data.load_prepare --dataset mnist --subset_ratio 0.1 --seed $(SEED) --save_loaded_data_folder data/.tmp --save_new_data_folder data/clean
	python -m data.load_prepare --dataset fmnist --subset_ratio 0.1 --seed $(SEED) --save_loaded_data_folder data/.tmp --save_new_data_folder data/clean
	python -m data.load_prepare --dataset cifar10 --subset_ratio 0.1 --seed $(SEED) --save_loaded_data_folder data/.tmp --save_new_data_folder data/clean

# Make sure that each dataset has been stored in the appropriate format by running make prepare_data
# If you have not created the model checkpoints for each dataset, leave the "model_ckpt_fp" argument unfilled and
# the model will be trained. The next time that you call the function, you can pass the checkpoint weights (.pt file)
run_evasion_attacks:
	python -m attack_generation.adversarials.run_attacks --data_name mnist --train_data_fp data/clean/mnist/$(SUBSET_FOLDER)/train.pt --test_data_fp data/clean/mnist/$(SUBSET_FOLDER)/test.pt --model_conf_fp configs/resnet/resnet_mnist.json --dir_suffix $(SUBSET_FOLDER) --seed $(SEED) --device cpu
	python -m attack_generation.adversarials.run_attacks --data_name fmnist --train_data_fp data/clean/fmnist/$(SUBSET_FOLDER)/train.pt --test_data_fp data/clean/fmnist/$(SUBSET_FOLDER)/test.pt --model_conf_fp configs/resnet/resnet_fmnist.json --dir_suffix $(SUBSET_FOLDER) --seed $(SEED) --device cpu
	python -m attack_generation.adversarials.run_attacks --data_name cifar10 --train_data_fp data/clean/cifar10/$(SUBSET_FOLDER)/train.pt --test_data_fp data/clean/cifar10/$(SUBSET_FOLDER)/test.pt --model_conf_fp configs/resnet/resnet_cifar10.json --dir_suffix $(SUBSET_FOLDER) --seed $(SEED) --device cpu


run_poison_attack:
	python -m attack_generation.poisons.run_attacks \
	--data_name $(DATA_NAME) \
	--train_data_fp data/clean/$(DATA_NAME)/$(SUBSET_FOLDER)/train.pt \
	--test_data_fp data/clean/$(DATA_NAME)/$(SUBSET_FOLDER)/test.pt \
	--model_conf_fp configs/resnet/resnet_$(DATA_NAME).json \
	--dir_suffix $(SUBSET_FOLDER) \
	$(if $(CKPT_NUMBER),--model_ckpt_fp "results/resnet20/$(DATA_NAME)/$(SUBSET_FOLDER)/clean/ckpts/checkpoint-$(CKPT_NUMBER).pt") \
	$(if $(SEED),--seed $(SEED),--seed 0) \
	$(if $(DEVICE),--device $(DEVICE),--device cpu) \
	$(if $(NUM_POISONS),--num_poisons $(NUM_POISONS),--num_poisons 1) \
	$(if $(NUM_TARGETS),--num_targets $(NUM_TARGETS),--num_targets 2) \
	$(if $(MAX_ITER),--max_iter $(MAX_ITER),--max_iter 3)


poison_influence:
	python -m poison_influence --data_name $(DATA_NAME) --model_name $(MODEL_NAME) --subset_id $(SUBSET_FOLDER) \
	--model_conf_fp configs/resnet/resnet_$(DATA_NAME).json

adv_influence:
	python -m test_adversarials_influence --attack $(ATTACK) --data_name mnist --model_name resnet20 --inf_fn_name tracin --subset_id $(SUBSET_FOLDER) --model_conf configs/resnet/resnet_mnist.json --inf_fn_conf configs/resnet/tracin_resnet.json --device cpu
	python -m test_adversarials_influence --attack $(ATTACK) --data_name fmnist --model_name resnet20 --inf_fn_name tracin --subset_id $(SUBSET_FOLDER) --model_conf configs/resnet/resnet_fmnist.json --inf_fn_conf configs/resnet/tracin_resnet.json --device cpu
	python -m test_adversarials_influence --attack $(ATTACK) --data_name cifar10 --model_name resnet20 --inf_fn_name tracin --subset_id $(SUBSET_FOLDER) --model_conf configs/resnet/resnet_cifar10.json --inf_fn_conf configs/resnet/tracin_resnet.json --device cpu
