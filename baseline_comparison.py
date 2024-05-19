
from pathlib import Path

import click
import numpy as np
import torch
import json 
import math
from tqdm import tqdm 

import torchvision
import matplotlib.pyplot as plt
from ibda.influence_functions.dynamic import tracin_torch
from ibda.models.model_dispatcher import dispatcher as model_dispatcher
from ibda.models.utils import set_model_weights
from ibda.utils.config_manager import ConfigManager
from ibda.signals import InfluenceErrorSignals
from sklearn.metrics import accuracy_score, average_precision_score


class BaselineComparison:

    def __init__(self, data_name, model_name, subset_id, attack_type):
        self.data_name = data_name
        self.model_name = model_name
        self.subset_id = subset_id
        self.attack_type = attack_type  
    
        target_bases_dict_path = Path("results", self.model_name, self.data_name, self.subset_id, "poisoned", self.attack_type, "target_bases.json")

        with open(target_bases_dict_path) as f:
            self.target_bases = json.load(f)
        
        self.results_save_dir = Path("baseline_comparison_results")
        self.results_save_dir.mkdir(parents=True, exist_ok=True)

    def compute_influence(self):
       
        data_name = self.data_name
        model_name = self.model_name
        subset_id = self.subset_id
        attack_type = self.attack_type
        ckpt_paths = [str(path) for path in list(Path("results", model_name, data_name, subset_id, "poisoned",attack_type, "ckpts/").glob("*.pt"))]
        
        ckpt_resampling = 5
        ckpt_paths = ckpt_paths[::ckpt_resampling]
        train_data_path = Path("data", "dirty", data_name, subset_id, attack_type,"poisoned_train.pt")
        test_data_path = Path("data", "clean", data_name, subset_id, "test.pt")

        self.train_data = torch.load(train_data_path)
        self.test_data = torch.load(test_data_path)

        self.y_train = self.train_data.tensors[1].numpy()
        self.y_test = self.test_data.tensors[1].numpy()

        num_classes = len(torch.unique(self.train_data.tensors[1]))
        input_shape = tuple(self.train_data.tensors[0].shape[1:])
        
        model_conf_fp = str(Path("configs", "resnet", f"resnet_{data_name}.json"))

        conf_mger = ConfigManager(model_training_conf=model_conf_fp)

        model_seed = conf_mger.model_training.random_seed
        model_name = conf_mger.model_training.name

        model = model_dispatcher[model_name](
            num_classes=num_classes,
            input_shape=input_shape,
            seed=model_seed,
            trainable_layers=conf_mger.model_training.trainable_layers,
        )
        layer_names = model.trainable_layer_names()

        model.eval()

        tracInObject = tracin_torch.TracInInfluenceTorch(
                model_instance=model,
                ckpts_file_paths=ckpt_paths,
                batch_size=128,
                fast_cp=True,
                layers=layer_names,
        )
        
        influence_save_dir = Path("results", model_name, data_name, subset_id,"poisoned",  attack_type,  "influence_matrices")
        influence_save_dir.mkdir(parents=True, exist_ok=True)
            

        if not (influence_save_dir / f"IM.npy").exists():
            self.influence_matrix = tracInObject.compute_train_to_test_influence(self.train_data, self.test_data)
            np.save(influence_save_dir / f"IM.npy", self.influence_matrix)
        else:
            print("Influence matrix already exists")
            self.influence_matrix = np.load(influence_save_dir / f"IM.npy")
        if not (influence_save_dir / f"SI_train.npy").exists():
            self.train_self_influence_array = tracInObject.compute_self_influence(self.train_data)
            np.save(influence_save_dir / f"SI_train.npy", self.train_self_influence_array)
        else:
            print("Self influence for train data already exists")
            self.train_self_influence_array = np.load(influence_save_dir / f"SI_train.npy")
        if not (influence_save_dir / f"SI_test.npy").exists():
            self.test_self_influence_array = tracInObject.compute_self_influence(self.test_data) 
            np.save(influence_save_dir / f"SI_test.npy", self.test_self_influence_array)
        else:   
            print("Self influence for test data already exists")
            self.test_self_influence_array = np.load(influence_save_dir / f"SI_test.npy")
        
        
    def get_target_ids(self):            
        target_ids = [element["target_id"][0] for element in self.target_bases]
        return target_ids
    
    def get_poison_ids(self):
        poison_ids = []
        for element in self.target_bases:
            poison_ids.extend(element["base_ids"])
        return poison_ids

    
            

    def renormalied_influence(self):
        renormalied_influence_matrix = np.empty(self.influence_matrix.shape)
        for idx_train,train_instance in enumerate(self.influence_matrix):
            train_SI = self.train_self_influence_array[idx_train]
            for idx_test,target_instance in enumerate(train_instance):
                test_SI = self.test_self_influence_array[idx_test]
                renormalied_influence_matrix[idx_train][idx_test] = target_instance/(np.sqrt(train_SI)*np.sqrt(test_SI))
        
        self.influence_matrix = renormalied_influence_matrix

    
    def get_signals(self):
        self.signalComputations = InfluenceErrorSignals(
            train_test_inf_mat=self.influence_matrix,
            y_train=self.y_train,
            y_test=self.y_test,
            compute_test_influence=False,
            )

        self.signals_datafame = self.signalComputations.compute_signals(verbose=False)
        self.signals_datafame["SI"] = self.train_self_influence_array 


    def get_avg_precision(self):
        
        poison_ids = self.get_poison_ids()
        poison_mask = np.zeros(self.y_train.shape[0], dtype=bool)
        poison_mask[poison_ids] = True
        

        self.avg_precision_dict = {}
        for col in self.signals_datafame.columns:
            avg_precision = average_precision_score(poison_mask, self.signals_datafame[col])
            self.avg_precision_dict[col] = avg_precision
        
        
    @staticmethod   
    def anom_score(array, c=2.2219):

        sorted_values = sorted(array)
        n = len(array)
        interpoint_distances = []

        for i in range(n):
            for j in range(i+1,n):
                interpoint_distances.append(c*abs(sorted_values[i]-sorted_values[j]))
        
        r = math.comb(math.floor(n/2)+1,2)
        Q = interpoint_distances[r-1]

        mean = array.mean()
        anom_scores = (array-mean)/Q

        return(anom_scores)


#    def two_class_distribution(self):
#        transposed = self.influence_matrix.T 
#        target_bases_tuples = []
#
#        for element in self.target_bases:
#            target_bases_tuples.append(((element["target_id"][0],element["target_class"]), (element["base_ids"],element["base_class"])))
#
#        first_one = target_bases_tuples[0]
#
#        first_one_base_class = first_one[1][1]
#        first_one_target_class = first_one[0][1]
#        
#
#
#        base_class_mask = self.y_train == first_one_base_class
#        
#        class_idx = []
#        for idx,label in enumerate(self.y_train):
#            if label == first_one_base_class:
#                class_idx.append(idx)
#      
#        
#        poison_ids = first_one[1][0] 
#        new_indexes = [class_idx.index(idx) for idx in poison_ids]
#
#        only_base_influence = self.influence_matrix[base_class_mask]
#        target_vector = only_base_influence.T[first_one[0][0]]
#        
#        self.poison_images = self.train_data.tensors[0][poison_ids]
#        self.poison_labels = self.train_data.tensors[1][poison_ids]
#        self.target_image = self.test_data.tensors[0][first_one[0][0]]
#        self.target_label = self.y_test[first_one[0][0]]
#
#        #make a scatter plot of target_vector values and color the indexes in new_indexes red 
#
#        mask = np.zeros(len(class_idx), dtype=bool)
#        mask[new_indexes] = True
#        poison_influence = target_vector[mask]
#
##        plt.scatter(range(len(target_vector)), target_vector, color="blue")
##        plt.scatter(new_indexes, poison_influence, color="red")
##        plt.xlabel(f"Train Instances, base class:{first_one_base_class}")
##        plt.ylabel("Influence Score")
##        plt.title(f"Influence in target {first_one[0][0]}, target_class: {first_one_target_class}, base_class: {first_one_base_class}")
##        plt.legend(["Non-poisoned", "Poisoned"])
##
##        plt.show()
#
#       
#


    def poison_statistic_detection(self):
            
            transposed = self.influence_matrix.T 
            target_bases_tuples = []


            for element in self.target_bases:
                target_bases_tuples.append((element["target_id"][0], element["base_ids"]))
            ##NOTE: I'm gonna take the first one of the attacks just to visualize the influence
            first_one = target_bases_tuples[1]
            
            dummy_target = transposed[first_one[0]]
            
            mask = np.zeros(self.y_train.shape[0], dtype=bool)
            poisons = first_one[1] 
            mask[poisons] = True
            dummy_target_poisons = dummy_target[mask]
            dummy_target_not_poisons = dummy_target[~mask][0:100]

            plt.scatter(range(len(dummy_target_poisons)), dummy_target_poisons, color="red")
            plt.scatter(range(len(dummy_target_not_poisons)), dummy_target_not_poisons, color="blue")
            
            plt.xlabel("Poisoned Instances")
            plt.ylabel("Influence Score")
            plt.title(f"Influence in target {first_one[0]}")
            plt.show()

            #Plot the dummy target scatter and for the indexes in poisons paint them red 
            
                        
            avg_precision = []
            for target,poisons in tqdm(target_bases_tuples):
                mask = np.zeros(self.y_train.shape[0], dtype=bool)
                mask[poisons] = True
                target_vector = transposed[target]
                anom_scores = self.anom_score(target_vector) 
                    
                 
                avg_precision.append(average_precision_score(mask, anom_scores))
                
            self.avg_precision_dict["poison_statistic"] = np.mean(avg_precision)  
           


    


    @staticmethod
    def plot_images(images, num_cols=5):
        grid_image = torchvision.utils.make_grid(images, nrow=int(len(images)**0.5))
        plt.imshow(grid_image.permute(1,2,0))
        plt.title("Poisoned Images")
        plt.show()


def run(data_name, model_name, subset_id, attack_type):
    scenario1 = BaselineComparison(data_name, model_name, subset_id, attack_type)
    scenario1.compute_influence()
   
    #NOTE:: NotNorm Signals 

    scenario1.get_signals()
    scenario1.get_avg_precision()
    
    with open(scenario1.results_save_dir / f"avg_precision_not_norm{subset_id}.json", "w") as f:
        json.dump(scenario1.avg_precision_dict, f)
    
    #NOTE:: Norm Signals

    #scenario1.renormalied_influence()
    scenario1.get_signals()
    scenario1.get_avg_precision()
#    scenario1.two_class_distribution()
#    scenario1.plot_images(scenario1.poison_images)
    print(scenario1.avg_precision_dict)




@click.command()
@click.option("--data_name", required=True, type=click.STRING)
@click.option("--model_name", required=True, type=click.STRING)
@click.option("--subset_id", required=True, type=click.STRING)
@click.option("--attack_type", required=True, type=click.STRING)
def main(data_name, model_name, subset_id, attack_type):
    run(data_name, model_name, subset_id, attack_type)


if __name__ == "__main__":
    main()
