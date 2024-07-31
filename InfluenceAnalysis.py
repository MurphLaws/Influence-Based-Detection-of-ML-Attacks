from pathlib import Path

import click
import numpy as np
import torch
import json 
import math
from tqdm import tqdm 
import sys
import os
import pandas as pd
import torchvision
import seaborn as sns
import matplotlib.pyplot as plt
from ibda.influence_functions.dynamic import tracin_torch
from ibda.models.model_dispatcher import dispatcher as model_dispatcher
from ibda.models.utils import set_model_weights
from ibda.utils.config_manager import ConfigManager
from ibda.signals import InfluenceErrorSignals
from sklearn.metrics import accuracy_score, average_precision_score
from main import CustomDataset
from poison_attacks.resnet_9_model import ResNet9
from torch.utils.data import Subset, TensorDataset

class InfluenceAnalysis:

    def __init__(self):

        self.target_instace = torch.load("poison_attacks/results/target_instance.pt")
        self.train_data = torch.load("poison_attacks/results/poisoned_train_ds.pt")
        self.test_data = torch.load("poison_attacks/results/poisoned_test_ds.pt")
        self.model = ResNet9(3, 10)


    def get_aggregated_influence(self,inf_path):
        inf = np.load("poison_attacks/results/influence/"+inf_path)
        inf = inf

        neg_inf = np.where(inf > 0, 0, inf)
        neg_inf = np.sum(neg_inf, axis=1)
        neg_inf = sum(neg_inf)

        pos_inf = np.where(inf < 0, 0, inf)
        pos_inf = np.sum(pos_inf, axis=1)
        pos_inf = sum(pos_inf)

        
        
        return pos_inf, neg_inf

    def compute_influence(self):


        ckpt_paths = [str(path) for path in list(Path("poison_attacks", "results", "models", "dirty" ).glob("*.pth"))]
        ckpt_paths = sorted(ckpt_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        #ckpt_paths = ckpt_paths[:2]

    

        for epoch,ckpt in enumerate(ckpt_paths):

            set_model_weights(self.model, ckpt)

           
            self.train_data = Subset(self.train_data, range(len(self.train_data)-100, len(self.train_data)))
             
            influence_matrix = np.zeros((len(self.train_data), len(self.test_data)))
            tracInObject = tracin_torch.TracInInfluenceTorch(
                     model_instance=self.model,
                    ckpts_file_paths=[ckpt],
                    batch_size=128,
                    fast_cp=True,
                    layers=["res2"],
                                )

            influence_matrix = tracInObject.compute_train_to_test_influence(self.train_data, self.test_data)
            np.save(f"poison_attacks/results/influence/{epoch}_full_IM.npy", influence_matrix)

   
   


        

    def class_wise_influence(self):
        
        for train_class in [0,2,5,7,9]:
                
                last_epoch = "poison_attacks/results/models/dirty/dirty_ckpt_29.pth"

                class_train_data = []
                class_train_labels = []

                for image, label in self.train_data:
                    if label == train_class:
                        class_train_data.append(image)
                        class_train_labels.append(label)


                class_train_data = TensorDataset(torch.stack(class_train_data), torch.tensor(class_train_labels))

                for test_class in [0,2,7,5,9]:
                    class_test_data = []
                    class_test_labels = []
                    for image, label in self.test_data:
                       
                        if label == test_class:
                            if image.shape == (3,32,32):
                                class_test_data.append(image)
                            else: 
                                class_test_data.append(image[0])
                            class_test_labels.append(label)
                    
                    class_test_data = TensorDataset(torch.stack(class_test_data), torch.tensor(class_test_labels))
                    class_test_data = Subset(class_test_data, range(len(class_test_data)-100, len(class_test_data)))

                    print(train_class, test_class)
                    print(len(class_train_data), len(class_test_data))


                    ckpt_paths = [str(path) for path in list(Path("poison_attacks", "results", "models", "dirty" ).glob("*.pth"))]
                    ckpt_paths = sorted(ckpt_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
                    #ckpt_paths = ckpt_paths[:7] #:Sumsampling of the ckpts


                    for ckpt in ckpt_paths:
                        set_model_weights(self.model, ckpt)
             
                        influence_matrix = np.zeros((len(class_train_data), len(class_test_data)))
                        tracInObject = tracin_torch.TracInInfluenceTorch(
                                model_instance=self.model,
                                ckpts_file_paths=[ckpt],
                                batch_size=128,
                                fast_cp=True,
                                layers=["res2"],
                                
                        )

                

                        influence_matrix_toadd = tracInObject.compute_train_to_test_influence(class_train_data, class_test_data)
                        influence_matrix += influence_matrix_toadd
                    np.save(f"poison_attacks/results/influence/class_{train_class}_to_class_{test_class}.npy", influence_matrix)

    def predict(self,data):
        load_ckpt =  "poison_attacks/results/models/dirty/dirty_ckpt_7.pth"
        model = ResNet9(3, 10)
        set_model_weights(model, load_ckpt)
        model.eval()
        with torch.no_grad():
            return model(data)

def PosNeg(inf_matrix):

    testMNI = inf_matrix.T
    testMNI = np.where(testMNI > 0, 0, testMNI)
    testMNI = np.sum(testMNI, axis=1)
    testMNI = abs(testMNI)
   
    def compute_signal(array):
        return(max(array*testMNI))

    result = np.apply_along_axis(compute_signal, axis=1, arr=inf_matrix)

    return result
 
    



def main():
    influence_analysis = InfluenceAnalysis()
    
    class0to0 = np.load("poison_attacks/results/influence/class_0_to_class_0.npy")
    class0to2 = np.load("poison_attacks/results/influence/class_0_to_class_2.npy")
    class0to5 = np.load("poison_attacks/results/influence/class_0_to_class_5.npy")
    class0to7 = np.load("poison_attacks/results/influence/class_0_to_class_7.npy")
    class0to9 = np.load("poison_attacks/results/influence/class_0_to_class_9.npy")

    class2to0 = np.load("poison_attacks/results/influence/class_2_to_class_0.npy")
    class2to2 = np.load("poison_attacks/results/influence/class_2_to_class_2.npy")
    class2to5 = np.load("poison_attacks/results/influence/class_2_to_class_5.npy")
    class2to7 = np.load("poison_attacks/results/influence/class_2_to_class_7.npy")
    class2to9 = np.load("poison_attacks/results/influence/class_2_to_class_9.npy")

    class5to0 = np.load("poison_attacks/results/influence/class_5_to_class_0.npy")
    class5to2 = np.load("poison_attacks/results/influence/class_5_to_class_2.npy")
    class5to5 = np.load("poison_attacks/results/influence/class_5_to_class_5.npy")
    class5to7 = np.load("poison_attacks/results/influence/class_5_to_class_7.npy")
    class5to9 = np.load("poison_attacks/results/influence/class_5_to_class_9.npy")

    class7to0 = np.load("poison_attacks/results/influence/class_7_to_class_0.npy")
    class7to2 = np.load("poison_attacks/results/influence/class_7_to_class_2.npy")
    class7to5 = np.load("poison_attacks/results/influence/class_7_to_class_5.npy")
    class7to7 = np.load("poison_attacks/results/influence/class_7_to_class_7.npy")
    class7to9 = np.load("poison_attacks/results/influence/class_7_to_class_9.npy")

    class9to0 = np.load("poison_attacks/results/influence/class_9_to_class_0.npy")
    class9to2 = np.load("poison_attacks/results/influence/class_9_to_class_2.npy")
    class9to5 = np.load("poison_attacks/results/influence/class_9_to_class_5.npy")
    class9to7 = np.load("poison_attacks/results/influence/class_9_to_class_7.npy")
    class9to9 = np.load("poison_attacks/results/influence/class_9_to_class_9.npy")


    class0tox = np.hstack((class0to0, class0to2, class0to5, class0to7, class0to9))
    class2tox = np.hstack((class2to0, class2to2, class2to5, class2to7, class2to9))
    class5tox = np.hstack((class5to0, class5to2, class5to5, class5to7, class5to9))
    class7tox = np.hstack((class7to0, class7to2, class7to5, class7to7, class7to9))
    class9tox = np.hstack((class9to0, class9to2, class9to5, class9to7, class9to9))

    full_matrix = np.vstack((class0tox, class2tox, class5tox, class7tox, class9tox))

    posneg = PosNeg(full_matrix)
    
    signals_mask = np.zeros(len(posneg))    
    random_mask = np.zeros(len(posneg))

    signals_mask[-50:] = 1
   

    random_idx = np.random.choice(len(random_mask), 50, replace=False)
    random_mask[random_idx] = 1
 


    avg_precision = average_precision_score(signals_mask, posneg)
    print(avg_precision)
    random_avg_precision = average_precision_score(random_mask, posneg)
    print(random_avg_precision)

    #Scatter of PosNeg distirbution
    fig, ax = plt.subplots()
    x = np.arange(len(posneg))
    y = posneg
    ax.scatter(x, y, color='blue')
    ax.scatter(x[-50:], y[-50:], color='red')
    ax.set_title(f'PosNeg Signal on all train samples')
    ax.set_xlabel('Train Samples')
    ax.set_ylabel('Signal Value')


    ax.legend(["Clean Samples", "Poison Samples"])
    plt.savefig("posneg.png")

    
    
    #influence_analysis.inf_trough_epochs()
    #influence_analysis.compute_influence()

  
if __name__ == "__main__":
    main()
