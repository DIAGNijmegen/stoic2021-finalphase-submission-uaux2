#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2022 Challenge STOIC2021 Team Augsburg,                       #
#                University of Augsburg, Germany                               #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import os
import pandas as pd
import numpy as np
import argparse
# TensorFlow libraries
import tensorflow as tf
# AUCMEDI libraries
from aucmedi import input_interface, DataGenerator, Neural_Network
from aucmedi.neural_network.architectures import supported_standardize_mode, architecture_dict
from aucmedi.data_processing.subfunctions import Padding, Crop, Clip, Chromer, Standardize
from aucmedi.data_processing.io_loader import sitk_loader
# Internal functions
import sys
sys.path.append(os.getcwd())
from algorithm.aucmedi.loss import focal_f1
from algorithm.aucmedi.omtrta.datagenerator import DataGenerator_MetaData
from algorithm.aucmedi.omtrta.resnet import Architecture_ResNet34

#-----------------------------------------------------#
#                      Argparser                      #
#-----------------------------------------------------#
parser = argparse.ArgumentParser(description="AUCMEDI Prediction for STOIC2021")
parser.add_argument("--data", help="Path to dataset (lungseg)",
                    required=True, type=str, dest="path_ct")
parser.add_argument("--sampling", help="Path to sampling (sampling.CV.csv)",
                    required=False, type=str, dest="path_sampling",
                    default="explore/results/sampling.CV.csv")
parser.add_argument("--meta", help="Path to metadata (ilr.csv)",
                    required=False, type=str, dest="path_meta",
                    default="explore/results/ilr.csv")
parser.add_argument("-g", "--gpu", help="GPU ID selection for multi cluster",
                    required=False, type=int, dest="gpu", default=0)
args = parser.parse_args()

#-----------------------------------------------------#
#                   Configurations                    #
#-----------------------------------------------------#
# Provide path to imaging data
path_ct = args.path_ct
# Provide path to the sampling list
path_sampling = args.path_sampling
# Provide path to the metadata
path_meta = args.path_meta

# Result/Model directory
path_model = "models.metadata.severity"

# Define some parameters
batch_size = 8
processes = 12
batch_queue_size = 10
threads = batch_size
tf_threads = 5

n_folds = 5
model_types = ["best.auc", "best.loss", "best.f1", "last"]

# Metadata columns
meta_cols = ["PatientAge", "PatientSex", "ilr"]

# Define architecture which should be processed
arch = "ResNet34.metadata"

# Define input shape
input_shape = (148, 180, 224)

#-----------------------------------------------------#
#              Setup of Tensorflow Stack              #
#-----------------------------------------------------#
# Set dynamic grwoth
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Fix GPU visibility
os.environ["CUDA_VISIBLE_DEVICES"] =  str(args.gpu)

# Fix tensorflow threads
os.environ["TF_NUM_INTRAOP_THREADS"] = str(tf_threads)
os.environ["TF_NUM_INTEROP_THREADS"] = str(tf_threads)
tf.config.threading.set_inter_op_parallelism_threads(tf_threads)
tf.config.threading.set_intra_op_parallelism_threads(tf_threads)
tf.config.set_soft_device_placement(True)

#-----------------------------------------------------#
#        AUCMEDI Classifier Setup for STOIC2021       #
#-----------------------------------------------------#
# Load sampling from disk
sampling = pd.read_csv(path_sampling)
sampling["index"] = sampling["index"].astype(str)

# Load metadata from disk
meta = pd.read_csv(path_meta)
meta["index"] = meta["index"].astype(str)
meta["PatientSex"] = meta["PatientSex"].replace({"F": 0, "M": 1})
age_map = {35:0, 45:1/5, 55:2/5, 65:3/5, 75:4/5, 85:5/5}
meta["PatientAge"].replace(age_map, inplace=True)

# Combine sampling and metadata
dataset = pd.merge(sampling, meta, on="index")

# Define Subfunctions
pp_rs = (2.10, 1.48, 1.48)
sf_list = [Clip(min=-1024, max=100),
           Standardize(mode="grayscale"),
           Padding(mode="constant", shape=input_shape),
           Crop(shape=input_shape, mode="center"),
           Chromer(target="rgb")]

# Initialize architecture
nn_arch = Architecture_ResNet34(meta_variables=len(meta_cols),
                                channels=3, input_shape=input_shape)
# Initialize model
model = Neural_Network(n_labels=2, channels=3, architecture=nn_arch,
                       workers=processes,
                       batch_queue_size=batch_queue_size,
                       multiprocessing=True)

# Obtain standardization mode for current architecture
sf_standardize = "tf"


# Iterate over each model type
for m in model_types:
    # Iterate over each kfold
    df_list = []
    for fold in range(0, n_folds):
        # Load model
        path_model_best = os.path.join(path_model, arch + "." + str(fold) + \
                                       ".model." + m + ".hdf5")
        model.load(path_model_best)

        # Select sampling dataframe to current CV fold
        test_dt = dataset[dataset["kfold"]== "fold_" + str(fold)]
        test_x = test_dt.loc[:,"index"].to_numpy()
        test_meta = test_dt.loc[:,["kfold", "sampling"]].to_numpy()
        test_y = pd.get_dummies(test_dt.loc[:,"probSevere"]).to_numpy()
        test_m = test_dt.loc[:,meta_cols].to_numpy()

        # Initialize training and validation Data Generators
        test_gen = DataGenerator_MetaData(test_x, path_ct, metadata=test_m, labels=None,
                                 batch_size=batch_size, img_aug=None,
                                 shuffle=False, subfunctions=sf_list,
                                 resize=None, standardize_mode=sf_standardize,
                                 grayscale=True, prepare_images=False,
                                 sample_weights=None, seed=None,
                                 image_format="mha", workers=threads,
                                 loader=sitk_loader, resampling=pp_rs)
        # Use fitted model for predictions
        preds = model.predict(test_gen)
        # Create prediction dataset
        df_index = pd.DataFrame(data={"index": test_x})
        df_meta = pd.DataFrame(data=test_meta, columns=["kfold", "sampling"])
        df_pd = pd.DataFrame(data=preds, columns=["pd_probSevere:0", "pd_probSevere:1"])
        df_gt = pd.DataFrame(data=np.argmax(test_y, axis=-1), columns=["gt_probSevere"])

        df_merged = pd.concat([df_meta, df_index, df_pd, df_gt],
                              axis=1, sort=False)
        # Append to df list
        df_list.append(df_merged)
    # Merge df list to final prediction dataframe
    df_final = pd.concat(df_list, axis=0, sort=False)
    # Store predictions to disk
    path_out = os.path.join(path_model, arch + ".preds." + m + ".csv")
    df_final.to_csv(path_out, index=False)
