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
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, \
                                       ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.metrics import AUC
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
# AUCMEDI libraries
from aucmedi import input_interface, DataGenerator, Neural_Network, Batchgenerators_Augmentation
from aucmedi.neural_network.architectures import supported_standardize_mode, architecture_dict
from aucmedi.utils.class_weights import compute_class_weights
from aucmedi.data_processing.subfunctions import Padding, Crop, Clip, Chromer, Standardize
from aucmedi.data_processing.io_loader import sitk_loader
# Internal functions
import sys
sys.path.append(os.getcwd())
from algorithm.aucmedi.loss import focal_f1

#-----------------------------------------------------#
#                      Argparser                      #
#-----------------------------------------------------#
parser = argparse.ArgumentParser(description="AUCMEDI Training for STOIC2021")
parser.add_argument("-f", "--fold", help="Cross-Validation Fold",
                    required=True, type=int, dest="fold")
parser.add_argument("--data", help="Path to dataset (lungseg)",
                    required=True, type=str, dest="path_ct")
parser.add_argument("--sampling", help="Path to sampling (sampling.CV.csv)",
                    required=False, type=str, dest="path_sampling",
                    default="explore/results/sampling.CV.csv")
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

# Result/Model directory
path_model = "models"

# Identify fold
k_fold = args.fold

# Define some parameters
batch_size = 8
processes = 12
batch_queue_size = 10
threads = batch_size
tf_threads = 5

# Define architecture which should be processed
arch = "3D.ResNet34"

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

# Select sampling dataframe to current CV fold
sampling = sampling[sampling["kfold"]== "fold_" + str(k_fold)]

# Extract training subset
train_dt = sampling[sampling["sampling"] == "train"]
train_x = train_dt.loc[:,"index"].to_numpy()
train_y = pd.get_dummies(train_dt.loc[:,["probCOVID", "probSevere"]]).to_numpy()
train_y = np.sum(train_y, axis=-1)
train_y = to_categorical(train_y, num_classes=3)
# Extract validation subset
val_dt = sampling[sampling["sampling"] == "val"]
val_x = val_dt.loc[:,"index"].to_numpy()
val_y = pd.get_dummies(val_dt.loc[:,["probCOVID", "probSevere"]]).to_numpy()
val_y = np.sum(val_y, axis=-1)
val_y = to_categorical(val_y, num_classes=3)

# Initialize Volume Augmentation
aug = Batchgenerators_Augmentation(image_shape=input_shape,
                mirror=True, rotate=True, scale=True,
                elastic_transform=True, gaussian_noise=False,
                brightness=False, contrast=False, gamma=True)
aug.aug_contrast_per_channel = False
aug.aug_brightness_per_channel = False
aug.aug_gamma_per_channel = False
aug.aug_mirror_p = 0.5
aug.aug_rotate_p = 0.5
aug.aug_scale_p = 0.5
aug.aug_gamma_p = 0.5
aug.aug_elasticTransform_p = 0.5
aug.build()

# Define Subfunctions
pp_rs = (2.10, 1.48, 1.48)
sf_list = [Clip(min=-1024, max=100),
           Standardize(mode="grayscale"),
           Padding(mode="constant", shape=input_shape),
           Crop(shape=input_shape, mode="random"),
           Chromer(target="rgb")]

# Compute class weights
class_weights, _ = compute_class_weights(ohe_array=train_y)

# Initialize architecture
nn_arch = architecture_dict[arch](channels=3, input_shape=input_shape)
# Initialize model
model = Neural_Network(n_labels=3, channels=3, architecture=nn_arch,
                       workers=processes,
                       batch_queue_size=batch_queue_size,
                       loss=focal_f1(class_weights),
                       metrics=[AUC(100), F1Score(num_classes=3, average="macro")],
                       pretrained_weights=True, multiprocessing=True)

# Obtain standardization mode for current architecture
sf_standardize = "tf"

# Initialize training and validation Data Generators
train_gen = DataGenerator(train_x, path_ct, labels=train_y,
                          batch_size=batch_size, img_aug=aug, shuffle=True,
                          subfunctions=sf_list, resize=None,
                          standardize_mode=sf_standardize,
                          grayscale=True, prepare_images=False,
                          sample_weights=None, seed=None,
                          image_format="mha", workers=threads,
                          loader=sitk_loader, resampling=pp_rs)
val_gen = DataGenerator(val_x, path_ct, labels=val_y, batch_size=batch_size,
                        img_aug=None, subfunctions=sf_list, shuffle=False,
                        standardize_mode=sf_standardize, resize=None,
                        grayscale=True, prepare_images=False,
                        sample_weights=None, seed=None,
                        image_format="mha", workers=threads,
                        loader=sitk_loader, resampling=pp_rs)

# Create model directory
if not os.path.exists(path_model) : os.mkdir(path_model)

# Define callbacks
cb_ml = ModelCheckpoint(os.path.join(path_model, arch + "." + str(k_fold) + ".model.best.loss.hdf5"),
                        monitor="val_loss", verbose=1,
                        save_best_only=True, mode="min")
cb_ma = ModelCheckpoint(os.path.join(path_model, arch + "." + str(k_fold) +  ".model.best.auc.hdf5"),
                        monitor="val_auc", verbose=1,
                        save_best_only=True, mode="max")
cb_mr = ModelCheckpoint(os.path.join(path_model, arch + "." + str(k_fold) + ".model.best.f1.hdf5"),
                        monitor="val_f1_score", verbose=1,
                        save_best_only=True, mode="max")
cb_cl = CSVLogger(os.path.join(path_model, arch +  "." + str(k_fold) + ".training.csv"),
                  separator=',', append=True)
cb_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8,
                          verbose=1, mode='min', min_lr=1e-7)
cb_es = EarlyStopping(monitor='val_loss', patience=36, verbose=1)
callbacks = [cb_ml, cb_ma, cb_mr, cb_cl, cb_lr, cb_es]

# Train model
model.train(train_gen, val_gen, epochs=250, iterations=None,
            callbacks=callbacks, transfer_learning=False)

# Dump latest model
model.dump(os.path.join(path_model, arch +  "." + str(k_fold) + ".model.last.hdf5"))
