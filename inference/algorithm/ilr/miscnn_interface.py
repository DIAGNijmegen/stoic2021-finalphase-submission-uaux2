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
from miscnn.data_loading.interfaces import Dictionary_interface
from miscnn import Data_IO, Preprocessor, Neural_Network
from miscnn.processing.subfunctions import Normalization, Clipping, Resampling, Padding
import os
import tensorflow as tf

#-----------------------------------------------------#
#              Setup of Tensorflow Stack              #
#-----------------------------------------------------#
def setup_tensorflow(num_threads=5):
    # Set dynamic grwoth
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # Fix tensorflow threads
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(num_threads)
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    tf.config.set_soft_device_placement(True)

#-----------------------------------------------------#
#               Setup of MIScnn Pipeline              #
#-----------------------------------------------------#
""" Function for setup all required MIScnn modules to obtain a functional MIScnn model.
    The model allows the function algorithm.ilr.inference.predict_ilr() to compute an ILR.

    The setup_miscnn() function should be called once in the main algorithm __init__() of the docker.

Args:
    path_model (String):                    File path to the trained MIScnn COVID-19 segmentation model.

Returns:
    model (MIScnn Neural_Network):          A functional instance of the MIScnn Neural_Network class with a preloaded weights.
"""
def setup_miscnn(path_model):
    # Setup tensorflow stack
    setup_tensorflow()
    # Initialize Data IO Interface for NIfTI data
    ## We are using 4 classes due to [background, lung_left, lung_right, covid-19]
    interface = Dictionary_interface({}, channels=1, classes=4)

    # Create Data IO object to load and write samples in the file structure
    data_io = Data_IO(interface, input_path="data", delete_batchDir=False)

    # Create a pixel value normalization Subfunction for z-score scaling
    sf_zscore = Normalization(mode="z-score")

    # Assemble Subfunction classes into a list
    sf = [sf_zscore]

    # Create and configure the Preprocessor class
    pp = Preprocessor(data_io, data_aug=None, batch_size=1, subfunctions=sf,
                      prepare_subfunctions=False, prepare_batches=False,
                      analysis="fullimage")

    # Create the Neural Network model
    model = Neural_Network(preprocessor=pp, batch_queue_size=1, workers=1)
    # Load model weights
    model.load(path_model)

    # Return miscnn model
    return model
