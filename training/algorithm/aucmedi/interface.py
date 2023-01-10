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
# TensorFlow libraries
import tensorflow as tf
# AUCMEDI libraries
from aucmedi import Neural_Network

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
#              Setup of AUCMEDI Pipeline              #
#-----------------------------------------------------#
""" Function for setup all required AUCMEDI modules to obtain a functional AUCMEDI model.
    The model allows the function algorithm.aucmedi.inference.predict() to utilize a deep computer vision model.

Args:
    path_model (String):                    File path to the trained AUCMEDI COVID-19 detection model.
    n_labels (Integer):                     Number of labels to predict (2 for severity / 3 for multi-class)

Returns:
    model (MIScnn Neural_Network):          A functional instance of the MIScnn Neural_Network class with a preloaded weights.
"""
def setup_aucmedi(path_model, n_labels=2):
    # Setup tensorflow stack
    setup_tensorflow()
    # Initialize model
    model = Neural_Network(n_labels=n_labels, channels=3, architecture=None,
                           workers=1, batch_queue_size=1, multiprocessing=False)
    # Load model
    model.load(path_model)

    # Return aucmedi model
    return model
