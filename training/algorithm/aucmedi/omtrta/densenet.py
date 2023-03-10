#==============================================================================#
#  Author:       Dominik MÃ¼ller                                                #
#  Copyright:    2022 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
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
#              REFERENCE IMPLEMENTATION:              #
# https://github.com/ZFTurbo/classification_models_3D #
#   Solovyev, Roman & Kalinin, Alexandr & Gabruseva,  #
#                  Tatiana. (2021).                   #
#    3D Convolutional Neural Networks for Stalled     #
#              Brain Capillary Detection.             #
#-----------------------------------------------------#
#                   REFERENCE PAPER:                  #
#                     25 Aug 2016.                    #
#      Densely Connected Convolutional Networks.      #
#    Gao Huang, Zhuang Liu, Laurens van der Maaten,   #
#                Kilian Q. Weinberger.                #
#           https://arxiv.org/abs/1608.06993          #
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from tensorflow.keras.models import Model
from tensorflow.keras import Input, layers
import tensorflow.keras.layers as layers
from classification_models_3D.tfkeras import Classifiers
# Internal libraries
from aucmedi.neural_network.architectures import Architecture_Base

#-----------------------------------------------------#
#           Architecture class: DenseNet121           #
#-----------------------------------------------------#
""" The classification variant of the DenseNet121 architecture.

Methods:
    __init__                Object creation function
    create_model:           Creating the DenseNet121 model for classification
"""
class Architecture_DenseNet121(Architecture_Base):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, meta_variables, channels, input_shape=(64, 64, 64)):
        self.input = input_shape + (channels,)
        self.meta_variables = meta_variables

    #---------------------------------------------#
    #                Create Model                 #
    #---------------------------------------------#
    def create_model(self, n_labels, fcl_dropout=True, out_activation="softmax",
                     pretrained_weights=False, multi_layer_head=True):
        # Get pretrained image weights from imagenet if desired
        if pretrained_weights : model_weights = "/opt/algorithm/models/densenet121_imagenet_weightsss.h5"
        else : model_weights = None

        # Obtain DenseNet121 as base model
        DenseNet121, preprocess_input = Classifiers.get("densenet121")
        base_model = DenseNet121(include_top=False, weights=model_weights,
                                 input_tensor=None, input_shape=self.input,
                                 pooling=None)
        top_model = base_model.output

        # Define metadata input
        meta_in = Input(shape=(self.meta_variables,))


        # Add classification head as top model
        top_model = layers.GlobalAveragePooling3D(name="avg_pool")(top_model)
        if fcl_dropout:
            top_model = layers.Dense(units=512)(top_model)
            top_model = layers.Dropout(0.3)(top_model)
        # Combine meta data and classification head
        head_model = layers.concatenate([top_model, meta_in])
        if multi_layer_head:
            head_model = layers.Dense(units=512)(head_model)
            head_model = layers.Dropout(0.3)(head_model)
            head_model = layers.LeakyReLU(0.1)(head_model)
            head_model = layers.Dense(units=256)(head_model)
            head_model = layers.Dropout(0.3)(head_model)
            head_model = layers.LeakyReLU(0.1)(head_model)
        # Add classification model head
        head_model = layers.Dense(n_labels, name="preds")(head_model)
        head_model = layers.Activation(out_activation, name="probs")(head_model)

    
        

        # Create model
        model = Model(inputs=[base_model.input, meta_in], outputs=head_model)

        # Return created model
        return model