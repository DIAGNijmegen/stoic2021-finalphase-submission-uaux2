#==============================================================================#
#  Author:       Dominik Müller                                                #
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
#                     10 Dec 2015.                    #
#    Deep Residual Learning for Image Recognition.    #
#  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. #
#           https://arxiv.org/abs/1512.03385          #
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
from tensorflow.keras.models import Model
from tensorflow.keras import layers, Input
from classification_models_3D.tfkeras import Classifiers
# Internal libraries
from aucmedi.neural_network.architectures import Architecture_Base

#-----------------------------------------------------#
#            Architecture class: ResNet34             #
#-----------------------------------------------------#
""" The classification variant of the ResNet34 architecture.

Methods:
    __init__                Object creation function
    create_model:           Creating the ResNet34 model for classification
"""
class Architecture_ResNet34(Architecture_Base):
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
                     pretrained_weights=False):
        # Get pretrained image weights from imagenet if desired
        if pretrained_weights : model_weights = "imagenet"
        else : model_weights = None

        # Obtain ResNet34 as base model
        ResNet34, preprocess_input = Classifiers.get("resnet34")
        base_model = ResNet34(include_top=False, weights=model_weights,
                              input_tensor=None, input_shape=self.input,
                              pooling=None)
        top_model = base_model.output

        # Define metadata input
        meta_in = Input(shape=(self.meta_variables,))

        # Add image classification head
        top_model = layers.GlobalAveragePooling3D(name="avg_pool")(top_model)
        if fcl_dropout:
            top_model = layers.Dense(units=512)(top_model)
            top_model = layers.Dropout(0.3)(top_model)
        # Combine image classification and metadata
        head_model = layers.concatenate([top_model, meta_in])
        # Add classification model head
        head_model = layers.Dense(n_labels, name="preds")(head_model)
        head_model = layers.Activation(out_activation, name="probs")(head_model)

        # Create model
        model = Model(inputs=[base_model.input, meta_in], outputs=head_model)

        # Return created model
        return model
