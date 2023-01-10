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
# External packages
import numpy as np
import tensorflow as tf
# Internal packages
from aucmedi.neural_network.loss_functions import categorical_focal_loss

#-----------------------------------------------------#
#                    F1-score loss                    #
#-----------------------------------------------------#
""" Compute macro-averaged (soft) F1-score.
    F1-score is also called Dice Similarity Coefficient.

    Arguments:
        y_true {tensor} : Ground truth labels, with shape (batch_size, number_of_classes).
        y_pred {tensor} : Model's predictions, with shape (batch_size, number_of_classes).

    Returns:
        loss {tensor}   : A tensor of soft F1 loss.
"""
def f1_soft(y_true, y_pred):
    # Cast to float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # Obtain confusion matrix
    tp = tf.reduce_sum(y_pred * y_true, axis=0)
    fp = tf.reduce_sum(y_pred * (1 - y_true), axis=0)
    fn = tf.reduce_sum((1 - y_pred) * y_true, axis=0)
    # Compute F1-score
    f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    # Macro average and return loss
    return 1.0 - tf.reduce_mean(f1)

#-----------------------------------------------------#
#                    Focal-F1 loss                    #
#-----------------------------------------------------#
""" Compute the sum of macro-averaged (soft) F1-score and categorical Focal loss.
    F1-score is also called Dice Similarity Coefficient.

    Arguments:
        y_true {tensor} : Ground truth labels, with shape (batch_size, number_of_classes).
        y_pred {tensor} : Model's predictions, with shape (batch_size, number_of_classes).

    Returns:
        loss {tensor}   : A tensor of focal-f1 loss.
"""
def focal_f1(alpha):
    alpha = np.array(alpha, dtype=np.float32)
    loss_focal = categorical_focal_loss(alpha=alpha)
    loss_f1 = f1_soft

    def compute_loss(y_true, y_pred):
        focal = loss_focal(y_true, y_pred)
        f1 = loss_f1(y_true, y_pred)
        return focal + f1
    return compute_loss
