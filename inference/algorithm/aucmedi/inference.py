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
import SimpleITK as sitk
import numpy as np
import os
# AUCMEDI packages
from aucmedi.data_processing.io_loader import cache_loader
from aucmedi.data_processing.subfunctions import Padding, Crop, Clip, Chromer, Standardize
from aucmedi import DataGenerator
from algorithm.aucmedi.omtrta.datagenerator import DataGenerator_MetaData
# Internal packages
import sys
sys.path.append(os.getcwd())
from algorithm.aucmedi.interface import setup_aucmedi
from algorithm.preprocess import resample

#-----------------------------------------------------#
#                 Inference Function                  #
#-----------------------------------------------------#
"""Applied on a original ITKsample without any preprocessing.

Args:
    sample_itk (SimpleITK Image):           A CT scan sample provided by itk.ReadImage(...).
    path_models (List):                     List of Strings. Pathes to a AUCMEDI model.
    meta_data                               Meta data of the sample.
    cv_queue (Queue):                       A multiprocessing Queue object to store the prediction float value in it.

Returns:
    pred (float):                           Computed severity predictions from CV models as percentage between [0.0,1.0].
"""
def predict_cv_meta(sample_itk, path_models, meta_data, cv_queue=None):
    # Resample image
    sample_itk_resampled = resample(sample_itk, new_spacing=(1.48, 1.48, 2.10))
    # Extract image
    volume = sitk.GetArrayFromImage(sample_itk_resampled)
    # Encode information as dictionary
    cache = {"my_ct_scan": volume}
    # Define AUCMEDI data processing pipeline
    sf_list = [Clip(min=-1024, max=100),
               Standardize(mode="grayscale"),
               Padding(mode="constant", shape=(148, 224, 224)),
               Crop(shape=(148, 224, 224), mode="center"),
               Chromer(target="rgb")]
    # Initialize inference Data Generator
    test_gen = DataGenerator_MetaData(["my_ct_scan"], None,
                             metadata=meta_data, labels=None,
                             batch_size=1, img_aug=None, shuffle=False,
                             subfunctions=sf_list, resize=None,
                             standardize_mode="torch",
                             grayscale=True, prepare_images=False,
                             sample_weights=None, seed=None,
                             image_format=None, two_dim=False,
                             loader=cache_loader, cache=cache)
    # Iterate over each cross-validation model
    pred_list = []
    for cv_m in path_models:
        # Setup AUCMEDI pipeline
        model = setup_aucmedi(cv_m, n_labels=2)
        # Compute CV predictions
        cv_pred = model.predict(test_gen)
        # Postprocess prediction
        cv_pred = np.squeeze(cv_pred, axis=0)
        cv_pred = cv_pred.tolist()[1]
        # Append to prediction list
        pred_list.append(cv_pred)
    # Compute final prediction by simple averaging
    pred = np.mean(pred_list)
    # Return score
    if cv_queue is None : return pred
    else : cv_queue.put(pred)

#-----------------------------------------------------#
#          Inference Function for multiclass          #
#-----------------------------------------------------#
"""The ITK sample has to be already lung segmented via algorithm.lungseg.apply.lung_segmentation()

Args:
    sample_itk (SimpleITK Image):           A CT scan sample provided by itk.ReadImage(...).
    path_model (String):                    Path to a AUCMEDI model.
    mc_queue (Queue):                       A multiprocessing Queue object to store the multiclass float values in it.

Returns:
    pred (float):                           Computed multiclass predictions (3x) from CV model as percentage between [0.0,1.0].
                                            Expected outcome:  [[float, float, float]] with shape (1,3)
                                            Return if mc_queue is None or not passed.
"""
def predict_mc(sample_itk, path_model, mc_queue=None):
    # Extract image
    volume = sitk.GetArrayFromImage(sample_itk)
    # Setup AUCMEDI pipeline
    model = setup_aucmedi(path_model, n_labels=3)
    # Encode information as dictionary
    cache = {"my_ct_scan": volume}
    # Define AUCMEDI data processing pipeline
    sf_list = [Clip(min=-1024, max=100),
               Standardize(mode="grayscale"),
               Padding(mode="constant", shape=(148, 180, 224)),
               Crop(shape=(148, 180, 224), mode="center"),
               Chromer(target="rgb")]
    # Initialize inference Data Generator
    test_gen = DataGenerator(["my_ct_scan"], None, labels=None,
                             batch_size=1, img_aug=None, shuffle=False,
                             subfunctions=sf_list, resize=None,
                             standardize_mode="tf",
                             grayscale=True, prepare_images=False,
                             sample_weights=None, seed=None,
                             image_format=None, two_dim=False,
                             loader=cache_loader, cache=cache)
    # Compute CV predictions
    pred = model.predict(test_gen)
    # Postprocess prediction
    pred = np.squeeze(pred, axis=0)
    pred = pred.tolist()
    # Return score
    if mc_queue is None : return pred
    else : mc_queue.put(pred)
