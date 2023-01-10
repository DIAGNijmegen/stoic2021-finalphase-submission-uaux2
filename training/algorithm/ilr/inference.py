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
import SimpleITK as sitk
import numpy as np
import os

import sys
sys.path.append(os.getcwd())
from algorithm.ilr.miscnn_interface import setup_miscnn
from algorithm.preprocess import resample, center_crop, clip_and_normalize

#-----------------------------------------------------#
#                     Compute ILR                     #
#-----------------------------------------------------#
""" Internal function to compute an ILR providing a segmentation mask.
    Note: This function will be called internally from predict_ilr()!

Args:
    mask (NumPy matrix):                    3D NumPy Matrix with shape (x,y,z) and a sparse encoding (0=class A; 1=class B; ...)

Returns:
    ilr (float):                            Computed Infection-Lung_Ratio as percentage between [0.0,1.0].
"""
def compute_ilr_from_mask(mask):
    # Obtain class prevalance
    lungs = 0
    covid = 0
    classes, counts = np.unique(mask, return_counts=True)
    for i, c in enumerate(classes):
        if c == 1 or c == 2 : lungs += counts[i]
        elif c == 3 : covid += counts[i]
    # Calculate ILR
    if covid != 0 : ilr = covid / (covid + lungs)
    else : ilr = 0.0
    # Return ILR
    return ilr

#-----------------------------------------------------#
#                 Inference Function                  #
#-----------------------------------------------------#
""" External function to compute an ILR providing an ITK sample (ct scan) and a MIScnn Neural_Network model.
    Note: This function should be called from the main predict() in process.py of the docker!

Args:
    sample_itk (SimpleITK Image):           A CT scan sample provided by itk.ReadImage(...).
    path_model (String):                    Path to a MIScnn model.
    ilr_queue (Queue):                      A multiprocessing Queue object to store the ILR float value in it.

Returns:
    ilr (float):                            Computed Infection-Lung_Ratio as percentage between [0.0,1.0].
                                            Return if ilr_queue is None or not passed.
"""
def predict_ilr(sample_itk, model, ilr_queue=None):
    # Perform resampling via provided function
    sample_itk = resample(sample_itk, new_spacing=(1.58, 1.58, 2.70))
    # Extract image
    volume = sitk.GetArrayFromImage(sample_itk)
    # Perform center cropping
    volume = center_crop(volume, new_shape=(80, 160, 160), outside_val=-1250)
    # Transpose volume to be identical to model input
    volume = np.transpose(volume, axes=(2,1,0))
    # Perform Clipping and normalization
    volume = clip_and_normalize(volume, clip_min=-1250, clip_max=250)
    # Setup MIScnn pipeline
    model = model #setup_miscnn(path_model)
    # Encode information as dictionary
    sample_dict = {"my_ct_scan": (volume, None)}
    # Update dictionary interface
    model.preprocessor.data_io.interface.dictionary = sample_dict
    # Compute segmentation prediction
    seg = model.predict(["my_ct_scan"], return_output=True)
    # Compute ILR
    ilr = compute_ilr_from_mask(seg)
    # Return score
    if ilr_queue is None : return ilr
    else : ilr_queue.put(ilr)
