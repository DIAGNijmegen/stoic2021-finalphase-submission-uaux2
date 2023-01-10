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
#                               REFERENCE PAPER:                               #
#                    Yamada, D., Ohde, S., Imai, R. et al.                     #
# Visual classification of three computed tomography lung patterns to predict  #
#                 prognosis of COVID-19: a retrospective study.                #
#                          BMC Pulm Med 22, 1 (2022).                          #
#                  https://doi.org/10.1186/s12890-021-01813-y                  #
#------------------------------------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External
import SimpleITK as sitk
import numpy as np
from lungmask import mask
# Internal
import sys
import os
sys.path.append(os.getcwd())
from algorithm.preprocess import resample

#-----------------------------------------------------#
#                Quantitative Analysis                #
#-----------------------------------------------------#
""" todo

Args:
    sample (SimpleITK):              A SimpleITK sample containing a thorax CT scan.
    qa_queue (Queue):                A multiprocessing Queue object to store the QA multiple float values in it.
    resampling (Tuple):              Tuple of floats for desired voxel spacing after resampling.

    todo

Returns:
    todo
"""
def lung_segmentation(sample_itk, path_out, path_lungseg_model,
                      resampling=(1.48, 1.48, 2.10)):
    # Resample image
    sample_itk_resampled = resample(sample_itk, new_spacing=resampling)
    # Load lungseg model into cache
    model_lungseg = mask.get_model(modeltype="unet", modelname=None,
                                   modelpath=path_lungseg_model, n_classes=3)
    # Perform lung segmentation
    mask_lungs = mask.apply(sample_itk_resampled, model=model_lungseg)
    mask_lungs = np.where(mask_lungs==2, 1, mask_lungs)
    # Convert sample from SimpleITK to NumPy
    volume = sitk.GetArrayFromImage(sample_itk_resampled)
    # Clip volume to specific minimum value of -1024
    min_value = -1024
    volume = np.clip(volume, a_min=min_value, a_max=None)
    # Extract lung volume according to mask
    vol_lungs = np.where(mask_lungs==1, volume, np.full(volume.shape, min_value))
    # Identify min and max for each axis
    cp = []
    for axis in [0,1,2]:
        # Identify minimum
        min = identify_position(mask_lungs, axis, True)
        max = identify_position(mask_lungs, axis, False)
        # Add cropping positions for this axis
        cp.append((min, max))

    # Crop volume according to identified cropping positions
    lung_seg = vol_lungs[cp[0][0]:cp[0][1],
                         cp[1][0]:cp[1][1],
                         cp[2][0]:cp[2][1]]

    # Create SimpleITK object of lung segmentation
    lungSeg_itk = sitk.GetImageFromArray(lung_seg)
    lungSeg_itk.SetSpacing(resampling)
    # Store additional segmentation information
    lungSeg_itk.SetMetaData("LungSize", str(np.sum(mask_lungs==1)))
    lungSeg_itk.SetMetaData("OriginalShape", str(volume.shape))

    # Write lung segmentation to disk
    sitk.WriteImage(lungSeg_itk, path_out)


""" Internal function for lung_segmentation().

    Identifies the min or max position in an array with a value unequal to zero along an axis.
    Allow edge identification for later cropping.
"""
def identify_position(mask, axis, min=True):
    # Initialize range object
    if min : ro = range(0, mask.shape[axis])
    else : ro = reversed(range(0, mask.shape[axis]))
    # Iterate over each slice
    pos = 0
    for s in ro:
        slice = np.take(mask, s, axis=axis)
        # Return current position if slice is not fully background
        if len(np.unique(slice)) != 1 : return s
