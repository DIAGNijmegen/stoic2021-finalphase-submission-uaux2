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
import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
import multiprocessing as mp

import random

import sys
sys.path.append(os.getcwd())
from algorithm.lungseg.apply import lung_segmentation

#-----------------------------------------------------#
#                    Configuration                    #
#-----------------------------------------------------#
# Which gpu to use
os.environ["CUDA_VISIBLE_DEVICES"] = str(2)

# STOIC21 CT data directory
path_stoic = "/share/stoic2021-training.v2/data/mha/"
# Result directory (requires extensive disk space!)
path_results = "/share/stoic2021-training.v2/data/lungseg/"
# Sampling file
path_sampling = "/data/stoic21/stoic2021-team-monai/explore/results/sampling.csv"

#-----------------------------------------------------#
#                       Runner                        #
#-----------------------------------------------------#
if __name__ == "__main__":
    # Set multiprocessing method to spawn
    try : mp.set_start_method("spawn")
    except RuntimeError : pass

    # Get global sample list
    sampling = pd.read_csv(path_sampling)
    sample_list = sampling["index"].astype(str).to_numpy()
    random.shuffle(sample_list)

    # Create result directory
    if not os.path.exists(path_results) : os.mkdir(path_results)

    # Information gathering
    infos = [[], [], []]

    # Iterate over each ct scan
    for index in sample_list:
        # Check if output already exists
        path_sample_out = os.path.join(path_results, index + ".mha")
        if os.path.exists(path_sample_out) : continue

        # Read sample
        path_sample_in = os.path.join(path_stoic, index + ".mha")
        input_image = sitk.ReadImage(path_sample_in)

        # Start process for lung segmentation
        p_ls = mp.Process(target=lung_segmentation,
                          args=(input_image, path_sample_out))
        p_ls.start()
        p_ls.join()

        # Validation
        sample_out = sitk.ReadImage(path_sample_out)
        soLS = sample_out.GetMetaData("LungSize")
        soOS = sample_out.GetMetaData("OriginalShape")
        soVol = sitk.GetArrayFromImage(sample_out).shape
        print(index, soLS, soOS, soVol)
        # Update infos
        infos[0].append(soVol[0])
        infos[1].append(soVol[1])
        infos[2].append(soVol[2])

    # Print out information
    print("X-Axis:", infos[0])
    print("Y-Axis:", infos[1])
    print("Z-Axis:", infos[2])
    print("Information - X - Minimum:", np.min(infos[0]))
    print("Information - X - Maximum:", np.max(infos[0]))
    print("Information - X - Median:", np.median(infos[0]))
    print("Information - X - Mean:", np.mean(infos[0]))
    print("Information - Y - Minimum:", np.min(infos[1]))
    print("Information - Y - Maximum:", np.max(infos[1]))
    print("Information - Y - Median:", np.median(infos[1]))
    print("Information - Y - Mean:", np.mean(infos[1]))
    print("Information - Z - Minimum:", np.min(infos[2]))
    print("Information - Z - Maximum:", np.max(infos[2]))
    print("Information - Z - Median:", np.median(infos[2]))
    print("Information - Z - Mean:", np.mean(infos[2]))
