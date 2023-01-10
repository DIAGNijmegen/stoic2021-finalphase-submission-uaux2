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
import csv
import pooch
import multiprocessing as mp

import sys
sys.path.append(os.getcwd())
from utils import convert_age
from algorithm.ilr.inference import predict_ilr

#-----------------------------------------------------#
#                    Configuration                    #
#-----------------------------------------------------#
# Which gpu to use
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

# STOIC21 CT data directory
path_stoic = "/share/stoic2021-training.v2/data/mha/"
# Result directory (repo/explore/results)
path_results = "/data/stoic21/stoic2021-team-monai/explore/results"

#-----------------------------------------------------#
#                       Runner                        #
#-----------------------------------------------------#
if __name__ == "__main__":
    # Set multiprocessing method to spawn
    try : mp.set_start_method("spawn")
    except RuntimeError : pass


    # Download MIScnn ILR model
    path_model_ilr = pooch.retrieve(url="https://mediastore.rz.uni-augsburg.de/get/4_60E_kPYu/",
                                    known_hash=None,
                                    progressbar=True)

    # Get global sample list
    path_sampling = os.path.join(path_results, "sampling.csv")
    sampling = pd.read_csv(path_sampling)
    sample_list = sampling["index"].astype(str).to_numpy()

    # Write header to ILR prediction result csv
    path_ilr = os.path.join(path_results, "ilr.csv")
    with open(path_ilr, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "PatientAge", "PatientSex", "ilr"])

    # Iterate over each ct scan
    for index in sample_list:
        # Read sample
        path_sample = os.path.join(path_stoic, index + ".mha")
        input_image = sitk.ReadImage(path_sample)

        # Initialize shared variable for MIScnn process
        ilr_queue = mp.Queue()
        # Start process for ILR estimation via MIScnn
        p_miscnn = mp.Process(target=predict_ilr,
                              args=(input_image, path_model_ilr, ilr_queue))
        p_miscnn.start()
        p_miscnn.join()
        # Obtain ILR estimation
        ilr = ilr_queue.get()

        # Read out metadata
        patient_sex = input_image.GetMetaData("PatientSex")
        patient_age_raw = input_image.GetMetaData("PatientAge")
        # Convert DICOM to normal age format
        patient_age = convert_age(patient_age_raw)

        # Store prediction
        with open(path_ilr, "a") as f:
            writer = csv.writer(f)
            writer.writerow([index, patient_age, patient_sex, ilr])
