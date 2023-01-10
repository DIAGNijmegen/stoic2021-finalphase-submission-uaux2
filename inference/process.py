from typing import Dict
from pathlib import Path
import SimpleITK
import multiprocessing as mp
import os
import numpy as np

from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

from utils import MultiClassAlgorithm, to_input_format, unpack_single_output, device, convert_age
# MIScnn ILR inference
from algorithm.ilr.inference import predict_ilr
# AUCMEDI predictions
from algorithm.aucmedi.inference import predict_mc, predict_cv_meta
# LungSeg preprocessing
from algorithm.lungseg.apply import lung_segmentation



COVID_OUTPUT_NAME = Path("probability-covid-19")
SEVERE_OUTPUT_NAME = Path("probability-severe-covid-19")


PATH_MODEL_LUNGSEG_FIXED = Path("/opt/algorithm/models/model.lungseg.pth")
PATH_MODEL_ILR_FIXED = Path("/opt/algorithm/models/model.miscnn.hdf5")
PATH_MODEL_AUCMEDI_MC = Path("/opt/algorithm/models/model.aucmedi.mc.hdf5")
PATH_MODEL_AUCMEDI = Path("/opt/algorithm/artifact/models.metadata.severity.densenet")

class StoicAlgorithm(MultiClassAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path=Path("/input/images/ct/"),
            output_path=Path("/output/")
        )

        # Set multiprocessing method to spawn
        mp.set_start_method("spawn")

    def prepare_meta_data(self, raw_age, raw_sex, sample_itk):
        # the be in line with the preprocessing in train.severity
        sex = 0  if raw_sex == "F" else 1
        age = (raw_age - 35)/50.
        # get ilr
        ilr_queue = mp.Queue()
        p_ilr = mp.Process(target=predict_ilr, args=(sample_itk, PATH_MODEL_ILR_FIXED, ilr_queue))
        p_ilr.start()
        p_ilr.join()
        ilr = ilr_queue.get()
        #print(np.array([[age,sex,ilr]]))
        return np.array([[age,sex,ilr]])


    def predict(self, *, input_image: SimpleITK.Image) -> Dict:
        # Read out metadata
        patient_sex = input_image.GetMetaData("PatientSex")
        patient_age_raw = input_image.GetMetaData("PatientAge")
        # Convert patient age from dicom format to normal integer
        patient_age = convert_age(patient_age_raw)


        # Apply lung segmentation
        path_sample_lungseg = "tmp.lungseg.mha"
        p_ls = mp.Process(target=lung_segmentation,
                          args=(input_image, path_sample_lungseg,
                                PATH_MODEL_LUNGSEG_FIXED))
        p_ls.start()
        p_ls.join()
        # Read preprocessed lungseg into RAM
        input_image_lungseg = SimpleITK.ReadImage(path_sample_lungseg)
        # Initialize shared variable for AUCMEDI MC process
        mc_queue = mp.Queue()
        # Start process for multi-class prediction via AUCMEDI
        p_aucmedi = mp.Process(target=predict_mc,
                               args=(input_image_lungseg, PATH_MODEL_AUCMEDI_MC,
                                     mc_queue))
        p_aucmedi.start()
        p_aucmedi.join()
        # Obtain AUCMEDI predictions
        mc_pred = mc_queue.get()       # outcome [float, float, float]
        # compute predictions from probCOVID inference
        prob_covid = mc_pred[1] + mc_pred[2]
        # Clean up
        os.remove(path_sample_lungseg)


        # Prepare metadata for CV Severity prediction
        meta_data = self.prepare_meta_data(patient_age, patient_sex, input_image)
        sample_itk = input_image

        # Initialize shared variable for AUCMEDI CV Severity process
        cv_queue = mp.Queue()
        # Start process for severity prediction via AUCMEDI
        PATH_MODEL_AUCMEDI_CV = list(PATH_MODEL_AUCMEDI.glob("*best.val_loss.hdf5"))
        print("Found number of models: ", len(PATH_MODEL_AUCMEDI_CV))
        p_aucmedi = mp.Process(target=predict_cv_meta,
                               args=(sample_itk, PATH_MODEL_AUCMEDI_CV, meta_data,
                                     cv_queue))
        p_aucmedi.start()
        p_aucmedi.join()
        # Obtain AUCMEDI prediction
        prob_severe = cv_queue.get()       # outcome float

        return {
            COVID_OUTPUT_NAME: prob_covid,
            SEVERE_OUTPUT_NAME: prob_severe
        }


if __name__ == "__main__":
    StoicAlgorithm().process()
