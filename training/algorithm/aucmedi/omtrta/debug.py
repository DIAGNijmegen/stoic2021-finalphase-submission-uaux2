import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from aucmedi.data_processing.subfunctions import Padding, Crop, Clip, Chromer, Standardize
from aucmedi.data_processing.io_loader import sitk_loader
from aucmedi import Neural_Network

import sys
sys.path.append("/data/stoic21/stoic2021-team-monai/")
from algorithm.aucmedi.omtrta.datagenerator import DataGenerator_MetaData
from algorithm.aucmedi.omtrta.resnet import Architecture_ResNet34

# Fix GPU visibility
import os
os.environ["CUDA_VISIBLE_DEVICES"] =  str(1)

#-----------------------------------------------------#
#                      Debugging                      #
#-----------------------------------------------------#
if __name__ == "__main__":
    # Provide path to imaging data
    path_ct = "/share/stoic2021-training.v2/data/lungseg/"
    # Provide path to the sampling list
    path_sampling = "/data/stoic21/stoic2021-team-monai/explore/results/sampling.CV.csv"
    # Provide path to the metadata
    path_metadata = "/data/stoic21/stoic2021-team-monai/explore/results/ilr.csv"

    # Load sampling from disk
    sampling = pd.read_csv(path_sampling)
    sampling["index"] = sampling["index"].astype(str)

    # Load metadata from disk
    ilr = pd.read_csv(path_metadata)
    ilr["index"] = ilr["index"].astype(str)
    ilr["PatientSex"] = ilr["PatientSex"].replace({"F": 0, "M": 1})
    age_map = {35:0, 45:1/5, 55:2/5, 65:3/5, 75:4/5, 85:5/5}
    ilr["PatientAge"].replace(age_map, inplace=True)

    # Subset
    sampling = sampling[sampling["kfold"]== "fold_" + str(1)]
    train_dt = sampling[sampling["sampling"] == "train"]
    train_dt = pd.merge(train_dt, ilr, on="index")

    # Extract subset
    train_x = train_dt.loc[:,"index"].to_numpy()
    train_y = pd.get_dummies(train_dt.loc[:,["probCOVID", "probSevere"]]).to_numpy()
    train_y = np.sum(train_y, axis=-1)
    train_y = to_categorical(train_y, num_classes=3)
    train_meta = train_dt.loc[:,["PatientAge", "PatientSex", "ilr"]].to_numpy()

    # Define Subfunctions
    pp_rs = (2.10, 1.48, 1.48)
    sf_list = [Clip(min=-1024, max=100),
               Standardize(mode="grayscale"),
               Padding(mode="constant", shape=(128,128,128)),
               Crop(shape=(128,128,128), mode="random"),
               Chromer(target="rgb")]

    ############################################################################

    my_gen = DataGenerator_MetaData(train_x, path_ct, train_meta,
                              labels=train_y,
                              batch_size=3, img_aug=None, shuffle=False,
                              subfunctions=sf_list, resize=None,
                              standardize_mode="tf",
                              grayscale=True, prepare_images=False,
                              sample_weights=None, seed=None,
                              image_format="mha", workers=1,
                              loader=sitk_loader, resampling=pp_rs)

    # Initialize architecture
    nn_arch = Architecture_ResNet34(meta_variables=3, channels=3,
                                    input_shape=(128,128,128))
    # Initialize model
    model = Neural_Network(n_labels=3, channels=3, architecture=nn_arch)

    model.train(my_gen, epochs=10)

    # for batch in my_gen:
    #     print(batch[0][0].shape, batch[0][1].shape, batch[1].shape)
