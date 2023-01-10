import multiprocessing

from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
# TensorFlow libraries
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, \
                                       ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.metrics import AUC
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
# AUCMEDI libraries
from aucmedi import input_interface, DataGenerator, Neural_Network, Batchgenerators_Augmentation
from aucmedi.neural_network.architectures import supported_standardize_mode, architecture_dict
from aucmedi.utils.class_weights import compute_class_weights
from aucmedi.data_processing.subfunctions import Padding, Crop, Clip, Chromer, Standardize
from aucmedi.data_processing.io_loader import sitk_loader
from aucmedi.sampling import sampling_split
# Internal functions
import sys
sys.path.append(os.getcwd())
from algorithm.aucmedi.loss import focal_f1
from algorithm.aucmedi.omtrta.datagenerator import DataGenerator_MetaData
from algorithm.aucmedi.omtrta.densenet import Architecture_DenseNet121
from algorithm.ilr.inference import predict_ilr
from algorithm.ilr.miscnn_interface import setup_miscnn
import SimpleITK

COMPLETE_META = "meta.csv"
META_COLS = ["PatientAge", "PatientSex", "ilr"]

MISCNN_MODEL_PATH = "/opt/algorithm/models/model.miscnn.hdf5"
import tempfile
TMP_DIR = '/scratch'


def convert_age(age_dicom):
    if age_dicom[-1] == "Y" : return int(age_dicom[:-1])
    else : return int(age_dicom)

def create_meta_data(data_dir, artifact_dir):
    data_dict = {"PatientID": [],  "PatientAge": [], "PatientSex": [], "ilr": []}
    data_dir = Path(data_dir)
    image_dir = data_dir / "data/mha/"
    output = Path(artifact_dir) / COMPLETE_META
    miscnn_model = setup_miscnn(MISCNN_MODEL_PATH)
    for mha_file in tqdm(list(image_dir.glob("*.mha"))): # TODO: IMPROVEMENT, paralellize
        img = SimpleITK.ReadImage(str(mha_file))
        patient_sex = img.GetMetaData("PatientSex")
        patient_age_raw = img.GetMetaData("PatientAge")
        # Convert patient age from dicom format to normal integer
        patient_age = convert_age(patient_age_raw)
        patient_id = mha_file.stem
        ilr = predict_ilr(img, miscnn_model)# TODO: Implement ILR calculation
        data_dict["PatientID"].append(patient_id)
        data_dict["PatientSex"].append(patient_sex)
        data_dict["PatientAge"].append(patient_age)
        data_dict["ilr"].append(ilr)
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(str(output), index=False)


def read_dataset(data_dir: str, artifact_dir: str):
    reference_path = os.path.join(data_dir, "metadata/reference.csv")
    df = pd.read_csv(reference_path)
    meta_path = os.path.join(artifact_dir, COMPLETE_META)
    meta_df = pd.read_csv(meta_path)
    df = df.merge(meta_df, on="PatientID")
    df["PatientSex"] = df["PatientSex"].replace({"F": 0, "M": 1})
    age_map = {35: 0, 45: 1 / 5, 55: 2 / 5, 65: 3 / 5, 75: 4 / 5, 85: 5 / 5}
    df["PatientAge"].replace(age_map, inplace=True)
    df["PatientID"] = df["PatientID"].astype(str)
    return df


def split_dataset(df: pd.DataFrame, random_state):
    df_labels = pd.get_dummies(df.loc[:, "probSevere"]).to_numpy()
    df_train, df_valid = train_test_split(df, test_size=0.05, random_state=random_state,
                                          stratify=df_labels)
    train_x = df_train["PatientID"].to_numpy()
    train_y = pd.get_dummies(df_train.loc[:, "probSevere"]).to_numpy()
    train_m = df_train[META_COLS].to_numpy()
    valid_x = df_valid["PatientID"].to_numpy()
    valid_y = pd.get_dummies(df_valid.loc[:, "probSevere"]).to_numpy()
    valid_m = df_valid[META_COLS].to_numpy()

    return train_x, train_y, train_m, valid_x, valid_y, valid_m


def do_learning_process(data_dir, artifact_dir, split_idx):
    batch_size = 7
    processes = 7
    threads = 2
    batch_queue_size = 10
    monitor_loss = 'val_loss'
    # tf_threads = 2
    # os.environ["TF_NUM_INTRAOP_THREADS"] = str(tf_threads)
    # os.environ["TF_NUM_INTEROP_THREADS"] = str(tf_threads)
    # tf.config.threading.set_inter_op_parallelism_threads(tf_threads)
    # tf.config.threading.set_intra_op_parallelism_threads(tf_threads)
    # tf.config.set_soft_device_placement(True)

    input_shape = (148, 224, 224)

    aug = Batchgenerators_Augmentation(image_shape=input_shape,
                                       mirror=True, rotate=True, scale=True,
                                       elastic_transform=True, gaussian_noise=False,
                                       brightness=False, contrast=False, gamma=True)
    aug.aug_contrast_per_channel = False
    aug.aug_brightness_per_channel = False
    aug.aug_gamma_per_channel = False
    aug.aug_mirror_p = 0.1
    aug.aug_rotate_p = 0.1
    aug.aug_scale_p = 0.1
    aug.aug_gamma_p = 0.1
    aug.aug_elasticTransform_p = 0.1
    aug.build()

    # Define Subfunctions
    pp_rs = (2.10, 1.48, 1.48)
    sf_list = [Clip(min=-1024, max=100),
               Standardize(mode="grayscale"),
               Padding(mode="constant", shape=input_shape),
               Crop(shape=input_shape, mode="random"),
               Chromer(target="rgb")]

    # Get the data
    path_model = f"models.metadata.severity.densenet"
    path_model = Path(artifact_dir) / path_model
    if not os.path.exists(path_model) : os.mkdir(path_model)
    path_ct = os.path.join(data_dir, "data/mha/")
    df = read_dataset(data_dir, artifact_dir)
    print("Running Split number: ", split_idx)
    train_x, train_y, train_m, val_x, val_y, val_m = split_dataset(df, split_idx)

    class_weights, _ = compute_class_weights(ohe_array=train_y)
    nn_arch = Architecture_DenseNet121(meta_variables=len(META_COLS),
                                       channels=3, input_shape=input_shape)
    model = Neural_Network(n_labels=2, channels=3, architecture=nn_arch,
                           workers=processes,
                           batch_queue_size=batch_queue_size,
                           loss=focal_f1(class_weights),
                           metrics=[AUC(100), F1Score(num_classes=2, average="macro")],
                           pretrained_weights=True, multiprocessing=False)
    print("Model Done")
    model.tf_epochs = 10
    sf_standardize = "torch"
    arch = "3D.DenseNet121.metadata"
    train_gen = DataGenerator_MetaData(train_x, path_ct, metadata=train_m, labels=train_y,
                                       batch_size=batch_size, img_aug=aug, shuffle=True,
                                       subfunctions=sf_list, resize=None,
                                       standardize_mode=sf_standardize,
                                       grayscale=True, prepare_images=True,
                                       sample_weights=None, seed=None,
                                       image_format="mha", workers=threads,
                                       loader=sitk_loader, resampling=pp_rs)
    val_gen = DataGenerator_MetaData(val_x, path_ct, metadata=val_m, labels=val_y, batch_size=batch_size,
                                     img_aug=None, subfunctions=sf_list, shuffle=False,
                                     standardize_mode=sf_standardize, resize=None,
                                     grayscale=True, prepare_images=True,
                                     sample_weights=None, seed=None,
                                     image_format="mha", workers=threads,
                                     loader=sitk_loader, resampling=pp_rs)
    print("Data Done")
    chkpoint_name = f".model.best.{monitor_loss}.hdf5"
    final_checkpoint_path = os.path.join(path_model, arch + f".split{split_idx}" + chkpoint_name)
    tmp_checkpoint_path = final_checkpoint_path
    if split_idx > 0:
        chkpoint_name = f".model.best.unfinished.{monitor_loss}.hdf5"
        tmp_checkpoint_path = os.path.join(path_model, arch + f".split{split_idx}" + chkpoint_name)
    cb_mr = ModelCheckpoint(tmp_checkpoint_path,
                            monitor=monitor_loss, verbose=1,
                            save_best_only=True, mode="min") # min for val_loss, right?
    cb_lr = ReduceLROnPlateau(monitor=monitor_loss, factor=0.1, patience=8,
                              verbose=1, mode='min', min_lr=1e-7)
    cb_es = EarlyStopping(monitor=monitor_loss, patience=36, verbose=1)
    callbacks = [cb_mr, cb_lr, cb_es]
    model.train(train_gen, val_gen, epochs=250, iterations=420, callbacks=callbacks, transfer_learning=True)
    if split_idx > 0:
        os.rename(tmp_checkpoint_path, final_checkpoint_path)
    os.system("rm -rf /scratch/aucmedi.tmp.*")


def do_learning(data_dir, artifact_dir):
    print("Start Meta Process")
    preprocess_p = multiprocessing.Process(target=create_meta_data, args=(data_dir, artifact_dir))
    #create_meta_data(data_dir, artifact_dir)
    preprocess_p.start()
    preprocess_p.join()
    print("Meta Done")
    for i in range(999):
        print("Running Split", i)
        p = multiprocessing.Process(target=do_learning_process, args=(data_dir, artifact_dir, i, ))
        p.start()
        p.join()
    #with multiprocessing.Pool(processes=1) as pool:
    #    for i in range(999):
    #        pool.apply(func=do_learning_process, args=(data_dir, artifact_dir, i, ))
    return []
