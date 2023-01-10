# Edit the base image here, e.g., to use
# TENSORFLOW (https://hub.docker.com/r/tensorflow/tensorflow/)
# or a different PYTORCH (https://hub.docker.com/r/pytorch/pytorch/) base image
FROM nvcr.io/nvidia/pytorch:21.10-py3

RUN apt-get update

RUN groupadd -r train && useradd -m --no-log-init -r -g train train

RUN mkdir -p /opt/train /input /output \
    && chown train:train /opt/train /input /output



RUN apt install wget -y

RUN mkdir -p /opt/algorithm/models
RUN wget https://mediastore.rz.uni-augsburg.de/get/4_60E_kPYu/ -O /opt/algorithm/models/model.miscnn.hdf5
RUN wget https://github.com/ZFTurbo/classification_models_3D/releases/download/v1.0/densenet121_inp_channel_3_tch_0_top_False.h5 -O /opt/algorithm/models/densenet121_imagenet_weightsss.h5
USER train
WORKDIR /opt/train

ENV PATH="/home/train/.local/bin:${PATH}"


RUN python -m pip install --user -U pip



COPY --chown=train:train requirements.txt /opt/train/
RUN python -m pip install --user -r requirements.txt

ENV TMP="/scratch"

COPY --chown=train:train config.py /opt/train/
COPY --chown=train:train ctdataset.py /opt/train/
COPY --chown=train:train main.py /opt/train/
COPY --chown=train:train metrics.py /opt/train/
COPY --chown=train:train train.py /opt/train/
COPY --chown=train:train algorithm/ /opt/train/algorithm/
COPY --chown=train:train config/ /opt/train/config/

ENTRYPOINT python -m main $0 $@



