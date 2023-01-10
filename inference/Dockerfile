# Edit the base image here, e.g., to use
# TENSORFLOW (https://hub.docker.com/r/tensorflow/tensorflow/)
# or a different PYTORCH (https://hub.docker.com/r/pytorch/pytorch/) base image
FROM nvcr.io/nvidia/pytorch:21.10-py3

RUN apt-get update
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip


RUN mkdir -p /opt/algorithm/models
RUN wget https://mediastore.rz.uni-augsburg.de/get/4_60E_kPYu/ -O /opt/algorithm/models/model.miscnn.hdf5
RUN wget https://github.com/JoHof/lungmask/releases/download/v0.0/unet_r231-d5d2fc3d.pth -O /opt/algorithm/models/model.lungseg.pth
RUN wget https://mediastore.rz.uni-augsburg.de/get/At3XprypsW/ -O /opt/algorithm/models/model.aucmedi.mc.hdf5

RUN python -m pip install git+https://github.com/JoHof/lungmask


COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -r requirements.txt

COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm utils.py /opt/algorithm/
COPY --chown=algorithm:algorithm algorithm/ /opt/algorithm/algorithm/
COPY --chown=algorithm:algorithm artifact/ /opt/algorithm/artifact/

ENTRYPOINT python -m process $0 $@

## ALGORITHM LABELS ##

# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=STOICAlgorithm
