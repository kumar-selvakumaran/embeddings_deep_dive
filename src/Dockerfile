
FROM  python:3.7.11

WORKDIR /app

#ENV CUDA_HOME=/usr/local/cuda-11.3

ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute ntainer


RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC


COPY . .

RUN apt-get update

RUN apt-get install -y make automake gcc g++ git libsm6 libxext6 libgl1 libglib2.0-0

# RUN git clone https://github.com/eriklindernoren/PyTorch-YOLOv3
WORKDIR PyTorch-YOLOv3/
# RUN pip3 install poetry --user
# RUN poetry install
# RUN ./weights/download_weights.sh
RUN pip3 install pytorchyolo --user

RUN pip uninstall -y torch torchvision

RUN pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# RUN git clone https://github.com/IDEA-Research/GroundingDINO.git

# WORKDIR /GroundingDINO

# RUN pip install -r requirements.txt


# RUN python setup.py build develop --user

# WORKDIR /app


# ############################################### jupyter testing ###########################################################

RUN pip install jupyter ipykernel

EXPOSE 8888

# jupyter notebook --ip=0.0.0.0 port=8888 NotebookApp.token='' --NotebookApp.password='' --allow-root
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--NotebookApp.token=''","--NotebookApp.password=''", "--allow-root"]

# ###########################################################################################################################
