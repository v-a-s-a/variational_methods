FROM tensorflow/tensorflow:nightly-py3

RUN apt-get -y update \
    && pip3 install --upgrade tfp-nightly jupyter


CMD tensorboard --logdir /logs --port 6006 & \
    jupyter-notebook --ip="*" --no-browser --allow-root 