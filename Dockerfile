FROM tensorflow/tensorflow:latest-py3

RUN apt-get -y update

CMD jupyter-notebook --ip="*" --no-browser --allow-root