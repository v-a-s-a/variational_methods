#!/bin/bash

 singularity shell \
    --pwd $PWD \
    --bind $(pwd)data/:/data/ \
    $(pwd)/singularity/tf-jupyter-2018-06-06-0fb2ab4dce61.img