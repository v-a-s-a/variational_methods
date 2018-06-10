#!/bin/bash

 singularity shell \
    --bind "/psych/ripke/vasa/variational_inference/differentiable_experiments/":"/data/" \
    "/psych/ripke/vasa/variational_inference//singularity/tf-jupyter-2018-06-06-0fb2ab4dce61.img"
