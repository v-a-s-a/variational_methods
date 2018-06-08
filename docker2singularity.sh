#!/bin/bash

docker run \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /Users/vasa/Projects/differentiable_experiments/singularity:/output \
    --privileged -t --rm \
    singularityware/docker2singularity:1.11 \
    tf-jupyter
