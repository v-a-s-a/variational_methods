#!/bin/bash

docker run \
	-it \
	--rm \
	--volume $(pwd):/notebooks \
	-p 8888:8888 tf-jupyter
