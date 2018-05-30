#!/bin/bash

docker run \
	-it \
	--rm \
	--memory 8G \
	--volume $(pwd):/notebooks \
	-p 8888:8888 tf-jupyter
