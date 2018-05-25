#!/bin/bash

docker run \
	-it \
	--rm \
	--memory 8192M \
	--volume $(pwd):/notebooks \
	-p 8888:8888 tf-jupyter
