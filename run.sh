#!/bin/bash

docker run \
	-it \
	--rm \
	--memory 4096M \
	--volume $(pwd):/notebooks \
	-p 8888:8888 tf-jupyter
