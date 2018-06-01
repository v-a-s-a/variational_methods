#!/bin/bash

docker run \
	-it \
	--rm \
	--memory 8G \
	--volume $(pwd):/notebooks \
	--volume $(pwd)/logs/:/logs \
	-p 8888:8888 -p 6006:6006 tf-jupyter 
