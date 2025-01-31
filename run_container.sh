#!/usr/bin/env bash
xhost +
docker run --gpus all --rm -it -e DISPLAY=${DISPLAY} -v /tmp:/tmp -v .:/mnt graspnet_ros bash
