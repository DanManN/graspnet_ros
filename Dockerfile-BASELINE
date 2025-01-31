FROM mzahana/ros-noetic-cuda11.4.2

# set user permissions
#RUN useradd user && \
RUN echo "user ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/user && \
	chmod 0440 /etc/sudoers.d/user && \
	mkdir -p /home/user && \
	chown user:user /home/user && \
	chsh -s /bin/bash user
RUN echo 'root:root' | chpasswd
RUN echo 'user:user' | chpasswd

# setup environment
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
RUN apt update && apt upgrade curl wget git -y

# add kitware repo to get latest cmake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
RUN echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
RUN curl -sSL https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add -
RUN apt update

# install packages
RUN apt update && apt install -y --no-install-recommends \
	cmake python3-pip libeigen3-dev libpcl-dev build-essential python3-dev libopenblas-dev \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /home/user

USER user
CMD /bin/bash
SHELL ["/bin/bash", "-c"]

# pip dependencies
RUN pip3 install ninja
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu113
RUN pip3 install -U MinkowskiEngine --install-option="--blas=openblas" -v --no-deps
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
RUN pip install graspnetAPI #git+https://github.com/graspnet/graspnetAPI.git
RUN pip3 install transformations

# setup workspace
RUN mkdir -p ~/graspnet_ws/src
RUN cd ~/graspnet_ws && git clone https://github.com/graspnet/graspnet-baseline.git

# patch graspnet-baseline/dataset/graspnet_dataset.py
RUN sed -i 's/from torch._six import container_abcs/import collections.abc as container_abcs/' \
	~/graspnet_ws/graspnet-baseline/dataset/graspnet_dataset.py

# install source dependencies
RUN cd ~/graspnet_ws/graspnet-baseline && pip3 install $(tail -n+2 requirements.txt)
# find arch number here: https://developer.nvidia.com/cuda-gpus
RUN cd ~/graspnet_ws/graspnet-baseline/pointnet2 && TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX" pip3 install .
RUN cd ~/graspnet_ws/graspnet-baseline/knn && pip3 install .

# Installing catkin package
COPY --chown=user . /home/user/graspnet_ws/src/graspnet_ros
RUN source /opt/ros/noetic/setup.bash && \
	cd ~/graspnet_ws && catkin_make

# update bashrc
RUN echo "source ~/graspnet_ws/devel/setup.bash" >> ~/.bashrc
RUN echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bashrc

CMD ["bash"]
