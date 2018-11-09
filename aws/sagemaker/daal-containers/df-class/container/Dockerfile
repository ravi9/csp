# Build an image that can do training and inference in SageMaker
# This is a Python 3 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM ubuntu:16.04

MAINTAINER Amazon AI <sage-learner@amazon.com>

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
		 tar \
		 bzip2 \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN cd ~ && wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b

RUN echo "export PATH=/root/miniconda3/bin:$PATH" > ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc"
ENV PATH=/root/miniconda3/bin:$PATH

RUN conda create -y -n daal4py -c intel -c intel/label/validation python=3.6 daal4py flask gevent gunicorn
RUN echo "source activate daal4py" > ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc"
ENV CONDA_PREFIX=/root/miniconda3/envs/daal4py
ENV PATH /root/miniconda3/envs/daal4py/bin:$PATH

# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.
# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY decisionForest /opt/program
WORKDIR /opt/program
