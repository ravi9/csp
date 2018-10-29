# Build an image that can do training and inference in SageMaker
# This is a Python 2 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM ubuntu:16.04

MAINTAINER Amazon AI <sage-learner@amazon.com>

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         bzip2 \
         python \
         nginx \
         tar \
         cpio \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN cd ~ && wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b

RUN echo "export PATH=/root/miniconda3/bin:$PATH" > ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc"
ENV PATH=/root/miniconda3/bin:$PATH

RUN conda create -y -n daal_py27 python=2.7

RUN conda install -y -n daal_py27 -c intel daal pydaal pandas
RUN conda install -y -n daal_py27 flask gevent gunicorn

RUN echo "source activate daal_py27" > ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc"
ENV PATH /root/miniconda3/envs/daal_py27/bin:$PATH

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
COPY LogisticRegression /opt/program
WORKDIR /opt/program

