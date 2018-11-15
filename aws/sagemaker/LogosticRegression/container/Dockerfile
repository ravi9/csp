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

RUN conda create -y -n daal4py -c intel python=3.6 daal4py pandas flask gevent gunicorn
RUN echo "source activate daal4py" > ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc"
ENV CONDA_PREFIX=/root/miniconda3/envs/daal4py
ENV PATH /root/miniconda3/envs/daal4py/bin:$PATH

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY LogisticRegression /opt/program
WORKDIR /opt/program
