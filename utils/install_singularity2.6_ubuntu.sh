#!/bin/bash

apt-get update
apt-get install -y python dh-autoreconf build-essential libarchive-dev squashfs-tools net-tools git 
git clone -b vault/release-2.6 https://github.com/sylabs/singularity.git singularity2.6
cd singularity2.6
./autogen.sh 
./configure 
make 
make install
