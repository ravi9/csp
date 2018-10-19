
#!/bin/sh

if [ "$EUID" -ne 0 ]
  then echo "Please run as root / sudo"
  exit
fi

# Required to build Singularity
sudo yum -y groupinstall "Development Tools"
sudo yum -y install libarchive-devel squashfs-tools git pssh

# Download and build Singularity from the GitHub master branch
git clone -b vault/release-2.6 https://github.com/singularityware/singularity.git
cd singularity
./autogen.sh
./configure
make dist
rpmbuild -ta singularity-*.tar.gz

# Install newly built Singularity RPM package
sudo yum -y install $HOME/rpmbuild/RPMS/x86_64/singularity-*.x86_64.rpm

# Install additional dependencies
sudo yum -y install epel-release debootstrap

# Install openmpi
sudo yum -y install openmpi openmpi-devel


grep -q '^StrictHostKeyChecking no' /etc/ssh/ssh_config || printf '\nStrictHostKeyChecking no\nUserKnownHostsFile /dev/null\n' > /etc/ssh/ssh_config
[ -d /mnt/resource ] && chmod 777 /mnt/resource

echo "$0 Completed"

