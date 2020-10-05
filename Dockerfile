FROM centos:7

RUN rpm -ivh https://download.fmi.fi/smartmet-open/rhel/7/x86_64/smartmet-open-release-17.9.28-1.el7.fmi.noarch.rpm \
	https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm

RUN yum -y install \
	eccodes-python git python-devel python2-pip gcc libSM libXrender libXext-devel libX11-devel hdf5-devel netcdf-devel

COPY . /nowcasting_fcst

# Centos 7 has too old numpy (it is installed as a requirement from eccodes-python), we'll
# install a newer with pip
RUN rpm -e --nodeps numpy

RUN cd nowcasting_fcst && \
	pip install --upgrade pip && \
	pip install --upgrade setuptools && \
	pip install -r requirements.txt

# Change testdata ownership so that test run succeeds
RUN chown -R 1459:10000 /nowcasting_fcst/testdata

# OpenShift user mapping
USER 1459:10000
