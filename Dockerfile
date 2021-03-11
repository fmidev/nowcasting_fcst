FROM centos:8

RUN rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm

RUN dnf -y install dnf-plugins-core && \
    dnf config-manager --set-enabled powertools && \
    dnf -y install git python38-pip python3-brlapi python38-numpy geos proj libSM libXrender eccodes s3cmd && \
    dnf -y clean all

COPY . /nowcasting_fcst

RUN cd nowcasting_fcst && pip3 install -r requirements.txt

# Change testdata ownership so that test run succeeds
RUN chown -R 1459:10000 /nowcasting_fcst/testdata

# OpenShift user mapping
USER 1459:10000
