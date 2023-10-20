FROM rockylinux/rockylinux:8

RUN rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm \
             https://download.fmi.fi/smartmet-open/rhel/8/x86_64/smartmet-open-release-latest-8.noarch.rpm

RUN dnf -y install dnf-plugins-core && \
    dnf config-manager --set-enabled powertools && \
    dnf config-manager --setopt="epel.exclude=eccodes*" --save && \
    dnf -y --setopt=install_weak_deps=False install python3.11 python3.11-pip libSM libXrender libXext libglvnd-devel eccodes git && \
    dnf -y clean all && rm -rf /var/cache/dnf

RUN git clone https://github.com/fmidev/nowcasting_fcst.git

RUN update-alternatives --set python3 /usr/bin/python3.11 && \
    python3 -m pip --no-cache-dir install -r nowcasting_fcst/requirements.txt && \
    python3 -m pip --no-cache-dir install s3cmd
