FROM registry.access.redhat.com/ubi8/ubi

RUN rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm \
             https://download.fmi.fi/smartmet-open/rhel/8/x86_64/smartmet-open-release-latest-8.noarch.rpm

RUN dnf -y install dnf-plugins-core && \
    dnf config-manager --set-enabled codeready-builder-for-rhel-8-x86_64-rpms && \
    dnf config-manager --setopt="epel.exclude=eccodes*" --save && \
    dnf -y --setopt=install_weak_deps=False install python3.11 python3.11-pip libSM libXrender libXext libglvnd-devel eccodes git && \
    dnf -y clean all && rm -rf /var/cache/dnf

RUN git clone https://github.com/fmidev/nowcasting_fcst.git

RUN update-alternatives --set python3 /usr/bin/python3.11 && \
    python3 -m pip --no-cache-dir install -r nowcasting_fcst/requirements.txt && \
    python3 -m pip --no-cache-dir install s3cmd

# Aggressive thinning of container image

RUN rpm -qa|egrep "dbus|systemd|pam|python3-|dnf|subscription|crack" | xargs rpm -e --nodeps 2>/dev/null
