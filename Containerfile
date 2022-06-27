FROM registry.access.redhat.com/ubi8/ubi

RUN rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm \
             https://download.fmi.fi/smartmet-open/rhel/8/x86_64/smartmet-open-release-21.3.26-2.el8.fmi.noarch.rpm

RUN dnf -y install dnf-plugins-core && \
    dnf -y module enable python38 && \
    dnf config-manager --set-enabled codeready-builder-for-rhel-8-x86_64-rpms && \
    dnf config-manager --setopt="epel.exclude=eccodes*" --save && \
    dnf -y --setopt=install_weak_deps=False install python38-pip libSM libXrender libXext eccodes && \
    dnf -y clean all && rm -rf /var/cache/dnf

COPY . /nowcasting_fcst

RUN update-alternatives --set python3 /usr/bin/python3.8 && \
    python3 -m pip --no-cache-dir install -r nowcasting_fcst/requirements.txt && \
    python3 -m pip --no-cache-dir install s3cmd

# Aggressive thinning of container image

RUN rm -rf /nowcasting_fcst/testdata /nowcasting_fcst/.git && \
    rpm -qa|egrep "dbus|systemd|pam|python3-|dnf|subscription|crack" | xargs rpm -e --nodeps 2>/dev/null
