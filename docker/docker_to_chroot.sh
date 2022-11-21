#!/bin/bash
# This script takes a Docker image and creates a chroot image (.img)

# Syntax: ./docker_to_chroot.sh kspacekelvin/fire-python fire-python-chroot.img

DOCKER_NAME=${1}
CHROOT_FILE=${2}
EXPORT_FILE=docker-export.tar

# Create a Docker container and export to a .tar file
echo ------------------------------------------------------------
echo Exporting Docker image ${DOCKER_NAME}
echo ------------------------------------------------------------

if test -f "${EXPORT_FILE}"; then
    echo "Warning -- ${EXPORT_FILE} exists and will be overwritten!"
    rm ${EXPORT_FILE}
fi

docker create --name tmpimage ${DOCKER_NAME}
docker export -o ${EXPORT_FILE} tmpimage
docker rm tmpimage

# Run a privileged Docker to create the chroot file 
docker run -it --rm          \
           --privileged=true \
           -v $(pwd):/share  \
           ubuntu            \
           /bin/bash -c "sed -i -e 's/\r//g' /share/docker_tar_to_chroot.sh && /share/docker_tar_to_chroot.sh /share/${EXPORT_FILE} /share/${CHROOT_FILE}"

rm ${EXPORT_FILE}
