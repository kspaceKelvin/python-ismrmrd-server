@echo off
rem This script takes a Docker image and creates a chroot image (.img)
rem Note that this script also requires docker_tar_to_chroot.sh to be located in the same folder
 
rem Syntax: docker_to_chroot.bat kspacekelvin/fire-python fire-python-chroot.img
 
if     "%1"=="" GOTO wrongargnum
if     "%2"=="" GOTO wrongargnum
if not "%3"=="" GOTO wrongargnum
 
set DOCKER_NAME=%1
set CHROOT_FILE=%2
set EXPORT_FILE=docker-export.tar
 
if exist %EXPORT_FILE% (
  echo Warning -- %EXPORT_FILE% exists and will be overwritten!
  del %EXPORT_FILE%
)
 
rem Create a Docker container and export to a .tar file
echo ------------------------------------------------------------
echo Exporting Docker image %DOCKER_NAME%
echo ------------------------------------------------------------
 
docker create --name tmpimage %DOCKER_NAME%
docker export -o %EXPORT_FILE% tmpimage
docker rm tmpimage
 
rem Run a privileged Docker to create the chroot file
docker run -it --rm          ^
           --privileged=true ^
           -v "%cd%":/share  ^
           ubuntu            ^
           /bin/bash -c "sed -i -e 's/\r//g' /share/docker_tar_to_chroot.sh && /share/docker_tar_to_chroot.sh /share/%EXPORT_FILE% /share/%CHROOT_FILE%"
 
del %EXPORT_FILE%
goto eof
 
:wrongargnum
echo Syntax: docker_to_chroot.bat docker_image_name chroot_file_name
 
:eof
