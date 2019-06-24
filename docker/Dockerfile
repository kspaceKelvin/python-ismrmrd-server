FROM python:3.7.1-alpine3.7
LABEL maintainer="Kelvin Chow (kelvin.chow@siemens.com)"


RUN     apk --no-cache add hdf5 hdf5-dev --repository http://dl-cdn.alpinelinux.org/alpine/edge/testing/ \
     && apk --no-cache add bash git openssh gcc libc-dev \
     && pip install --no-cache-dir pyxb h5py

RUN  mkdir -p /opt/code

RUN  cd /opt/code \
     && git clone https://github.com/ismrmrd/ismrmrd-python.git \
     && cd /opt/code/ismrmrd-python \
     && python3 setup.py install

RUN  cd /opt/code \
     && git clone https://github.com/ismrmrd/ismrmrd-python-tools.git \
     && cd /opt/code/ismrmrd-python-tools \
     && python3 setup.py install

RUN  cd /opt/code \
     && git clone https://github.com/kspaceKelvin/python-ismrmrd-server.git

RUN  apk del gcc git

CMD /usr/local/bin/python3 /opt/code/python-ismrmrd-server/main.py -v -H 0.0.0.0 -p 9002
