# ----- First stage to build BART -----
FROM python:3.12.0-slim AS bart_build
ARG  DEBIAN_FRONTEND=noninteractive
ENV  TZ=America/Chicago

RUN  apt-get update && apt-get install -y git cmake g++ libfftw3-dev liblapacke-dev libpng-dev gfortran
RUN  mkdir -p /opt/code

# BART (static linked)
RUN cd /opt/code                                   && \
    git clone https://github.com/mrirecon/bart.git --branch v0.9.00 && \
    cd bart                                        && \
    make -j $(nproc)                               && \
    make install

# ----- Main stage without build dependencies -----
# Re-use already built Docker image, but the contents of \docker\Dockerfile can also be
# recapitulated here instead to ensure the latest build
FROM kspacekelvin/fire-python

# this does *not* work in the chroot image!
ENV PYTHONPATH=/opt/code/bart/python

# Copy BART from previous stage
COPY --from=bart_build /usr/local/bin/bart             /usr/local/bin/
COPY --from=bart_build /usr/local/lib/bart/commands    /usr/local/lib/bart/commands
COPY --from=bart_build /usr/local/share/doc/bart       /usr/local/share/doc/bart
COPY --from=bart_build /opt/code/bart                  /opt/code/bart

# Install BART runtime dependencies
RUN  apt-get update && apt-get install -y libfftw3-single3 libgomp1 liblapacke libpng16-16 libblas3 libunwind8
