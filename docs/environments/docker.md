# Docker

Docker is useful for portable execution of the MRD server and client.

[Docker](https://www.docker.com/products/docker-desktop) provides isolated containers that package a reconstruction program and its libraries without requiring manual installation on the host. A complete working environment is available as the [`kspacekelvin/fire-python` Docker image](https://hub.docker.com/r/kspacekelvin/fire-python). A native Python environment is recommended for development.

## Pull image

```bash
docker pull kspacekelvin/fire-python
```

## Run server container

Windows:

```bash
docker run -p=9002:9002 --rm -it -v C:\tmp:/tmp kspacekelvin/fire-python
```

Linux/macOS:

```bash
docker run -p=9002:9002 --rm -it -v /tmp:/tmp kspacekelvin/fire-python
```

## Run client container

Windows:

```bash
docker run --rm -it --add-host=host.docker.internal:host-gateway -v C:\tmp:/tmp kspacekelvin/fire-python /bin/bash
```

Linux/macOS:

```bash
docker run --rm -it --add-host=host.docker.internal:host-gateway -v /tmp:/tmp kspacekelvin/fire-python /bin/bash
```

The server command options are:

```text
-p=9002:9002   Map host port 9002 to container port 9002.
-it             Run interactively; Ctrl+C stops the server.
--rm            Remove the container after it stops.
-v HOST:/tmp   Map a host folder to /tmp in the container for logs and debug files.
```

The `-p` option maps the container's port to a host port; change the first port number to use a different host port. The `-it` option enables interactive mode with a pseudo-terminal, which is needed for Ctrl+C to stop the server. The `--rm` option removes the container after it stops. The `-v` option maps a host directory to `/tmp` inside the container, where log and debug files are stored.

Inside the client container, run:

```bash
python /opt/code/python-ismrmrd-server/client.py -a host.docker.internal -p 9002 -o /tmp/phantom_img.h5 /tmp/phantom_raw.h5
```

The `host.docker.internal` name resolves to the Docker host. The client container can also generate phantom data:

```bash
python /opt/code/python-ismrmrd-server/generate_cartesian_shepp_logan_dataset.py -o /tmp/phantom_raw.h5
```

The client invocation uses:

```text
-a host.docker.internal   Address of the Docker host.
-p 9002                   Port mapped to the server container.
-o /tmp/phantom_img.h5    Output MRD file.
-G dataset                Optional output group name.
```

The client container requires `--add-host=host.docker.internal:host-gateway` with current Docker versions so the host name resolves correctly.
