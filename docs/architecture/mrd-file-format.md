# Quickstart

## Reconstruct a phantom raw data set

1. Set up a working environment using either [Conda](../environments/conda.md), [uv](../environments/uv.md), or [Docker](../environments/docker.md).
2. Generate a sample raw dataset:

```bash
python generate_cartesian_shepp_logan_dataset.py -o phantom_raw.h5
```

MRD data is stored in the HDF5 file format in a hierarchical structure in groups. The example above creates a `dataset` group containing:

```text
/dataset/data   Raw k-space data
/dataset/xml    MRD header
```

3. Start the server in verbose mode:

```bash
python main.py -v
```

4. Start the client and send data for reconstruction:

```bash
python client.py -G dataset -o phantom_img.h5 phantom_raw.h5
```

The `-G` argument specifies the group name in the output file, the `-o` argument specifies the output file, and the last argument is the input file.

5. Generate a GIF preview:

```bash
python mrd2gif.py phantom_img.h5
```

The [mrd2gif.py](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/mrd2gif.py) program converts an MRD Image file into an animated GIF for quick previewing.

A GIF file, animated when multiple images are present, is generated in the same folder as the MRD file. Its name uses the MRD base filename with the group and subgroups appended.

## Data layout notes

MRD data is stored in HDF5 groups. Common output paths include:

- `/dataset/data` for raw k-space data
- `/dataset/xml` for MRD header
- `/dataset/image_0/data` for image data
- `/dataset/image_0/header` for image headers
- `/dataset/image_0/attributes` for image meta attributes

The `-G` argument specifies the output group name. If it is omitted, a group name based on the current date and time is created. This is useful when multiple client runs need to be stored in one HDF5 file.

When using Conda, run the client in a new command prompt after activating the environment. With Docker, the client can run in the same container used to generate the phantom data.

Images are grouped by series index under `image_x`, where `x` is the `image_series_index` in the ImageHeader.

The reconstructed HDF5 file can be opened with an HDF viewer such as [HDFView](https://www.hdfgroup.org/downloads/hdfview/). The [ismrmrd-python-tools](https://github.com/ismrmrd/ismrmrd-python-tools) repository also provides an interactive `imageviewer` tool:

```bash
python imageviewer.py phantom_img.h5
```

In MATLAB, load and display the image with:

```matlab
img = h5read('/tmp/phantom_img.h5', '/dataset/image_0/data');
figure, imagesc(img), axis image, colormap(gray)
```

For Docker, generate the phantom in the same [Docker container as the client](../environments/docker.md) and run the server in a separate [Docker container](../environments/docker.md).
