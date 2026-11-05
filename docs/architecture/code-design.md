# Code Design

This code is structured to enable integration of custom reconstruction and analysis modules into the MRD streaming framework without the need to modify the streaming framework code itself.  There are a set of core files that implement the client/server of the MRD streaming framework itself and a set of example analysis modules.  Custom modules are added as additional .py files which are selected at runtime by the client via the config string.  There are also a set of utility functions that perform common tasks.

## Data flow

In the [MRD streaming framework](https://ismrmrd.readthedocs.io/en/latest/mrd_streaming_protocol.html), the client is defined as having the source data and the server analyzes the data and sends it back to the client.  A [`client.py`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/client.py) is included in this repo and reads data from an MRD HDF5 file, but this role may be played by an MRI scanner in deployment.

The [`server`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/server.py) receives data from the client over a TCP network socket.  The server selects a Python module from the client `config` value. For example, `invertcontrast` resolves to `invertcontrast.py`.  The `--defaultConfig` (`-d`) value in [`main.py`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/7cafe0a52e8d43557574d184042c9216558edae4/main.py#L45) is used to specify a default config file if real-time selection fails.  When packaged in a Docker application, it can also specify the intended default configuration of the application.  The [`--savedata`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/7cafe0a52e8d43557574d184042c9216558edae4/main.py#L48) option can be used to have the server save a copy of the data it receives from the server.  The resulting MRD file can be used for subsequent calls by `client.py`.  This workflow may be useful when saving data streamed from an external source, e.g. directly from an MRI scanner.

## Module design

Several example module files are provided, with [`invertcontrast.py`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/invertcontrast.py) being the most general example.  Each module must have the entry point function [`process(connection, config, mrdHeader)`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/7cafe0a52e8d43557574d184042c9216558edae4/invertcontrast.py#L21), which receives a continuous stream of data from the client.

Inside the `process()` function for modules is a `for` loop that iterates over each data received from the client.
```python
for item in connection:
    # ----------------------------------------------------------
    # Raw k-space data messages
    # ----------------------------------------------------------
    if isinstance(item, ismrmrd.Acquisition):
        # Accumulate all imaging readouts in a group
        if (not item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT) and
            not item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION) and
            not item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA) and
            not item.is_flag_set(ismrmrd.ACQ_IS_NAVIGATION_DATA)):
            acqGroup.append(item)

        # When this criteria is met, run process_raw() on the accumulated
        # data, which returns images that are sent back to the client.
        if item.is_flag_set(ismrmrd.ACQ_LAST_IN_MEASUREMENT):
            logging.info("Processing a group of k-space data")
            image = process_raw(acqGroup, connection, config, mrdHeader)
            connection.send_image(image)
            acqGroup = []
```

This basic design separates out each data item (readout, image, waveform), collects the data into a list (`acqGroup`), and determines whether or not to trigger an analysis on the accumulated data via the `process_raw()` function.  The filtering (e.g. excluding noise data, phase correction data, etc.) is application specific and should be adjusted as needed.  The triggering condition (`ACQ_LAST_IN_MEASUREMENT` in this example), can be used to perform "real-time" analysis of the data before all data is collected.  For example, if the criteria were changed to [`ACQ_LAST_IN_SLICE`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/7cafe0a52e8d43557574d184042c9216558edae4/bartfire.py#L54), `process_raw()` would be triggered to perform a reconstruction as soon as acquisition for each slice has completed.


## Core files

- [`main.py`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/main.py): Main entrypoint for starting the MRD server via the `Server` class
- [`server.py`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/server.py): Handles communication with the client via the [MRD streaming session protocol](https://ismrmrd.readthedocs.io/en/latest/mrd_streaming_protocol.html), select an analysis module based on the requested config, and executes processing
- [`connection.py`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/connection.py): Handles parsing of individual [MRD message types](https://ismrmrd.readthedocs.io/en/latest/mrd_messages.html) within an MRD streaming session
- [`constants.py`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/constants.py): Definition of MRD message type constants
- [`mrdhelper.py`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/mrdhelper.py): Helper utilities for MRD metadata and image handling
- [`client.py`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/client.py): Main entry point for starting the MRD client, which sends MRD data and receives results

## Example modules

- [`invertcontrast.py`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/invertcontrast.py): Primary example module.  Accepts raw k-space acquisitions (2D/3D) or image data, inverts image contrast (with support for JSON configuration parameters like colormaps/RGB output, pixel shifting, and ROI overlays), and returns the processed images
- [`simplefft.py`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/simplefft.py): Simplified alternate code structure for modules. Performs basic Cartesian 2D FFT image reconstruction from raw k-space acquisitions, including sum-of-squares coil combination, intensity normalization, and oversampling removal
- [`analyzeflow.py`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/analyzeflow.py): Image analysis module. Processes phase-contrast velocity images, performs background noise masking based on mean temporal differences across cardiac phases, and outputs images grouped by flow encoding direction
- [`report.py`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/report.py): Generates a synthetic summary report image with acquisition parameters and measurement information rendered via matplotlib, embedding values into MetaAttributes for downstream parsing
- [`bartfire.py`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/bartfire.py): Reconstructs raw k-space data by interfacing with the [BART (Berkeley Advanced Reconstruction Toolbox)](https://mrirecon.codeberg.page/) library (e.g., ESPIRiT coil calibration and parallel imaging reconstruction).  Note that this requires that [BART library](https://codeberg.org/mrirecon/bart) be compiled, such as with [`docker\bart\Dockerfile`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/docker/bart/Dockerfile)
- [`spectroscopy.py`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/spectroscopy.py): Processes single-voxel MR spectroscopy (MRS) or MR spectroscopic imaging (MRSI) data, including coil selection, oversampling removal, spectral/spatial reconstruction, and result plotting.
- [`filter.py`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/filter.py): An example custom module derived from invertcontrast.py that demonstrates applying a SciPy median filter to reconstructed image data.

## Utility files

- [`generate_cartesian_shepp_logan_dataset.py`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/generate_cartesian_shepp_logan_dataset.py): Creates Cartesian Shepp-Logan phantom raw data
- [`dicom2mrd.py`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/dicom2mrd.py): Converts DICOM folders to MRD image files
- [`mrd2dicom.py`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/mrd2dicom.py): Converts MRD image files to DICOM folders
- [`mrd2gif.py`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/mrd2gif.py): Creates animated GIF previews from MRD images
