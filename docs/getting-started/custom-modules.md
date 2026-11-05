# Custom Modules

The server is modular. The client config value (for example `-c invertcontrast`) selects the module name used by the server.

The config passed by the client is interpreted by the server as the module that should parse incoming data. Additional modules can be added by creating an appropriately named `.py` file in the Python path, such as the current folder. It is recommended that a default config be set in the Dockerfile `CMD` line when building an image, to indicate the intended config to be run.

## How module selection works

- If config is `invertcontrast`, the server looks for `invertcontrast.py`.
- If no matching module is found, the server can use a default config.
- The default can be provided to `main.py` with `--defaultConfig` (`-d`).

## Add a raw k-space filter

1. Copy [invertcontrast.py](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/invertcontrast.py) to `filterkspace.py`.
2. In `process_raw()`, add your filtering logic before FFT.
3. Run the client using:

```bash
python client.py -c filterkspace -o phantom_img.h5 phantom_raw.h5
```

## Add an image processing filter

1. Copy [invertcontrast.py](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/invertcontrast.py) to `filterimage.py`.
2. Edit `process_image()` to apply your image filter.
3. Run the client using:

```bash
python client.py -c filterimage -o phantom_img.h5 phantom_raw.h5
```

The NumPy library provides a [Hanning filter](https://numpy.org/doc/stable/reference/generated/numpy.hanning.html) for the raw k-space example.

In the `process_raw()` function, find the [section where raw k-space data is sorted into a Cartesian grid just before the Fourier transform](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/27454bd9f1a2c7fd3928bfa0767840b0d015d988/invertcontrast.py#L177). Replace the Fourier-transform section with the filter followed by the transform:

### Raw k-space filter example

In `process_raw()`, add a Hanning filter after the raw data has been sorted into a Cartesian grid and immediately before the Fourier transform:

```python
# Apply Hanning filter
logging.info("Applying Hanning filter to k-space data")
filt = np.sqrt(np.outer(np.hanning(data.shape[1]), np.hanning(data.shape[2])))
filt = np.expand_dims(filt, axis=(0,3))
data = np.multiply(data,filt)
np.save(debugFolder + "/" + "rawFilt.npy", data)

# Fourier Transform
data = fft.fftshift( data, axes=(1, 2))
data = fft.ifft2(    data, axes=(1, 2))
data = fft.ifftshift(data, axes=(1, 2))
```

### Image filter example

Pillow's [FIND_EDGES](https://pythontic.com/image-processing/pillow/edge-detection) filter can be applied after normalizing image data to 0-255:

Add `from PIL import Image, ImageFilter` to the new module. In `process_image()`, replace the normalization and inversion sections with the following filter workflow:

```python
# Normalize to range 0-255
data = data.astype(np.float64)
data *= 255/data.max()

# Apply a 2D high-pass filter for each image
logging.info("Applying high-pass image filter")
from PIL import Image, ImageFilter
for iImg in range(data.shape[-1]):
	im = Image.fromarray(np.squeeze(data[...,iImg])).convert('RGB')
	im = im.filter(ImageFilter.FIND_EDGES)
	data[:,:,0,0,iImg] = np.asarray(im)[...,0]

# Rescale back to 16-bit
data = data * maxVal/data.max()
data = data.astype(np.int16)
np.save(debugFolder + "/" + "imgFiltered.npy", data)
```

After creating a module, pass its name with the client's `-c` option. If the requested module cannot be found, the server falls back to the default configuration.

Start the server with [main.py](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/6684b4d17c0591e64b34bc06fdd06d78a2d8c659/main.py), then run the client in a separate window with the new config. Create a GIF preview with [mrd2gif.py](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/mrd2gif.py).

The server implementation looks for matching modules in [server.py](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/server.py), and the client config option is defined in [client.py](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/client.py). The example [simplefft.py](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/simplefft.py) is another starting point for reconstruction workflows, while [analyzeflow.py](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/analyzeflow.py) demonstrates image analysis.
