# Custom Docker Image

You can integrate a custom module with minimal changes by extending the existing Docker image.

## Example flow

1. Copy an example module such as [invertcontrast.py](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/invertcontrast.py) and implement your logic.
2. Add an optional JSON config file.
3. Create a lightweight Dockerfile based on `kspacekelvin/fire-python`.

See the [`custom/` folder](https://github.com/kspaceKelvin/python-ismrmrd-server/tree/master/custom) for a working example:

- [`custom/filter.py`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/custom/filter.py)
- [`custom/filter.json`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/custom/filter.json)
- [`custom/custom.dockerfile`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/custom/custom.dockerfile)

Build from the `custom/` folder:

```bash
docker build --no-cache -t fire-python-custom -f custom.dockerfile ./
```

The base images `kspacekelvin/fire-python` and `kspacekelvin/fire-python-devcon` can be used as starting points. The custom Dockerfile generally only needs to copy the module and optional JSON file and install additional dependencies.

The example [filter.py](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/custom/filter.py) applies the [SciPy median filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html). Compare it with [invertcontrast.py from the reference commit](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/3f52bf1504b1fed56b28d29b9fff560c5138e9f3/invertcontrast.py). The accompanying [filter.json](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/custom/filter.json) configures the median-filter window size.
