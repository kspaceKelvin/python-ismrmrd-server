# Handling 3D Volumes

<!-- snippet: 3d-volumes, image-stack, volume-data -->

MRD image data is stored with the shape `[channels, z, y, x]`.  The channels dimension refers to receiver channels except when `image_type = 6 (MRD_IMTYPE_RGB)` is set, as described in [RGB color images](rgb-images.md).  In the case of three dimensional data, either a single 3D MRD image object can be used, or a list of 2D MRD images.

When using a single 3D MRD image, the `position` field in the [ImageHeader](https://ismrmrd.readthedocs.io/en/latest/mrd_image_data.html#imageheader) is defined as the center of the 3D volume.  The `field_of_view[2]` defines the total z extent of the volume.

If a 3D volume is split into separate 2D images, the `position` field must be updated to the center of each individual slice.  Determine each position from the slice geometry and `slice_dir`.  The `field_of_view[2]` defines the thickness of an individual slice.

When creating MRD images from NumPy arrays using `ismrmrd.Image.from_array()`, use the MRD image layout `[channels, z, y, x]` with `transpose=False`. Single-channel data may omit the channel dimension and use `[z, y, x]` or `[y, x]`.

Examples in this repo use the data shape [`[CHA, PE, RO]`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/7cafe0a52e8d43557574d184042c9216558edae4/invertcontrast.py#L187-L191) or [`[CHA, PE, RO, PAR]`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/7cafe0a52e8d43557574d184042c9216558edae4/invertcontrast.py#L206-L211).  This is consistent with the convention of having readouts as the horizontal (columns) dimension and phase encodes as the vertical (rows) dimension. After coil combination, the channel dimension is removed, so 2D `data` has shape `[PE, RO]` and 3D `data` has shape `[PE, RO, PAR]`. The following converts it to and MRD-compatible single-channel layout `[y, x]`, or `[z, y, x]`:

```python
if mrdHeader.encoding[0].reconSpace.matrixSize.z > 1:
    # For 3D data, send back the entire volume as one image
    tmpImg = ismrmrd.Image.from_array(data.transpose((2, 0, 1)), transpose=False)
else:
    # For 2D data, send back individual 2D images
    tmpImg = ismrmrd.Image.from_array(data, transpose=False)
```

## Reference

- **Source file:** `invertcontrast.py` — `process_raw()`
- **Permalink:** [invertcontrast.py#L293-L303](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/7cafe0a52e8d43557574d184042c9216558edae4/invertcontrast.py#L293-L303)
- **Search anchor:** `ismrmrd.Image.from_array`
