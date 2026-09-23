# RGB Color Images

<!-- snippet: rgb-image-conversion, colormap, matplotlib, jet, uint16 -->

Color (RGB) images may be desirable for plots, image overlays, or other visualization.  These are distinct from scalar images with a colormap, where each pixel has a single value and the visualized color is determined from the colormap and window/level.  In contrast, color images have an RGB color for each pixel and cannot be window/leveled.

RGB images in MRD indicated with an [`image_type = 6 (MRD_IMTYPE_RGB)`](https://ismrmrd.readthedocs.io/en/latest/mrd_image_data.html#image-types) and [`channels = 3`](https://ismrmrd.readthedocs.io/en/latest/mrd_image_data.html#imageheader).  It must have [`data_type = MRD_USHORT (uint16)`](https://ismrmrd.readthedocs.io/en/latest/mrd_image_data.html#data-types) with each color channel being in the range (0, 255).

```python
# MRD RGB images must be uint16 in range (0, 255)
rgb *= 255
data = rgb.astype(np.uint16)

# Set RGB parameters
mrdHeader.image_type = 6  # To be defined as ismrmrd.IMTYPE_RGB
mrdHeader.channels   = 3  # RGB "channels".  This is set by from_array, but need to be explicit as we're copying the old header instead

# RGB images have no windowing
del tmpMeta['WindowCenter']
del tmpMeta['WindowWidth']

# RGB images shouldn't undergo further processing, e.g. orientation or distortion correction
tmpMeta['InternalSend'] = 1
```

## Reference

- **Source file:** `invertcontrast.py` — `process_image()`
- **Permalink:** invertcontrast.py [L331-L334](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/3972fea/invertcontrast.py#L331-L334), [L358-L359](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/3972fea/invertcontrast.py#L358-L359), [L394-L399](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/3972fea/invertcontrast.py#L394-L399)
- **Search anchor:** `mrdhelper.get_json_config_param(config, 'options') == 'rgb'`
