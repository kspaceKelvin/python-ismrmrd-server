# Creating ROI Overlays

<!-- snippet: roi-overlay, region-of-interest, heart-shape, parametric-curve -->

Regions of interest polygons can be stored in image [MetaAttributes](https://ismrmrd.readthedocs.io/en/latest/mrd_image_data.html#metaattributes).  In addition to (x,y) points, ROIs properties include an RGB color, thickness, style, and visibility.  [invertcontrast.py](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/3972fea/invertcontrast.py#L439-L456) contains an example of generating an ROI and adding it to the MetaAttributes.

```python
def create_example_roi(img_size):
    t = np.linspace(0, 2*np.pi)
    x = 16*np.power(np.sin(t), 3)
    y = -13*np.cos(t) + 5*np.cos(2*t) + 2*np.cos(3*t) + np.cos(4*t)

    # Place ROI in bottom right of image, offset and scaled to 10% of the image size
    x = (x-np.min(x)) / (np.max(x) - np.min(x))
    y = (y-np.min(y)) / (np.max(y) - np.min(y))
    x = (x * 0.10*np.min(img_size[:2])) + (img_size[1]-0.2*np.min(img_size[:2]))
    y = (y * 0.10*np.min(img_size[:2])) + (img_size[0]-0.2*np.min(img_size[:2]))

    rgb = (1,0,0)  # Red, green, blue color -- normalized to 1
    thickness  = 1  # Line thickness
    style      = 0  # Line style (0 = solid, 1 = dashed)
    visibility = 1  # Line visibility (0 = false, 1 = true)

    roi = mrdhelper.create_roi(x, y, rgb, thickness, style, visibility)
    return roi
```

```python
if mrdhelper.get_json_config_param(config, 'options') == 'roi':
    # Example for sending ROIs
    logging.info("Creating ROI_example")
    tmpMeta['ROI_example'] = create_example_roi(data.shape)
```

**Notes:**
- The ROI is stored in metadata under a key starting with `ROI_` (e.g. `tmpMeta['ROI_example'] = roi`).
- Coordinates are in pixels with `(0, 0)` at the top-left corner.
- See [`mrdhelper.create_roi()`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/3972fea/mrdhelper.py#L185-L213) for the serialization format and [`mrdhelper.parse_roi()`](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/3972fea/mrdhelper.py#L215-L241) for deserialization.

## Reference

- **Source file:** `invertcontrast.py` — `create_example_roi()`
- **Permalink:** invertcontrast.py [#L439-L456](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/3972fea/invertcontrast.py#L439-L456) and [#L380-L383](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/3972fea/invertcontrast.py#L380-L383)
- **Search anchor:** `create_example_roi`
