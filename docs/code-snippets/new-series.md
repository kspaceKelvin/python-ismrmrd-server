# Creating New Series

<!-- snippet: series-number, image_series_index, image_index, series-description -->

DICOM images within a study are grouped in series, which contain a set of related images.  When doing analysis, the results (derived images, reports, etc.) should generally be grouped into a new DICOM series.  The MRD ImageHeader field [`image_series_index`](https://ismrmrd.readthedocs.io/en/latest/mrd_image_data.html#imageheader) corresponds to the DICOM [SeriesNumber](https://dicom.innolitics.com/ciods/mr-image/general-series/00200011).  All MRD images with the same `image_series_index` are grouped together in the same DICOM `SeriesNumber`.  When used with FIRE/OpenRecon, absolute `image_series_index` are not used -- the next available `SeriesNumber` is used.

The MRD ImageHeader field [`image_index`](https://ismrmrd.readthedocs.io/en/latest/mrd_image_data.html#imageheader) corresponds to the DICOM [InstanceNumber](https://dicom.innolitics.com/ciods/enhanced-mr-image/enhanced-mr-image-multi-frame-functional-groups/00200013) and is used to order images within the same `SeriesNumber`.

The MRD MetaAttribute [`SeriesDescription` and `SeriesDescriptionAdditional`](https://ismrmrd.readthedocs.io/en/latest/mrd_image_data.html#metaattributes) are used to modify the DICOM [`SeriesDescription`](https://dicom.innolitics.com/ciods/mr-image/general-series/0008103e).  MRD `SeriesDescription`, if present, replaces the entirety of the DICOM `SeriesDescription`.  `SeriesDescriptionAdditional`, if present, appends text to the existing DICOM `SeriesDescription`.  It can either be a string or an array of strings that will be concatenated with '_' when converting to DICOM.

These fields are all supported by the [mrd2dicom.py](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/mrd2dicom.py) conversion tool.

```python
tmpMeta['SequenceDescriptionAdditional']  = 'FIRE'
tmpImg.image_series_index = 99
```

## Reference

- **Source file:** `invertcontrast.py` — `process_image()`
- **Permalink:** invertcontrast.py [L384](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/3972fea/invertcontrast.py#L384), [L434](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/3972fea/invertcontrast.py#L434)
- **Search anchor:** `image_series_index` and `tmpMeta['SequenceDescriptionAdditional']`
