# Storing Arbitrary Data in MetaAttributes

<!-- snippet: metaattributes-arbitrary-data, tmpMeta.update, report-fields -->

It may be helpful to store results or debug data in the image [MetaAttributes](https://ismrmrd.readthedocs.io/en/latest/mrd_image_data.html#metaattributes).  These are stored per-image and allow for information to be programmatically extracted from a set of data using a script.  Data is stored in arbitrary key/value data in MRD MetaAttributes by updating the metadata object with a dictionary.  Note that all MetaAttributes are converted to text during serialization to XML.

```python
# Create a dictionary of values to report
data = {}
data['protocolName'] = mrdHeader.measurementInformation.protocolName
data['scanner']      = f'{mrdHeader.acquisitionSystemInformation.systemVendor} {mrdHeader.acquisitionSystemInformation.systemModel} {mrdHeader.acquisitionSystemInformation.systemFieldStrength_T:{".1f" if mrdHeader.acquisitionSystemInformation.systemFieldStrength_T > 1 else ".2f"}}T'
data['fieldOfView']  = f'{mrdHeader.encoding[0].encodedSpace.fieldOfView_mm.x:.1f} x {mrdHeader.encoding[0].encodedSpace.fieldOfView_mm.y:.1f} x {mrdHeader.encoding[0].encodedSpace.fieldOfView_mm.z:.1f} mm^3'
data['matrixSize']   = f'{mrdHeader.encoding[0].encodedSpace.matrixSize.x} x {mrdHeader.encoding[0].encodedSpace.matrixSize.y} x {mrdHeader.encoding[0].encodedSpace.matrixSize.z}'
```

```python
# Add all of the report data to the MetaAttributes so they can be parsed from the resulting images
tmpMeta.update(data)
```

**Notes:**
- `data` can contain arbitrary report fields (for example protocol, scanner, or matrix size).
- Keys and values are serialized into `attribute_string` when `tmpMeta.serialize()` is called.

## Reference

- **Source file:** `report.py` — `process_data()`
- **Permalink:** report.py [#L150-L155](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/3972fea/report.py#L150-L155) and [#L254-L255](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/3972fea/report.py#L254-L255)
- **Search anchor:** `Add all of the report data to the MetaAttributes so they can be parsed from the resulting images`
