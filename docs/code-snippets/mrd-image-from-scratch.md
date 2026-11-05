# Creating an MRD Image from Scratch

<!-- snippet: mrd-image-from-scratch, report-image, minimal-image-header -->
When creating reports or other visualization graphics, it may be necessary to create the MRD [ImageHeader](https://ismrmrd.readthedocs.io/en/latest/mrd_image_data.html) from scratch without copying information from existing acquisition or image headers.  This code sets the minimum and recommended header fields.

```python
# Create new MRD instance for data with shape [y x]
mrdImg = ismrmrd.Image.from_array(imgGray, transpose=False)

# Set the minimum appropriate ImageHeader information without using a reference acquisition/image as a starting point
mrdImg.field_of_view = (mrdImg.getHead().matrix_size[0], mrdImg.getHead().matrix_size[1], mrdImg.getHead().matrix_size[2])

# Set image orientation dimensions.  Note that the default initialized values (0,0,0)
# are invalid because they are not unit vectors
mrdImg.read_dir  = (1, 0, 0)
mrdImg.phase_dir = (0, 1, 0)
mrdImg.slice_dir = (0, 0, 1)

# mrdImg.position is optional, but is relative to the patient_table_position
# Setting patient_table_position is recommended, otherwise the report may
# significantly shifted from other images in the series_
mrdImg.patient_table_position = group[0].patient_table_position

# Optional, but recommended.  Default value (0) corresponds to midnight
mrdImg.acquisition_time_stamp = group[0].acquisition_time_stamp

# Default value of image_type (0) is invalid
mrdImg.image_type = ismrmrd.IMTYPE_MAGNITUDE

# Use a different image_series_index to have a separate series than the main
# images.  Absolute value does not matter, but images with the same
# image_series_index are grouped together in the same DICOM SeriesNumber
mrdImg.image_series_index = 0

# DICOM InstanceNumber. Should be incremented if multiple images in a series
mrdImg.image_index = 0

# Set MRD MetaAttributes
tmpMeta = ismrmrd.Meta()
tmpMeta['DataRole']               = 'Image'
tmpMeta['ImageProcessingHistory'] = ['FIRE', 'PYTHON']
tmpMeta['Keep_image_geometry']    = 1

# Add image orientation directions to MetaAttributes
# Note that DICOM image orientation is in LPS coordinates, so if another set of image directional
# cosines are chosen, they may be flipped/rotated to bring them into LPS coordinate space
tmpMeta['ImageRowDir']    = ["{:.18f}".format(mrdImg.read_dir[0]),  "{:.18f}".format(mrdImg.read_dir[1]),  "{:.18f}".format(mrdImg.read_dir[2])]
tmpMeta['ImageColumnDir'] = ["{:.18f}".format(mrdImg.phase_dir[0]), "{:.18f}".format(mrdImg.phase_dir[1]), "{:.18f}".format(mrdImg.phase_dir[2])]

mrdImg.attribute_string = xml
imagesOut.append(mrdImg)
```

**Notes:**
- This pattern is useful when no source image/acquisition header exists to copy from.
- Always use `transpose=False` with `Image.from_array()`.
- `read_dir`, `phase_dir`, and `slice_dir` should be valid unit vectors.

## Reference

- **Source file:** `report.py` — `process_data()`
- **Permalink:** [report.py#L206-L260](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/3972fea/report.py#L206-L260)
- **Search anchor:** `Create new MRD instance for the report image`
