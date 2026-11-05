# Spectroscopy-Specific Settings

<!-- snippet: spectroscopy-settings, spectro-metadata, real-dwell-time -->

Spectroscopy data may require additional metadata to correctly integrate with the Siemens ICE pipeline.

- Spectroscopy data that is intended to be stored as [DICOM Spectroscopy](https://dicom.nema.org/dicom/2013/output/chtml/part03/sect_A.36.html#sect_A.36.3) formatted data must have the Siemens ICE MiniHeader field `SpectroData` set to `true`. This can be set in the MetaAttributes, as in [spectroscopy.py#L243](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/3972fea/spectroscopy.py#L243):
    ```python
    tmpMeta['SiemensControl_SpectroData'] = ['bool', 'true']
    ```

- When sending back DICOM spectroscopy data, ensure that the [MRD ImageHeader `image_type`](https://ismrmrd.readthedocs.io/en/latest/mrd_image_data.html#image-types) is set to `MRD_IMTYPE_COMPLEX` (5) in order to correct downstream header fields:
    ```python
    tmpImg.image_type = ismrmrd.IMTYPE_COMPLEX
    ```

- In Siemens convention, unprocessed FIDs are stored as [raw AcquisitionData](https://ismrmrd.readthedocs.io/en/latest/mrd_raw_data.html) while processed (coil combined, averaged, etc.) FIDs are stored as [ImageData](https://ismrmrd.readthedocs.io/en/latest/mrd_image_data.html).  In the ImageData format, a single readout would use the `x` dimension for the readout time dimension while `y` and `z` dimensions are 1.  2D spectroscopy imaging data is supported where `x` is time, `y` is the lines (LIN) dimension and `z` is the segments (SEG) dimension.

- The spectral width for the acquisition is often necessary for analysis.  Depending on how the data is converted to MRD format, it may be in one of two formats:
  - In the MRD header as a `userParameters.userDouble` with name `SpectralWidth`. [dicom2mrd.py#L187-L193](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/3972fea/dicom2mrd.py#L187-L193) will populate this field from the DICOM using the [(0018,9052) SpectralWidth](https://dicom.innolitics.com/ciods/mr-spectroscopy/mr-spectroscopy/00189052) DICOM field.  This is the spectral width of the complete FID readout in Hz.
  - In the MRD header as a `userParameters.userDouble` with name `DwellTime_0`. This is the dwell time for a single oversampled readout point in microseconds.

    The spectral width is used in the [LCModel-MRD-App](https://github.com/kspaceKelvin/LCModel-MRD-App) in [lcmodel.py#L252-L255](https://github.com/kspaceKelvin/LCModel-MRD-App/blob/0d66d2e/lcmodel.py#L252-L255):
    ```python
    try:
        deltaT = 1/mrdhelper.get_userParameterDouble_value(mrdHeader, 'SpectralWidth')
    except:
        deltaT = mrdhelper.get_userParameterDouble_value(mrdHeader, 'DwellTime_0') * 2 * 1e-6
    ```

- If the module removes 2x readout oversampling, the dwell time must be updated for correct header generation by the Siemens ICE pipeline.  This is done by populating the MiniHeader field `RealDwellTime`, which is units of nanoseconds, as in [spectroscopy.py#L253](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/3972fea/spectroscopy.py#L253):
    ```python
    tmpMeta['SiemensDicom_RealDwellTime'] = ['int', str(int(dwellTime*1000*2))]
    ```
