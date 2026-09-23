# DICOM Input Data

For image processing workflows, DICOM can be used as input by converting to MRD. The [dicom2mrd.py](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/dicom2mrd.py) script converts DICOMs to MRD, while [mrd2dicom.py](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/mrd2dicom.py) performs the inverse.

## Convert DICOM to MRD

```bash
python dicom2mrd.py -o dicom_img.h5 dicoms
```

## Process with client/server

```bash
python client.py -c invertcontrast -o dicom_img_inverted.h5 dicom_img.h5
```

## Convert MRD back to DICOM

```bash
python mrd2dicom.py dicom_img_inverted.h5
```

The input folder may contain `.ima` or `.dcm` files, including files organized in subfolders. The output MRD file contains MRD-formatted images, and the final command writes the processed images to a DICOM folder.

Start the server first with [main.py](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/main.py), then run the client with [client.py](https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/client.py).
