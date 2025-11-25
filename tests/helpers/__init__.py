import sys

import ismrmrd
import numpy as np
import pydicom
import pydicom.data

import dicom2mrd

MAX = 4095  # 12-bit data


def write_example(outdir):
    mrd_h5 = outdir / "checker.h5"

    # 1. create a 4x4x4 dicom image with siemens header (copied from pydicom.data)
    ds = pydicom.dcmread(pydicom.data.get_testdata_file("MR_small.dcm"))

    # 2. make a checkerboard pattern of min / max values in the 3D image
    checker_data = np.zeros((4, 4, 4), dtype=np.uint16)
    min_val = 0
    max_val = MAX
    ds.Rows = 4
    ds.Columns = 4
    ds.NumberOfFrames = 4
    ds.SeriesDescription = "Test Checkerboard"
    ds.MagneticFieldStrength = 1.5
    ds.AcquisitionTime = "120000.000000"
    ds.SeriesNumber = 1

    for z in range(ds.NumberOfFrames):
        for y in range(ds.Columns):
            for x in range(ds.Rows):
                if (x + y + z) % 2 == 0:
                    checker_data[z, y, x] = max_val
                else:
                    checker_data[z, y, x] = min_val

    # Update DICOM dataset with checkerboard data and add missing fields
    ds.PixelData = checker_data.tobytes()

    # Add missing required fields

    # Create temporary DICOM folder and file
    temp_dicom_dir = outdir / "temp_dicoms"
    temp_dicom_dir.mkdir()
    temp_dicom = temp_dicom_dir / "temp.dcm"
    ds.save_as(temp_dicom)

    # 3. use dicom2mrd to make a mrd h5 file
    args = dicom2mrd.argparse.Namespace(
        folder=temp_dicom_dir, outFile=mrd_h5, outGroup="dataset"
    )
    dicom2mrd.main(args)

    return mrd_h5


def mrd_data(filename, group="dataset"):
    """Read MRD image data from file"""
    dataset = ismrmrd.Dataset(filename, group, False)

    # Check what image groups are available
    dataset_list = dataset.list()
    image_groups = [name for name in dataset_list if name.startswith("image")]
    return np.squeeze(dataset.read_image(image_groups[0], 0).data)
