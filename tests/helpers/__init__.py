import sys

import ismrmrd
import numpy as np
import pydicom
import pydicom.data

import dicom2mrd

MAX = 4095  # 12-bit data


def checkers(nz, ny, nx):
    checker_data = np.zeros((nz,ny, nx), dtype=np.uint16)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if (x + y + z) % 2 == 0:
                    checker_data[z, y, x] = MAX
                else:
                    checker_data[z, y, x] = 0
    return checker_data


def write_example(data, outdir, series_number=1):
    """
    Write example data files in mdr.h5 and dicom/ formats
    :param outdir: where to save, likely `tmp_path` from pytest
    :param series_number: dicom header information
    :returns: dict with 'mrd' and 'dcmdir' keys
    """
    mrd_h5 = outdir / "checker.h5"

    ds = pydicom.dcmread(pydicom.data.get_testdata_file("MR_small.dcm"))

    #ds.PixelData = checkers(ds.NumberOfFrames,ds.Columns,ds.Rows).tobytes()
    ds.PixelData = data.tobytes()
    (ds.NumberOfFrames, ds.Columns, ds.Rows) = data.shape
    ds.SeriesDescription = "Test Checkerboard"
    ds.MagneticFieldStrength = 1.5
    ds.AcquisitionTime = "120000.000000"
    ds.SeriesNumber = series_number

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

    return {'mrd': mrd_h5, 'dcmdir': temp_dicom_dir}


def mrd_data(filename, group="dataset"):
    """Read MRD image data from file"""
    dataset = ismrmrd.Dataset(filename, group, False)

    # Check what image groups are available
    dataset_list = dataset.list()
    image_groups = [name for name in dataset_list if name.startswith("image")]
    return np.squeeze(dataset.read_image(image_groups[0], 0).data)
