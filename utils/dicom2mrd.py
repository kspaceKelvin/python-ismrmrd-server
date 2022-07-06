import pydicom
import argparse
import ismrmrd
import numpy as np
import os
import ctypes
import re
import base64

# Defaults for input arguments
defaults = {
    'outGroup':       'dataset',
}

# Lookup table between DICOM and MRD image types
imtype_map = {'M': ismrmrd.IMTYPE_MAGNITUDE,
              'P': ismrmrd.IMTYPE_PHASE,
              'R': ismrmrd.IMTYPE_REAL,
              'I': ismrmrd.IMTYPE_IMAG}

# Lookup table between DICOM and Siemens flow directions
venc_dir_map = {'rl'  : 'FLOW_DIR_R_TO_L',
                'lr'  : 'FLOW_DIR_L_TO_R',
                'ap'  : 'FLOW_DIR_A_TO_P',
                'pa'  : 'FLOW_DIR_P_TO_A',
                'fh'  : 'FLOW_DIR_F_TO_H',
                'hf'  : 'FLOW_DIR_H_TO_F',
                'in'  : 'FLOW_DIR_TP_IN',
                'out' : 'FLOW_DIR_TP_OUT'}

def CreateMrdHeader(dset):
    """Create MRD XML header from a DICOM file"""

    mrdHead = ismrmrd.xsd.ismrmrdHeader()

    mrdHead.measurementInformation                             = ismrmrd.xsd.measurementInformationType()
    mrdHead.measurementInformation.measurementID               = dset.SeriesInstanceUID
    mrdHead.measurementInformation.patientPosition             = dset.PatientPosition
    mrdHead.measurementInformation.protocolName                = dset.SeriesDescription
    mrdHead.measurementInformation.frameOfReferenceUID         = dset.FrameOfReferenceUID

    mrdHead.acquisitionSystemInformation                       = ismrmrd.xsd.acquisitionSystemInformationType()
    mrdHead.acquisitionSystemInformation.systemVendor          = dset.Manufacturer
    mrdHead.acquisitionSystemInformation.systemModel           = dset.ManufacturerModelName
    mrdHead.acquisitionSystemInformation.systemFieldStrength_T = float(dset.MagneticFieldStrength)
    mrdHead.acquisitionSystemInformation.institutionName       = dset.InstitutionName
    try:
        mrdHead.acquisitionSystemInformation.stationName       = dset.StationName
    except:
        pass

    mrdHead.experimentalConditions                             = ismrmrd.xsd.experimentalConditionsType()
    mrdHead.experimentalConditions.H1resonanceFrequency_Hz     = int(dset.MagneticFieldStrength*4258e4)

    enc = ismrmrd.xsd.encodingType()
    enc.trajectory                                              = ismrmrd.xsd.trajectoryType('cartesian')
    encSpace                                                    = ismrmrd.xsd.encodingSpaceType()
    encSpace.matrixSize                                         = ismrmrd.xsd.matrixSizeType()
    encSpace.matrixSize.x                                       = dset.Columns
    encSpace.matrixSize.y                                       = dset.Rows
    encSpace.matrixSize.z                                       = 1
    encSpace.fieldOfView_mm                                     = ismrmrd.xsd.fieldOfViewMm()
    if dset.SOPClassUID.name == 'Enhanced MR Image Storage':
        encSpace.fieldOfView_mm.x                               =       dset.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing[0]*dset.Rows
        encSpace.fieldOfView_mm.y                               =       dset.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing[1]*dset.Columns
        encSpace.fieldOfView_mm.z                               = float(dset.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SliceThickness)
    else:
        encSpace.fieldOfView_mm.x                               =       dset.PixelSpacing[0]*dset.Rows
        encSpace.fieldOfView_mm.y                               =       dset.PixelSpacing[1]*dset.Columns
        encSpace.fieldOfView_mm.z                               = float(dset.SliceThickness)
    enc.encodedSpace                                            = encSpace
    enc.reconSpace                                              = encSpace
    enc.encodingLimits                                          = ismrmrd.xsd.encodingLimitsType()
    enc.parallelImaging                                         = ismrmrd.xsd.parallelImagingType()

    enc.parallelImaging.accelerationFactor                      = ismrmrd.xsd.accelerationFactorType()
    if dset.SOPClassUID.name == 'Enhanced MR Image Storage':
        enc.parallelImaging.accelerationFactor.kspace_encoding_step_1 = dset.SharedFunctionalGroupsSequence[0].MRModifierSequence[0].ParallelReductionFactorInPlane
        enc.parallelImaging.accelerationFactor.kspace_encoding_step_2 = dset.SharedFunctionalGroupsSequence[0].MRModifierSequence[0].ParallelReductionFactorOutOfPlane
    else:
        enc.parallelImaging.accelerationFactor.kspace_encoding_step_1 = 1
        enc.parallelImaging.accelerationFactor.kspace_encoding_step_2 = 1

    mrdHead.encoding.append(enc)

    mrdHead.sequenceParameters                                  = ismrmrd.xsd.sequenceParametersType()

    return mrdHead

def GetDicomFiles(directory):
    """Get path to all DICOMs in a directory and its sub-directories"""
    for entry in os.scandir(directory):
        if entry.is_file() and (entry.path.lower().endswith(".dcm") or entry.path.lower().endswith(".ima")):
            yield entry.path
        elif entry.is_dir():
            yield from GetDicomFiles(entry.path)


def main(args):
    dsetsAll = []
    for entryPath in GetDicomFiles(args.folder):
        dsetsAll.append(pydicom.dcmread(entryPath))

    # Group by series number
    uSeriesNum = np.unique([dset.SeriesNumber for dset in dsetsAll])
    print("Found %d unique series from %d files in folder %s" % (len(uSeriesNum), len(dsetsAll), args.folder))

    print("Creating MRD XML header from file %s" % dsetsAll[0].filename)
    mrdHead = CreateMrdHeader(dsetsAll[0])
    print(mrdHead.toXML())

    imgAll = [None]*len(uSeriesNum)

    for iSer in range(len(uSeriesNum)):
        dsets = [dset for dset in dsetsAll if dset.SeriesNumber == uSeriesNum[iSer]]

        imgAll[iSer] = [None]*len(dsets)

        # Sort images by instance number, as they may be read out of order
        def get_instance_number(item):
            return item.InstanceNumber
        dsets = sorted(dsets, key=get_instance_number)

        # Build a list of unique SliceLocation and TriggerTimes, as the MRD
        # slice and phase counters index into these
        uSliceLoc = np.unique([dset.SliceLocation for dset in dsets])
        if dsets[0].SliceLocation != uSliceLoc[0]:
            uSliceLoc = uSliceLoc[::-1]

        try:
            # This field may not exist for non-gated sequences
            uTrigTime = np.unique([dset.TriggerTime for dset in dsets])
            if dsets[0].TriggerTime != uTrigTime[0]:
                uTrigTime = uTrigTime[::-1]
        except:
            uTrigTime = np.zeros_like(uSliceLoc)

        print("Series %d has %d images with %d slices and %d phases" % (uSeriesNum[iSer], len(dsets), len(uSliceLoc), len(uTrigTime)))

        for iImg in range(len(dsets)):
            tmpDset = dsets[iImg]

            # Create new MRD image instance.
            # pixel_array data has shape [row col], i.e. [y x].
            # from_array() should be called with 'transpose=False' to avoid warnings, and when called
            # with this option, can take input as: [cha z y x], [z y x], or [y x]
            tmpMrdImg = ismrmrd.Image.from_array(tmpDset.pixel_array, transpose=False)
            tmpMeta   = ismrmrd.Meta()

            try:
                tmpMrdImg.image_type                = imtype_map[tmpDset.ImageType[2]]
            except:
                print("Unsupported ImageType %s -- defaulting to IMTYPE_MAGNITUDE" % tmpDset.ImageType[2])
                tmpMrdImg.image_type                = ismrmrd.IMTYPE_MAGNITUDE

            tmpMrdImg.field_of_view            = (tmpDset.PixelSpacing[0]*tmpDset.Rows, tmpDset.PixelSpacing[1]*tmpDset.Columns, tmpDset.SliceThickness)
            tmpMrdImg.position                 = tuple(np.stack(tmpDset.ImagePositionPatient))
            tmpMrdImg.read_dir                 = tuple(np.stack(tmpDset.ImageOrientationPatient[0:3]))
            tmpMrdImg.phase_dir                = tuple(np.stack(tmpDset.ImageOrientationPatient[3:7]))
            tmpMrdImg.slice_dir                = tuple(np.cross(np.stack(tmpDset.ImageOrientationPatient[0:3]), np.stack(tmpDset.ImageOrientationPatient[3:7])))
            tmpMrdImg.acquisition_time_stamp   = round((int(tmpDset.AcquisitionTime[0:2])*3600 + int(tmpDset.AcquisitionTime[2:4])*60 + int(tmpDset.AcquisitionTime[4:6]) + float(tmpDset.AcquisitionTime[6:]))*1000/2.5)
            try:
                tmpMrdImg.physiology_time_stamp[0] = round(int(tmpDset.TriggerTime/2.5))
            except:
                pass

            try:
                ImaAbsTablePosition = tmpDset.get_private_item(0x0019, 0x13, 'SIEMENS MR HEADER').value
                tmpMrdImg.patient_table_position = (ctypes.c_float(ImaAbsTablePosition[0]), ctypes.c_float(ImaAbsTablePosition[1]), ctypes.c_float(ImaAbsTablePosition[2]))
            except:
                pass

            tmpMrdImg.image_series_index     = uSeriesNum.tolist().index(tmpDset.SeriesNumber)
            tmpMrdImg.image_index            = tmpDset.get('InstanceNumber', 0)
            tmpMrdImg.slice                  = uSliceLoc.tolist().index(tmpDset.SliceLocation)
            try:
                tmpMrdImg.phase                  = uTrigTime.tolist().index(tmpDset.TriggerTime)
            except:
                pass

            try:
                res  = re.search(r'(?<=_v).*$',     tmpDset.SequenceName)
                venc = re.search(r'^\d+',           res.group(0))
                dir  = re.search(r'(?<=\d)[^\d]*$', res.group(0))

                tmpMeta['FlowVelocity']   = float(venc.group(0))
                tmpMeta['FlowDirDisplay'] = venc_dir_map[dir.group(0)]
            except:
                pass

            tmpMeta['SequenceDescription'] = tmpDset.SeriesDescription

            # Remove pixel data from pydicom class
            del tmpDset['PixelData']

            # Store the complete base64, json-formatted DICOM header so that non-MRD fields can be
            # recapitulated when generating DICOMs from MRD images
            tmpMeta['DicomJson'] = base64.b64encode(tmpDset.to_json().encode('utf-8')).decode('utf-8')

            tmpMrdImg.attribute_string = tmpMeta.serialize()
            imgAll[iSer][iImg] = tmpMrdImg

    # Create an MRD file
    print("Creating MRD file %s with group %s" % (args.outFile, args.outGroup))
    mrdDset = ismrmrd.Dataset(args.outFile, args.outGroup)
    mrdDset._file.require_group(args.outGroup)

    # Write MRD Header
    mrdDset.write_xml_header(bytes(mrdHead.toXML(), 'utf-8'))

    # Write all images
    for iSer in range(len(imgAll)):
        for iImg in range(len(imgAll[iSer])):
            mrdDset.append_image("images_%d" % imgAll[iSer][iImg].image_series_index, imgAll[iSer][iImg])

    mrdDset.close()

if __name__ == '__main__':
    """Basic conversion of a folder of DICOM files to MRD .h5 format"""

    parser = argparse.ArgumentParser(description='Convert DICOMs to MRD file',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('folder',            help='Input folder of DICOMs')
    parser.add_argument('-o', '--outFile',  help='Output MRD file')
    parser.add_argument('-g', '--outGroup', help='Group name in output MRD file')

    parser.set_defaults(**defaults)

    args = parser.parse_args()

    if args.outFile is None:
        args.outFile = os.path.basename(args.folder) + '.h5'

    main(args)
