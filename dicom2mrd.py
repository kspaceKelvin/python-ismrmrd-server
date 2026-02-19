import pydicom
import argparse
import ismrmrd
import numpy as np
import os
import ctypes
import re
import base64

import dateutil.parser
from datetime import datetime

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

def CalcFieldOfView(dset):
    if dset.SOPClassUID.name == 'Enhanced MR Image Storage':
        try:
            PixelMeasuresSequence = dset.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
        except:
            PixelMeasuresSequence = dset.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0]

            uSliceThickness  = set([float(s.PixelMeasuresSequence[0].SliceThickness)  for s in dset.PerFrameFunctionalGroupsSequence])
            uPixelSpacingRow = set([float(s.PixelMeasuresSequence[0].PixelSpacing[0]) for s in dset.PerFrameFunctionalGroupsSequence])
            uPixelSpacingCol = set([float(s.PixelMeasuresSequence[0].PixelSpacing[1]) for s in dset.PerFrameFunctionalGroupsSequence])

            if (len(uSliceThickness) > 1) or (len(uPixelSpacingRow) > 1) or (len(uPixelSpacingCol) > 1):
                print('Warning: Enhanced DICOM has frames with different PixelSpacing or SliceThickness -- only using information from first frame for MRD header')

        return (      PixelMeasuresSequence[0].PixelSpacing[1]*dset.Rows,
                      PixelMeasuresSequence[0].PixelSpacing[0]*dset.Columns,
                float(PixelMeasuresSequence[0].SliceThickness))

    elif dset.SOPClassUID.name == 'MR Image Storage':
        return (      dset.PixelSpacing[1]*dset.Rows,
                      dset.PixelSpacing[0]*dset.Columns,
                float(dset.SliceThickness))
    elif dset.SOPClassUID.name == 'MR Spectroscopy Storage':
        return (dset.VolumeLocalizationSequence[0].SlabThickness,
                dset.VolumeLocalizationSequence[1].SlabThickness,
                dset.VolumeLocalizationSequence[2].SlabThickness)


def CreateMrdHeader(dset):
    """Create MRD XML header from a DICOM file"""

    mrdHead = ismrmrd.xsd.ismrmrdHeader()

    # -------------------- studyInformation --------------------
    mrdHead.studyInformation = ismrmrd.xsd.studyInformationType()
    try:
        studyDateTime = dateutil.parser.parse(getattr(dset, 'StudyDate', '1970-01-01') + ' ' + getattr(dset, 'StudyTime', ''))
        mrdHead.studyInformation.studyDate = studyDateTime.strftime('%Y-%m-%d')
        mrdHead.studyInformation.studyTime = studyDateTime.strftime('%H:%M:%S')
    except:
        pass

    mrdHead.studyInformation.studyID                = getattr(dset, 'StudyID',                None)
    mrdHead.studyInformation.accessionNumber        = getattr(dset, 'AccessionNumber',        None)
    # mrdHead.studyInformation.referringPhysicianName = getattr(dset, 'ReferringPhysicianName', None)
    mrdHead.studyInformation.studyDescription       = getattr(dset, 'StudyDescription',       None)
    mrdHead.studyInformation.studyInstanceUID       = getattr(dset, 'StudyInstanceUID',       None)
    mrdHead.studyInformation.bodyPartExamined       = getattr(dset, 'BodyPartExamined',       None)

    # -------------------- measurementInformation --------------------
    mrdHead.measurementInformation                             = ismrmrd.xsd.measurementInformationType()
    mrdHead.measurementInformation.measurementID               = getattr(dset, 'SeriesInstanceUID',   None)
    mrdHead.measurementInformation.patientPosition             = getattr(dset, 'PatientPosition',     None)
    mrdHead.measurementInformation.protocolName                = getattr(dset, 'SeriesDescription',   None)
    mrdHead.measurementInformation.frameOfReferenceUID         = getattr(dset, 'FrameOfReferenceUID', None)

    # -------------------- acquisitionSystemInformation --------------------
    mrdHead.acquisitionSystemInformation                       = ismrmrd.xsd.acquisitionSystemInformationType()
    mrdHead.acquisitionSystemInformation.systemVendor          =       getattr(dset, 'Manufacturer',          None)
    mrdHead.acquisitionSystemInformation.systemModel           =       getattr(dset, 'ManufacturerModelName', None)
    mrdHead.acquisitionSystemInformation.systemFieldStrength_T = float(getattr(dset, 'MagneticFieldStrength', '0'))
    mrdHead.acquisitionSystemInformation.institutionName       =       getattr(dset, 'InstitutionName',       None)
    mrdHead.acquisitionSystemInformation.stationName           =       getattr(dset, 'StationName',           None)

    # -------------------- experimentalConditions --------------------
    mrdHead.experimentalConditions                             = ismrmrd.xsd.experimentalConditionsType()
    if hasattr(dset, 'TransmitterFrequency'):
        mrdHead.experimentalConditions.H1resonanceFrequency_Hz = int(getattr(dset, 'TransmitterFrequency')*1e6)
    elif hasattr(dset, 'ImagingFrequency'):
        mrdHead.experimentalConditions.H1resonanceFrequency_Hz = int(getattr(dset, 'ImagingFrequency')*1e6)
    else:
        mrdHead.experimentalConditions.H1resonanceFrequency_Hz = int(getattr(dset, 'MagneticFieldStrength')*4258e4)

    # -------------------- encodingType --------------------
    enc = ismrmrd.xsd.encodingType()
    enc.trajectory          = ismrmrd.xsd.trajectoryType('cartesian')

    encSpace                = ismrmrd.xsd.encodingSpaceType()
    encSpace.matrixSize     = ismrmrd.xsd.matrixSizeType()
    encSpace.matrixSize.x   = dset.Columns
    encSpace.matrixSize.y   = dset.Rows
    encSpace.matrixSize.z   = 1
    encSpace.fieldOfView_mm = ismrmrd.xsd.fieldOfViewMm(*CalcFieldOfView(dset))

    enc.encodedSpace = encSpace
    enc.reconSpace   = encSpace

    enc.encodingLimits                     = ismrmrd.xsd.encodingLimitsType()
    enc.parallelImaging                    = ismrmrd.xsd.parallelImagingType()
    enc.parallelImaging.accelerationFactor = ismrmrd.xsd.accelerationFactorType()
    if hasattr(dset, 'SharedFunctionalGroupsSequence'):
        if dset.SharedFunctionalGroupsSequence[0].MRModifierSequence[0].ParallelAcquisition == 'NO':
            enc.parallelImaging.accelerationFactor.kspace_encoding_step_1 = 1
            enc.parallelImaging.accelerationFactor.kspace_encoding_step_2 = 1
        else:
            enc.parallelImaging.accelerationFactor.kspace_encoding_step_1 = dset.SharedFunctionalGroupsSequence[0].MRModifierSequence[0].ParallelReductionFactorInPlane
            enc.parallelImaging.accelerationFactor.kspace_encoding_step_2 = dset.SharedFunctionalGroupsSequence[0].MRModifierSequence[0].ParallelReductionFactorOutOfPlane
    else:
        enc.parallelImaging.accelerationFactor.kspace_encoding_step_1 = 1
        enc.parallelImaging.accelerationFactor.kspace_encoding_step_2 = 1

    mrdHead.encoding.append(enc)

    mrdHead.sequenceParameters               = ismrmrd.xsd.sequenceParametersType()
    if hasattr(dset, 'SharedFunctionalGroupsSequence'):
        mrdHead.sequenceParameters.TR            = float(dset.SharedFunctionalGroupsSequence[0].MRTimingAndRelatedParametersSequence[0].RepetitionTime)
        mrdHead.sequenceParameters.flipAngle_deg = float(dset.SharedFunctionalGroupsSequence[0].MRTimingAndRelatedParametersSequence[0].FlipAngle)
        mrdHead.sequenceParameters.TE            =       dset.SharedFunctionalGroupsSequence[0].MREchoSequence[0].EffectiveEchoTime
    else:
        mrdHead.sequenceParameters.TR            = float(dset.RepetitionTime)
        mrdHead.sequenceParameters.flipAngle_deg = float(dset.FlipAngle)
        mrdHead.sequenceParameters.TE            = float(dset.EchoTime)

    # -------------------- User parameters --------------------
    userParameters = ismrmrd.xsd.userParametersType()

    # Water suppression
    try:
        if hasattr(dset, 'SharedFunctionalGroupsSequence'):
            MeasurementOptions = dset.SharedFunctionalGroupsSequence[0][0x002110FE][0][0x0021105C].value
        else:
            MeasurementOptions = dset[0x0021105C].value

        if isinstance(MeasurementOptions, str):
            if MeasurementOptions == 'WS':
                userParameterString = ismrmrd.xsd.userParameterStringType('FatWaterContrast', 'WATER_SATURATION')
                userParameters.userParameterString.append(userParameterString)
        else:
            if 'WS' in list(MeasurementOptions):
                userParameterString = ismrmrd.xsd.userParameterStringType('FatWaterContrast', 'WATER_SATURATION')
                userParameters.userParameterString.append(userParameterString)
    except:
        pass

    # Spectroscopy readout points (without oversampling)
    try:
        SpecVectorSize = dset.SharedFunctionalGroupsSequence[0].MRSpectroscopyFOVGeometrySequence[0].SpectroscopyAcquisitionDataColumns
        userParameterLong = ismrmrd.xsd.userParameterLongType('SpecVectorSize', SpecVectorSize)
        userParameters.userParameterLong.append(userParameterLong)
    except:
        pass

    # Readout oversampling
    try:
        if hasattr(dset, 'SharedFunctionalGroupsSequence'):
            ReadoutOS = dset.SharedFunctionalGroupsSequence[0][0x002110FE][0][0x00211012].value
        else:
            ReadoutOS = dset[0x00211012].value
        userParameterDouble = ismrmrd.xsd.userParameterDoubleType('ReadoutOS', ReadoutOS)
        userParameters.userParameterDouble.append(userParameterDouble)
    except:
        pass

    # Spectral Width (Hz)
    try:
        SpectralWidth = dset.SpectralWidth
        userParameterDouble = ismrmrd.xsd.userParameterDoubleType('SpectralWidth', SpectralWidth)
        userParameters.userParameterDouble.append(userParameterDouble)
    except:
        pass

    # Dwell time (oversampled)
    try:
        DwellTime = 1e6 / SpectralWidth / ReadoutOS
        userParameterDouble = ismrmrd.xsd.userParameterDoubleType('DwellTime_0', DwellTime)
        userParameters.userParameterDouble.append(userParameterDouble)
    except:
        pass

    # Spectroscopy volume of interest dimensions
    try:
        for dim in dset.VolumeLocalizationSequence:
            # Determine if this is x, y, or z
            if all(np.abs(np.cross([0, 0, 1], np.array(dim.SlabOrientation))) < 1e-5):
                name = 'SpecVoiThickness'
            elif all(np.abs(np.cross([1, 0, 0], np.array(dim.SlabOrientation))) < 1e-5):
                name = 'SpecVoiPhaseFOV'
            elif all(np.abs(np.cross([0, 1, 0], np.array(dim.SlabOrientation))) < 1e-5):
                name = 'SpecVoiReadoutFOV'
            else:
                print(f'Could not determine spectroscopy VOI dimension for orientation {dim.SlabOrientation}')
                continue

            userParameterDouble = ismrmrd.xsd.userParameterDoubleType(name, dim.SlabThickness)
            userParameters.userParameterDouble.append(userParameterDouble)
    except:
        pass

    mrdHead.userParameters = userParameters
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

    # Re-group series that were split during conversion from multi-frame to single-frame DICOMs
    if all(uSeriesNum > 1000):
        for i in range(len(dsetsAll)):
            dsetsAll[i].SeriesNumber = int(np.floor(dsetsAll[i].SeriesNumber / 1000))
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
        try:
            uSliceLoc = np.unique([dset.SliceLocation for dset in dsets])
            if dsets[0].SliceLocation != uSliceLoc[0]:
                uSliceLoc = uSliceLoc[::-1]
        except:
            uSliceLoc = np.zeros(len(uSeriesNum))

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
            # from_array() should be called with 'transpose=False' to avoid warnings, and when called
            # with this option, can take input as: [cha z y x], [z y x], or [y x]

            if hasattr(tmpDset, 'pixel_array'):
                # pixel_array data has shape [row col], i.e. [y x].
                tmpMrdImg = ismrmrd.Image.from_array(tmpDset.pixel_array, transpose=False)
            elif hasattr(tmpDset, 'SpectroscopyData'):
                tmpMrdImg = ismrmrd.Image.from_array(np.frombuffer(tmpDset.SpectroscopyData, dtype=np.complex64), transpose=False)
            else:
                print(f'Error: Could not find imaging or spectroscopy data for file {tmpDset.filename}')
                continue

            tmpMeta   = ismrmrd.Meta()

            try:
                tmpMrdImg.image_type                = imtype_map[tmpDset.ImageType[2]]
            except:
                print("Unsupported ImageType %s -- defaulting to IMTYPE_MAGNITUDE" % tmpDset.ImageType[2])
                tmpMrdImg.image_type                = ismrmrd.IMTYPE_MAGNITUDE

            if hasattr(tmpDset, 'PerFrameFunctionalGroupsSequence'):
                ImagePositionPatient    = tmpDset.PerFrameFunctionalGroupsSequence[0].PlanePositionSequence[0].ImagePositionPatient
                ImageOrientationPatient = tmpDset.PerFrameFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient
                AcquisitionTime         = tmpDset.PerFrameFunctionalGroupsSequence[0].FrameContentSequence[0].FrameAcquisitionDateTime[8:]  # Strip out date
                try:
                    TriggerTime = tmpDset.PerFrameFunctionalGroupsSequence[0].CardiacSynchronizationSequence[0].NominalCardiacTriggerDelayTime
                except:
                    TriggerTime = None
            else:
                ImagePositionPatient    = tmpDset.ImagePositionPatient
                ImageOrientationPatient = tmpDset.ImageOrientationPatient
                AcquisitionTime         = tmpDset.AcquisitionTime
                try:
                    TriggerTime = float(tmpDset.TriggerTime)
                except:
                    TriggerTime = None

            tmpMrdImg.field_of_view            = CalcFieldOfView(tmpDset)
            tmpMrdImg.position                 = tuple(np.stack(ImagePositionPatient))
            tmpMrdImg.read_dir                 = tuple(np.stack(ImageOrientationPatient[0:3]))
            tmpMrdImg.phase_dir                = tuple(np.stack(ImageOrientationPatient[3:7]))
            tmpMrdImg.slice_dir                = tuple(np.cross(np.stack(ImageOrientationPatient[0:3]), np.stack(ImageOrientationPatient[3:7])))
            tmpMrdImg.acquisition_time_stamp   = round((int(AcquisitionTime[0:2])*3600 + int(AcquisitionTime[2:4])*60 + int(AcquisitionTime[4:6]) + float(AcquisitionTime[6:]))*1000/2.5)
            if TriggerTime:
                tmpMrdImg.physiology_time_stamp[0] = round(int(TriggerTime/2.5))

            try:
                ImaAbsTablePosition = tmpDset.get_private_item(0x0019, 0x13, 'SIEMENS MR HEADER').value
                tmpMrdImg.patient_table_position = (ctypes.c_float(ImaAbsTablePosition[0]), ctypes.c_float(ImaAbsTablePosition[1]), ctypes.c_float(ImaAbsTablePosition[2]))
            except:
                pass

            tmpMrdImg.image_series_index     = uSeriesNum.tolist().index(tmpDset.SeriesNumber)
            tmpMrdImg.image_index            = tmpDset.get('InstanceNumber', 0)
            tmpMrdImg.slice                  = uSliceLoc.tolist().index(getattr(tmpDset, 'SliceLocation', 0))
            try:
                tmpMrdImg.phase                  = uTrigTime.tolist().index(tmpDset.TriggerTime)
            except:
                pass

            try:
                tmpMeta['ImageType'] = tmpDset.ImageType
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

            try:
                tmpMeta['ImageComments'] = tmpDset.ImageComments
            except:
                pass

            tmpMeta['SequenceDescription'] = tmpDset.SeriesDescription

            # Remove pixel data from pydicom class before serializing metadata
            if hasattr(tmpDset, 'PixelData'):
                del tmpDset['PixelData']

            if hasattr(tmpDset, 'SpectroscopyData'):
                del tmpDset['SpectroscopyData']

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
            mrdDset.append_image("image_%d" % imgAll[iSer][iImg].image_series_index, imgAll[iSer][iImg])

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
