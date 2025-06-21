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
    try:
        mrdHead.acquisitionSystemInformation.institutionName       = dset.InstitutionName
    except:
        mrdHead.acquisitionSystemInformation.institutionName       = 'Virtual'
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
    
    # Use helper functions to get pixel spacing and slice thickness
    pixel_spacing = get_pixel_spacing(dset)
    slice_thickness = get_slice_thickness(dset)
    
    encSpace.fieldOfView_mm.x = pixel_spacing[0] * dset.Rows
    encSpace.fieldOfView_mm.y = pixel_spacing[1] * dset.Columns
    encSpace.fieldOfView_mm.z = slice_thickness
    
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

def get_slice_location(dset):
    """Extract slice location from various DICOM formats"""
    # Check for standard SliceLocation
    if hasattr(dset, 'SliceLocation'):
        return dset.SliceLocation
    
    # Check for enhanced DICOM with PerFrameFunctionalGroupsSequence
    if hasattr(dset, 'SOPClassUID') and dset.SOPClassUID.name == 'Enhanced MR Image Storage':
        try:
            # Extract from PlanePositionSequence
            if hasattr(dset.PerFrameFunctionalGroupsSequence[0], 'PlanePositionSequence'):
                pos = dset.PerFrameFunctionalGroupsSequence[0].PlanePositionSequence[0].ImagePositionPatient
                return float(pos[2])  # Z-coordinate
        except:
            pass
    
    # Last resort: use ImagePositionPatient
    if hasattr(dset, 'ImagePositionPatient'):
        return float(dset.ImagePositionPatient[2])
    
    # If all else fails, return None
    return None

def get_pixel_spacing(dset):
    """Extract pixel spacing from various DICOM formats"""
    # Check for standard PixelSpacing
    if hasattr(dset, 'PixelSpacing'):
        return dset.PixelSpacing
    
    # Check for enhanced DICOM with PerFrameFunctionalGroupsSequence
    if hasattr(dset, 'SOPClassUID') and dset.SOPClassUID.name == 'Enhanced MR Image Storage':
        try:
            # Extract from PixelMeasuresSequence
            if hasattr(dset.PerFrameFunctionalGroupsSequence[0], 'PixelMeasuresSequence'):
                return dset.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing
            # Check SharedFunctionalGroupsSequence as an alternative
            elif hasattr(dset, 'SharedFunctionalGroupsSequence'):
                if hasattr(dset.SharedFunctionalGroupsSequence[0], 'PixelMeasuresSequence'):
                    return dset.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing
        except:
            pass
    
    # If all else fails, return default values
    print(f"Warning: No pixel spacing found for file {dset.filename if hasattr(dset, 'filename') else 'unknown'}")
    return [1.0, 1.0]  # Default 1mm spacing

def get_slice_thickness(dset):
    """Extract slice thickness from various DICOM formats"""
    # Check for standard SliceThickness
    if hasattr(dset, 'SliceThickness'):
        return float(dset.SliceThickness)
    
    # Check for enhanced DICOM with PerFrameFunctionalGroupsSequence
    if hasattr(dset, 'SOPClassUID') and dset.SOPClassUID.name == 'Enhanced MR Image Storage':
        try:
            # Extract from PixelMeasuresSequence
            if hasattr(dset.PerFrameFunctionalGroupsSequence[0], 'PixelMeasuresSequence'):
                return float(dset.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SliceThickness)
            # Check SharedFunctionalGroupsSequence as an alternative
            elif hasattr(dset, 'SharedFunctionalGroupsSequence'):
                if hasattr(dset.SharedFunctionalGroupsSequence[0], 'PixelMeasuresSequence'):
                    return float(dset.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SliceThickness)
        except:
            pass
    
    # If all else fails, return default value
    print(f"Warning: No slice thickness found for file {dset.filename if hasattr(dset, 'filename') else 'unknown'}")
    return 1.0  # Default 1mm thickness

def get_image_position(dset):
    """Extract image position from various DICOM formats"""
    # Check for standard ImagePositionPatient
    if hasattr(dset, 'ImagePositionPatient'):
        return np.stack(dset.ImagePositionPatient)
    
    # Check for enhanced DICOM with PerFrameFunctionalGroupsSequence
    if hasattr(dset, 'SOPClassUID') and dset.SOPClassUID.name == 'Enhanced MR Image Storage':
        try:
            # Extract from PlanePositionSequence
            if hasattr(dset.PerFrameFunctionalGroupsSequence[0], 'PlanePositionSequence'):
                return np.stack(dset.PerFrameFunctionalGroupsSequence[0].PlanePositionSequence[0].ImagePositionPatient)
            # Check SharedFunctionalGroupsSequence as an alternative
            elif hasattr(dset, 'SharedFunctionalGroupsSequence'):
                if hasattr(dset.SharedFunctionalGroupsSequence[0], 'PlanePositionSequence'):
                    return np.stack(dset.SharedFunctionalGroupsSequence[0].PlanePositionSequence[0].ImagePositionPatient)
        except:
            pass
    
    # If all else fails, return default values
    print(f"Warning: No image position found for file {dset.filename if hasattr(dset, 'filename') else 'unknown'}")
    return np.array([0.0, 0.0, 0.0])  # Default position at origin

def get_image_orientation(dset):
    """Extract image orientation from various DICOM formats"""
    # Check for standard ImageOrientationPatient
    if hasattr(dset, 'ImageOrientationPatient'):
        return np.stack(dset.ImageOrientationPatient)
    
    # Check for enhanced DICOM with PerFrameFunctionalGroupsSequence
    if hasattr(dset, 'SOPClassUID') and dset.SOPClassUID.name == 'Enhanced MR Image Storage':
        try:
            # Extract from PlaneOrientationSequence
            if hasattr(dset.PerFrameFunctionalGroupsSequence[0], 'PlaneOrientationSequence'):
                return np.stack(dset.PerFrameFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient)
            # Check SharedFunctionalGroupsSequence as an alternative
            elif hasattr(dset, 'SharedFunctionalGroupsSequence'):
                if hasattr(dset.SharedFunctionalGroupsSequence[0], 'PlaneOrientationSequence'):
                    return np.stack(dset.SharedFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient)
        except:
            pass
    
    # If all else fails, return default values (axial orientation)
    print(f"Warning: No image orientation found for file {dset.filename if hasattr(dset, 'filename') else 'unknown'}")
    return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])  # Default axial orientation

def get_acquisition_time(dset):
    """Extract acquisition time from various DICOM formats"""
    # Check for standard AcquisitionTime
    if hasattr(dset, 'AcquisitionTime') and dset.AcquisitionTime:
        return dset.AcquisitionTime
    
    # Check for enhanced DICOM - look in various functional groups
    if hasattr(dset, 'SOPClassUID') and dset.SOPClassUID.name == 'Enhanced MR Image Storage':
        try:
            # Try to get from PerFrameFunctionalGroupsSequence
            if hasattr(dset, 'PerFrameFunctionalGroupsSequence'):
                if hasattr(dset.PerFrameFunctionalGroupsSequence[0], 'MRAcquisitionSequence'):
                    return dset.PerFrameFunctionalGroupsSequence[0].MRAcquisitionSequence[0].AcquisitionDateTime.split('T')[1]
                
            # Try SharedFunctionalGroupsSequence
            if hasattr(dset, 'SharedFunctionalGroupsSequence'):
                if hasattr(dset.SharedFunctionalGroupsSequence[0], 'MRAcquisitionSequence'):
                    return dset.SharedFunctionalGroupsSequence[0].MRAcquisitionSequence[0].AcquisitionDateTime.split('T')[1]
                
            # Try ContentTime as fallback
            if hasattr(dset, 'ContentTime'):
                return dset.ContentTime
        except:
            pass
    
    # Try other common time fields as fallbacks
    for time_field in ['AcquisitionDateTime', 'ContentTime', 'InstanceCreationTime', 'SeriesTime', 'StudyTime']:
        if hasattr(dset, time_field) and getattr(dset, time_field):
            # If it's a datetime field, extract just the time part
            time_value = getattr(dset, time_field)
            if 'T' in time_value:
                return time_value.split('T')[1]
            return time_value
    
    # If all else fails
    print(f"Warning: No acquisition time found for file {dset.filename if hasattr(dset, 'filename') else 'unknown'}")
    return "000000.000000"  # Midnight as default

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
        slice_locations = [get_slice_location(dset) for dset in dsets]
        
        # If any slice locations are None, create artificial locations
        if None in slice_locations:
            print("Warning: Some images missing slice location - using sequential numbering")
            slice_locations = list(range(len(dsets)))
        
        uSliceLoc = np.unique(slice_locations)
        if slice_locations[0] != uSliceLoc[0]:
            uSliceLoc = uSliceLoc[::-1]

        try:
            # This field may not exist for non-gated sequences
            uTrigTime = np.unique([getattr(dset, 'TriggerTime', 0) for dset in dsets])
            if hasattr(dsets[0], 'TriggerTime') and dsets[0].TriggerTime != uTrigTime[0]:
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
                tmpMrdImg.image_type = imtype_map[tmpDset.ImageType[2]]
            except:
                print("Unsupported ImageType %s -- defaulting to IMTYPE_MAGNITUDE" % tmpDset.ImageType[2])
                tmpMrdImg.image_type = ismrmrd.IMTYPE_MAGNITUDE

            try:
                # Get pixel spacing and slice thickness with proper handling for enhanced DICOM
                pixel_spacing = get_pixel_spacing(tmpDset)
                slice_thickness = get_slice_thickness(tmpDset)
                
                tmpMrdImg.field_of_view = (
                    pixel_spacing[0] * tmpDset.Rows,
                    pixel_spacing[1] * tmpDset.Columns,
                    slice_thickness
                )
            except Exception as e:
                print(f"Error setting field_of_view: {e} - using defaults")
                tmpMrdImg.field_of_view = (
                    tmpDset.Rows,  # Default to 1mm spacing
                    tmpDset.Columns,
                    1.0  # Default slice thickness
                )
                
            try:
                # Get image position and orientation with proper handling for enhanced DICOM
                image_position = get_image_position(tmpDset)
                image_orientation = get_image_orientation(tmpDset)
                
                tmpMrdImg.position = tuple(image_position)
                tmpMrdImg.read_dir = tuple(image_orientation[0:3])
                tmpMrdImg.phase_dir = tuple(image_orientation[3:6])
                tmpMrdImg.slice_dir = tuple(np.cross(image_orientation[0:3], image_orientation[3:6]))
            except Exception as e:
                print(f"Error setting position/orientation: {e} - using defaults")
                # Default to standard orientation (axial)
                tmpMrdImg.position = (0.0, 0.0, 0.0)
                tmpMrdImg.read_dir = (1.0, 0.0, 0.0)
                tmpMrdImg.phase_dir = (0.0, 1.0, 0.0)
                tmpMrdImg.slice_dir = (0.0, 0.0, 1.0)
            
            try:
                # Get acquisition time with proper handling for enhanced DICOM
                acq_time = get_acquisition_time(tmpDset)
                
                # Parse the acquisition time string into hours, minutes, seconds
                if len(acq_time) >= 6:  # Make sure we have at least HHMMSS format
                    hours = int(acq_time[0:2])
                    minutes = int(acq_time[2:4])
                    seconds = float(acq_time[4:])
                    tmpMrdImg.acquisition_time_stamp = round((hours*3600 + minutes*60 + seconds)*1000/2.5)
                else:
                    # If format is unexpected, use a default timestamp
                    print(f"Warning: Unexpected acquisition time format: {acq_time}")
                    tmpMrdImg.acquisition_time_stamp = 0
            except Exception as e:
                print(f"Error setting acquisition_time_stamp: {e} - using default")
                tmpMrdImg.acquisition_time_stamp = 0
            
            try:
                tmpMrdImg.physiology_time_stamp[0] = round(int(getattr(tmpDset, 'TriggerTime', 0)/2.5))
            except:
                pass

            try:
                ImaAbsTablePosition = tmpDset.get_private_item(0x0019, 0x13, 'SIEMENS MR HEADER').value
                tmpMrdImg.patient_table_position = (ctypes.c_float(ImaAbsTablePosition[0]), ctypes.c_float(ImaAbsTablePosition[1]), ctypes.c_float(ImaAbsTablePosition[2]))
            except:
                pass

            tmpMrdImg.image_series_index     = uSeriesNum.tolist().index(tmpDset.SeriesNumber)
            tmpMrdImg.image_index            = tmpDset.get('InstanceNumber', 0)
            
            # Use the same slice location extraction for consistency
            loc = get_slice_location(tmpDset)
            if loc is not None:
                tmpMrdImg.slice = np.where(uSliceLoc == loc)[0][0]
            else:
                tmpMrdImg.slice = iImg % len(uSliceLoc)
                
            try:
                if hasattr(tmpDset, 'TriggerTime'):
                    tmpMrdImg.phase = uTrigTime.tolist().index(tmpDset.TriggerTime)
                else:
                    tmpMrdImg.phase = 0
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
