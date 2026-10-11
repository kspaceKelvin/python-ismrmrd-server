import pydicom
import argparse
import ismrmrd
import ismrmrd.xsd
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

class DicomImage:
    def __init__(self, dset, frame_idx=0):
        self.dset = dset
        self.frame_idx = frame_idx
        self.is_enhanced = (dset.SOPClassUID.name == 'Enhanced MR Image Storage')
        
    def get_group(self, group_name):
        if not self.is_enhanced:
            return None
            
        # Check PerFrame
        if group_name in self.dset.PerFrameFunctionalGroupsSequence[self.frame_idx]:
            return self.dset.PerFrameFunctionalGroupsSequence[self.frame_idx][group_name][0]
        # Check Shared
        if group_name in self.dset.SharedFunctionalGroupsSequence[0]:
            return self.dset.SharedFunctionalGroupsSequence[0][group_name][0]
        return None

    @property
    def pixel_array(self):
        if self.is_enhanced:
            return self.dset.pixel_array[self.frame_idx]
        else:
            return self.dset.pixel_array

    @property
    def PixelSpacing(self):
        if self.is_enhanced:
            measure = self.get_group('PixelMeasuresSequence')
            if measure:
                return measure.PixelSpacing
            return [1.0, 1.0]
        else:
            return self.dset.PixelSpacing

    @property
    def SliceThickness(self):
        if self.is_enhanced:
            measure = self.get_group('PixelMeasuresSequence')
            if measure:
                return measure.SliceThickness
            return 1.0
        else:
            return self.dset.SliceThickness

    @property
    def ImagePositionPatient(self):
        if self.is_enhanced:
            pos = self.get_group('PlanePositionSequence')
            if pos:
                return pos.ImagePositionPatient
            return [0.0, 0.0, 0.0]
        else:
            return self.dset.ImagePositionPatient

    @property
    def ImageOrientationPatient(self):
        if self.is_enhanced:
            orient = self.get_group('PlaneOrientationSequence')
            if orient:
                return orient.ImageOrientationPatient
            return [1, 0, 0, 0, 1, 0]
        else:
            return self.dset.ImageOrientationPatient

    @property
    def SliceLocation(self):
        # Calculate SliceLocation from Position and Orientation to be consistent
        # SliceLocation is the projection of Position onto the normal vector of the slice.
        try:
            pos = np.array(self.ImagePositionPatient, dtype=float)
            orient = np.array(self.ImageOrientationPatient, dtype=float)
            normal = np.cross(orient[0:3], orient[3:6])
            return np.dot(pos, normal)
        except:
            if not self.is_enhanced and 'SliceLocation' in self.dset:
                return self.dset.SliceLocation
            return 0.0

    @property
    def TemporalPositionIndex(self):
        if self.is_enhanced:
             content = self.get_group('FrameContentSequence')
             if content and 'TemporalPositionIndex' in content:
                 return content.TemporalPositionIndex
        return 0

    @property
    def InStackPositionNumber(self):
        if self.is_enhanced:
            content = self.get_group('FrameContentSequence')
            if content and 'InStackPositionNumber' in content:
                return int(content.InStackPositionNumber)
        return None

    @property
    def StackID(self):
        if self.is_enhanced:
            content = self.get_group('FrameContentSequence')
            if content and 'StackID' in content:
                return str(content.StackID)
        return None

    @property
    def TriggerTime(self):
        if self.is_enhanced:
            cardiac = self.get_group('CardiacSynchronizationSequence')
            if cardiac and 'NominalCardiacTriggerDelayTime' in cardiac:
                return float(cardiac.NominalCardiacTriggerDelayTime)
            
            # Fallback to FrameContentSequence -> TemporalPositionIndex if needed?
            # Or just return 0.0
            return 0.0
        else:
            return float(self.dset.get('TriggerTime', 0.0))

    @property
    def AcquisitionTime(self):
        if self.is_enhanced:
             content = self.get_group('FrameContentSequence')
             if content and 'FrameAcquisitionDateTime' in content:
                 dt = content.FrameAcquisitionDateTime
                 # DT is YYYYMMDDHHMMSS.FFFFFF
                 if len(dt) > 8:
                    return dt[8:] 
             return self.dset.get('AcquisitionTime', '000000.00')
        else:
            return self.dset.get('AcquisitionTime', '000000.00')

    @property
    def InstanceNumber(self):
        if self.is_enhanced:
            return (self.dset.InstanceNumber * 10000) + self.frame_idx + 1
        else:
            return self.dset.InstanceNumber
            
    @property
    def SeriesNumber(self):
        return self.dset.SeriesNumber
        
    @property
    def SeriesDescription(self):
        return self.dset.get('SeriesDescription', '')
        
    @property
    def ImageType(self):
        if self.is_enhanced:
             content = self.get_group('MRImageFrameTypeSequence')
             if content:
                 return content.FrameType
             return self.dset.get('ImageType', ['','','MAGNITUDE'])
        else:
            return self.dset.get('ImageType', ['','','MAGNITUDE'])

    @property
    def Rows(self):
        return self.dset.Rows
    
    @property
    def Columns(self):
        return self.dset.Columns
        
    @property
    def ImageComments(self):
        return self.dset.get('ImageComments', '')
        
    @property
    def SequenceName(self):
        return self.dset.get('SequenceName', '')
        
    def get_private_item(self, group, element, creator):
        try:
            return self.dset.get_private_item(group, element, creator)
        except:
            return None

    def to_json(self):
        return self.dset.to_json()


def _normalize(vec):
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


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
    try:
        mrdHead.acquisitionSystemInformation.systemFieldStrength_T = float(dset.MagneticFieldStrength)
    except:
        pass
        
    try:
        mrdHead.acquisitionSystemInformation.institutionName       = dset.InstitutionName
    except:
        mrdHead.acquisitionSystemInformation.institutionName       = 'Virtual'
    try:
        mrdHead.acquisitionSystemInformation.stationName       = dset.StationName
    except:
        pass

    mrdHead.experimentalConditions                             = ismrmrd.xsd.experimentalConditionsType()
    try:
        mrdHead.experimentalConditions.H1resonanceFrequency_Hz     = int(dset.MagneticFieldStrength*4258e4)
    except:
        pass

    enc = ismrmrd.xsd.encodingType()
    enc.trajectory                                              = ismrmrd.xsd.trajectoryType('cartesian')
    encSpace                                                    = ismrmrd.xsd.encodingSpaceType()
    encSpace.matrixSize                                         = ismrmrd.xsd.matrixSizeType()
    encSpace.matrixSize.x                                       = dset.Columns
    encSpace.matrixSize.y                                       = dset.Rows
    encSpace.matrixSize.z                                       = 1
    encSpace.fieldOfView_mm                                     = ismrmrd.xsd.fieldOfViewMm()
    
    if dset.SOPClassUID.name == 'Enhanced MR Image Storage':
        # Use first frame as reference
        groups = dset.SharedFunctionalGroupsSequence[0]
        if 'PixelMeasuresSequence' not in groups:
             groups = dset.PerFrameFunctionalGroupsSequence[0]
             
        # PixelSpacing is [RowSpacing, ColSpacing] -> [Y_spacing, X_spacing]
        pixel_spacing = groups.PixelMeasuresSequence[0].PixelSpacing
        slice_thickness = float(groups.PixelMeasuresSequence[0].SliceThickness)
        
        encSpace.fieldOfView_mm.x                               =       pixel_spacing[1]*dset.Columns
        encSpace.fieldOfView_mm.y                               =       pixel_spacing[0]*dset.Rows
        encSpace.fieldOfView_mm.z                               =       slice_thickness
    else:
        pixel_spacing = dset.PixelSpacing
        slice_thickness = float(dset.SliceThickness)

        encSpace.fieldOfView_mm.x                               =       pixel_spacing[1]*dset.Columns
        encSpace.fieldOfView_mm.y                               =       pixel_spacing[0]*dset.Rows
        encSpace.fieldOfView_mm.z                               =       slice_thickness
    enc.encodedSpace                                            = encSpace
    enc.reconSpace                                              = encSpace
    enc.encodingLimits                                          = ismrmrd.xsd.encodingLimitsType()
    enc.parallelImaging                                         = ismrmrd.xsd.parallelImagingType()

    enc.parallelImaging.accelerationFactor                      = ismrmrd.xsd.accelerationFactorType()
    if dset.SOPClassUID.name == 'Enhanced MR Image Storage':
        # Try to find MRModifierSequence
        found_accel = False
        if 'MRModifierSequence' in dset.SharedFunctionalGroupsSequence[0]:
             mod = dset.SharedFunctionalGroupsSequence[0].MRModifierSequence[0]
             if 'ParallelReductionFactorInPlane' in mod:
                 enc.parallelImaging.accelerationFactor.kspace_encoding_step_1 = mod.ParallelReductionFactorInPlane
                 enc.parallelImaging.accelerationFactor.kspace_encoding_step_2 = mod.ParallelReductionFactorOutOfPlane
                 found_accel = True
        
        if not found_accel:
             enc.parallelImaging.accelerationFactor.kspace_encoding_step_1 = 1
             enc.parallelImaging.accelerationFactor.kspace_encoding_step_2 = 1
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
        try:
            dset = pydicom.dcmread(entryPath)
            dsetsAll.append(dset)
        except Exception as e:
            print(f"Error reading {entryPath}: {e}")

    if not dsetsAll:
        print(f"No DICOM files found in {args.folder}")
        return

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
        # Get all files for this series
        series_dsets = [dset for dset in dsetsAll if dset.SeriesNumber == uSeriesNum[iSer]]
        
        # Expand to DicomImage objects (handling Enhanced DICOM frames)
        images = []
        for dset in series_dsets:
            if dset.SOPClassUID.name == 'Enhanced MR Image Storage':
                nFrames = getattr(dset, 'NumberOfFrames', 1)
                for i in range(nFrames):
                    images.append(DicomImage(dset, i))
            else:
                images.append(DicomImage(dset))

        # Group by Phase (TemporalPositionIndex for Enhanced, TriggerTime for others)
        is_enhanced_series = any(img.is_enhanced for img in images)
        
        if is_enhanced_series:
             key_func = lambda img: img.TemporalPositionIndex
        else:
             key_func = lambda img: img.TriggerTime

        uKeys = sorted(list(set(key_func(img) for img in images)))
        
        imgAll[iSer] = []
        
        for iPhase, key in enumerate(uKeys):
            # Get images for this phase
            phase_imgs = [img for img in images if key_func(img) == key]
            
            # Sort slices robustly:
            # 1) Enhanced frame stack coordinates when available
            # 2) Geometric slice position along slice normal
            # 3) InstanceNumber as final tie-breaker
            def sort_key(img):
                in_stack = img.InStackPositionNumber
                stack_id = img.StackID or ''
                try:
                    slice_loc = float(img.SliceLocation)
                except Exception:
                    slice_loc = 0.0
                return (
                    0 if in_stack is not None else 1,
                    stack_id,
                    int(in_stack) if in_stack is not None else 0,
                    slice_loc,
                    int(img.InstanceNumber),
                )

            phase_imgs.sort(key=sort_key)

            # Ensure slice index increases in the same physical direction as slice_dir.
            if len(phase_imgs) > 1:
                row_dir = _normalize(phase_imgs[0].ImageOrientationPatient[0:3])
                col_dir = _normalize(phase_imgs[0].ImageOrientationPatient[3:6])
                slice_dir = _normalize(np.cross(row_dir, col_dir))

                for i_check in range(1, len(phase_imgs)):
                    pos0 = np.asarray(phase_imgs[0].ImagePositionPatient, dtype=float)
                    pos1 = np.asarray(phase_imgs[i_check].ImagePositionPatient, dtype=float)
                    step = float(np.dot(pos1 - pos0, slice_dir))
                    if step < 0:
                        phase_imgs.reverse()
                        break
                    if step > 0:
                        break
            
            if not phase_imgs:
                continue
            
            # Calculate total FOV for the volume
            refImg = phase_imgs[0]
            num_slices = len(phase_imgs)
            total_fov_z = float(refImg.SliceThickness * num_slices)
            
            # Create separate MRD Image for each slice
            for iSlice, sliceImg in enumerate(phase_imgs):
                # Create MRD Image from single slice.
                # Keep DICOM's [rows, cols] layout to match direction vectors and spacing.
                tmpMrdImg = ismrmrd.Image.from_array(sliceImg.pixel_array, transpose=False)
                tmpMeta   = ismrmrd.Meta()

                try:
                    # ImageType is a list/tuple. Index 2 is usually M/P/R/I
                    # Enhanced: FrameType
                    itype = sliceImg.ImageType
                    if len(itype) > 2:
                        tmpMrdImg.image_type = imtype_map.get(itype[2], ismrmrd.IMTYPE_MAGNITUDE)
                    elif len(itype) > 0:
                         # Try to guess from first char of first element?
                         tmpMrdImg.image_type = imtype_map.get(itype[0][0], ismrmrd.IMTYPE_MAGNITUDE)
                    else:
                        tmpMrdImg.image_type = ismrmrd.IMTYPE_MAGNITUDE
                except:
                    print("Unsupported ImageType %s -- defaulting to IMTYPE_MAGNITUDE" % sliceImg.ImageType)
                    tmpMrdImg.image_type = ismrmrd.IMTYPE_MAGNITUDE

                # DICOM PixelSpacing is [row_spacing, col_spacing] => [Y, X].
                tmpMrdImg.field_of_view = (
                    float(sliceImg.PixelSpacing[1] * sliceImg.Columns),
                    float(sliceImg.PixelSpacing[0] * sliceImg.Rows),
                    float(sliceImg.SliceThickness),
                )
                
                # Note: matrix_size is read-only and derived from data shape (cols, rows, 1 for 2D slices)
                # The slice index indicates position within the volume
                
                row_dir = _normalize(sliceImg.ImageOrientationPatient[0:3])
                col_dir = _normalize(sliceImg.ImageOrientationPatient[3:6])
                slice_dir = _normalize(np.cross(row_dir, col_dir))

                tmpMrdImg.position = tuple(np.asarray(sliceImg.ImagePositionPatient, dtype=float))
                tmpMrdImg.read_dir = tuple(row_dir)
                tmpMrdImg.phase_dir = tuple(col_dir)
                tmpMrdImg.slice_dir = tuple(slice_dir)
                
                # AcquisitionTime HHMMSS.FFFFFF
                acq_time = sliceImg.AcquisitionTime
                try:
                    # Handle potential empty or malformed time
                    if acq_time and len(acq_time) >= 6:
                        h = int(acq_time[0:2])
                        m = int(acq_time[2:4])
                        s = int(acq_time[4:6])
                        f = float(acq_time[6:]) if len(acq_time) > 6 else 0.0
                        tmpMrdImg.acquisition_time_stamp = round((h*3600 + m*60 + s + f)*1000/2.5)
                    else:
                        tmpMrdImg.acquisition_time_stamp = 0
                except:
                    tmpMrdImg.acquisition_time_stamp = 0

                try:
                    tmpMrdImg.physiology_time_stamp[0] = int(round(sliceImg.TriggerTime / 2.5))
                except:
                    pass

                try:
                    item = sliceImg.get_private_item(0x0019, 0x13, 'SIEMENS MR HEADER')
                    if item:
                        ImaAbsTablePosition = item.value
                        tmpMrdImg.patient_table_position = (ctypes.c_float(ImaAbsTablePosition[0]), ctypes.c_float(ImaAbsTablePosition[1]), ctypes.c_float(ImaAbsTablePosition[2]))
                except:
                    pass

                tmpMrdImg.image_series_index     = uSeriesNum.tolist().index(sliceImg.SeriesNumber)
                tmpMrdImg.image_index            = sliceImg.InstanceNumber
                
                tmpMrdImg.slice = iSlice
                tmpMrdImg.phase = iPhase

                # Store original DICOM ImageType in metadata
                try:
                    itype = sliceImg.ImageType
                    if itype:
                        tmpMeta['ImageType'] = '\\'.join(str(x) for x in itype)
                except:
                    tmpMeta['ImageType'] = ''

                try:
                    seq_name = sliceImg.SequenceName
                    res  = re.search(r'(?<=_v).*$',     seq_name)
                    if res:
                        venc = re.search(r'^\d+',           res.group(0))
                        dir  = re.search(r'(?<=\d)[^\d]*$', res.group(0))
                        if venc and dir:
                            tmpMeta['FlowVelocity']   = float(venc.group(0))
                            tmpMeta['FlowDirDisplay'] = venc_dir_map.get(dir.group(0), '')
                except:
                    pass

                try:
                    tmpMeta['ImageComments'] = sliceImg.ImageComments
                except:
                    pass

                tmpMeta['SeriesDescription'] = sliceImg.SeriesDescription
                tmpMeta['SequenceDescription'] = sliceImg.SeriesDescription
                tmpMeta['Keep_image_geometry'] = 1

                tmpMrdImg.attribute_string = tmpMeta.serialize()
                
                imgAll[iSer].append(tmpMrdImg)
        
        print("Series %d: Created %d 2D images (slices x phases)" % (uSeriesNum[iSer], len(imgAll[iSer])))

    # Create an MRD file
    print("Creating MRD file %s with group %s" % (args.outFile, args.outGroup))
    mrdDset = ismrmrd.Dataset(args.outFile, args.outGroup)
    mrdDset._file.require_group(args.outGroup)

    # Write MRD Header
    mrdDset.write_xml_header(bytes(mrdHead.toXML(), 'utf-8'))

    # Write all images
    for iSer in range(len(imgAll)):
        for iImg in range(len(imgAll[iSer])):
            if imgAll[iSer][iImg] is not None:
                mrdDset.append_image("image_%d" % imgAll[iSer][iImg].image_series_index, imgAll[iSer][iImg])

    if len(imgAll) > 0 and len(imgAll[0]) > 0 and imgAll[0][0] is not None:
        first_img = imgAll[0][0]
        print(f"First Image FOV: {first_img.field_of_view}")
        print(f"First Image Matrix Size: {first_img.matrix_size}")

    print(f"Header FOV: {mrdHead.encoding[0].encodedSpace.fieldOfView_mm.x} {mrdHead.encoding[0].encodedSpace.fieldOfView_mm.y} {mrdHead.encoding[0].encodedSpace.fieldOfView_mm.z}")
    
    fov = mrdHead.encoding[0].encodedSpace.fieldOfView_mm
    matrix = mrdHead.encoding[0].encodedSpace.matrixSize
    print(f"Computed Voxel Size: {fov.x/matrix.x} {fov.y/matrix.y} {fov.z/matrix.z}")

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
