#!/usr/bin/env python3
"""
NIfTI to ISMRMRD Converter
Converts NIfTI files to ISMRMRD format for testing the OpenRecon pipeline
adapted from: https://github.com/jlautman1/open-recon-fetal-brain-measurements/blob/main/nifti_to_ismrmrd_converter.py
"""

import os
import sys
import argparse
import re
import numpy as np
import nibabel as nib
import json
from pathlib import Path

try:
    import ismrmrd
    print("✅ Successfully imported ismrmrd module")
except ImportError as e:
    print(f"❌ Failed to import ismrmrd: {e}")
    print("   Creating mock ISMRMRD classes for testing...")
    
    # Create mock ISMRMRD classes for testing
    class MockImage:
        def __init__(self, data):
            # Keep data as-is, don't convert to complex
            self.data = data
            self.meta = {}
            self.attribute_string = ""
            self.image_type = 1  # IMTYPE_MAGNITUDE
            self.image_index = 0
            self.image_series_index = 1
        
        @classmethod
        def from_array(cls, data, transpose=True):
            return cls(data)
    
    class MockMeta:
        def __init__(self):
            self._data = {}
        
        def __setitem__(self, key, value):
            self._data[key] = value
        
        def __getitem__(self, key):
            return self._data[key]
        
        def get(self, key, default=None):
            return self._data.get(key, default)
        
        def serialize(self):
            return json.dumps(self._data)
    
    # Create a mock ismrmrd module
    import types
    ismrmrd = types.ModuleType('ismrmrd')
    ismrmrd.Image = MockImage
    ismrmrd.Meta = MockMeta
    IMTYPE_MAGNITUDE = 1


def extract_orientation_from_affine(affine, shape):
    """
    Extract position and direction vectors from NIfTI affine matrix
    
    The affine matrix transforms from voxel coordinates to world coordinates:
    [x_world]   [r11 r12 r13 tx]   [i]
    [y_world] = [r21 r22 r23 ty] * [j]
    [z_world]   [r31 r32 r33 tz]   [k]
    [   1   ]   [ 0   0   0   1]   [1]
    
    Returns:
        position: [x, y, z] position of the first voxel center
        read_dir: [x, y, z] direction vector for readout (columns)
        phase_dir: [x, y, z] direction vector for phase encoding (rows)
        slice_dir: [x, y, z] direction vector for slice (slices)
    """
    print("🧭 Extracting orientation from affine matrix...")
    print(f"   Affine matrix:\n{affine}")
    
    # Extract the rotation/scaling part and translation
    # First 3 columns are the direction vectors scaled by voxel size
    rotation_scale = affine[:3, :3]
    translation = affine[:3, 3]
    
    # Extract direction vectors (columns of the rotation matrix)
    # These need to be normalized to get unit direction vectors
    col0 = rotation_scale[:, 0]  # First axis (usually X/readout)
    col1 = rotation_scale[:, 1]  # Second axis (usually Y/phase)
    col2 = rotation_scale[:, 2]  # Third axis (usually Z/slice)
    
    # Calculate voxel sizes from the direction vectors
    voxel_size_x = np.linalg.norm(col0)
    voxel_size_y = np.linalg.norm(col1)
    voxel_size_z = np.linalg.norm(col2)
    
    print(f"   Voxel sizes from affine: [{voxel_size_x:.4f}, {voxel_size_y:.4f}, {voxel_size_z:.4f}] mm")
    
    # Normalize to get unit direction vectors
    read_dir = col0 / voxel_size_x if voxel_size_x > 0 else col0
    phase_dir = col1 / voxel_size_y if voxel_size_y > 0 else col1
    slice_dir = col2 / voxel_size_z if voxel_size_z > 0 else col2
    
    # Position is the translation (position of first voxel)
    position = translation.copy()

    # Convert from RAS (NIfTI) to LPS (ISMRMRD/DICOM)
    # Flip X and Y coordinates
    print("🔄 Converting from RAS to LPS orientation (flipping X and Y)...")
    position[0] = -position[0]
    position[1] = -position[1]
    
    read_dir[0] = -read_dir[0]
    read_dir[1] = -read_dir[1]
    
    phase_dir[0] = -phase_dir[0]
    phase_dir[1] = -phase_dir[1]
    
    slice_dir[0] = -slice_dir[0]
    slice_dir[1] = -slice_dir[1]
    
    print(f"   Position: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}] mm")
    print(f"   Read direction:  [{read_dir[0]:.4f}, {read_dir[1]:.4f}, {read_dir[2]:.4f}]")
    print(f"   Phase direction: [{phase_dir[0]:.4f}, {phase_dir[1]:.4f}, {phase_dir[2]:.4f}]")
    print(f"   Slice direction: [{slice_dir[0]:.4f}, {slice_dir[1]:.4f}, {slice_dir[2]:.4f}]")
    
    return {
        'position': position.tolist(),
        'read_dir': read_dir.tolist(),
        'phase_dir': phase_dir.tolist(),
        'slice_dir': slice_dir.tolist(),
        'voxel_size': [voxel_size_x, voxel_size_y, voxel_size_z]
    }


def extract_metadata_from_filename(nifti_path):
    """Extract patient and series metadata from NIfTI filename"""
    filename = os.path.basename(nifti_path)
    print(f"🏷️ Extracting metadata from filename: {filename}")
    
    # Default metadata
    metadata = {
        'config': 'openrecon',
        'enable_measurements': True,
        'enable_reporting': True,
        'confidence_threshold': 0.5,
        'PatientName': 'TEST^PATIENT',
        'StudyDescription': 'OPENRECON TEST',
        'SeriesDescription': 'TEST_SERIES',
        'PixelSpacing': [0.8, 0.8],
        'SliceThickness': 0.8,
        'PatientID': 'TESTPAT001',
        'SeriesNumber': 1
    }
    
    # Try to parse filename format: Pat[PatientID]_Se[SeriesNumber]_Res[X]_[Y]_Spac[Z].nii.gz
    if filename.startswith('Pat') and '_Se' in filename:
        try:
            # Remove either .nii.gz or .nii extension flexibly
            base_name = re.sub(r'\.nii(\.gz)?$', '', filename, flags=re.IGNORECASE)
            parts = base_name.split('_')
            for part in parts:
                if part.startswith('Pat'):
                    patient_id = part[3:]  # Remove 'Pat' prefix
                    metadata['PatientID'] = patient_id
                    metadata['PatientName'] = f'PATIENT^{patient_id}'
                elif part.startswith('Se'):
                    series_num = int(part[2:])  # Remove 'Se' prefix
                    metadata['SeriesNumber'] = series_num
                elif part.startswith('Res'):
                    # Next part should be the Y resolution
                    idx = parts.index(part)
                    if idx + 1 < len(parts):
                        x_res = float(part[3:])  # Remove 'Res' prefix
                        y_res = float(parts[idx + 1])
                        metadata['PixelSpacing'] = [x_res, y_res]
                elif part.startswith('Spac'):
                    slice_thickness = float(part[4:])  # Remove 'Spac' prefix
                    metadata['SliceThickness'] = slice_thickness
            
            print(f"✅ Parsed metadata from filename:")
            print(f"   Patient ID: {metadata['PatientID']}")
            print(f"   Series: {metadata['SeriesNumber']}")
            print(f"   Resolution: {metadata['PixelSpacing']}")
            print(f"   Slice thickness: {metadata['SliceThickness']}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not parse filename completely: {e}")
            print("   Using default metadata values")
    
    return metadata


def convert_nifti_to_ismrmrd(nifti_path, output_path=None):
    """Convert NIfTI file to ISMRMRD format"""
    
    if not os.path.exists(nifti_path):
        raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")
    
    print(f"🔄 Converting NIfTI to ISMRMRD format")
    print(f"   Input: {nifti_path}")
    
    # Load NIfTI data
    print("📖 Loading NIfTI file...")
    nii = nib.load(nifti_path)
    data = nii.get_fdata()
    affine = nii.affine
    
    print(f"📐 Original data shape: {data.shape}")
    print(f"🔢 Value range: {data.min():.2f} - {data.max():.2f}")
    print(f"📊 Data type: {data.dtype}")
    
    # Extract orientation information from affine matrix
    orientation_info = extract_orientation_from_affine(affine, data.shape)
    
    # Normalize data to reasonable range for medical imaging
    if data.max() > 4095:  # If values are very high, normalize
        data = (data / data.max()) * 4095
        print(f"🔧 Normalized data to range: {data.min():.2f} - {data.max():.2f}")
    
    # Ensure we have 3D data
    if len(data.shape) == 2:
        data = data[:, :, np.newaxis]
        print(f"📝 Expanded 2D to 3D: {data.shape}")
    elif len(data.shape) == 4:
        data = data[:, :, :, 0]  # Take first volume
        print(f"📝 Reduced 4D to 3D: {data.shape}")
    
    # Create ISMRMRD Image object
    print("🏗️ Creating ISMRMRD Image object...")
    
    # For magnitude/T2W images, keep as real float32 data
    if np.iscomplexobj(data):
        # If complex, take magnitude
        ismrmrd_data = np.abs(data).astype(np.float32)
        print("🔧 Converted complex data to magnitude (float32)")
    else:
        # Keep as real float32
        ismrmrd_data = data.astype(np.float32)
    
    print(f"📊 ISMRMRD data type: {ismrmrd_data.dtype}")
    print(f"📐 NIfTI data shape: {ismrmrd_data.shape}")
    
    # Create ISMRMRD image - no transpose, keep data as-is
    # Let ISMRMRD handle storage format internally
    try:
        ismrmrd_image = ismrmrd.Image.from_array(ismrmrd_data, transpose=False)
    except TypeError:
        # Older versions don't support transpose parameter
        ismrmrd_image = ismrmrd.Image.from_array(ismrmrd_data)
    
    print(f"📐 ISMRMRD image data shape: {ismrmrd_image.data.shape}")
    
    # Set basic image properties
    if hasattr(ismrmrd_image, 'image_type'):
        ismrmrd_image.image_type = IMTYPE_MAGNITUDE if 'IMTYPE_MAGNITUDE' in globals() else 1
    if hasattr(ismrmrd_image, 'image_series_index'):
        ismrmrd_image.image_series_index = 1
    if hasattr(ismrmrd_image, 'image_index'):
        ismrmrd_image.image_index = 0
    
    # Extract metadata from filename
    metadata = extract_metadata_from_filename(nifti_path)
    
    # Add orientation information to metadata
    metadata['position'] = orientation_info['position']
    metadata['read_dir'] = orientation_info['read_dir']
    metadata['phase_dir'] = orientation_info['phase_dir']
    metadata['slice_dir'] = orientation_info['slice_dir']
    
    # Update pixel spacing from actual affine-derived voxel sizes
    voxel_size = orientation_info['voxel_size']
    print(f"🔧 Voxel spacing from affine matrix: {voxel_size}")
    
    print(f"📏 Original data shape: {ismrmrd_data.shape}")
    print(f"📏 ISMRMRD image.data shape: {ismrmrd_image.data.shape}")
    
    # Check if ISMRMRD image has matrix_size attribute
    if hasattr(ismrmrd_image, 'matrix_size'):
        print(f"📏 ISMRMRD matrix_size attribute: {ismrmrd_image.matrix_size}")
    
    # CRITICAL: ISMRMRD stores data in reversed/transposed order (column-major Fortran order)
    # Original data: [624, 512, 416] but ISMRMRD reads it as [416, 512, 624]
    # So we need to reverse the FOV array to match: [Z_fov, Y_fov, X_fov]
    # Correction: Based on testing, the order should be [Y_fov, Z_fov, X_fov]
    field_of_view = [
        ismrmrd_data.shape[1] * voxel_size[1],  # Y
        ismrmrd_data.shape[2] * voxel_size[2],  # Z
        ismrmrd_data.shape[0] * voxel_size[0]   # X
    ]
    
    metadata['PixelSpacing'] = [voxel_size[0], voxel_size[1]]
    metadata['SliceThickness'] = voxel_size[2]
    metadata['field_of_view'] = field_of_view
    
    print(f"📏 Voxel spacing: [{voxel_size[0]}, {voxel_size[1]}, {voxel_size[2]}] mm")
    print(f"📏 Field of view (reversed for ISMRMRD): {field_of_view} mm")
    print(f"📏 Target matrix: {ismrmrd_data.shape[0]} x {ismrmrd_data.shape[1]} x {ismrmrd_data.shape[2]}")
    # print(f"📏 ISMRMRD stores as: 416 x 512 x 624")
    print(f"📏 Expected voxel: [{field_of_view[0]/ismrmrd_data.shape[1]:.8f}, {field_of_view[1]/ismrmrd_data.shape[2]:.8f}, {field_of_view[2]/ismrmrd_data.shape[0]:.8f}]")
    
    # Set image metadata
    if hasattr(ismrmrd_image, 'meta'):
        ismrmrd_image.meta = metadata
    
    # Set field_of_view on the image header if available
    if hasattr(ismrmrd_image, 'field_of_view'):
        ismrmrd_image.field_of_view[:] = field_of_view
    
    # Set position and orientation on the image header if available
    if hasattr(ismrmrd_image, 'position'):
        ismrmrd_image.position[:] = orientation_info['position']
        print(f"✅ Set image position: {orientation_info['position']}")
    
    if hasattr(ismrmrd_image, 'read_dir'):
        ismrmrd_image.read_dir[:] = orientation_info['read_dir']
        print(f"✅ Set read direction: {orientation_info['read_dir']}")
    
    if hasattr(ismrmrd_image, 'phase_dir'):
        ismrmrd_image.phase_dir[:] = orientation_info['phase_dir']
        print(f"✅ Set phase direction: {orientation_info['phase_dir']}")
    
    if hasattr(ismrmrd_image, 'slice_dir'):
        ismrmrd_image.slice_dir[:] = orientation_info['slice_dir']
        print(f"✅ Set slice direction: {orientation_info['slice_dir']}")
    
    # Create XML metadata string for ISMRMRD
    meta_obj = ismrmrd.Meta()
    for key, value in metadata.items():
        if isinstance(value, (list, tuple)):
            meta_obj[key] = list(value)
        else:
            meta_obj[key] = str(value)
    
    meta_obj['DataRole'] = 'Image'
    meta_obj['ImageProcessingHistory'] = ['NIfTI_CONVERSION']
    meta_obj['Keep_image_geometry'] = 1
    meta_obj['orientation_extracted'] = 'true'
    
    if hasattr(ismrmrd_image, 'attribute_string'):
        ismrmrd_image.attribute_string = meta_obj.serialize()
    
    print(f"✅ Successfully created ISMRMRD Image")
    print(f"   Data shape: {ismrmrd_image.data.shape}")
    print(f"   Data type: {ismrmrd_image.data.dtype}")
    
    # Save to file if requested
    if output_path:
        print(f"💾 Saving to: {output_path}")
        
        # Remove existing file if it exists to avoid corruption
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"🗑️  Removed existing file: {output_path}")
        
        try:
            # Check if we have the real ismrmrd module (not mock)
            if hasattr(ismrmrd, 'Dataset'):
                print(f"📝 Creating proper ISMRMRD Dataset file...")
                
                # Ensure parent directory exists
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                
                # Create ISMRMRD Dataset (this creates proper structure)
                mrdDset = ismrmrd.Dataset(output_path, 'dataset')
                mrdDset._file.require_group('dataset')
                
                # Create XML header with proper metadata
                print(f"📋 Creating ISMRMRD XML header...")
                mrdHead = ismrmrd.xsd.ismrmrdHeader()
                
                # Study Information
                mrdHead.studyInformation = ismrmrd.xsd.studyInformationType()
                mrdHead.studyInformation.studyDescription = metadata.get('StudyDescription', 'NIFTI_CONVERSION')
                
                # Patient Information
                mrdHead.subjectInformation = ismrmrd.xsd.subjectInformationType()
                mrdHead.subjectInformation.patientName = metadata.get('PatientName', 'TEST^PATIENT')
                mrdHead.subjectInformation.patientID = metadata.get('PatientID', 'TEST001')
                
                # Acquisition System Information
                mrdHead.acquisitionSystemInformation = ismrmrd.xsd.acquisitionSystemInformationType()
                mrdHead.acquisitionSystemInformation.systemVendor = 'NIfTI_Converter'
                mrdHead.acquisitionSystemInformation.systemModel = 'Virtual'
                mrdHead.acquisitionSystemInformation.institutionName = 'Test'
                
                # Encoding information
                encoding = ismrmrd.xsd.encodingType()
                encoding.trajectory = ismrmrd.xsd.trajectoryType.CARTESIAN
                
                # Get voxel size and FOV from metadata
                voxel_size = orientation_info['voxel_size']
                
                # Encoded space (acquisition space)
                encoding.encodedSpace = ismrmrd.xsd.encodingSpaceType()
                encoding.encodedSpace.matrixSize = ismrmrd.xsd.matrixSizeType()
                encoding.encodedSpace.matrixSize.x = int(ismrmrd_data.shape[0])
                encoding.encodedSpace.matrixSize.y = int(ismrmrd_data.shape[1])
                encoding.encodedSpace.matrixSize.z = int(ismrmrd_data.shape[2])
                
                encoding.encodedSpace.fieldOfView_mm = ismrmrd.xsd.fieldOfViewMm()
                encoding.encodedSpace.fieldOfView_mm.x = float(ismrmrd_data.shape[0] * voxel_size[0])
                encoding.encodedSpace.fieldOfView_mm.y = float(ismrmrd_data.shape[1] * voxel_size[1])
                encoding.encodedSpace.fieldOfView_mm.z = float(ismrmrd_data.shape[2] * voxel_size[2])
                
                # Recon space (same as encoded for NIfTI conversion)
                encoding.reconSpace = ismrmrd.xsd.encodingSpaceType()
                encoding.reconSpace.matrixSize = ismrmrd.xsd.matrixSizeType()
                encoding.reconSpace.matrixSize.x = int(ismrmrd_data.shape[0])
                encoding.reconSpace.matrixSize.y = int(ismrmrd_data.shape[1])
                encoding.reconSpace.matrixSize.z = int(ismrmrd_data.shape[2])
                
                encoding.reconSpace.fieldOfView_mm = ismrmrd.xsd.fieldOfViewMm()
                encoding.reconSpace.fieldOfView_mm.x = float(ismrmrd_data.shape[0] * voxel_size[0])
                encoding.reconSpace.fieldOfView_mm.y = float(ismrmrd_data.shape[1] * voxel_size[1])
                encoding.reconSpace.fieldOfView_mm.z = float(ismrmrd_data.shape[2] * voxel_size[2])
                
                # Encoding limits
                encoding.encodingLimits = ismrmrd.xsd.encodingLimitsType()
                
                mrdHead.encoding.append(encoding)
                
                # Sequence parameters
                mrdHead.sequenceParameters = ismrmrd.xsd.sequenceParametersType()
                mrdHead.sequenceParameters.TR = [1.0]
                mrdHead.sequenceParameters.TE = [1.0]
                
                print(f"   Matrix size: {ismrmrd_data.shape[0]} x {ismrmrd_data.shape[1]} x {ismrmrd_data.shape[2]}")
                print(f"   FOV: {encoding.encodedSpace.fieldOfView_mm.x:.2f} x {encoding.encodedSpace.fieldOfView_mm.y:.2f} x {encoding.encodedSpace.fieldOfView_mm.z:.2f} mm")
                print(f"   Voxel size: {voxel_size[0]:.4f} x {voxel_size[1]:.4f} x {voxel_size[2]:.4f} mm")
                
                # Write XML header
                mrdDset.write_xml_header(mrdHead.toXML('utf-8'))
                print(f"✅ Written XML header")
                
                # Create image with proper metadata
                tmpMeta = ismrmrd.Meta()
                for key, value in metadata.items():
                    if isinstance(value, (list, tuple)):
                        tmpMeta[key] = list(value)
                    else:
                        tmpMeta[key] = str(value)
                
                tmpMeta['DataRole'] = 'Image'
                tmpMeta['ImageProcessingHistory'] = ['NIFTI_CONVERSION']
                tmpMeta['Keep_image_geometry'] = 1
                
                ismrmrd_image.attribute_string = tmpMeta.serialize()
                
                # Set image series index
                ismrmrd_image.image_series_index = metadata.get('SeriesNumber', 1)
                
                # Write each slice as a separate image
                print(f"💾 Writing {ismrmrd_data.shape[2]} slices as separate images...")
                
                for slice_idx in range(ismrmrd_data.shape[2]):
                    # Extract 2D slice [X, Y]
                    slice_data = ismrmrd_data[:, :, slice_idx].astype(np.float32)
                    
                    # Create ISMRMRD image for this slice
                    try:
                        slice_image = ismrmrd.Image.from_array(slice_data, transpose=False)
                    except TypeError:
                        slice_image = ismrmrd.Image.from_array(slice_data)
                    
                    # Set image properties
                    slice_image.image_type = IMTYPE_MAGNITUDE if 'IMTYPE_MAGNITUDE' in globals() else 1
                    slice_image.image_series_index = metadata.get('SeriesNumber', 1)
                    slice_image.image_index = slice_idx
                    
                    # Calculate position for this slice
                    # Position of slice = position of first slice + (slice_idx * slice_spacing * slice_direction)
                    slice_position = (
                        np.array(orientation_info['position']) + 
                        slice_idx * voxel_size[2] * np.array(orientation_info['slice_dir'])
                    )
                    
                    # Set position and orientation on the slice
                    if hasattr(slice_image, 'position'):
                        slice_image.position[:] = slice_position.tolist()
                    
                    if hasattr(slice_image, 'read_dir'):
                        slice_image.read_dir[:] = orientation_info['read_dir']
                    
                    if hasattr(slice_image, 'phase_dir'):
                        slice_image.phase_dir[:] = orientation_info['phase_dir']
                    
                    if hasattr(slice_image, 'slice_dir'):
                        slice_image.slice_dir[:] = orientation_info['slice_dir']
                    
                    # Set field_of_view for 2D slice (only X and Y, Z is single slice)
                    # Correction: Based on testing, the order should be [Y_fov, X_fov, Thickness]
                    slice_fov = [
                        ismrmrd_data.shape[1] * voxel_size[1],
                        ismrmrd_data.shape[0] * voxel_size[0],
                        voxel_size[2]  # Single slice thickness
                    ]
                    
                    if hasattr(slice_image, 'field_of_view'):
                        slice_image.field_of_view[:] = slice_fov
                    
                    # Create metadata for this slice
                    slice_meta = ismrmrd.Meta()
                    for key, value in metadata.items():
                        if isinstance(value, (list, tuple)):
                            slice_meta[key] = list(value)
                        else:
                            slice_meta[key] = str(value)
                    
                    slice_meta['DataRole'] = 'Image'
                    slice_meta['ImageProcessingHistory'] = ['NIFTI_CONVERSION']
                    slice_meta['Keep_image_geometry'] = 1
                    slice_meta['slice_number'] = slice_idx
                    slice_meta['position'] = slice_position.tolist()
                    
                    slice_image.attribute_string = slice_meta.serialize()
                    
                    # Set slice and phase index
                    slice_image.slice = slice_idx
                    slice_image.phase = 0
                    slice_image.acquisition_time_stamp = 0

                    # Write slice to dataset
                    # Use series index as key to group all slices in one series
                    series_index = metadata.get('SeriesNumber', 1)
                    mrdDset.append_image("image_%d" % series_index, slice_image)
                    
                    if (slice_idx + 1) % 50 == 0 or slice_idx == ismrmrd_data.shape[2] - 1:
                        print(f"   Written {slice_idx + 1}/{ismrmrd_data.shape[2]} slices...")
                
                # Close dataset
                mrdDset.close()
                
                print(f"✅ Saved {ismrmrd_data.shape[2]} slices to {output_path}")
            else:
                raise ImportError("ismrmrd.Dataset not available - cannot create proper ISMRMRD file")
            
        except (ImportError, AttributeError, Exception) as e:
            print(f"❌ Could not save as ISMRMRD file: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    return ismrmrd_image, metadata


def main():
    """Main function to test the converter"""
    print("🧪 NIfTI to ISMRMRD Converter")
    print("=" * 50)
    
    # CLI arguments
    parser = argparse.ArgumentParser(description="Convert a NIfTI file to ISMRMRD format for OpenRecon testing")
    parser.add_argument(
        "-i", "--input",
        dest="nifti_file",
        help="Path to the NIfTI file to convert, e.g. Pat[PatientID]_Se[SeriesNumber]_Res[X]_[Y]_Spac[Z].nii.gz",
        default="Pat[PatientID]_Se[SeriesNumber]_Res[X]_[Y]_Spac[Z].nii.gz"
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_path",
        help="Optional output path for serialized ISMRMRD data (pickle)",
        default="test_ismrmrd_output.h5"
    )
    args = parser.parse_args()

    # Resolve inputs
    nifti_file = args.nifti_file
    output_path = args.output_path
    
    if not os.path.exists(nifti_file):
        print(f"❌ Test file not found: {nifti_file}")
        print("   Please check the file path")
        return False
    
    try:
        # Convert NIfTI to ISMRMRD
        print(f"➡️  Using input: {nifti_file}")
        print(f"➡️  Output path: {output_path}")
        ismrmrd_image, metadata = convert_nifti_to_ismrmrd(nifti_file, output_path)
        
        print("\n📋 Conversion Summary:")
        print(f"   Input file: {nifti_file}")
        print(f"   Original 3D volume shape: {ismrmrd_image.data.shape}")
        print(f"   Number of 2D slices saved: {ismrmrd_image.data.shape[2]}")
        print(f"   Each slice shape: [{ismrmrd_image.data.shape[0]}, {ismrmrd_image.data.shape[1]}]")
        print(f"   Patient ID: {metadata.get('PatientID', 'Unknown')}")
        print(f"   Series: {metadata.get('SeriesNumber', 'Unknown')}")
        print(f"   PixelSpacing: {metadata.get('PixelSpacing', 'Unknown')}")
        print(f"   SliceThickness: {metadata.get('SliceThickness', 'Unknown')}")
        print(f"\n🧭 Orientation Information:")
        print(f"   First slice position: {metadata.get('position', 'Unknown')}")
        print(f"   Read direction: {metadata.get('read_dir', 'Unknown')}")
        print(f"   Phase direction: {metadata.get('phase_dir', 'Unknown')}")
        print(f"   Slice direction: {metadata.get('slice_dir', 'Unknown')}")
        print(f"   Slice spacing: {metadata.get('SliceThickness', 'Unknown')} mm")
        
        return ismrmrd_image, metadata
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = main()
    if result:
        print("\n🎉 Conversion completed successfully!")
    else:
        print("\n💥 Conversion failed!")
