# MRD Helper functions
import ismrmrd
import re
import logging
import base64

def update_img_header_from_raw(imgHead, rawHead):
    """Populate ImageHeader fields from AcquisitionHeader"""

    if rawHead is None:
        return imgHead

    # # These fields are not translated from the raw header, but filled in
    # # during image creation by from_array
    # imgHead.data_type            = 
    # imgHead.matrix_size          = 
    # imgHead.channels             = 

    # # This is mandatory, but must be filled in from the XML header, 
    # # not from the acquisition header
    # imgHead.field_of_view        = 

    imgHead.version                = rawHead.version
    imgHead.flags                  = rawHead.flags
    imgHead.measurement_uid        = rawHead.measurement_uid

    imgHead.position               = rawHead.position
    imgHead.read_dir               = rawHead.read_dir
    imgHead.phase_dir              = rawHead.phase_dir
    imgHead.slice_dir              = rawHead.slice_dir
    imgHead.patient_table_position = rawHead.patient_table_position

    imgHead.average                = rawHead.idx.average
    imgHead.slice                  = rawHead.idx.slice
    imgHead.contrast               = rawHead.idx.contrast
    imgHead.phase                  = rawHead.idx.phase
    imgHead.repetition             = rawHead.idx.repetition
    imgHead.set                    = rawHead.idx.set

    imgHead.acquisition_time_stamp = rawHead.acquisition_time_stamp
    imgHead.physiology_time_stamp  = rawHead.physiology_time_stamp

    # Defaults, to be updated by the user
    imgHead.image_type             = ismrmrd.IMTYPE_MAGNITUDE
    imgHead.image_index            = 1
    imgHead.image_series_index     = 0

    imgHead.user_float             = rawHead.user_float
    imgHead.user_int               = rawHead.user_int

    return imgHead


def get_userParameterLong_value(metadata, name):
    """Get a value from MRD Header userParameterLong (returns None if key not found)"""
    for param in metadata.userParameters.userParameterLong:
        if param.name == name:
            return int(param.value)
    return None

def get_userParameterDouble_value(metadata, name):
    """Get a value from MRD Header userParameterDouble (returns None if key not found)"""
    for param in metadata.userParameters.userParameterDouble:
        if param.name == name:
            return float(param.value)
    return None

def get_userParameterString_value(metadata, name):
    """Get a value from MRD Header userParameterDouble (returns None if key not found)"""
    for param in metadata.userParameters.userParameterDouble:
        if param.name == name:
            return float(param.value)
    return None

def get_userParameterBase64_value(metadata, name):
    """Get a value from MRD Header userParameterBase64 (returns None if key not found)"""
    for param in metadata.userParameters.userParameterBase64:
        if param.name == name:
            return base64.b64decode(param.value).decode('utf-8')
    return None

def get_meta_value(meta, key):
    """Get a value from MRD Meta Attributes (returns None if key not found)"""
    if key in meta.keys():
        return meta[key]
    else:
        return None

def extract_minihead_bool_param(miniHead, name):
    """Extract a bool parameter from the serialized text of the ICE MiniHeader"""
    val = extract_minihead_param(miniHead, name, 'ParamBool')

    if val is None:
        return False
    elif val.strip('" ').lower() == 'true'.lower():
        return True
    else:
        return False

def extract_minihead_long_param(miniHead, name):
    """Extract a long parameter from the serialized text of the ICE MiniHeader"""
    val = extract_minihead_param(miniHead, name, 'ParamLong')

    if val is None:
        return int(0)
    else:
        return int(val)

def extract_minihead_double_param(miniHead, name):
    """Extract a double parameter from the serialized text of the ICE MiniHeader"""
    val = extract_minihead_param(miniHead, name, 'ParamDouble')

    if val is None:
        return float(0)
    else:
        return float(val)

def extract_minihead_string_param(miniHead, name):
    """Extract a string parameter from the serialized text of the ICE MiniHeader"""
    val = extract_minihead_param(miniHead, name, 'ParamString')

    return val.strip(' "')

def extract_minihead_param(miniHead, name, strType):
    """Extract a string parameter from the serialized text of the ICE MiniHeader"""
    expr = r'(?<=<' + strType + r'."' + name + r'">)\s*[^}]*\s*'
    res = re.search(expr, miniHead)

    if res is None:
        return None

    # Strip off beginning '{' and whitespace, then split on newlines
    values = res.group(0).strip('{\n ').split('\n')

    # Lines beginning with <> are properties -- ignore them
    values = [val for val in values if bool(re.search(r'^\s*<\w+>', val)) is False]

    if len(values) != 1:
        return None
    else:
        return values[0]

def meta_fix_imagedirs(meta, header):
    """Add image orientation directions to MetaAttributes if not already present"""
    if meta.get('ImageRowDir') is None:
        meta['ImageRowDir'] = ["{:.18f}".format(header.read_dir[0]), "{:.18f}".format(header.read_dir[1]), "{:.18f}".format(header.read_dir[2])]

    if meta.get('ImageColumnDir') is None:
        meta['ImageColumnDir'] = ["{:.18f}".format(header.phase_dir[0]), "{:.18f}".format(header.phase_dir[1]), "{:.18f}".format(header.phase_dir[2])]

def check_metadata(metadata):
    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier
    try:
        # Disabled due to incompatibility between PyXB and Python 3.8:
        # https://github.com/pabigot/pyxb/issues/123
        # # logging.info("Metadata: \n%s", metadata.toxml('utf-8'))
        logging.info("Incoming dataset contains %d encodings", len(metadata.encoding))
        logging.info("First encoding is of type '%s', with a field of view of (%s x %s x %s)mm^3 and a matrix size of (%s x %s x %s)",
            metadata.encoding[0].trajectory,
            metadata.encoding[0].encodedSpace.matrixSize.x,
            metadata.encoding[0].encodedSpace.matrixSize.y,
            metadata.encoding[0].encodedSpace.matrixSize.z,
            metadata.encoding[0].encodedSpace.fieldOfView_mm.x,
            metadata.encoding[0].encodedSpace.fieldOfView_mm.y,
            metadata.encoding[0].encodedSpace.fieldOfView_mm.z)
    except:
        logging.info("Improperly formatted metadata: \n%s", metadata)

def create_roi(x, y, rgb = (1, 0, 0), thickness = 1, style = 0, visibility = 1):
    """
    Create an MRD-formatted ROI
        Parameters:
            - x (1D ndarray)     : x coordinates in units of pixels, with (0,0) at the top left
            - y (1D ndarray)     : y coordinates in units of pixels, matching the length of x
            - rgb (3 item tuple) : Colour as an (red, green, blue) tuple normalized to 1
            - thickness (float)  : Line thickness
            - style (int)        : Line style (0 = solid, 1 = dashed)
            - visibility (int)   : Line visibility (0 = false, 1 = true)
        Returns:
            - roi (string list)  : MRD-formatted ROI, intended to be stored as a MetaAttribute
                                   with field name starting with "ROI_"
    """
    xy = [(x[i], y[i]) for i in range(0, len(x))]  # List of (x,y) tuples

    roi = []
    roi.append('%f' % rgb[0])
    roi.append('%f' % rgb[1])
    roi.append('%f' % rgb[2])
    roi.append('%f' % thickness)
    roi.append('%f' % style)
    roi.append('%f' % visibility)

    for i in range(0, len(xy)):
        roi.append('%f' % xy[i][0])
        roi.append('%f' % xy[i][1])

    return roi
