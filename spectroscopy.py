import ismrmrd
import os
import itertools
import logging
import traceback
import numpy as np
import numpy.fft as fft
import xml.dom.minidom
import base64
import ctypes
import re
import mrdhelper
import constants
from time import perf_counter
import matplotlib.pyplot as plt
import tempfile
import subprocess
from PIL import Image

# Folder for debug output files
debugFolder = "/tmp/share/debug"
 
def process(connection, config, metadata):
    logging.info("Config: \n%s", config)
 
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
 
    # Continuously parse incoming data parsed from MRD messages
    currentSeries = 0
    acqGroup = []
    imgGroup = []
    waveformGroup = []
    try:
        for item in connection:
            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):
                # Accumulate all imaging readouts in a group
                if (not item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT) and
                    not item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION) and
                    not item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA) and
                    not item.is_flag_set(ismrmrd.ACQ_IS_NAVIGATION_DATA)):
                    acqGroup.append(item)
 
                # When this criteria is met, run process_raw() on the accumulated
                # data, which returns images that are sent back to the client.
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE):
                    logging.info("Processing a group of k-space data")
                    image = process_raw(acqGroup, connection, config, metadata)
                    connection.send_image(image)
                    acqGroup = []
 
            # ----------------------------------------------------------
            # Image data messages
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Image):
                # Only process magnitude images -- send phase images back without modification (fallback for images with unknown type)
                if (item.image_type is ismrmrd.IMTYPE_MAGNITUDE) or (item.image_type == 0):
                    imgGroup.append(item)
                else:
                    tmpMeta = ismrmrd.Meta.deserialize(item.attribute_string)
                    tmpMeta['Keep_image_geometry']    = 1
                    item.attribute_string = tmpMeta.serialize()
 
                    connection.send_image(item)
                    continue
 
                # When this criteria is met, run process_group() on the accumulated
                # data, which returns images that are sent back to the client.
                # e.g. when the series number changes:
                if item.image_series_index != currentSeries:
                    logging.info("Processing a group of images because series index changed to %d", item.image_series_index)
                    currentSeries = item.image_series_index
                    image = process_image(imgGroup, connection, config, metadata)
                    connection.send_image(image)
                    imgGroup = []
 
            # ----------------------------------------------------------
            # Waveform data messages
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Waveform):
                waveformGroup.append(item)
 
            elif item is None:
                break
 
            else:
                logging.error("Unsupported data  type %s", type(item).__name__)
 
        # Extract raw ECG waveform data. Basic sorting to make sure that data
        # is time-ordered, but no additional checking for missing data.
        # ecgData has shape (5 x timepoints)
        if len(waveformGroup) > 0:
            waveformGroup.sort(key = lambda item: item.time_stamp)
            ecgData = [item.data for item in waveformGroup if item.waveform_id == 0]
            ecgData = np.concatenate(ecgData,1)
 
        # Process any remaining groups of raw or image data.  This can
        # happen if the trigger condition for these groups are not met.
        # This is also a fallback for handling image data, as the last
        # image in a series is typically not separately flagged.
        if len(acqGroup) > 0:
            logging.info("Processing a group of k-space data (untriggered)")
            image = process_raw(acqGroup, connection, config, metadata)
            connection.send_image(image)
            acqGroup = []
 
        if len(imgGroup) > 0:
            logging.info("Processing a group of images (untriggered)")
            image = process_image(imgGroup, connection, config, metadata)
            connection.send_image(image)
            imgGroup = []
 
    except Exception as e:
        logging.error(traceback.format_exc())
        connection.send_logging(constants.MRD_LOGGING_ERROR, traceback.format_exc())
 
    finally:
        connection.send_close()
 

def process_raw(group, connection, config, metadata):
    # Format data into a [cha RO ave lin seg] array
    nAve = int(metadata.encoding[0].encodingLimits.average.maximum                - metadata.encoding[0].encodingLimits.average.minimum)                + 1
    nLin = int(metadata.encoding[0].encodingLimits.kspace_encoding_step_1.maximum - metadata.encoding[0].encodingLimits.kspace_encoding_step_1.minimum) + 1
    nSeg = int(metadata.encoding[0].encodingLimits.segment.maximum                - metadata.encoding[0].encodingLimits.segment.minimum)                + 1
    nRO  = mrdhelper.get_userParameterLong_value(metadata, 'SpecVectorSize')

    if nRO is None:
        nRO = int((group[0].data.shape[1] - group[0].discard_pre - group[0].discard_post) / 2)  # 2x readout oversampling
        logging.warning("Could not find SpecVectorSize in header -- using size %d from data", nRO)

    # 2x readout oversampling
    nRO = nRO * 2

    logging.info("MRD header: %d averages, %d lines, %d segments" % (nAve, nLin, nSeg))

    aves = [acquisition.idx.average              for acquisition in group]
    lins = [acquisition.idx.kspace_encode_step_1 for acquisition in group]
    segs = [acquisition.idx.segment              for acquisition in group]

    # Sanity check on nLin
    if nLin > np.max(lins)*5:
        logging.warning("Overriding implusible nLin from %d to %d" % (nLin, np.max(lins)+1))
        nLin = np.max(lins)+1

    data = np.zeros((group[0].data.shape[0], 
                     nRO,
                     nAve, 
                     nLin, 
                     nSeg), 
                    group[0].data.dtype)

    for acq, ave, lin, seg in zip(group, aves, lins, segs):
        data[:,:,ave,lin,seg] = acq.data[:,acq.discard_pre:(acq.data.shape[1]-acq.discard_post)]

    logging.info("Incoming raw spectroscopy data is shape %s" % (data.shape,))

    # Select coil with the best SNR
    indBestCoil = np.argmax(np.mean(np.abs(data[:,:,0:9,0,0]),axis=(1,2)))
    data = data[np.newaxis,indBestCoil,...]

    # Remove readout oversampling
    data = fft.fft(data, axis=1)
    data = np.delete(data, np.arange(int(data.shape[1]*1/4),int(data.shape[1]*3/4)), axis=1)
    data = fft.ifft( data, axis=1)

    # Match Siemens convention of complex conjugate representation
    data = np.conj(data)

    # Match Siemens data scaling
    data = data * 2**25

    # Image recon for spectroscopic imaging
    if (data.shape[3] > 1) and (data.shape[4] > 1):
        data = fft.fftshift( data, axes=(3, 4))
        data = fft.ifft2(    data, axes=(3, 4))
        data = fft.ifftshift(data, axes=(3, 4))

    # Combine averages
    data = np.mean(data, axis=2, keepdims=True)

    # Collapse into shape [RO lin seg]
    data = np.squeeze(data)

    # Send data back as complex singles
    data = data.astype(np.complex64)

    # Transpose to shape [SEG LIN COL]
    data = data.transpose()
    logging.info("Outgoing spectroscopy data is shape %s" % (data.shape,))

    # Create new MRD instance for the processed image
    # from_array() should be called with 'transpose=False' to avoid warnings, and when called
    # with this option, can take input as: [cha z y x], [z y x], [y x], or [x]
    # For spectroscopy data, dimensions are: [z y t], i.e. [SEG LIN COL] (PAR would be 3D)
    tmpImg = ismrmrd.Image.from_array(data, transpose=False)
 
    # Set the header information
    tmpImg.setHead(mrdhelper.update_img_header_from_raw(tmpImg.getHead(), group[0].getHead()))

    if data.ndim > 1:
        # 2D spectroscopic imaging
        tmpImg.field_of_view = (ctypes.c_float(data.shape[2]/data.shape[1]*metadata.encoding[0].reconSpace.fieldOfView_mm.y),
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y),
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
    else:
        # Single voxel
        tmpImg.field_of_view = (ctypes.c_float(data.shape[0]*metadata.encoding[0].reconSpace.fieldOfView_mm.y/2),
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y/2),
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))

    tmpImg.image_index   = 1
    tmpImg.image_type    = ismrmrd.IMTYPE_COMPLEX
    tmpImg.flags         = 2**5   # IMAGE_LAST_IN_AVERAGE
 
    logging.info("Outgoing spectroscopy data is field_of_view %s, %s, %s" % (np.double(tmpImg.field_of_view[0]), np.double(tmpImg.field_of_view[1]), np.double(tmpImg.field_of_view[2])))
    logging.info("Outgoing spectroscopy data is matrix_size   %s, %s, %s" % (tmpImg.getHead().matrix_size[0], tmpImg.getHead().matrix_size[1], tmpImg.getHead().matrix_size[2]))

    # Set ISMRMRD Meta Attributes
    tmpMeta = ismrmrd.Meta()
    tmpMeta['DataRole']                            = 'Spectroscopy'
    tmpMeta['ImageProcessingHistory']              = ['FIRE', 'SPECTRO', 'PYTHON']
    tmpMeta['Keep_image_geometry']                 = 1
    tmpMeta['SiemensControl_SpectroData']          = ['bool', 'true']
    #tmpMeta['SiemensControl_Suffix4DataFileName']  = ['string', '-1_1_1_1_1_1']

    # Change dwell time to account for removal of readout oversampling
    dwellTime = mrdhelper.get_userParameterDouble_value(metadata, 'DwellTime_0')  # in ms

    if dwellTime is None:
        logging.error("Could not find DwellTime_0 in MRD header")
    else:
        logging.info("Found acquisition dwell time from header: " + str(dwellTime*1000))
        tmpMeta['SiemensDicom_RealDwellTime']         = ['int', str(int(dwellTime*1000*2))]
 
    xml = tmpMeta.serialize()
    logging.debug("Image MetaAttributes: %s", xml)
    tmpImg.attribute_string = xml

    images = [tmpImg]

    roiImg = plot_spectra(tmpImg, connection, config, metadata)
    if roiImg is not None:
        images.append(roiImg)

    return images
 

def process_image(images, connection, config, metadata):
    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")
 
    logging.debug("Processing data with %d images of type %s", len(images), ismrmrd.get_dtype_from_data_type(images[0].data_type))

    roiImg = plot_spectra(images[0], connection, config, metadata)

    return roiImg

def plot_spectra(img, connection, config, metadata):
    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    # For 2D trajectories, create both an ROI vectorized figure and a PNG figure
    if (img.data.shape[1] > 1) or (img.data.shape[2] > 1):
        return None

    # ---------- Send back an MRD image with the spectrum as ROIs ----------
    roiMeta = ismrmrd.Meta()
    roiMeta['SequenceDescriptionAdditional']  = 'SPEC_PLOT'
    roiMeta['WindowCenter']                   = '16384'
    roiMeta['WindowWidth']                    = '32768'
    roiMeta['Keep_image_geometry']            = 1
    roiMeta['InternalSend']                   = ['bool', 'true']

    # Size of blank dummy image
    imgX = 1280
    imgY = 1280

    # Fraction of image to use
    widthX = 0.9
    heightY = 0.2
    offsetY = 0.4

    # Image coordinates have origin at top left
    y =  fft.fftshift(fft.fft(np.squeeze(img.data), axis=0))
    y = np.abs(y)
    y = ((1-(y/np.max(y)))*heightY+offsetY) * imgY
    x = np.linspace(-widthX/2, widthX/2, len(y))*imgX + imgX/2

    # Plot options
    rgb = (1,0,0)  # Red, green, blue color -- normalized to 1
    thickness  = 1 # Line thickness
    style      = 0 # Line style (0 = solid, 1 = dashed)
    visibility = 1 # Line visibility (0 = false, 1 = true)

    roiMeta['ROI_spectra'] = mrdhelper.create_roi(x, y, rgb, thickness, style, visibility)

    # Additional ROI for x-axis
    xAxis = np.array((-widthX/2, widthX/2))*imgX + imgX/2
    yAxis = (np.array((offsetY,offsetY))+heightY) * imgY
    roiMeta['ROI_axis']    = mrdhelper.create_roi(xAxis, yAxis, (0,0,1), thickness, style, visibility)

    # Blank MRD image
    roiImg = ismrmrd.Image.from_array(np.zeros((imgX, imgY), dtype=np.int16), transpose=False)

    # Set the header information
    tmpHead = img.getHead()
    tmpHead.data_type     = roiImg.data_type
    tmpHead.field_of_view = (ctypes.c_float( imgX), ctypes.c_float( imgY), ctypes.c_float(10))  # Dummy FOV because the spectroscopy FOV isn't appropriate
    tmpHead.matrix_size   = (ctypes.c_ushort(imgX), ctypes.c_ushort(imgY), ctypes.c_ushort(1))
    roiImg.setHead(tmpHead)

    roiImg.image_index        = 1
    roiImg.image_series_index = 1
    roiImg.attribute_string   = roiMeta.serialize()

    return roiImg