import ismrmrd
import os
import itertools
import logging
import traceback
import numpy as np
import numpy.fft as fft
import base64
import re
import mrdhelper
import constants
from time import perf_counter

# Folder for debug output files
debugFolder = "/tmp/share/debug"

def process(connection, config, mrdHeader):
    logging.info("Config: \n%s", config)

    # mrdHeader should be xml formatted MRD header, but may be a string
    # if it failed conversion earlier
    try:
        # Disabled due to incompatibility between PyXB and Python 3.8:
        # https://github.com/pabigot/pyxb/issues/123
        # # logging.info("MRD header: \n%s", mrdHeader.toxml('utf-8'))

        logging.info("Incoming dataset contains %d encodings", len(mrdHeader.encoding))
        logging.info("First encoding is of type '%s', with a matrix size of (%s x %s x %s) and a field of view of (%s x %s x %s)mm^3", 
            mrdHeader.encoding[0].trajectory, 
            mrdHeader.encoding[0].encodedSpace.matrixSize.x, 
            mrdHeader.encoding[0].encodedSpace.matrixSize.y, 
            mrdHeader.encoding[0].encodedSpace.matrixSize.z, 
            mrdHeader.encoding[0].encodedSpace.fieldOfView_mm.x, 
            mrdHeader.encoding[0].encodedSpace.fieldOfView_mm.y, 
            mrdHeader.encoding[0].encodedSpace.fieldOfView_mm.z)

    except:
        logging.info("Improperly formatted MRD header: \n%s", mrdHeader)

    # Continuously parse incoming data parsed from MRD messages
    imgGroup = []
    waveformGroup = []
    try:
        for item in connection:
            # ----------------------------------------------------------
            # Image data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Image):
                # Only process phase images
                if item.image_type is ismrmrd.IMTYPE_PHASE:
                    imgGroup.append(item)
                else:
                    connection.send_image(item)
                    continue

            # ----------------------------------------------------------
            # Waveform data messages
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Waveform):
                waveformGroup.append(item)

            # ----------------------------------------------------------
            # Ignore raw k-space data
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Acquisition):
                strWarn = "Received an ismrmrd.Acquisition which is ignored by this analysis"
                logging.warning(strWarn)
                connection.send_logging(constants.MRD_LOGGING_INFO, strWarn)

            elif item is None:
                break

            else:
                logging.error("Unsupported data type %s", type(item).__name__)

        # Extract raw ECG waveform data. Basic sorting to make sure that data 
        # is time-ordered, but no additional checking for missing data.
        # ecgData has shape (5 x timepoints)
        if len(waveformGroup) > 0:
            waveformGroup.sort(key = lambda item: item.time_stamp)
            ecgData = [item.data for item in waveformGroup if item.waveform_id == 0]
            if len(ecgData) > 0:
                ecgData = np.concatenate(ecgData,1)

        # Process any remaining groups of raw or image data.  This can 
        # happen if the trigger condition for these groups are not met.
        # This is also a fallback for handling image data, as the last
        # image in a series is typically not separately flagged.
        if len(imgGroup) > 0:
            logging.info("Processing a group of images (untriggered)")
            image = process_image(imgGroup, connection, config, mrdHeader)
            logging.debug("Sending images to client")
            connection.send_image(image)
            imgGroup = []

    except Exception as e:
        logging.error(traceback.format_exc())
        connection.send_logging(constants.MRD_LOGGING_ERROR, traceback.format_exc())

    finally:
        connection.send_close()

def process_image(imgGroup, connection, config, mrdHeader):
    if len(imgGroup) == 0:
        return []

    logging.info(f'-----------------------------------------------')
    logging.info(f'     process_image called with {len(imgGroup)} images')
    logging.info(f'-----------------------------------------------')

    # Start timer
    tic = perf_counter()

    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    logging.debug("Processing data with %d images of type %s", len(imgGroup), imgGroup[0].data.dtype)

    # Display MetaAttributes for first image
    tmpMeta = ismrmrd.Meta.deserialize(imgGroup[0].attribute_string)
    logging.debug("MetaAttributes[0]: %s", ismrmrd.Meta.serialize(tmpMeta))

    # Optional serialization of ICE MiniHeader
    if 'IceMiniHead' in tmpMeta:
        logging.debug("IceMiniHead[0]: %s", base64.b64decode(tmpMeta['IceMiniHead']).decode('utf-8'))

    # Extract some indices for the images
    slice = [img.slice for img in imgGroup]
    phase = [img.phase for img in imgGroup]

    # Process each group of venc directions separately
    unique_venc_dir = np.unique([ismrmrd.Meta.deserialize(img.attribute_string)['FlowDirDisplay'] for img in imgGroup])

    # Measure processing time
    toc = perf_counter()
    strProcessTime = "Total processing time: %.2f ms" % ((toc-tic)*1000.0)
    logging.info(strProcessTime)

    # Send this as a text message back to the client
    connection.send_logging(constants.MRD_LOGGING_INFO, strProcessTime)

    # Start the phase images at series 10.  When interpreted by FIRE, images
    # with the same image_series_index are kept in the same series, but the
    # absolute series number isn't used and can be arbitrary
    last_series = 10
    imagesOut = []
    for venc_dir in unique_venc_dir:
        # data array has dimensions [row col sli phs], i.e. [y x sli phs]
        # info lists has dimensions [sli phs]
        data = np.zeros((imgGroup[0].data.shape[2], imgGroup[0].data.shape[3], max(slice)+1, max(phase)+1), imgGroup[0].data.dtype)
        head = [[None]*(max(phase)+1) for _ in range(max(slice)+1)]
        meta = [[None]*(max(phase)+1) for _ in range(max(slice)+1)]

        for img, sli, phs in zip(imgGroup, slice, phase):
            if ismrmrd.Meta.deserialize(img.attribute_string)['FlowDirDisplay'] == venc_dir:
                # print("sli phs", sli, phs)
                data[:,:,sli,phs] = img.data
                head[sli][phs]    = img.getHead()
                meta[sli][phs]    = ismrmrd.Meta.deserialize(img.attribute_string)

        logging.debug("Phase data with venc encoding %s is size %s" % (venc_dir, data.shape,))
        np.save(debugFolder + "/" + "data_" + venc_dir + ".npy", data)

        # Mask out data with high mean temporal diff
        threshold = 250
        data_meandiff = np.mean(np.abs(np.diff(data,3)),3)
        data_masked = data
        data_masked[(data_meandiff > threshold)] = 2048
        np.save(debugFolder + "/" + "data_masked_" + venc_dir + ".npy", data_masked)

        # Determine max value (12 or 16 bit)
        BitsStored = 12
        if (mrdhelper.get_userParameterLong_value(mrdHeader, "BitsStored") is not None):
            BitsStored = mrdhelper.get_userParameterLong_value(mrdHeader, "BitsStored")
        maxVal = 2**BitsStored - 1

        # Normalize and convert to int16
        data_masked = (data_masked.astype(np.float64) - 2048)*maxVal/2048
        data_masked = np.around(data_masked).astype(np.int16)

        # Re-slice back into 2D images
        for sli in range(data_masked.shape[2]):
            for phs in range(data_masked.shape[3]):
                # Create new MRD instance for the processed image
                # data has shape [y x sli phs]
                # from_array() should be called with 'transpose=False' to avoid warnings, and when called
                # with this option, can take input as: [cha z y x], [z y x], or [y x]
                tmpImg = ismrmrd.Image.from_array(data_masked[...,sli,phs], transpose=False)

                # Set the header information
                tmpHead = head[sli][phs]
                tmpHead.data_type          = tmpImg.getHead().data_type
                tmpHead.image_index        = phs + sli*data_masked.shape[3]
                tmpHead.image_series_index = last_series
                tmpImg.setHead(tmpHead)

                # Set ISMRMRD Meta Attributes
                tmpMeta = meta[sli][phs]
                tmpMeta['DataRole']               = 'Image'
                tmpMeta['ImageProcessingHistory'] = ['FIRE', 'PYTHON']
                tmpMeta['WindowCenter']           = str((maxVal+1)/2)
                tmpMeta['WindowWidth']            = str((maxVal+1))
                tmpMeta['Keep_image_geometry']    = 1

                # Add image orientation directions to MetaAttributes if not already present
                if tmpMeta.get('ImageRowDir') is None:
                    tmpMeta['ImageRowDir'] = ["{:.18f}".format(tmpHead.read_dir[0]), "{:.18f}".format(tmpHead.read_dir[1]), "{:.18f}".format(tmpHead.read_dir[2])]

                if tmpMeta.get('ImageColumnDir') is None:
                    tmpMeta['ImageColumnDir'] = ["{:.18f}".format(tmpHead.phase_dir[0]), "{:.18f}".format(tmpHead.phase_dir[1]), "{:.18f}".format(tmpHead.phase_dir[2])]

                xml = tmpMeta.serialize()
                logging.debug("Image MetaAttributes: %s", xml)
                tmpImg.attribute_string = xml
                imagesOut.append(tmpImg)

        last_series += 1
    return imagesOut
