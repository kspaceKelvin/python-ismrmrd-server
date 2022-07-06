import base64
import ismrmrd
import logging
import numpy as np
import os
from time import perf_counter

import config
import constants
import mrdhelper

# TODO Prevent issuing a warning due to ignoring magnitude data
SETTINGS = config.Settings(False, True, False, [], [], [ismrmrd.IMTYPE_PHASE], True)


def process_image(images, index, connection, metadata, debug_folder):

    # Start timer
    tic = perf_counter()

    logging.debug("Processing data with %d images of type %s", len(images), ismrmrd.get_dtype_from_data_type(images[0].data_type))

    # Display MetaAttributes for first image
    tmpMeta = ismrmrd.Meta.deserialize(images[0].attribute_string)
    logging.debug("MetaAttributes[0]: %s", ismrmrd.Meta.serialize(tmpMeta))

    # Optional serialization of ICE MiniHeader
    if 'IceMiniHead' in tmpMeta:
        logging.debug("IceMiniHead[0]: %s", base64.b64decode(tmpMeta['IceMiniHead']).decode('utf-8'))

    # Extract some indices for the images
    slice = [img.slice for img in images]
    phase = [img.phase for img in images]

    # Process each group of venc directions separately
    unique_venc_dir = np.unique([ismrmrd.Meta.deserialize(img.attribute_string)['FlowDirDisplay'] for img in images])

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
        data = np.zeros((images[0].data.shape[2], images[0].data.shape[3], max(slice)+1, max(phase)+1), images[0].data.dtype)
        head = [[None]*(max(phase)+1) for _ in range(max(slice)+1)]
        meta = [[None]*(max(phase)+1) for _ in range(max(slice)+1)]

        for img, sli, phs in zip(images, slice, phase):
            if ismrmrd.Meta.deserialize(img.attribute_string)['FlowDirDisplay'] == venc_dir:
                # print("sli phs", sli, phs)
                data[:,:,sli,phs] = img.data
                head[sli][phs]    = img.getHead()
                meta[sli][phs]    = ismrmrd.Meta.deserialize(img.attribute_string)

        logging.debug("Phase data with venc encoding %s is size %s" % (venc_dir, data.shape,))
        np.save(os.path.join(debug_folder, "data_" + venc_dir + ".npy"), data)

        # Mask out data with high mean temporal diff
        threshold = 250
        data_meandiff = np.mean(np.abs(np.diff(data,3)),3)
        data_masked = data
        data_masked[(data_meandiff > threshold)] = 2048
        np.save(os.path.join(debug_folder, "data_masked_" + venc_dir + ".npy"), data_masked)

        # Normalize and convert to int16
        data_masked = (data_masked.astype(np.float64) - 2048)*32767/2048
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
                tmpMeta['WindowCenter']           = '16384'
                tmpMeta['WindowWidth']            = '32768'
                tmpMeta['Keep_image_geometry']    = 1

                mrdhelper.meta_fix_imagedirs(tmpMeta, tmpHead)

                xml = tmpMeta.serialize()
                logging.debug("Image MetaAttributes: %s", xml)
                tmpImg.attribute_string = xml
                imagesOut.append(tmpImg)

        last_series += 1
    return imagesOut
