
import ctypes
import base64
import ismrmrd
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import xml.dom.minidom

import config
import mrdhelper

SETTINGS = config.Settings(True, True, False,
                           [ismrmrd.ACQ_IS_NOISE_MEASUREMENT,
                            ismrmrd.ACQ_IS_PARALLEL_CALIBRATION,
                            ismrmrd.ACQ_IS_PHASECORR_DATA],
                           [ismrmrd.ACQ_LAST_IN_SLICE],
                           [ismrmrd.IMTYPE_MAGNITUDE, 0],
                           ['slice'])


def process_acquisition(group, index, connection, metadata):

    # Format data into single [cha PE RO phs] array
    lin = [acquisition.idx.kspace_encode_step_1 for acquisition in group]
    phs = [acquisition.idx.phase                for acquisition in group]

    # Use the zero-padded matrix size
    data = np.zeros((group[0].data.shape[0],
                     metadata.encoding[0].encodedSpace.matrixSize.y,
                     metadata.encoding[0].encodedSpace.matrixSize.x,
                     max(phs)+1),
                    group[0].data.dtype)

    rawHead = [None]*(max(phs)+1)

    for acq, lin, phs in zip(group, lin, phs):
        if (lin < data.shape[1]) and (phs < data.shape[3]):
            # TODO: Account for asymmetric echo in a better way
            data[:,lin,-acq.data.shape[1]:,phs] = acq.data

            # center line of k-space is encoded in user[5]
            if (rawHead[phs] is None) or (np.abs(acq.getHead().idx.kspace_encode_step_1 - acq.getHead().idx.user[5]) < np.abs(rawHead[phs].idx.kspace_encode_step_1 - rawHead[phs].idx.user[5])):
                rawHead[phs] = acq.getHead()

    # Flip matrix in RO/PE to be consistent with ICE
    data = np.flip(data, (1, 2))
    logging.debug("Raw data is size %s" % (data.shape,))

    debug_dir = os.path.join(config.SHAREDIR, 'debug', 'rgb')
    try:
        os.makedirs(debug_dir)
    except FileExistsError:
        pass
    np.save(os.path.join(debug_dir, "raw" + str(index) + ".npy"), data)

    # Fourier Transform
    data = fft.fftshift( data, axes=(1, 2))
    data = fft.ifft2(    data, axes=(1, 2))
    data = fft.ifftshift(data, axes=(1, 2))

    # Sum of squares coil combination
    # Data will be [PE RO phs]
    data = np.abs(data)
    data = np.square(data)
    data = np.sum(data, axis=0)
    data = np.sqrt(data)

    logging.debug("Image data is size %s" % (data.shape,))
    np.save(os.path.join(debug_dir, "img" + str(index) + ".npy"), data)

    # Normalize and convert to int16
    data *= 32767/data.max()
    data = np.around(data)
    data = data.astype(np.int16)

    # Remove readout oversampling
    offset = int((data.shape[1] - metadata.encoding[0].reconSpace.matrixSize.x)/2)
    data = data[:,offset:offset+metadata.encoding[0].reconSpace.matrixSize.x]

    # Remove phase oversampling
    offset = int((data.shape[0] - metadata.encoding[0].reconSpace.matrixSize.y)/2)
    data = data[offset:offset+metadata.encoding[0].reconSpace.matrixSize.y,:]

    logging.debug("Image without oversampling is size %s" % (data.shape,))
    np.save(os.path.join(debug_dir, "img" + str(index) + "Crop.npy"), data)

    # Format as ISMRMRD image data
    imagesOut = []
    for phs in range(data.shape[2]):
        # Create new MRD instance for the processed image
        # data has shape [PE RO phs], i.e. [y x].
        # from_array() should be called with 'transpose=False' to avoid warnings, and when called
        # with this option, can take input as: [cha z y x], [z y x], or [y x]
        tmpImg = ismrmrd.Image.from_array(data[...,phs], transpose=False)

        # Set the header information
        tmpImg.setHead(mrdhelper.update_img_header_from_raw(tmpImg.getHead(), rawHead[phs]))
        tmpImg.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x),
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y),
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
        tmpImg.image_index = phs

        # Set ISMRMRD Meta Attributes
        tmpMeta = ismrmrd.Meta()
        tmpMeta['DataRole']               = 'Image'
        tmpMeta['ImageProcessingHistory'] = ['FIRE', 'PYTHON']
        tmpMeta['WindowCenter']           = '16384'
        tmpMeta['WindowWidth']            = '32768'
        tmpMeta['Keep_image_geometry']    = 1

        xml = tmpMeta.serialize()
        logging.debug("Image MetaAttributes: %s", xml)
        tmpImg.attribute_string = xml
        imagesOut.append(tmpImg)

    # Call process_image() to create RGB images
    imagesOut = process_image(imagesOut, 0, connection, metadata)

    return imagesOut



def process_image(images, index, connection, metadata):

    logging.debug("Processing data with %d images of type %s", len(images), ismrmrd.get_dtype_from_data_type(images[0].data_type))

    # Note: The MRD Image class stores data as [cha z y x]

    # Extract image data into a 5D array of size [img cha z y x]
    data = np.stack([img.data                              for img in images])
    head = [img.getHead()                                  for img in images]
    meta = [ismrmrd.Meta.deserialize(img.attribute_string) for img in images]

    # Reformat data to [y x z cha img], i.e. [row col] for the first two dimensions
    data = data.transpose((3, 4, 2, 1, 0))

    # Display MetaAttributes for first image
    logging.debug("MetaAttributes[0]: %s", ismrmrd.Meta.serialize(meta[0]))

    # Optional serialization of ICE MiniHeader
    if 'IceMiniHead' in meta[0]:
        logging.debug("IceMiniHead[0]: %s", base64.b64decode(meta[0]['IceMiniHead']).decode('utf-8'))

    logging.debug("Original image data is size %s" % (data.shape,))

    debug_dir = os.path.join(config.SHAREDIR, 'debug', 'rgb')
    try:
        os.makedirs(debug_dir)
    except FileExistsError:
        pass
    np.save(os.path.join(debug_dir, "imgOrig.npy"), data)

    if data.shape[3] != 1:
        logging.error("Multi-channel data is not supported")
        return []

    # Normalize to (0.0, 1.0) as expected by get_cmap()
    data = data.astype(float)
    data -= data.min()
    data *= 1/data.max()

    # Apply colormap
    cmap = plt.get_cmap('jet')
    rgb = cmap(data)

    # Remove alpha channel
    # Resulting shape is [row col z rgb img]
    rgb = rgb[...,0:-1]
    rgb = rgb.transpose((0, 1, 2, 5, 4, 3))
    rgb = np.squeeze(rgb, 5)

    # MRD RGB images must be uint16 in range (0, 255)
    rgb *= 255
    data = rgb.astype(np.uint16)
    np.save(os.path.join(debug_dir, "imgRGB.npy"), data)

    # Re-slice back into 2D images
    imagesOut = [None] * data.shape[-1]
    for iImg in range(data.shape[-1]):
        # Create new MRD instance for the inverted image
        # Transpose from convenience shape of [y x z cha] to MRD Image shape of [cha z y x]
        # from_array() should be called with 'transpose=False' to avoid warnings, and when called
        # with this option, can take input as: [cha z y x], [z y x], or [y x]
        imagesOut[iImg] = ismrmrd.Image.from_array(data[...,iImg].transpose((3, 2, 0, 1)), transpose=False)
        data_type = imagesOut[iImg].data_type

        # Create a copy of the original fixed header and update the data_type
        # (we changed it to int16 from all other types)
        newHeader = head[iImg]
        newHeader.data_type = data_type

        # Set RGB parameters
        newHeader.image_type = 6  # To be defined as ismrmrd.IMTYPE_RGB
        newHeader.channels   = 3  # RGB "channels".  This is set by from_array, but need to be explicit as we're copying the old header instead

        imagesOut[iImg].setHead(newHeader)

        # Create a copy of the original ISMRMRD Meta attributes and update
        tmpMeta = meta[iImg]
        tmpMeta['DataRole']                       = 'Image'
        tmpMeta['ImageProcessingHistory']         = ['PYTHON', 'RGB']
        tmpMeta['SequenceDescriptionAdditional']  = 'FIRE_RGB'
        tmpMeta['Keep_image_geometry']            = 1

        # Add image orientation directions to MetaAttributes if not already present
        mrdhelper.meta_fix_imagedirs(tmpMeta, newHeader)
        metaXml = tmpMeta.serialize()
        logging.debug("Image MetaAttributes: %s", xml.dom.minidom.parseString(metaXml).toprettyxml())
        logging.debug("Image data has %d elements", imagesOut[iImg].data.size)

        imagesOut[iImg].attribute_string = metaXml

    return imagesOut
