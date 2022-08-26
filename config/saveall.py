import ismrmrd
import logging
import numpy as np
import os

import config

# TODO Eventually get this to save waveforms as well
SETTINGS = config.Settings(True, True, False,
                           [],
                           [ismrmrd.ACQ_LAST_IN_SLICE],
                           [ismrmrd.IMTYPE_MAGNITUDE,
                            ismrmrd.IMTYPE_PHASE,
                            ismrmrd.IMTYPE_REAL,
                            ismrmrd.IMTYPE_IMAG,
                            ismrmrd.IMTYPE_COMPLEX],
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
    logging.info("Raw data is size %s" % (data.shape,))

    debug_dir = os.path.join(config.SHAREDIR, 'debug', 'saveall')
    try:
        os.makedirs(debug_dir)
    except FileExistsError:
        pass
    np.save(os.path.join(debug_dir, "raw" + str(index) + ".npy"), data)

    return group



def process_image(images, index, connection, metadata):

    logging.debug("Processing data with %d images of type %s", len(images), ismrmrd.get_dtype_from_data_type(images[0].data_type))

    # Note: The MRD Image class stores data as [cha z y x]

    # Extract image data into a 5D array of size [img cha z y x]
    data = np.stack([img.data                              for img in images])
    head = [img.getHead()                                  for img in images]
    meta = [ismrmrd.Meta.deserialize(img.attribute_string) for img in images]

    # Reformat data to [y x z cha img], i.e. [row col] for the first two dimensions
    data = data.transpose((3, 4, 2, 1, 0))
    logging.debug("Original image data is size %s" % (data.shape,))

    debug_dir = os.path.join(config.SHAREDIR, 'debug', 'saveall')
    try:
        os.makedirs(debug_dir)
    except FileExistsError:
        pass
    np.save(os.path.join(debug_dir, "image" + str(index) + ".npy"), data)

    return images
