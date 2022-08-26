import ismrmrd
import logging
import numpy as np
import os

import config

SETTINGS = config.Settings(False, True, False, [], [],
                           [ismrmrd.IMTYPE_MAGNITUDE,
                            ismrmrd.IMTYPE_PHASE,
                            ismrmrd.IMTYPE_REAL,
                            ismrmrd.IMTYPE_IMAG,
                            ismrmrd.IMTYPE_COMPLEX],
                           ['slice'])


def process_image(images, index, connection, metadata):

    logging.debug("Processing data with %d images of type %s", len(images), ismrmrd.get_dtype_from_data_type(images[0].data_type))

    # Note: The MRD Image class stores data as [cha z y x]
    # Extract image data into a 5D array of size [img cha z y x]
    data = np.stack([img.data for img in images])

    # Reformat data to [x y z cha img]
    data = data.transpose((4, 3, 2, 1, 0)).squeeze()
    logging.debug("Image data is size %s" % (data.shape,))

    debug_dir = os.path.join(config.SHAREDIR, 'debug', 'saveimages')
    try:
        os.makedirs(debug_dir)
    except FileExistsError:
        pass
    np.save(os.path.join(debug_dir, "image" + str(index) + ".npy"), data)

    return images
