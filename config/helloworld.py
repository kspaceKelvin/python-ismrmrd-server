import ismrmrd
import logging
import numpy as np
import os
import xml.dom.minidom

import config
import mrdhelper

SETTINGS = config.Settings(False, True, False, [], [], [ismrmrd.IMTYPE_MAGNITUDE], ['slice'])


def process_image(images_in, index, connection, metadata, debug_folder):

    logging.debug("Processing data with %d images of type %s", len(images_in), ismrmrd.get_dtype_from_data_type(images_in[0].data_type))

    # Note: The MRD Image class stores data as [cha z y x]

    # Extract image data into a 5D array of size [img cha z y x]
    data = np.stack([img.data                              for img in images_in])
    head = [img.getHead()                                  for img in images_in]
    meta = [ismrmrd.Meta.deserialize(img.attribute_string) for img in images_in]

    logging.debug("Original image data is size %s" % (data.shape,))
    np.save(os.path.join(debug_folder, "image" + str(index) + ".npy"), data)

    # Re-slice back into 2D images
    images_out = [None] * data.shape[0]
    for image_index in range(data.shape[0]):
        # Create new MRD instance for the image
        logging.info('Data shape for from_array(): ' + str(data[image_index].shape))
        images_out[image_index] = ismrmrd.Image.from_array(data[image_index], transpose=False)
        images_out[image_index].setHead(head[image_index])

        # Create a copy of the original ISMRMRD Meta attributes and update
        new_meta = meta[image_index]
        new_meta['DataRole']                      = 'Image'
        new_meta['ImageProcessingHistory']        = ['PYTHON', 'HELLOWORLD']
        new_meta['SequenceDescriptionAdditional'] = 'FIRE'
        new_meta['Keep_image_geometry']           = 1

        mrdhelper.meta_fix_imagedirs(new_meta, head[image_index])

        metaXml = new_meta.serialize()
        logging.debug("Image MetaAttributes: %s", xml.dom.minidom.parseString(metaXml).toprettyxml())
        logging.debug("Image data has %d elements", images_out[image_index].data.size)

        images_out[image_index].attribute_string = metaXml

    return images_out
