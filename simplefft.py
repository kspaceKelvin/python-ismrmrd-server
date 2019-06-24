
import ismrmrd
import os
import itertools
import logging
import numpy as np
import numpy.fft as fft


def groups(iterable, predicate):
    group = []
    for item in iterable:
        group.append(item)

        if predicate(item):
            yield group
            group = []


def process(connection, config, params):
    logging.info("Processing connection.")
    logging.info("Config: \n%s", config.decode("utf-8"))
    logging.info("Params: \n%s", params.decode("utf-8"))

    for group in groups(connection, lambda acq: acq.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE)):
        image = process_group(group, config, params)

        logging.info("Sending image to client:\n%s", image)
        connection.send_image(image)


def process_group(group, config, params):

    # Folder for debug output files
    debugFolder = "/tmp/share/dependency"

    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.info("Created folder " + debugFolder + " for debug output files")

    # Format data into single [cha RO PE] array
    data = [acquisition.data for acquisition in group]
    data = np.stack(data, axis=-1)

    logging.info("Raw data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "raw.npy", data)

    # Fourier Transform
    data = fft.fftshift(data, axes=(1, 2))
    data = fft.ifft2(data)
    data = fft.ifftshift(data, axes=(1, 2))

    # Sum of squares coil combination
    data = np.abs(data)
    data = np.square(data)
    data = np.sum(data, axis=0)
    data = np.sqrt(data)

    logging.info("Image data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "img.npy", data)

    # Normalize and convert to int16
    data *= 32768/data.max()
    data = np.around(data)
    data = data.astype(np.int16)

    # Remove phase oversampling
    nRO = np.size(data,0);
    data = data[int(nRO/4):int(nRO*3/4),:]
    logging.info("Image without oversampling is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "img_crop.npy", data)

    # Format as ISMRMRD image data
    image = ismrmrd.Image.from_array(data, acquisition=group[0])
    image.image_index = 1

    # Set ISMRMRD Meta Attributes
    meta = ismrmrd.Meta({'GADGETRON_DataRole': 'Image',
                         'GADGETRON_ImageProcessingHistory': ['FIRE', 'PYTHON'],
                         'GADGETRON_WindowCenter': '16384',
                         'GADGETRON_WindowWidth': '32768'})
    xml = meta.serialize()
    logging.info("XML: %s", xml)
    logging.info("Image data has %d elements", image.data.size)

    image.attribute_string = xml
    return image


