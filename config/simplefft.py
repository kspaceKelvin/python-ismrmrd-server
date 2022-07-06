
import ctypes
import ismrmrd
import logging
import numpy as np
import numpy.fft as fft
import os

import config
import mrdhelper

SETTINGS = config.Settings(True, False, False,
                           [ismrmrd.ACQ_IS_PHASECORR_DATA,
                            ismrmrd.ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA],
                           [ismrmrd.ACQ_LAST_IN_SLICE],
                           [],
                           True)


def process_acquisition(group, index, connection, metadata, debug_folder):

    # Format data into single [cha RO PE] array
    data = [acquisition.data for acquisition in group]
    data = np.stack(data, axis=-1)

    logging.debug("Raw data is size %s" % (data.shape,))
    np.save(os.path.join(debug_folder, "raw" + str(index) + ".npy"), data)

    # Remove readout oversampling
    data = fft.ifft(data, axis=1)
    data = np.delete(data, np.arange(int(data.shape[1]*1/4),int(data.shape[1]*3/4)), 1)
    data = fft.fft( data, axis=1)

    logging.debug("Raw data is size after readout oversampling removal %s" % (data.shape,))
    np.save(os.path.join(debug_folder, "raw" + str(index) + "NoOS.npy"), data)

    # Fourier Transform
    data = fft.fftshift( data, axes=(1, 2))
    data = fft.ifft2(    data, axes=(1, 2))
    data = fft.ifftshift(data, axes=(1, 2))

    # Sum of squares coil combination
    data = np.abs(data)
    data = np.square(data)
    data = np.sum(data, axis=0)
    data = np.sqrt(data)

    logging.debug("Image data is size %s" % (data.shape,))
    np.save(os.path.join(debug_folder, "img" + str(index) + ".npy"), data)

    # Normalize and convert to int16
    data *= 32767/data.max()
    data = np.around(data)
    data = data.astype(np.int16)

    # Remove readout oversampling
    offset = int((data.shape[0] - metadata.encoding[0].reconSpace.matrixSize.x)/2)
    data = data[offset:offset+metadata.encoding[0].reconSpace.matrixSize.x,:]

    # Remove phase oversampling
    offset = int((data.shape[1] - metadata.encoding[0].reconSpace.matrixSize.y)/2)
    data = data[:,offset:offset+metadata.encoding[0].reconSpace.matrixSize.y]

    logging.debug("Image without oversampling is size %s" % (data.shape,))
    np.save(os.path.join(debug_folder, "img" + str(index) + "Crop.npy"), data)

    # Format as ISMRMRD image data
    # data has shape [RO PE], i.e. [x y].
    # from_array() should be called with 'transpose=False' to avoid warnings, and when called
    # with this option, can take input as: [cha z y x], [z y x], or [y x]
    image = ismrmrd.Image.from_array(data.transpose(), acquisition=group[0], transpose=False)
    image.image_index = 1

    # Set field of view
    image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x),
                           ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                           ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))

    # Set ISMRMRD Meta Attributes
    meta = ismrmrd.Meta({'DataRole':               'Image',
                         'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                         'WindowCenter':           '16384',
                         'WindowWidth':            '32768'})

    mrdhelper.meta_fix_imagedirs(meta, image.getHead())
    xml = meta.serialize()
    logging.debug("Image MetaAttributes: %s", xml)
    logging.debug("Image data has %d elements", image.data.size)

    image.attribute_string = xml
    return image
