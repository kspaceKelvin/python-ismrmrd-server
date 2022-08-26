import ismrmrd
import logging
import numpy as np
import os

import config

SETTINGS = config.Settings(True, True, True,
                           [],
                           [ismrmrd.ACQ_LAST_IN_SLICE],
                           [0,
                            ismrmrd.IMTYPE_MAGNITUDE,
                            ismrmrd.IMTYPE_PHASE,
                            ismrmrd.IMTYPE_REAL,
                            ismrmrd.IMTYPE_IMAG,
                            ismrmrd.IMTYPE_COMPLEX],
                           ['measurement_uid', 'average', 'slice', 'contrast', 'phase', 'repetition', 'set', 'image_type'])
                            # Only separate process_image() calls for 'image_series_index'


def process_acquisition(group, index, connection, metadata):

    logging.info('Running process_acquisition() for index ' + str(index) + ': ' + str(len(group)) + ' entries in group')
    # if all(item.data.shape == group[0].data.shape for item in group[1:]):
    #     logging.info('All entries in group are of shape: ' + group[0].data.shape)
    # else:
    #     logging.info('Dimensions of entries in group:')
    #     for item in group:
    #         logging.info(item.data.shape)
    for index, item in enumerate(group):
        txt = str(index) + ': '
        hdr = item.getHead()
        for item in [hdr.flags, hdr.measurement_uid, hdr.scan_counter, hdr.number_of_samples, hdr.active_channels,
                     (hdr.idx.kspace_encode_step_1, hdr.idx.kspace_encode_step_2, hdr.idx.average, hdr.idx.slice, hdr.idx.contrast, hdr.idx.phase, hdr.idx.repetition, hdr.idx.set, hdr.idx.segment)]:
            txt += ' - ' + str(item)
        logging.info(txt)

    return []



def process_image(images, index, connection, metadata):

    logging.info('Running process_image() for index ' + str(index) + ': ' + str(len(images)) + ' entries in group')
    for index, item in enumerate(images):
        txt = str(index) + ': '
        hdr = item.getHead()
        for item in [hdr.flags, hdr.measurement_uid, tuple(hdr.matrix_size), hdr.channels, hdr.average, hdr.slice, hdr.contrast, hdr.phase, hdr.repetition, hdr.set, hdr.image_type, hdr.image_index]:
            txt += ' - ' + str(item)
        logging.info(txt)

    return []


# TODO Provide information about waveforms
