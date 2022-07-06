import importlib
import ismrmrd
import logging
import numpy as np
import os
import traceback
from collections import namedtuple

import constants
import mrdhelper

# Root folder for debug output files
# Put contents written by a specific config to a named sub-directory
DEBUGFOLDER = "/tmp/share/debug"

DEFAULTCONFIG = 'null'

# Control what information the individual configs need to specify
# - "keep_acq": bool: Store acquisition data in RAM, call module's process_acquisition() function when appropriate
# - "keep_image": bool: Store image data in RAM, call module's process_image() function when appropriate
# - "keep_waveform": bool: Store waveform data in RAM; not currently utilised
# - "acq_ignore": list: Specify ismrmrd flags relating to acquisition data that should be ignored during data load
# - "acq_trigger": list: Specify ismrmrd flags relating to acquisition data that should trigger the module's process_acquisition() function once completed
# - "image_select": list: Specify ismrmrd flags relating to image data that should be retained during data load
# - "image_trigger": bool: Controls whether module's process_image() function should be triggered for individual images
Settings = namedtuple('Settings', ['keep_acq', 'keep_image', 'keep_waveform', 'acq_ignore', 'acq_trigger', 'image_select', 'image_trigger'])

# TODO Hypothetically, could have "image_trigger" be not a bool, but a set of conditions upon which process_image() will be called
# For instance, if magnitude & phase data were to come in interleaved, but you wanted to retain only magnitude data,
#   and not load the phase data / data across all Images, you'd need to indicate that you want to trigger based on a
#   series index change but not a data type change.


def process(connection, config, metadata):

    logging.info("Config: %s", config)
    try:
        module = importlib.import_module('config.' + config)
        logging.info("Opened config %s", config)
    except ImportError as exc:
        if exc.__traceback__.tb_next is not None:
            logging.error('Failure attempting to open config %s:', config)
            logging.error(str(exc))
            raise
        logging.info("Unrecognised config '%s'; falling back to '%s'", config, DEFAULTCONFIG)
        module = importlib.import_module('config.' + DEFAULTCONFIG)
    settings = module.SETTINGS
    logging.debug('Settings for selected config: %s', settings)

    mrdhelper.check_metadata(metadata)

    # Create folder, if necessary
    config_debug_folder = os.path.join(DEBUGFOLDER, config)
    if not os.path.exists(config_debug_folder):
        os.makedirs(config_debug_folder)
        logging.info("Created folder " + config_debug_folder + " for debug output files")
    else:
        logging.debug("Debugging folder \"" + config_debug_folder + "\" already exists")

    # Continuously parse incoming data parsed from MRD messages
    # Only store in RAM those data that are of interest to the config
    acquisition_group = []
    acquisition_index = 0
    image_group = []
    image_series_index = 0
    image_series_type = None
    waveform_group = []

    acq_ignored = 0
    acq_wrongtype = 0
    image_ignored = 0
    image_wrongtype = 0
    waveform_ignored = 0

    # TODO What about the scenario where one wants to collect all waveform data before processing
    #   k-space data?
    # - Settings.acq_trigger can prevent calling the recon function until all data have been imported,
    #   but still need a way to get the waveform data to the config function
    try:
        for item in connection:

            if isinstance(item, ismrmrd.Acquisition):

                if settings.keep_acq:
                    if not any(item.is_flag_set(flag) for flag in settings.acq_ignore):
                        acquisition_group.append(item)

                        # When this criteria is met, run process_acquisition() on the accumulated
                        # data, which returns images that are sent back to the client.
                        if any(item.is_flag_set(flag) for flag in settings.acq_trigger):
                            logging.info("Processing a group of k-space data (explicitly triggered)")
                            image = module.process_acquisition(acquisition_group, acquisition_index, connection, metadata, config_debug_folder)
                            if image:
                                connection.send_image(image)
                            acquisition_group = []
                            acquisition_index += 1
                    else:
                        acq_wrongtype += 1
                else:
                    acq_ignored += 1

            elif isinstance(item, ismrmrd.Image):

                if settings.keep_image:
                    if any(item.is_flag_set(flag) for flag in settings.image_select):

                        if image_series_type is None:
                            image_series_type = item.image_type
                        # When this criteria is met, run process_image() on the accumulated
                        # data, which returns images that are sent back to the client.
                        # e.g. when the series number changes:
                        if settings.image_trigger and (item.image_type != image_series_type or item.image_series_index != image_series_index):
                            logging.info("Processing a group of images; previously was <type %d index %d>; now <type %d index %d>",
                                image_series_type, image_series_index, item.image_type, item.image_series_index)
                            image = module.process_image(image_group, image_series_index, connection, metadata, config_debug_folder)
                            if image:
                                connection.send_image(image)
                            image_series_index = item.image_series_index
                            image_series_type = item.image_type
                            image_group = []
                        image_group.append(item)
                    else:
                        image_wrongtype += 1
                else:
                    image_ignored += 1

            elif isinstance(item, ismrmrd.Waveform):
                if settings.keep_waveform:
                    waveform_group.append(item)
                else:
                    waveform_ignored += 1

            elif item is None:
                break

            else:
                logging.error("Unsupported data type %s", type(item).__name__)

        # # Extract raw ECG waveform data. Basic sorting to make sure that data
        # # is time-ordered, but no additional checking for missing data.
        # # ecg_data has shape (5 x timepoints)
        # if waveform_group:
        #     waveform_group.sort(key = lambda item: item.time_stamp)
        #     ecg_data = [item.data for item in waveform_group if item.waveform_id == 0]
        #     ecg_data = np.concatenate(ecg_data, 1)

        # Process any remaining groups of raw or image data.  This can
        # happen if the trigger condition for these groups are not met.
        # This is also a fallback for handling image data, as the last
        # image in a series is typically not separately flagged.
        if acquisition_group:
            logging.info("Processing a group of k-space data (end of data stream)")
            image = module.process_acquisition(acquisition_group, acquisition_index, connection, metadata, config_debug_folder)
            if image:
                connection.send_image(image)
            acquisition_group = []

        if image_group:
            logging.info("Processing a group of images (end of data stream)")
            image = module.process_image(image_group, image_series_index, connection, metadata, config_debug_folder)
            if image:
                connection.send_image(image)
            image_group = []

        for count, text in [(acq_ignored, "ismrmrd.Acquisition"),
                            (acq_wrongtype, "ismrmrd.Acquisition of ignored types"),
                            (image_ignored, "ismrmrd.Image"),
                            (image_wrongtype, "ismrmrd.Image of incompatible datatypes"),
                            (waveform_ignored, "ismrmrd.Waveform")]:
            if count:
                str_warn = "Received " + str(count) + " instances of " + text + ", which were ignored by this analysis"
                logging.warning(str_warn)
                connection.send_logging(constants.MRD_LOGGING_INFO, str_warn)

    except Exception as e:
        logging.error(traceback.format_exc())
        connection.send_logging(constants.MRD_LOGGING_ERROR, traceback.format_exc())

    finally:
        connection.send_close()
