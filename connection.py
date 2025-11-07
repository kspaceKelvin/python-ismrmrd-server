import constants
import ismrmrd
import ctypes
import os
from datetime import datetime
import h5py
import random
import threading

import logging
import socket
import numpy as np

class Connection:
    def __init__(self, socket, savedata, savedataFile = "", savedataFolder = "", savedataGroup = "dataset"):
        self.savedata       = savedata
        self.savedataFile   = savedataFile
        self.savedataFolder = savedataFolder
        self.savedataGroup  = savedataGroup
        self.mrdFilePath    = None
        self.dset           = None
        self.socket         = socket
        self.is_exhausted   = False
        self.sentAcqs       = 0
        self.sentImages     = 0
        self.sentWaveforms  = 0
        self.recvAcqs       = 0
        self.recvImages     = 0
        self.recvWaveforms  = 0
        self.lock           = threading.Lock()
        self.handlers       = {
            constants.MRD_MESSAGE_CONFIG_FILE:         self.read_config_file,
            constants.MRD_MESSAGE_CONFIG_TEXT:         self.read_config_text,
            constants.MRD_MESSAGE_METADATA_XML_TEXT:   self.read_metadata,
            constants.MRD_MESSAGE_CLOSE:               self.read_close,
            constants.MRD_MESSAGE_TEXT:                self.read_text,
            constants.MRD_MESSAGE_ISMRMRD_ACQUISITION: self.read_acquisition,
            constants.MRD_MESSAGE_ISMRMRD_WAVEFORM:    self.read_waveform,
            constants.MRD_MESSAGE_ISMRMRD_IMAGE:       self.read_image
        }

    def create_save_file(self):
        if self.savedata is True:
            # Create savedata folder, if necessary
            if ((self.savedataFolder) and (not os.path.exists(self.savedataFolder))):
                os.makedirs(self.savedataFolder)
                logging.debug("Created folder " + self.savedataFolder + " to save incoming data")

            if (self.savedataFile):
                self.mrdFilePath = self.savedataFile
            else:
                self.mrdFilePath = os.path.join(self.savedataFolder, "MRD_input_" + datetime.now().strftime("%Y-%m-%d-%H%M%S" + "_" + str(random.randint(0,100)) + ".h5"))

            # Create HDF5 file to store incoming MRD data
            logging.info("Incoming data will be saved to: '%s' in group '%s'", self.mrdFilePath, self.savedataGroup)
            self.dset = ismrmrd.Dataset(self.mrdFilePath, self.savedataGroup)
            self.dset._file.require_group(self.savedataGroup)

    def save_additional_config(self, configAdditionalText):
        if self.savedata is True:
            if self.dset is None:
                self.create_save_file()

            self.dset._file.require_group("dataset")
            dsetConfigAdditional = self.dset._dataset.require_dataset('configAdditional',shape=(1,), dtype=h5py.special_dtype(vlen=bytes))
            dsetConfigAdditional[0] = bytes(configAdditionalText, 'utf-8')

    def send_logging(self, level, contents):
        try:
            formatted_contents = "%s %s" % (level, contents)
        except:
            logging.warning("Unsupported logging level: " + level)
            formatted_contents = contents

        self.send_text(formatted_contents)

    def __iter__(self):
        while not self.is_exhausted:
            yield self.next()

    def __next__(self):
        return self.next()

    def read(self, nbytes):
        return self.socket.recv(nbytes, socket.MSG_WAITALL)

    def peek(self, nbytes):
        return self.socket.recv(nbytes, socket.MSG_PEEK)

    def next(self):
        with self.lock:
            id = self.read_mrd_message_identifier()

            if (self.is_exhausted == True):
                return

            handler = self.handlers.get(id, lambda: Connection.unknown_message_identifier(id))
            return handler()

    def shutdown_close(self):
        # Encapsulate shutdown in a try block because the socket may have
        # already been closed on the other side
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except:
            pass
        self.socket.close()
        logging.info("Socket closed")

    @staticmethod
    def unknown_message_identifier(identifier):
        logging.error("Received unknown message type: %d", identifier)
        raise StopIteration

    def read_mrd_message_identifier(self):
        try:
            identifier_bytes = self.read(constants.SIZEOF_MRD_MESSAGE_IDENTIFIER)
        except ConnectionResetError:
            logging.error("Connection closed unexpectedly")
            self.is_exhausted = True
            return

        if (len(identifier_bytes) == 0):
            self.is_exhausted = True
            return

        return constants.MrdMessageIdentifier.unpack(identifier_bytes)[0]

    def peek_mrd_message_identifier(self):
        try:
            identifier_bytes = self.peek(constants.SIZEOF_MRD_MESSAGE_IDENTIFIER)
        except ConnectionResetError:
            logging.error("Connection closed unexpectedly")
            self.is_exhausted = True
            return

        if (len(identifier_bytes) == 0):
            self.is_exhausted = True
            return

        return constants.MrdMessageIdentifier.unpack(identifier_bytes)[0]

    def read_mrd_message_length(self):
        length_bytes = self.read(constants.SIZEOF_MRD_MESSAGE_LENGTH)
        return constants.MrdMessageLength.unpack(length_bytes)[0]

    # ----- MRD_MESSAGE_CONFIG_FILE (1) ----------------------------------------
    # This message contains the file name of a configuration file used for 
    # image reconstruction/post-processing.  The file must exist on the server.
    # Message consists of:
    #   ID               (   2 bytes, unsigned short)
    #   Config file name (1024 bytes, char          )
    def send_config_file(self, filename):
        with self.lock:
            logging.info("--> Sending MRD_MESSAGE_CONFIG_FILE (1)")
            self.socket.send(constants.MrdMessageIdentifier.pack(constants.MRD_MESSAGE_CONFIG_FILE))
            self.socket.send(constants.MrdMessageConfigurationFile.pack(filename.encode()))

    def read_config_file(self):
        logging.info("<-- Received MRD_MESSAGE_CONFIG_FILE (1)")
        config_file_bytes = self.read(constants.SIZEOF_MRD_MESSAGE_CONFIGURATION_FILE)
        config_file = constants.MrdMessageConfigurationFile.unpack(config_file_bytes)[0]
        config_file = config_file.split(b'\x00',1)[0].decode('utf-8')  # Strip off null terminators in fixed 1024 size

        logging.debug("    " + config_file)
        if (config_file == "savedataonly"):
            logging.info("Save data, but no processing based on config")
            if self.savedata is True:
                logging.debug("Saving data is already enabled")
            else:
                self.savedata = True
                self.create_save_file()

        if self.savedata is True:
            if self.dset is None:
                self.create_save_file()

            self.dset._file.require_group("dataset")
            dsetConfigFile = self.dset._dataset.require_dataset('config_file',shape=(1,), dtype=h5py.special_dtype(vlen=bytes))
            dsetConfigFile[0] = bytes(config_file, 'utf-8')

        return config_file

    # ----- MRD_MESSAGE_CONFIG_TEXT (2) --------------------------------------
    # This message contains the configuration information (text contents) used 
    # for image reconstruction/post-processing.  Text is null-terminated.
    # Message consists of:
    #   ID               (   2 bytes, unsigned short)
    #   Length           (   4 bytes, uint32_t      )
    #   Config text data (  variable, char          )
    def send_config_text(self, contents):
        with self.lock:
            logging.info("--> Sending MRD_MESSAGE_CONFIG_TEXT (2)")
            self.socket.send(constants.MrdMessageIdentifier.pack(constants.MRD_MESSAGE_CONFIG_TEXT))
            contents_with_nul = '%s\0' % contents # Add null terminator
            self.socket.send(constants.MrdMessageLength.pack(len(contents_with_nul.encode())))
            self.socket.send(contents_with_nul.encode())

    def read_config_text(self):
        logging.info("<-- Received MRD_MESSAGE_CONFIG_TEXT (2)")
        length = self.read_mrd_message_length()
        config = self.read(length)
        config = config.split(b'\x00',1)[0].decode('utf-8')  # Strip off null teminator

        logging.debug("    " + config)

        if self.savedata is True:
            if self.dset is None:
                self.create_save_file()

            self.dset._file.require_group("dataset")
            dsetConfig = self.dset._dataset.require_dataset('config',shape=(1,), dtype=h5py.special_dtype(vlen=bytes))
            dsetConfig[0] = bytes(config, 'utf-8')

        return config

    # ----- MRD_MESSAGE_METADATA_XML_TEXT (3) -----------------------------------
    # This message contains the metadata for the entire dataset, formatted as
    # MRD XML flexible data header text.  Text is null-terminated.
    # Message consists of:
    #   ID               (   2 bytes, unsigned short)
    #   Length           (   4 bytes, uint32_t      )
    #   Text xml data    (  variable, char          )
    def send_metadata(self, contents):
        with self.lock:
            logging.info("--> Sending MRD_MESSAGE_METADATA_XML_TEXT (3)")
            self.socket.send(constants.MrdMessageIdentifier.pack(constants.MRD_MESSAGE_METADATA_XML_TEXT))
            contents_with_nul = '%s\0' % contents # Add null terminator
            self.socket.send(constants.MrdMessageLength.pack(len(contents_with_nul.encode())))
            self.socket.send(contents_with_nul.encode())

    def read_metadata(self):
        logging.info("<-- Received MRD_MESSAGE_METADATA_XML_TEXT (3)")
        length = self.read_mrd_message_length()
        metadata = self.read(length)
        metadata = metadata.split(b'\x00',1)[0].decode('utf-8')  # Strip off null teminator

        if self.savedata is True:
            if self.dset is None:
                self.create_save_file()

            logging.debug("    Saving XML header to file")
            self.dset.write_xml_header(bytes(metadata, 'utf-8'))

        return metadata

    # ----- MRD_MESSAGE_CLOSE (4) ----------------------------------------------
    # This message signals that all data has been sent (either from server or client).
    def send_close(self):
        with self.lock:
            logging.info("--> Sending MRD_MESSAGE_CLOSE (4)")
            self.socket.send(constants.MrdMessageIdentifier.pack(constants.MRD_MESSAGE_CLOSE))

    def read_close(self):
        logging.info("<-- Received MRD_MESSAGE_CLOSE (4)")
        logging.info("    Total received acquisitions: %5d", self.recvAcqs)
        logging.info("    Total received images:       %5d", self.recvImages)
        logging.info("    Total received waveforms:    %5d", self.recvWaveforms)
        logging.info("------------------------------------------")

        if self.savedata is True:
            if self.dset is None:
                self.create_save_file()

            logging.debug("Closing file %s", self.dset._file.filename)
            self.dset.close()
            self.dset = None

        self.is_exhausted = True
        return

    # ----- MRD_MESSAGE_TEXT (5) -----------------------------------
    # This message contains arbitrary text data.
    # Message consists of:
    #   ID               (   2 bytes, unsigned short)
    #   Length           (   4 bytes, uint32_t      )
    #   Text data        (  variable, char          )
    def send_text(self, contents):
        with self.lock:
            logging.info("--> Sending MRD_MESSAGE_TEXT (5)")
            logging.info("    %s", contents)
            self.socket.send(constants.MrdMessageIdentifier.pack(constants.MRD_MESSAGE_TEXT))
            contents_with_nul = '%s\0' % contents # Add null terminator
            self.socket.send(constants.MrdMessageLength.pack(len(contents_with_nul.encode())))
            self.socket.send(contents_with_nul.encode())

    def read_text(self):
        logging.info("<-- Received MRD_MESSAGE_TEXT (5)")
        length = self.read_mrd_message_length()
        text = self.read(length)
        text = text.split(b'\x00',1)[0].decode('utf-8')  # Strip off null teminator
        logging.info("    %s", text)
        return text

    # ----- MRD_MESSAGE_ISMRMRD_ACQUISITION (1008) -----------------------------
    # This message contains raw k-space data from a single readout.
    # Message consists of:
    #   ID               (   2 bytes, unsigned short)
    #   Fixed header     ( 340 bytes, mixed         )
    #   Trajectory       (  variable, float         )
    #   Raw k-space data (  variable, float         )
    def send_acquisition(self, acquisition):
        with self.lock:
            self.sentAcqs += 1
            if (self.sentAcqs == 1) or (self.sentAcqs % 100 == 0):
                logging.info("--> Sending MRD_MESSAGE_ISMRMRD_ACQUISITION (1008) (total: %d)", self.sentAcqs)

            self.socket.send(constants.MrdMessageIdentifier.pack(constants.MRD_MESSAGE_ISMRMRD_ACQUISITION))
            acquisition.serialize_into(self.socket.send)

    def read_acquisition(self):
        self.recvAcqs += 1
        if (self.recvAcqs == 1) or (self.recvAcqs % 100 == 0):
            logging.info("<-- Received MRD_MESSAGE_ISMRMRD_ACQUISITION (1008) (total: %d)", self.recvAcqs)

        acq = ismrmrd.Acquisition.deserialize_from(self.read)

        if self.savedata is True:
            if self.dset is None:
                self.create_save_file()

            self.dset.append_acquisition(acq)

        return acq

    # ----- MRD_MESSAGE_ISMRMRD_IMAGE (1022) -----------------------------------
    # This message contains a single [x y z cha] image.
    # Message consists of:
    #   ID               (   2 bytes, unsigned short)
    #   Fixed header     ( 198 bytes, mixed         )
    #   Attribute length (   8 bytes, uint64_t      )
    #   Attribute data   (  variable, char          )
    #   Image data       (  variable, variable      )
    def send_image(self, images):
        with self.lock:
            if not isinstance(images, list):
                images = [images]

            logging.info("--> Sending MRD_MESSAGE_ISMRMRD_IMAGE (1022) (%d images)", len(images))
            for image in images:
                if image is None:
                    continue

                self.sentImages += 1
                self.socket.send(constants.MrdMessageIdentifier.pack(constants.MRD_MESSAGE_ISMRMRD_IMAGE))
                image.serialize_into(self.socket.send)

            # Explicit version of serialize_into() for more verbose debugging
            # self.socket.send(image.getHead())
            # self.socket.send(constants.MrdMessageAttribLength.pack(len(image.attribute_string)))
            # self.socket.send(bytes(image.attribute_string, 'utf-8'))
            # self.socket.send(bytes(image.data))

    def read_image(self):
        self.recvImages += 1
        logging.info("<-- Received MRD_MESSAGE_ISMRMRD_IMAGE (1022)")
        # return ismrmrd.Image.deserialize_from(self.read)

        # Explicit version of deserialize_from() for more verbose debugging
        logging.debug("   Reading in %d bytes of image header", ctypes.sizeof(ismrmrd.ImageHeader))
        header_bytes = self.read(ctypes.sizeof(ismrmrd.ImageHeader))

        attribute_length_bytes = self.read(ctypes.sizeof(ctypes.c_uint64))
        attribute_length = ctypes.c_uint64.from_buffer_copy(attribute_length_bytes)
        logging.debug("   Reading in %d bytes of attributes", attribute_length.value)

        attribute_bytes = self.read(attribute_length.value)
        if (attribute_length.value > 25000):
            logging.debug("   Attributes (truncated): %s", attribute_bytes[0:24999].decode('utf-8'))
        else:
            logging.debug("   Attributes: %s", attribute_bytes.decode('utf-8'))

        image = ismrmrd.Image(header_bytes, attribute_bytes.split(b'\x00',1)[0].decode('utf-8'))  # Strip off null teminator

        logging.info("    Image is size %d x %d x %d with %d channels of type %s", image.getHead().matrix_size[0], image.getHead().matrix_size[1], image.getHead().matrix_size[2], image.channels, image.data.dtype)
        def calculate_number_of_entries(nchannels, xs, ys, zs):
            return nchannels * xs * ys * zs

        nentries = calculate_number_of_entries(image.channels, *image.getHead().matrix_size)
        nbytes = nentries * image.data.dtype.itemsize

        logging.debug("Reading in %d bytes of image data", nbytes)
        data_bytes = self.read(nbytes)

        image.data.ravel()[:] = np.frombuffer(data_bytes, dtype=image.data.dtype)

        if self.savedata is True:
            if self.dset is None:
                self.create_save_file()
            self.dset.append_image("image_%d" % image.image_series_index, image)

        return image

    # ----- MRD_MESSAGE_ISMRMRD_WAVEFORM (1026) -----------------------------
    # This message contains abitrary (e.g. physio) waveform data.
    # Message consists of:
    #   ID               (   2 bytes, unsigned short)
    #   Fixed header     ( 240 bytes, mixed         )
    #   Waveform data    (  variable, uint32_t      )
    def send_waveform(self, waveform):
        with self.lock:
            self.sentWaveforms += 1
            if (self.sentWaveforms == 1) or (self.sentWaveforms % 100 == 0):
                logging.info("--> Sending MRD_MESSAGE_ISMRMRD_WAVEFORM (1026) (total: %d)", self.sentWaveforms)

            self.socket.send(constants.MrdMessageIdentifier.pack(constants.MRD_MESSAGE_ISMRMRD_WAVEFORM))
            waveform.serialize_into(self.socket.send)

    def read_waveform(self):
        self.recvWaveforms += 1
        if (self.recvWaveforms == 1) or (self.recvWaveforms % 100 == 0):
            logging.info("<-- Received MRD_MESSAGE_ISMRMRD_WAVEFORM (1026) (total: %d)", self.recvWaveforms)

        waveform = ismrmrd.Waveform.deserialize_from(self.read)

        if self.savedata is True:
            if self.dset is None:
                self.create_save_file()

            self.dset.append_waveform(waveform)

        return waveform

