
import constants
from connection import Connection

import socket
import logging
import multiprocessing
import ismrmrd.xsd
import importlib
import os
import json
import signal

import simplefft
import invertcontrast
import analyzeflow

class Server:
    """
    Something something docstring.
    """

    def __init__(self, address, port, defaultConfig, savedata, savedataFolder, multiprocessing):
        logging.info("Starting server and listening for data at %s:%d", address, port)

        logging.info("Default config is %s", defaultConfig)
        if (savedata is True):
            logging.debug("Saving incoming data is enabled.")

        if (multiprocessing is True):
            logging.debug("Multiprocessing is enabled.")

        self.defaultConfig = defaultConfig
        self.multiprocessing = multiprocessing
        self.savedata = savedata
        self.savedataFolder = savedataFolder
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((address, port))

    def serve(self):
        logging.debug("Serving... ")
        self.socket.listen(0)

        while True:
            try:
                signal.siginterrupt(signal.SIGTERM, True)
                signal.siginterrupt(signal.SIGINT, True)
            except AttributeError:
                # signal.siginterrupt is not available in Windows
                pass

            sock, (remote_addr, remote_port) = self.socket.accept()

            logging.info("Accepting connection from: %s:%d", remote_addr, remote_port)

            if (self.multiprocessing is True):
                process = multiprocessing.Process(target=self.handle, args=[sock])
                process.daemon = True
                process.start()
                logging.debug("Spawned process %d to handle connection.", process.pid)
            else:
                self.handle(sock)

    def handle(self, sock):

        try:
            connection = Connection(sock, self.savedata, "", self.savedataFolder, "dataset")

            # First message is the config (file or text)
            config = next(connection)

            # Break out if a connection was established but no data was received
            if ((config is None) & (connection.is_exhausted is True)):
                logging.info("Connection closed without any data received")
                return

            # Second messages is the metadata (text)
            metadata_xml = next(connection)

            # Break if no MRD header was received before a close message (e.g. Gadgetron dependency query)
            if ((metadata_xml is None) & (connection.is_exhausted is True)):
                logging.info("Connection closed without an MRD header received")
                return

            logging.debug("XML Metadata: %s", metadata_xml)
            try:
                metadata = ismrmrd.xsd.CreateFromDocument(metadata_xml)
                if (metadata.acquisitionSystemInformation.systemFieldStrength_T != None):
                    logging.info("Data is from a %s %s at %1.1fT", metadata.acquisitionSystemInformation.systemVendor, metadata.acquisitionSystemInformation.systemModel, metadata.acquisitionSystemInformation.systemFieldStrength_T)
            except:
                logging.warning("Metadata is not a valid MRD XML structure.  Passing on metadata as text")
                metadata = metadata_xml

            # Support additional config parameters passed through a JSON text message
            if connection.peek_mrd_message_identifier() == constants.MRD_MESSAGE_TEXT:
                configAdditionalText = next(connection)
                logging.info("Received additional config text: %s", configAdditionalText)
                connection.save_additional_config(configAdditionalText)
                try:
                    configAdditional = json.loads(configAdditionalText)

                    if ('parameters' in configAdditional):
                        if ('config' in configAdditional['parameters']):
                            logging.info("Changing config to: %s", configAdditional['parameters']['config'])
                            config = configAdditional['parameters']['config']

                        if ('customconfig' in configAdditional['parameters']) and (configAdditional['parameters']['customconfig'] != ""):
                            logging.info("Changing config to: %s", configAdditional['parameters']['customconfig'])
                            config = configAdditional['parameters']['customconfig']
                except:
                    logging.error("Failed to parse as JSON")
            else:
                configAdditional = config

            # Decide what program to use based on config
            # If not one of these explicit cases, try to load file matching name of config
            if (config == "simplefft"):
                logging.info("Starting simplefft processing based on config")
                simplefft.process(connection, configAdditional, metadata)
            elif (config == "invertcontrast"):
                logging.info("Starting invertcontrast processing based on config")
                invertcontrast.process(connection, configAdditional, metadata)
            elif (config == "analyzeflow"):
                logging.info("Starting analyzeflow processing based on config")
                analyzeflow.process(connection, configAdditional, metadata)
            elif (config == "null"):
                logging.info("No processing based on config")
                try:
                    for msg in connection:
                        if msg is None:
                            break
                finally:
                    connection.send_close()
            elif (config == "savedataonly"):
                # Dummy loop with no processing
                try:
                    for msg in connection:
                        if msg is None:
                            break
                finally:
                    connection.send_close()
            else:
                usedConfig = config
                if importlib.util.find_spec(config) is None:
                    logging.error("Could not find config module '%s' -- falling back to default config: %s", config, self.defaultConfig)
                    usedConfig = self.defaultConfig

                try:
                    # Load module from file having exact name as config
                    module = importlib.import_module(usedConfig)
                    logging.info("Starting config %s", usedConfig)
                    module.process(connection, configAdditional, metadata)
                except ImportError as e:
                    logging.error("Failed to load config '%s' with error:\n  %s", usedConfig, e)
                    if usedConfig != self.defaultConfig:
                        logging.info("Falling back to default config: '%s'", self.defaultConfig)
                        try:
                            module = importlib.import_module(self.defaultConfig)
                            logging.info("Starting config %s", self.defaultConfig)
                            module.process(connection, configAdditional, metadata)
                        except ImportError as e:
                            logging.error("Failed to load default config '%s' with error:\n  %s", self.defaultConfig, e)

        except Exception as e:
            logging.exception(e)

        finally:
            connection.shutdown_close()

            # Dataset may not be closed properly if a close message is not received
            if connection.savedata is True:
                try:
                    connection.dset.close()
                except:
                    pass

                if (connection.savedataFile == ""):
                    try:
                        # Ensure ismrmrd package has a context manager
                        if not (hasattr(ismrmrd.Dataset, '__enter__') and hasattr(ismrmrd.Dataset, '__exit__')):
                            raise Exception("Current ismrmrd Python package does not support context manager as required by this code.  Please update to 1.14.1 or newer")

                        # Rename the saved file to use the protocol name
                        with ismrmrd.Dataset(connection.mrdFilePath, connection.savedataGroup, False) as dset:
                            groups = dset.list()

                            if ('xml' in groups):
                                xml_header = dset.read_xml_header()
                                xml_header = xml_header.decode("utf-8")
                                mrdHead = ismrmrd.xsd.CreateFromDocument(xml_header)

                        if (mrdHead.measurementInformation.protocolName != ""):
                            newFilePath = connection.mrdFilePath.replace("MRD_input_", mrdHead.measurementInformation.protocolName + "_")
                            os.rename(connection.mrdFilePath, newFilePath)
                            connection.mrdFilePath = newFilePath
                    except:
                        pass

                if connection.mrdFilePath is not None:
                    logging.info("Incoming data was saved at %s", connection.mrdFilePath)
