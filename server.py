import importlib
import ismrmrd.xsd
import logging
import multiprocessing
import os
import socket

import config
from connection import Connection


class Server:
    """
    Something something docstring.
    """

    def __init__(self, address, port, savedata, savedataFolder, multiprocessing):
        logging.info("Starting server and listening for data at %s:%d", address, port)
        if (savedata is True):
            logging.debug("Saving incoming data is enabled.")

        if (multiprocessing is True):
            logging.debug("Multiprocessing is enabled.")

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
            # TODO Code currently does not handle case where this is a file,
            #   but the client code suggests that this is a valid use case
            # This seemingly includes the default operation, in which case file
            #   "default.xml" is passed
            selected_config = next(connection)

            # Break out if a connection was established but no data was received
            if ((selected_config is None) & (connection.is_exhausted is True)):
                logging.info("Connection closed without any data received")
                return

            # Second messages is the metadata (text)
            metadata_xml = next(connection)
            logging.debug("XML Metadata: %s", metadata_xml)
            try:
                metadata = ismrmrd.xsd.CreateFromDocument(metadata_xml)
                if (metadata.acquisitionSystemInformation.systemFieldStrength_T != None):
                    logging.info("Data is from a %s %s at %1.1fT", metadata.acquisitionSystemInformation.systemVendor, metadata.acquisitionSystemInformation.systemModel, metadata.acquisitionSystemInformation.systemFieldStrength_T)
            except:
                logging.warning("Metadata is not a valid MRD XML structure; passing on metadata as text")
                metadata = metadata_xml

            config.process(connection, selected_config, metadata)

        except Exception as e:
            logging.exception(e)

        finally:
            # Encapsulate shutdown in a try block because the socket may have
            # already been closed on the other side
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except:
                pass
            sock.close()
            logging.info("Socket closed")

            # Dataset may not be closed properly if a close message is not received
            if connection.savedata is True:
                try:
                    connection.dset.close()
                except:
                    pass

                if (connection.savedataFile == ""):
                    try:
                        # Rename the saved file to use the protocol name
                        dset = ismrmrd.Dataset(connection.mrdFilePath, connection.savedataGroup, False)
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
