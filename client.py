#!/usr/bin/python3

# from server import Server

import argparse
import logging
import datetime
import h5py
import socket
import sys
import ismrmrd
import multiprocessing
from connection import Connection
import time
import os

defaults = {
    'address':        'localhost',
    'port':           9002, 
    'outfile':        'out.h5',
    'out_group':      str(datetime.datetime.now()),
    'config':         'default.xml',
    'send_waveforms': False
}

def connection_receive_loop(sock, outfile, outgroup, verbose, logfile, recvAcqs, recvImages, recvWaveforms):
    """Start a Connection instance to receive data, generally run in a separate thread"""

    if verbose:
        verbosity = logging.DEBUG
    else:
        verbosity = logging.INFO

    if logfile:
        logging.basicConfig(filename=logfile, format='%(asctime)s - %(message)s', level=verbosity)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    else:
        logging.basicConfig(format='%(asctime)s - %(message)s', level=verbosity)

    incoming_connection = Connection(sock, True, outfile, "", outgroup)

    try:
        for msg in incoming_connection:
            if msg is None:
                break
    finally:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except:
            pass
        sock.close()
        logging.debug("Socket closed (reader)")

        # Dataset may not be closed properly if a close message is not received
        try:
            incoming_connection.dset.close()
        except:
            pass

    recvAcqs.value      = incoming_connection.recvAcqs
    recvImages.value    = incoming_connection.recvImages
    recvWaveforms.value = incoming_connection.recvWaveforms

def main(args):
    # ----- Load and validate file ---------------------------------------------
    if (args.config_local):
        if not os.path.exists(args.config_local):
            logging.error("Could not find local config file %s", args.config_local)
            return

    dset = h5py.File(args.filename, 'r')
    if not dset:
        logging.error("Not a valid dataset: %s" % args.filename)
        return

    dsetNames = dset.keys()
    logging.info("File %s contains %d groups:", args.filename, len(dset.keys()))
    print(" ", "\n  ".join(dsetNames))

    if not args.in_group:
        if len(dset.keys()) == 1:
            args.in_group = list(dset.keys())[0]
        else:
            logging.error("Input group not specified and multiple groups are present")
            return


    if args.in_group not in dset:
        logging.error("Could not find group %s", args.in_group)
        return

    group = dset.get(args.in_group)

    logging.info("Reading data from group '%s' in file '%s'", args.in_group, args.filename)

    # ----- Determine type of data stored --------------------------------------
    # Raw data is stored as:
    #   /group/config      text of recon config parameters (optional)
    #   /group/xml         text of ISMRMRD flexible data header
    #   /group/data        array of IsmsmrdAcquisition data + header
    #   /group/waveforms   array of waveform (e.g. PMU) data

    # Image data is stored as:
    #   /group/config              text of recon config parameters (optional)
    #   /group/xml                 text of ISMRMRD flexible data header (optional)
    #   /group/image_0/data        array of IsmrmrdImage data
    #   /group/image_0/header      array of ImageHeader
    #   /group/image_0/attributes  text of image MetaAttributes
    isRaw   = False
    isImage = False
    hasWaveforms = False

    if ( ('data' in group) and ('xml' in group) ):
        isRaw = True
    else:
        isImage = True
        imageNames = group.keys()
        logging.info("Found %d image sub-groups: %s", len(imageNames), ", ".join(imageNames))
        # print(" ", "\n  ".join(imageNames))

        for imageName in imageNames:
            if ((imageName == 'xml') or (imageName == 'config') or (imageName == 'config_file')):
                continue

            image = group[imageName]
            if not (('data' in image) and ('header' in image) and ('attributes' in image)):
                isImage = False

    if ('waveforms' in group):
        hasWaveforms = True

    dset.close()

    if ((isRaw is False) and (isImage is False)):
        logging.error("File does not contain properly formatted MRD raw or image data")
        return

    # ----- Open connection to server ------------------------------------------
    # Spawn a thread to connect and handle incoming data
    logging.info("Connecting to MRD server at %s:%d" % (args.address, args.port))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.address, args.port))

    recvAcqs      = multiprocessing.Value('i', 0)
    recvImages    = multiprocessing.Value('i', 0)
    recvWaveforms = multiprocessing.Value('i', 0)
    process = multiprocessing.Process(target=connection_receive_loop, args=(sock, args.outfile, args.out_group, args.verbose, args.logfile, recvAcqs, recvImages, recvWaveforms))
    process.daemon = True
    process.start()

    # This connection is only used for outgoing data.  It should not be used for
    # writing to the HDF5 file as multi-threading issues can occur
    connection = Connection(sock, False)

    # --------------- Send config -----------------------------
    if (args.config_local):
        with open(args.config_local, "r") as fid:
            config_text = fid.read()
        logging.info("Sending local config file '%s' with text:", args.config_local)
        logging.info(config_text)
        connection.send_config_text(config_text)
    else:
        logging.info("Sending remote config file name '%s'", args.config)
        connection.send_config_file(args.config)

    dset = ismrmrd.Dataset(args.filename, args.in_group, False)

    # --------------- Send MRD metadata -----------------------
    groups = dset.list()
    if ('xml' in groups):
        xml_header = dset.read_xml_header()
        xml_header = xml_header.decode("utf-8")
    else:
        logging.warning("Could not find MRD metadata xml in file")
        xml_header = "Dummy XML header"
    connection.send_metadata(xml_header)

    # --------------- Send waveform data ----------------------
    # TODO: Interleave waveform and other data so they arrive chronologically
    if hasWaveforms:
        if args.send_waveforms:
            logging.info("Sending waveform data")
            logging.info("Found %d waveforms", dset.number_of_waveforms())

            for idx in range(0, dset.number_of_waveforms()):
                wav = dset.read_waveform(idx)
                try:
                    connection.send_waveform(wav)
                except:
                    logging.error('Failed to send waveform %d' % idx)
        else:
            logging.info("Waveform data present, but send-waveforms option turned off")

    # --------------- Send raw data ----------------------
    if isRaw:
        logging.info("Starting raw data session")
        logging.info("Found %d raw data readouts", dset.number_of_acquisitions())

        for idx in range(dset.number_of_acquisitions()):
            acq = dset.read_acquisition(idx)
            try:
                connection.send_acquisition(acq)
            except:
                logging.error('Failed to send acquisition %d' % idx)

    # --------------- Send image data ----------------------
    else:
        logging.info("Starting image data session")
        for group in groups:
            if ( (group == 'config') or (group == 'config_file') or (group == 'xml') ):
                logging.info("Skipping group %s", group)
                continue

            logging.info("Reading images from '/" + args.in_group + "/" + group + "'")

            for imgNum in range(0, dset.number_of_images(group)):
                image = dset.read_image(group, imgNum)

                if not isinstance(image.attribute_string, str):
                    image.attribute_string = image.attribute_string.decode('utf-8')

                logging.debug("Sending image %d of %d", imgNum, dset.number_of_images(group)-1)
                connection.send_image(image)

    dset.close()
    connection.send_close()

    # Wait for incoming data and cleanup
    logging.debug("Waiting for threads to finish")
    process.join()

    sock.close()
    logging.info("Socket closed (writer)")

    # Save a copy of the MRD XML header now that the connection thread is finished with the file
    logging.debug("Writing MRD metadata to file")
    dset = ismrmrd.Dataset(args.outfile, args.out_group)
    dset.write_xml_header(bytes(xml_header, 'utf-8'))
    dset.close()

    logging.info("---------------------- Summary ----------------------")
    logging.info("Sent %4d acquisitions  |  Received %4d acquisitions", connection.sentAcqs,      recvWaveforms.value)
    logging.info("Sent %4d images        |  Received %4d images",       connection.sentImages,    recvImages.value)
    logging.info("Sent %4d waveforms     |  Received %4d waveforms",    connection.sentWaveforms, recvWaveforms.value)
    logging.info("Session complete")

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example client for MRD streaming format',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename',                                    help='Input file')
    parser.add_argument('-a', '--address',                             help='Address (hostname) of MRD server')
    parser.add_argument('-p', '--port',           type=int,            help='Port')
    parser.add_argument('-o', '--outfile',                             help='Output file')
    parser.add_argument('-g', '--in-group',                            help='Input data group')
    parser.add_argument('-G', '--out-group',                           help='Output group name')
    parser.add_argument('-c', '--config',                              help='Remote configuration file')
    parser.add_argument('-C', '--config-local',                        help='Local configuration file')
    parser.add_argument('-w', '--send-waveforms', action='store_true', help='Send waveform (physio) data')
    parser.add_argument('-v', '--verbose',        action='store_true', help='Verbose mode')
    parser.add_argument('-l', '--logfile',        type=str,            help='Path to log file')

    parser.set_defaults(**defaults)

    args = parser.parse_args()

    if args.logfile:
        print("Logging to file: ", args.logfile)
        logging.basicConfig(filename=args.logfile, format='%(asctime)s - %(message)s', level=logging.WARNING)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    else:
        print("No logfile provided")
        logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.WARNING)

    if args.verbose:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(logging.INFO)

    main(args)
