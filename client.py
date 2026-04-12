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
import json

defaults = {
    'filename':           '',
    'in_group':           '',
    'address':            'localhost',
    'port':               9002,
    'outfile':            None,
    'out_group':          str(datetime.datetime.now()),
    'config':             'invertcontrast',
    'config_local':       '',
    'ignore_json_config': False,
    'send_waveforms':     False,
    'verbose':            False,
    'logfile':            '',
    'quiet':              False,
    'mrd2gif':            False
}

def connection_receive_loop(sock, outfile, outgroup, verbose, logfile, quiet, recvAcqs, recvImages, recvWaveforms):
    """Start a Connection instance to receive data, generally run in a separate thread"""

    if verbose:
        verbosity = logging.DEBUG
    else:
        verbosity = logging.INFO

    if logfile:
        logging.basicConfig(filename=logfile, format='%(asctime)s - %(message)s', level=verbosity)
        if not quiet:
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
    # ----- Set up logging ---------------------------------------------
    if args.logfile:
        print("Logging to file: ", args.logfile)
        logging.basicConfig(filename=args.logfile, format='%(asctime)s - %(message)s', level=logging.WARNING)
        if not args.quiet:
            logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    else:
        print("No logfile provided")
        logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.WARNING)

    if args.verbose:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(logging.INFO)

    # Use an output filename based on the input file if not provided
    if args.outfile is None:
        base, ext = os.path.splitext(args.filename)
        args.outfile = base + '_results' + ext
        logging.info("Output file not specified -- writing results to %s", args.outfile)

    # If a config is specified via the command line arguments, then set ignore_json_config to True
    if ('-c' in sys.argv) or ('--config' in sys.argv):
        args.ignore_json_config = True

    # ----- Load and validate file ---------------------------------------------
    if (args.config_local):
        if not os.path.exists(args.config_local):
            logging.error("Could not find local config file %s", args.config_local)
            return

    localConfigAdditionalText = None
    if (args.config) and (not args.config_local):
        configAdditionalFile = args.config + '.json'
        if os.path.exists(configAdditionalFile):
            logging.info("Found additional config file %s", configAdditionalFile)

            fid = open(configAdditionalFile, 'r')
            localConfigAdditionalText = fid.read()
            fid.close()

    with h5py.File(args.filename, 'r') as dset:
        if not dset:
            logging.error("Not a valid dataset: %s" % args.filename)
            return
        dsetNames = dset.keys()
        logging.info("File %s contains %d groups:", args.filename, len(dset.keys()))
        logging.info("\n  ".join(dsetNames))

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
        hasRaw   = False
        hasImage = False
        hasWaveforms = False

        if ('data' in group):
            hasRaw = True

        if len([key for key in group.keys() if (key.startswith('image_') or key.startswith('images_'))]) > 0:
            hasImage = True

        if ('waveforms' in group):
            hasWaveforms = True

    if ((hasRaw is False) and (hasImage is False)):
        logging.error("File does not contain properly formatted MRD raw or image data")
        return

    # ----- Open connection to server ------------------------------------------
    # Spawn a thread to connect and handle incoming data
    logging.info("Connecting to MRD server at %s:%d" % (args.address, args.port))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    attempt     = 0
    maxAttempts = 5
    success     = False
    while attempt < maxAttempts:
        try:
            sock.connect((args.address, args.port))
        except socket.error as error:
            logging.warning("Failed to connect (%d/%d): %s" % (attempt+1, maxAttempts, error))
            time.sleep(1)
            attempt += 1
        else:
            success = True
            attempt = maxAttempts

    if not success:
        sock.close()
        logging.error("... Aborting")
        return

    recvAcqs      = multiprocessing.Value('i', 0)
    recvImages    = multiprocessing.Value('i', 0)
    recvWaveforms = multiprocessing.Value('i', 0)
    process = multiprocessing.Process(target=connection_receive_loop, args=(sock, args.outfile, args.out_group, args.verbose, args.logfile, args.quiet, recvAcqs, recvImages, recvWaveforms))
    process.daemon = True
    process.start()

    # This connection is only used for outgoing data.  It should not be used for
    # writing to the HDF5 file as multi-threading issues can occur
    connection = Connection(sock, False)

    # --------------- Send config -----------------------------
    if (args.config_local):
        fid = open(args.config_local, "r")
        config_text = fid.read()
        fid.close()
        logging.info("Sending local config file '%s' with text:", args.config_local)
        logging.info(config_text)
        connection.send_config_text(config_text)
    else:
        logging.info("Sending remote config file name '%s'", args.config)
        connection.send_config_file(args.config)

    # Ensure ismrmrd package has a context manager
    if not (hasattr(ismrmrd.Dataset, '__enter__') and hasattr(ismrmrd.Dataset, '__exit__')):
        raise Exception("Current ismrmrd Python package does not support context manager as required by this code.  Please update to 1.14.1 or newer")

    with ismrmrd.Dataset(args.filename, args.in_group, create_if_needed=False) as dset:
        # --------------- Send MRD metadata -----------------------
        groups = dset.list()
        if ('xml' in groups):
            xml_header = dset.read_xml_header()
            xml_header = xml_header.decode("utf-8")
        else:
            logging.warning("Could not find MRD metadata xml in file")
            xml_header = "Dummy XML header"
        connection.send_metadata(xml_header)

        # --------------- Send additional config -----------------------
        groups = dset.list()
        if localConfigAdditionalText is None:
            if ('configAdditional' in groups):
                configAdditionalText = dset._dataset['configAdditional'][0]
                configAdditionalText = configAdditionalText.decode("utf-8")

                if args.ignore_json_config:
                    # Remove the config specified in the JSON, allowing the config passed via command line to the client to be used
                    configAdditional = json.loads(configAdditionalText)
                    if ('parameters' in configAdditional):
                        if ('config' in configAdditional['parameters']):
                            logging.warning(f"Input file contains JSON configAdditional that specifies config '{configAdditional['parameters']['config']}', but will be ignored because '--ignore-json-config' was specified!")
                            del configAdditional['parameters']['config']

                        if ('customconfig' in configAdditional['parameters']):
                            if configAdditional['parameters']['customconfig'] != '':
                                logging.warning(f"Input file contains JSON configAdditional that specifies customconfig '{configAdditional['parameters']['customconfig']}', but will be ignored because '--ignore-json-config' was specified!")
                            del configAdditional['parameters']['customconfig']

                        configAdditionalText = json.dumps(configAdditional, indent=2)

                logging.info("Sending configAdditional found in file %s:\n%s", args.filename, configAdditionalText)
                connection.send_text(configAdditionalText)
            else:
                # Do nothing -- no additional config in local .json file or in MRD file
                pass
        else:
            if ('configAdditional' in groups):
                logging.warning("configAdditional found in file %s, but is overriden by local file %s!", args.filename, configAdditionalFile)

            if args.ignore_json_config:
                # Remove the config specified in the JSON, allowing the config passed via command line to the client to be used
                localConfigAdditional = json.loads(localConfigAdditionalText)
                if ('parameters' in localConfigAdditional):
                    if ('config' in localConfigAdditional['parameters']):
                        logging.warning(f"configAdditional file '{configAdditionalFile}' specifies config '{localConfigAdditional['parameters']['config']}', but will be ignored because '--ignore-json-config' was specified!")
                        del localConfigAdditional['parameters']['config']

                    if ('customconfig' in localConfigAdditional['parameters']):
                        if localConfigAdditional['parameters']['customconfig'] != '':
                            logging.warning(f"configAdditional file '{configAdditionalFile}' specifies customconfig '{localConfigAdditional['parameters']['customconfig']}', but will be ignored because '--ignore-json-config' was specified!")
                        del localConfigAdditional['parameters']['customconfig']

                    localConfigAdditionalText = json.dumps(localConfigAdditional, indent=2)

            logging.info("Sending configAdditional found in file %s:\n%s", configAdditionalFile, localConfigAdditionalText)
            connection.send_text(localConfigAdditionalText)

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
                        logging.error('Failed to send waveform %d -- aborting!' % idx)
                        break
            else:
                logging.info("Waveform data present, but send-waveforms option turned off")

        # --------------- Send raw data ----------------------
        if hasRaw:
            logging.info("Starting raw data session")
            logging.info("Found %d raw data readouts", dset.number_of_acquisitions())

            for idx in range(dset.number_of_acquisitions()):
                acq = dset.read_acquisition(idx)
                try:
                    connection.send_acquisition(acq)
                except:
                    logging.error('Failed to send acquisition %d -- aborting!' % idx)
                    break

        # --------------- Send image data ----------------------
        if hasImage:
            logging.info("Starting image data session")
            for group in [key for key in groups if (key.startswith('image_') or key.startswith('images_'))]:
                logging.info("Reading images from '/" + args.in_group + "/" + group + "'")

                for imgNum in range(0, dset.number_of_images(group)):
                    image = dset.read_image(group, imgNum)

                    if not isinstance(image.attribute_string, str):
                        image.attribute_string = image.attribute_string.decode('utf-8')

                    logging.debug("Sending image %d of %d", imgNum, dset.number_of_images(group)-1)
                    try:
                        connection.send_image(image)
                    except:
                        logging.error('Failed to send image %d -- aborting!' % imgNum)
                        break

    try:
        connection.send_close()
    except:
        logging.error('Failed to send close message!')

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
    logging.info("Sent %5d acquisitions  |  Received %5d acquisitions", connection.sentAcqs,      recvWaveforms.value)
    logging.info("Sent %5d images        |  Received %5d images",       connection.sentImages,    recvImages.value)
    logging.info("Sent %5d waveforms     |  Received %5d waveforms",    connection.sentWaveforms, recvWaveforms.value)
    logging.info("Results written to %s", args.outfile)
    logging.info("Session complete")

    if args.mrd2gif:
        try:
            import mrd2gif
            from types import SimpleNamespace

            mrd2gifargs = SimpleNamespace(**mrd2gif.defaults)
            mrd2gifargs.filename = args.outfile
            mrd2gifargs.in_group = args.out_group

            logging.info('Calling mrd2gif...')
            mrd2gif.main(mrd2gifargs)
        except:
            logging.error('Failed to import or call mrd2gif')

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example client for MRD streaming format',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename',                                        help='Input file')
    parser.add_argument('-a', '--address',                                 help='Address (hostname) of MRD server')
    parser.add_argument('-p', '--port',               type=int,            help='Port')
    parser.add_argument('-o', '--outfile',                                 help='Output file')
    parser.add_argument('-g', '--in-group',                                help='Input data group')
    parser.add_argument('-G', '--out-group',                               help='Output group name')
    parser.add_argument('-c', '--config',                                  help='Remote configuration file')
    parser.add_argument('-C', '--config-local',                            help='Local configuration file')
    parser.add_argument('-w', '--send-waveforms',     action='store_true', help='Send waveform (physio) data')
    parser.add_argument('-v', '--verbose',            action='store_true', help='Verbose mode')
    parser.add_argument('-l', '--logfile',            type=str,            help='Path to log file')
    parser.add_argument('-q', '--quiet',              action='store_true', help='Suppress stdout logging')
    parser.add_argument(      '--ignore-json-config', action='store_true', help='Ignore config specified in JSON')
    parser.add_argument(      '--mrd2gif',            action='store_true', help='Run mrd2gif on output file')

    parser.set_defaults(**defaults)

    args = parser.parse_args()

    main(args)
