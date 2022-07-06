#!/usr/bin/python3

import os
import re
import argparse
import h5py
import ismrmrd
import numpy as np
import pydicom
import pynetdicom
import base64

# Lookup table between DICOM and MRD mrdImg types
imtype_map = {ismrmrd.IMTYPE_MAGNITUDE : 'M',
              ismrmrd.IMTYPE_PHASE     : 'P',
              ismrmrd.IMTYPE_REAL      : 'R',
              ismrmrd.IMTYPE_IMAG      : 'I',
              0                        : 'M'} # Fallback for unset value

# Lookup table between DICOM and Siemens flow directions
venc_dir_map = {'FLOW_DIR_R_TO_L' : 'rl',
                'FLOW_DIR_L_TO_R' : 'lr',
                'FLOW_DIR_A_TO_P' : 'ap',
                'FLOW_DIR_P_TO_A' : 'pa',
                'FLOW_DIR_F_TO_H' : 'fh',
                'FLOW_DIR_H_TO_F' : 'hf',
                'FLOW_DIR_TP_IN'  : 'in',
                'FLOW_DIR_TP_OUT' : 'out'}

def main(args):
    dset = h5py.File(args.filename, 'r')
    if not dset:
        print("Not a valid dataset: %s" % (args.filename))
        return

    dsetNames = dset.keys()
    print("File %s contains %d groups:" % (args.filename, len(dset.keys())))
    print(" ", "\n  ".join(dsetNames))

    if not args.in_group:
        if len(dset.keys()) > 1:
            print("Input group not specified -- selecting most recent")
        args.in_group = list(dset.keys())[-1]

    if not args.out_folder:
        args.out_folder = re.sub('.h5$', '', args.filename)
        print("Output folder not specified -- using %s" % args.out_folder)

    if args.in_group not in dset:
        print("Could not find group %s" % (args.in_group))
        return

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    group = dset.get(args.in_group)
    print("Reading data from group '%s' in file '%s'" % (args.in_group, args.filename))

    # mrdImg data is stored as:
    #   /group/config              text of recon config parameters (optional)
    #   /group/xml                 text of ISMRMRD flexible data header (optional)
    #   /group/image_0/data        array of IsmrmrdImage data
    #   /group/image_0/header      array of ImageHeader
    #   /group/image_0/attributes  text of mrdImg MetaAttributes

    isImage = True
    imageNames = group.keys()
    print("Found %d mrdImg sub-groups: %s" % (len(imageNames), ", ".join(imageNames)))

    for imageName in imageNames:
        if ((imageName == 'xml') or (imageName == 'config') or (imageName == 'config_file')):
            continue

        mrdImg = group[imageName]
        if not (('data' in mrdImg) and ('header' in mrdImg) and ('attributes' in mrdImg)):
            isImage = False

    dset.close()

    if (isImage is False):
        print("File does not contain properly formatted MRD raw or mrdImg data")
        return

    dset = ismrmrd.Dataset(args.filename, args.in_group, False)

    groups = dset.list()

    if ('xml' in groups):
        xml_header = dset.read_xml_header()
        xml_header = xml_header.decode("utf-8")
        mrdHead = ismrmrd.xsd.CreateFromDocument(xml_header)

    for group in groups:
        if ( (group == 'config') or (group == 'config_file') or (group == 'xml') ):
            continue

        print("Reading images from '/" + args.in_group + "/" + group + "'")

        for imgNum in range(0, dset.number_of_images(group)):
            mrdImg = dset.read_image(group, imgNum)
            meta = ismrmrd.Meta.deserialize(mrdImg.attribute_string)

            if ((mrdImg.data.shape[0] == 3) and (mrdImg.getHead().image_type == 6)):
                # RGB images
                print("RGB data not yet supported")
                continue
            else:
                if (mrdImg.data.shape[1] != 1):
                    print("Multi-slice data not yet supported")
                    continue

                if (mrdImg.data.shape[0] != 1):
                    print("Multi-channel data not yet supported")
                    continue

                # Use previously JSON serialized header as a starting point, if available
                if meta.get('DicomJson') is not None:
                    dicomDset = pydicom.dataset.Dataset.from_json(base64.b64decode(meta['DicomJson']))
                else:
                    dicomDset = pydicom.dataset.Dataset()

                # Enforce explicit little endian for written DICOM files
                dicomDset.file_meta                            = pydicom.dataset.FileMetaDataset()
                dicomDset.file_meta.TransferSyntaxUID          = pydicom.uid.ExplicitVRLittleEndian
                dicomDset.file_meta.MediaStorageSOPClassUID    = pynetdicom.sop_class.MRImageStorage
                dicomDset.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
                pydicom.dataset.validate_file_meta(dicomDset.file_meta)
                # FileMetaInformationGroupLength is still missing?
                dicomDset.is_little_endian                     = True
                dicomDset.is_implicit_VR                       = False

                # ----- Update DICOM header from MRD header -----
                try:
                    if mrdHead.measurementInformation is None:
                        pass
                        # print("  MRD header does not contain measurementInformation section")
                    else:
                        # print("---------- Old -------------------------")
                        # print("SeriesInstanceUID  : %s" % dicomDset.SeriesInstanceUID   )
                        # print("PatientPosition    : %s" % dicomDset.PatientPosition     )
                        # print("SeriesDescription  : %s" % dicomDset.SeriesDescription   )
                        # print("FrameOfReferenceUID: %s" % dicomDset.FrameOfReferenceUID )

                        if mrdHead.measurementInformation.measurementID       is not None: dicomDset.SeriesInstanceUID   = mrdHead.measurementInformation.measurementID
                        if mrdHead.measurementInformation.patientPosition     is not None: dicomDset.PatientPosition     = mrdHead.measurementInformation.patientPosition.name
                        if mrdHead.measurementInformation.protocolName        is not None: dicomDset.SeriesDescription   = mrdHead.measurementInformation.protocolName
                        if mrdHead.measurementInformation.frameOfReferenceUID is not None: dicomDset.FrameOfReferenceUID = mrdHead.measurementInformation.frameOfReferenceUID

                        # print("---------- New -------------------------")
                        # print("SeriesInstanceUID  : %s" % dicomDset.SeriesInstanceUID   )
                        # print("PatientPosition    : %s" % dicomDset.PatientPosition     )
                        # print("SeriesDescription  : %s" % dicomDset.SeriesDescription   )
                        # print("FrameOfReferenceUID: %s" % dicomDset.FrameOfReferenceUID )
                except:
                    print("Error setting header information from MRD header's measurementInformation section")

                try:
                    # print("---------- Old -------------------------")
                    # print("mrdHead.acquisitionSystemInformation.systemVendor         : %s" % mrdHead.acquisitionSystemInformation.systemVendor          )
                    # print("mrdHead.acquisitionSystemInformation.systemModel          : %s" % mrdHead.acquisitionSystemInformation.systemModel           )
                    # print("mrdHead.acquisitionSystemInformation.systemFieldStrength_T: %s" % mrdHead.acquisitionSystemInformation.systemFieldStrength_T )
                    # print("mrdHead.acquisitionSystemInformation.institutionName      : %s" % mrdHead.acquisitionSystemInformation.institutionName       )
                    # print("mrdHead.acquisitionSystemInformation.stationName          : %s" % mrdHead.acquisitionSystemInformation.stationName           )

                    if mrdHead.acquisitionSystemInformation.systemVendor          is not None: dicomDset.Manufacturer          = mrdHead.acquisitionSystemInformation.systemVendor         
                    if mrdHead.acquisitionSystemInformation.systemModel           is not None: dicomDset.ManufacturerModelName = mrdHead.acquisitionSystemInformation.systemModel          
                    if mrdHead.acquisitionSystemInformation.systemFieldStrength_T is not None: dicomDset.MagneticFieldStrength = mrdHead.acquisitionSystemInformation.systemFieldStrength_T
                    if mrdHead.acquisitionSystemInformation.institutionName       is not None: dicomDset.InstitutionName       = mrdHead.acquisitionSystemInformation.institutionName      
                    if mrdHead.acquisitionSystemInformation.stationName           is not None: dicomDset.StationName           = mrdHead.acquisitionSystemInformation.stationName

                    # print("---------- New -------------------------")
                    # print("mrdHead.acquisitionSystemInformation.systemVendor         : %s" % mrdHead.acquisitionSystemInformation.systemVendor          )
                    # print("mrdHead.acquisitionSystemInformation.systemModel          : %s" % mrdHead.acquisitionSystemInformation.systemModel           )
                    # print("mrdHead.acquisitionSystemInformation.systemFieldStrength_T: %s" % mrdHead.acquisitionSystemInformation.systemFieldStrength_T )
                    # print("mrdHead.acquisitionSystemInformation.institutionName      : %s" % mrdHead.acquisitionSystemInformation.institutionName       )
                    # print("mrdHead.acquisitionSystemInformation.stationName          : %s" % mrdHead.acquisitionSystemInformation.stationName           )
                except:
                    print("Error setting header information from MRD header's acquisitionSystemInformation section")

                # Set mrdImg pixel data from MRD mrdImg
                dicomDset.PixelData = np.squeeze(mrdImg.data).tobytes() # mrdImg.data is [cha z y x] -- squeeze to [y x] for [row col]
                dicomDset.Rows      = mrdImg.data.shape[2]
                dicomDset.Columns   = mrdImg.data.shape[3]

                if (mrdImg.data.dtype == 'uint16') or (mrdImg.data.dtype == 'int16'):
                    dicomDset.BitsAllocated = 16
                    dicomDset.BitsStored    = 16
                    dicomDset.HighBit       = 15
                elif (mrdImg.data.dtype == 'uint32') or (mrdImg.data.dtype == 'int') or (mrdImg.data.dtype == 'float32'):
                    dicomDset.BitsAllocated = 32
                    dicomDset.BitsStored    = 32
                    dicomDset.HighBit       = 31
                elif (mrdImg.data.dtype == 'float64'):
                    dicomDset.BitsAllocated = 64
                    dicomDset.BitsStored    = 64
                    dicomDset.HighBit       = 63
                else:
                    print("Unsupported data type: ", mrdImg.data.dtype)

                dicomDset.SeriesNumber               = mrdImg.image_series_index
                dicomDset.InstanceNumber             = mrdImg.image_index

                # ----- Set some mandatory default values -----
                if not 'SamplesPerPixel' in dicomDset:
                    dicomDset.SamplesPerPixel = 1

                if not 'PhotometricInterpretation' in dicomDset:
                    dicomDset.PhotometricInterpretation = 'MONOCHROME2'

                if not 'PixelRepresentation' in dicomDset:
                    dicomDset.PixelRepresentation = 0  # Unsigned integer

                if not 'ImageType' in dicomDset:
                    dicomDset.ImageType = ['ORIGINAL', 'PRIMARY', 'M']

                if not 'SeriesNumber' in dicomDset:
                    dicomDset.SeriesNumber = 1

                if not 'SeriesDescription' in dicomDset:
                    dicomDset.SeriesDescription = ''

                if not 'InstanceNumber' in dicomDset:
                    dicomDset.InstanceNumber = 1

                # ----- Update DICOM header from MRD ImageHeader -----
                dicomDset.ImageType[2]               = imtype_map[mrdImg.image_type]
                dicomDset.PixelSpacing               = [float(mrdImg.field_of_view[0]) / mrdImg.data.shape[2], float(mrdImg.field_of_view[1]) / mrdImg.data.shape[3]]
                dicomDset.SliceThickness             = mrdImg.field_of_view[2]
                dicomDset.ImagePositionPatient       = [mrdImg.position[0], mrdImg.position[1], mrdImg.position[2]]
                dicomDset.ImageOrientationPatient    = [mrdImg.read_dir[0], mrdImg.read_dir[1], mrdImg.read_dir[2], mrdImg.phase_dir[0], mrdImg.phase_dir[1], mrdImg.phase_dir[2]]

                time_sec = mrdImg.acquisition_time_stamp/1000/2.5
                hour = int(np.floor(time_sec/3600))
                min  = int(np.floor((time_sec - hour*3600)/60))
                sec  = time_sec - hour*3600 - min*60
                dicomDset.AcquisitionTime            = "%02.0f%02.0f%02.6f" % (hour, min, sec)
                dicomDset.TriggerTime                = mrdImg.physiology_time_stamp[0] / 2.5

                # ----- Update DICOM header from MRD Image MetaAttributes -----
                if meta.get('SeriesDescription') is not None:
                    dicomDset.SeriesDescription = meta['SeriesDescription']

                if meta.get('SeriesDescriptionAdditional') is not None:
                    dicomDset.SeriesDescription = dicomDset.SeriesDescription + meta['SeriesDescriptionAdditional']

                if meta.get('ImageComment') is not None:
                    dicomDset.ImageComment = "_".join(meta['ImageComment'])

                if meta.get('ImageType') is not None:
                    dicomDset.ImageType = meta['ImageType']

                if (meta.get('ImageRowDir') is not None) and (meta.get('ImageColumnDir') is not None):
                    dicomDset.ImageOrientationPatient = [float(meta['ImageRowDir'][0]), float(meta['ImageRowDir'][1]), float(meta['ImageRowDir'][2]), float(meta['ImageColumnDir'][0]), float(meta['ImageColumnDir'][1]), float(meta['ImageColumnDir'][2])]

                if meta.get('RescaleIntercept') is not None:
                    dicomDset.RescaleIntercept = meta['RescaleIntercept']

                if meta.get('RescaleSlope') is not None:
                    dicomDset.RescaleSlope = meta['RescaleSlope']

                if meta.get('WindowCenter') is not None:
                    dicomDset.WindowCenter = meta['WindowCenter']

                if meta.get('WindowWidth') is not None:
                    dicomDset.WindowWidth = meta['WindowWidth']

                if meta.get('EchoTime') is not None:
                    dicomDset.EchoTime = meta['EchoTime']

                if meta.get('InversionTime') is not None:
                    dicomDset.InversionTime = meta['InversionTime']

                # Unhandled fields:
                # LUTFileName
                # ROI

                # Write DICOM files
                fileName = "%02.0f_%s_%03.0f.dcm" % (dicomDset.SeriesNumber, dicomDset.SeriesDescription, dicomDset.InstanceNumber)
                print("  Writing file %s" % fileName)
                dicomDset.save_as(os.path.join(args.out_folder, fileName))
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MRD image file to DICOM files',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename',           help='Input file')
    parser.add_argument('-g', '--in-group',   help='Input data group')
    parser.add_argument('-o', '--out-folder', help='Output folder')

    args = parser.parse_args()

    main(args)
