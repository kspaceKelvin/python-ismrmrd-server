#!/usr/bin/python3

import os
import argparse
import h5py
import ismrmrd
import numpy as np
import mrdhelper
import inspect
from PIL import Image, ImageDraw

defaults = {
    'in_group':         '',
    'rescale':          1,
    'no_mosaic_slices': False,
    'mosaic_less_than': 6
}

def main(args):
    with h5py.File(args.filename, 'r') as dset:
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

        if args.in_group not in dset:
            print("Could not find group %s" % (args.in_group))
            return

        group = dset.get(args.in_group)
        print("Reading data from group '%s' in file '%s'" % (args.in_group, args.filename))

        # Image data is stored as:
        #   /group/config              text of recon config parameters (optional)
        #   /group/xml                 text of ISMRMRD flexible data header (optional)
        #   /group/image_0/data        array of IsmrmrdImage data
        #   /group/image_0/header      array of ImageHeader
        #   /group/image_0/attributes  text of image MetaAttributes
        isImage = True
        imageNames = group.keys()
        print("Found %d image sub-groups: %s" % (len(imageNames), ", ".join(imageNames)))

        for imageName in imageNames:
            if ((imageName == 'xml') or (imageName == 'config') or (imageName == 'config_file')):
                continue

            image = group[imageName]
            if not (('data' in image) and ('header' in image) and ('attributes' in image)):
                isImage = False

    if (isImage is False):
        print("File does not contain properly formatted MRD image data")
        return

    if 'mode' in inspect.signature(ismrmrd.Dataset).parameters:
        modeargs = {'mode': 'r'}
    else:
        modeargs = {}

    with ismrmrd.Dataset(args.filename, args.in_group, create_if_needed=False, **modeargs) as dset:
        groups = dset.list()
        for group in groups:
            if ( (group == 'config') or (group == 'config_file') or (group == 'xml') ):
                continue

            print("Reading images from '/" + args.in_group + "/" + group + "'")

            images = []
            rois   = []
            heads  = []
            metas  = []
            for imgNum in range(0, dset.number_of_images(group)):
                image = dset.read_image(group, imgNum)

                if ((image.data.shape[0] == 3) and (image.getHead().image_type == 6)):
                    # RGB images
                    data = np.squeeze(image.data.transpose((2, 3, 0, 1))) # Transpose to [row col rgb]
                    data = data.astype(np.uint8)                          # Stored as uint16 as per MRD specification, but uint8 required for PIL
                    images.append(Image.fromarray(data, mode='RGB'))
                else:
                    data = image.data
                    if np.any(np.iscomplex(data)):
                        print("  Converting image %d from complex to magnitude for display" % imgNum)
                        data = np.abs(data)

                    for cha in range(data.shape[0]):
                        for sli in range(data.shape[1]):
                            images.append(Image.fromarray(np.squeeze(data[cha,sli,...])))  # data is [cha z y x] -- squeeze to [y x] for [row col]

                if image.data.shape[0] > 1:
                    if image.getHead().image_type == 6:
                        print("  Image %d is RGB" % imgNum)
                    else:
                        print("  Image %d has %d channels" % (imgNum, image.data.shape[0]))

                if image.data.shape[1] > 2:
                    print("  Image %d is a 3D volume with %d slices" % (imgNum, image.data.shape[1]))

                # Read ROIs
                meta = ismrmrd.Meta.deserialize(image.attribute_string)
                imgRois = []
                for key in meta.keys():
                    if not key.startswith('ROI_') and not key.startswith('GT_ROI_'):
                        continue

                    roi = meta[key]
                    x, y, rgb, thickness, style, visibility = mrdhelper.parse_roi(roi)

                    if visibility == 0:
                        continue

                    imgRois.append((x, y, rgb, thickness))

                # Same ROIs for each channel and slice (in a single MRD image)
                for chasli in range(image.data.shape[0]*image.data.shape[1]):
                    rois.append(imgRois)

                # MRD ImageHeader
                for chasli in range(image.data.shape[0]*image.data.shape[1]):
                    heads.append(image.getHead())

                for chasli in range(image.data.shape[0]*image.data.shape[1]):
                    metas.append(meta)

            print("  Read in %s images of shape %s" % (len(images), images[0].size[::-1]))

            hasRois = any([len(x) > 0 for x in rois])

            # Window/level for all images in series
            seriesMaxVal = np.median([np.percentile(np.array(img), 95) for img in images])
            seriesMinVal = np.median([np.percentile(np.array(img),  5) for img in images])

            # Special case for "sparse" images, usually just text
            if seriesMaxVal == seriesMinVal:
                seriesMaxVal = np.median([np.max(np.array(img)) for img in images])
                seriesMinVal = np.median([np.min(np.array(img)) for img in images])

            imagesWL = []
            for img, roi, meta in zip(images, rois, metas):
                # Use window/level from MetaAttributes if available
                minVal = seriesMinVal
                maxVal = seriesMaxVal

                if (('WindowCenter' in meta) and ('WindowWidth' in meta)):
                    minVal = float(meta['WindowCenter']) - float(meta['WindowWidth'])/2
                    maxVal = float(meta['WindowCenter']) + float(meta['WindowWidth'])/2
                elif (('GADGETRON_WindowCenter' in meta) and ('GADGETRON_WindowWidth' in meta)):
                    minVal = float(meta['GADGETRON_WindowCenter']) - float(meta['GADGETRON_WindowWidth'])/2
                    maxVal = float(meta['GADGETRON_WindowCenter']) + float(meta['GADGETRON_WindowWidth'])/2

                if ('LUTFileName' in meta) or ('GADGETRON_ColorMap' in meta):
                    LUTFileName = meta['LUTFileName'] if 'LUTFileName' in meta else meta['GADGETRON_ColorMap']

                    # Replace extension with '.npy'
                    LUTFileName = os.path.splitext(LUTFileName)[0] + '.npy'

                    # LUT file is a (256,3) numpy array of RGB values between 0 and 255
                    if os.path.exists(LUTFileName):
                        palette = np.load(LUTFileName)
                        palette = palette.flatten().tolist()  # As required by PIL
                    # Look in subdirectory 'colormaps' if not found in current directory
                    elif os.path.exists(os.path.join('colormaps', LUTFileName)):
                        palette = np.load(os.path.join('colormaps', LUTFileName))
                        palette = palette.flatten().tolist()  # As required by PIL
                    else:
                        print("LUT file %s specified by MetaAttributes, but not found" % (LUTFileName))
                        palette = None
                else:
                    palette = None

                if img.mode != 'RGB':
                    if hasRois:
                        # Convert to RGB mode to allow colored ROI overlays
                        data = np.array(img).astype(float)
                        data -= minVal
                        if maxVal != minVal:
                            data *= 255/(maxVal - minVal)
                        data = np.clip(data, 0, 255)
                        if palette is not None:
                            tmpImg = Image.fromarray(data.astype(np.uint8), mode='P')
                            tmpImg.putpalette(palette)
                            tmpImg = tmpImg.convert('RGB')  # Needed in order to draw ROIs
                        else:
                            tmpImg = Image.fromarray(np.repeat(data[:,:,np.newaxis],3,axis=2).astype(np.uint8), mode='RGB')

                        # Gadgetron compatibility: x/y are swapped in ROIs if Correct_image_orientation is true
                        if ('Correct_image_orientation' in meta) and (meta['Correct_image_orientation'] != 0):
                            print('  Swapping x/y dimension of ROIs due to Correct_image_orientation...')
                            for i in range(len(roi)):
                                roi[i] = tuple((roi[i][1], roi[i][0], roi[i][2], roi[i][3]))

                        if args.rescale != 1:
                            tmpImg = tmpImg.resize(tuple(args.rescale*x for x in tmpImg.size))
                            for i in range(len(roi)):
                                roi[i] = tuple(([args.rescale*x for x in roi[i][0]], [args.rescale*y for y in roi[i][1]], roi[i][2], roi[i][3]))

                        for (x, y, rgb, thickness) in roi:
                            draw = ImageDraw.Draw(tmpImg)
                            draw.line(list(zip(x, y)), fill=(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255), 255), width=int(thickness))
                        imagesWL.append(tmpImg)
                    else:
                        data = np.array(img).astype(float)
                        data -= minVal
                        data *= 255/(maxVal - minVal)
                        data = np.clip(data, 0, 255)

                        if palette is not None:
                            tmpImg = Image.fromarray(data.astype(np.uint8), mode='P')
                            tmpImg.putpalette(palette)
                            imagesWL.append(tmpImg)
                        else:
                            imagesWL.append(Image.fromarray(data))
                else:
                    imagesWL.append(img)

            # Combine multiple slices into a mosaic
            isMosaic = False
            if not args.no_mosaic_slices:
                slices = [head.slice for head in heads]
                contrasts = [head.contrast for head in heads]

                if np.unique(slices).size > 1:
                    # Create a list where each element is all images from a given slice
                    imagesWLSplit = []
                    for slice in np.unique(slices):
                        imagesWLSplit.append([img for img, sli in zip(imagesWL, slices) if sli == slice])

                    if np.unique([len(imgs) for imgs in imagesWLSplit]).size > 1:
                        print('  ERROR: Failed to create mosaic because not all slices have the same number of images -- skipping mosaic!')
                    else:
                        print(f'  Creating a mosaic of {len(imagesWLSplit[0])} images with {np.unique(slices).size} slices in each')

                        # Loop over non-slice dimension
                        imagesWLMosaic = []
                        for idx in range(len(imagesWLSplit[0])):
                            imgMode = imagesWLSplit[0][idx].mode
                            tmpImg = Image.fromarray(np.hstack([img[idx] for img in imagesWLSplit]), mode=imgMode)
                            if imgMode == 'P':
                                palette = imagesWLSplit[0][0].getpalette()
                                tmpImg.putpalette(palette)
                            imagesWLMosaic.append(tmpImg)

                        imagesWL = imagesWLMosaic
                    isMosaic = True
                elif np.unique(contrasts).size > 1:
                    # Create a list where each element is all images from a given slice
                    imagesWLSplit = []
                    for contrast in np.unique(contrasts):
                        imagesWLSplit.append([img for img, con in zip(imagesWL, contrasts) if con == contrast])

                    if np.unique([len(imgs) for imgs in imagesWLSplit]).size > 1:
                        print('  ERROR: Failed to create mosaic because not all contrasts have the same number of images -- skipping mosaic!')
                    else:
                        print(f'  Creating a mosaic of {len(imagesWLSplit[0])} images with {np.unique(contrasts).size} contrasts in each')

                        # Loop over non-contrast dimension
                        imagesWLMosaic = []
                        for idx in range(len(imagesWLSplit[0])):
                            imgMode = imagesWLSplit[0][idx].mode
                            tmpImg = Image.fromarray(np.hstack([img[idx] for img in imagesWLSplit]), mode=imgMode)
                            if imgMode == 'P':
                                palette = imagesWLSplit[0][0].getpalette()
                                tmpImg.putpalette(palette)
                            imagesWLMosaic.append(tmpImg)

                        imagesWL = imagesWLMosaic
                    isMosaic = True

            # Mosaic a small number of images instead of making an animated GIF
            if (len(imagesWL) > 1) and (len(imagesWL) < args.mosaic_less_than) and not isMosaic:
                print(f'  Creating a mosaic of {len(imagesWL)} images (number of images in series is <{args.mosaic_less_than})')

                # Make sure all images have the same mode
                modes = set([img.mode for img in imagesWL])
                if len(modes) > 1:
                    print('  Warning: Series has images with mixed modes of type: ' + ', '.join(modes) + ' -- converting to P type')
                    if 'P' not in modes:
                        raise Exception('Unhandled case of mixed modes without a P type')

                    for i, img in enumerate(imagesWL):
                        if img.mode != 'P':
                            imagesWL[i] = imagesWL[i].convert('P')

                imagesWLMosaic = []

                imgMode = imagesWL[0].mode
                tmpImg = Image.fromarray(np.hstack([img for img in imagesWL]), mode=imgMode)
                if imgMode == 'P':
                    palette = imagesWL[0].getpalette()
                    tmpImg.putpalette(palette)
                imagesWLMosaic.append(tmpImg)

                imagesWL = imagesWLMosaic

            # Add SequenceDescriptionAdditional to filename, if present
            image = dset.read_image(group, 0)
            meta = ismrmrd.Meta.deserialize(image.attribute_string)
            if 'SequenceDescriptionAdditional' in meta.keys():
                seqDescription = '_' + meta['SequenceDescriptionAdditional']
            elif 'GADGETRON_SeqDescription' in meta.keys():
                if isinstance(meta['GADGETRON_SeqDescription'], str):
                    seqDescription = '_' + meta['GADGETRON_SeqDescription'].lstrip('_')
                else:
                    seqDescription = '_' + '_'.join([s.lstrip('_') for s in meta['GADGETRON_SeqDescription']])
            else:
                seqDescription = ''

            # Make valid file name 
            gifFileName = os.path.splitext(os.path.basename(args.filename))[0] + '_' + args.in_group + '_' + group + seqDescription + '.gif'
            gifFileName = "".join(c for c in gifFileName if c.isalnum() or c in (' ','.','_')).rstrip()
            gifFileName = gifFileName.replace(" ", "_")
            gifFilePath = os.path.join(os.path.dirname(args.filename), gifFileName)

            print("  Writing image: %s " % (gifFilePath))
            if len(imagesWL) > 1:
                imagesWL[0].save(gifFilePath, save_all=True, append_images=imagesWL[1:], loop=0, duration=40)
            else:
                imagesWL[0].save(gifFilePath, save_all=True, append_images=imagesWL[1:])

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MRD image file to animated GIF',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename',                                      help='Input file')
    parser.add_argument('-g', '--in-group',                              help='Input data group')
    parser.add_argument('-r', '--rescale',          type=int,            help='Rescale factor (integer) for output images')
    parser.add_argument(      '--no-mosaic-slices', action='store_true', help='Do not mosaic images along slice dimension')
    parser.add_argument(      '--mosaic-less-than', type=int,            help='Mosaic images with less than this number of images in series')

    parser.set_defaults(**defaults)

    args = parser.parse_args()

    main(args)
