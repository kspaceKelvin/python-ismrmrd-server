#!/usr/bin/python3

import contextlib
import io
import os
import argparse
import h5py
import ismrmrd
import numpy as np
import mrdhelper
import matplotlib.pyplot as plt
import inspect
from typing import List, Tuple, Dict, Optional, Any
from PIL import Image, ImageDraw, PngImagePlugin

defaults = {
    'in_group':         '',
    'rescale':          1,
    'fps':              25,
    'no_mosaic_slices': False,
    'mosaic_less_than': 6,
    'filetype':         'gif',
    'series':           '',
    'quiet':            False
}

def ReadMrdImageSeries(dset: ismrmrd.Dataset, group: str) -> Tuple[List[Image.Image], List[List[Tuple]], List[Any], List[Any]]:
    """
    Read all images and metadata from a given series in an MRD dataset.

    Args:
        dset: Handle to open MRD dataset.
        group: Group (series) of images to read.
    
    Returns:
        Tuple of (images, rois, heads, metas) where:
        - images: Images in PIL.Image format
        - rois: ROI data for each image (tuples of x, y, rgb, thickness)
        - heads: ImageHeaders for each image
        - metas: MetaAttributes for each image
    """
    images = []
    rois   = []
    heads  = []
    metas  = []

    warnIfComplex = True
    for imgNum in range(0, dset.number_of_images(group)):
        image = dset.read_image(group, imgNum)

        if ((image.data.shape[0] == 3) and (image.getHead().image_type == 6)):
            # RGB images
            data = np.squeeze(image.data.transpose((2, 3, 0, 1))) # Transpose to [row col rgb]
            data = data.astype(np.uint8)                          # Stored as uint16 as per MRD specification, but uint8 required for PIL
            images.append(Image.fromarray(data, mode='RGB'))
        else:
            data = image.data
            if np.iscomplexobj(data):
                if warnIfComplex:
                    print("  Converting images in series %s from complex to magnitude" % group)
                    warnIfComplex = False
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

        # Don't use consider channels dimension for RGB images
        if image.getHead().image_type == 6:
            numchasli = image.data.shape[1]
        else:
            numchasli = image.data.shape[0]*image.data.shape[1]

        # Same ROIs for each channel and slice (in a single MRD image)
        for chasli in range(numchasli):
            rois.append(imgRois)

        # MRD ImageHeader
        for chasli in range(numchasli):
            heads.append(image.getHead())

        for chasli in range(numchasli):
            metas.append(meta)

    return (images, rois, heads, metas)

def ApplyWindowColormapROI(images, rois, heads, metas, rescale):
    """
    Apply window/level, colormaps, and ROIs to images if applicable.

    Window/level from MetaAttributes are used when available, otherwise
    percentile scaling is used.  Colormaps are loaded from correspondingly
    named .npy files and applied as a palette.  ROIs are drawn into the
    image pixel data.
    
    Args:
        images: Images in PIL.Image format.
        rois: ROI data for each image (tuples of x, y, rgb, thickness).
        heads: ImageHeaders for each image.
        metas: MetaAttributes for each image.
        rescale: Rescale image dimensions by this factor.

    Returns:
        Images after windowing, colormap, and ROI processing.
    """

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
            # LUT file is a (256,3) numpy array of RGB values between 0 and 255
            LUTFileName = os.path.splitext(LUTFileName)[0] + '.npy'

            LUTPath = None
            dirs = ['',
                    'colormaps',
                    os.path.dirname(os.path.abspath(__file__)), 
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'colormaps')]

            for dir in dirs:
                testPath = os.path.join(dir, LUTFileName)
                if os.path.exists(testPath):
                    LUTPath = testPath
                    break

            if LUTPath:
                palette = np.load(LUTPath)
                palette = palette.flatten().tolist()  # As required by PIL
                print("  Applying LUT file %s" % (LUTPath))
            elif os.path.splitext(LUTFileName)[0] in plt.colormaps():
                cmap = plt.get_cmap(os.path.splitext(LUTFileName)[0])
                palette = cmap(np.linspace(0, 1, 256))[:,:3] * 255
                palette = palette.astype(int).flatten().tolist()
                print("  Applying LUT %s from matplotlib colormap library" % (os.path.splitext(LUTFileName)[0]))
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

                if rescale != 1:
                    tmpImg = tmpImg.resize(tuple(rescale*x for x in tmpImg.size))
                    for i in range(len(roi)):
                        roi[i] = tuple(([rescale*x for x in roi[i][0]], [rescale*y for y in roi[i][1]], roi[i][2], roi[i][3]))

                for (x, y, rgb, thickness) in roi:
                    draw = ImageDraw.Draw(tmpImg)
                    draw.line(list(zip(x, y)), fill=(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255), 255), width=int(thickness))
                imagesWL.append(tmpImg)
            else:
                data = np.array(img).astype(float)
                data -= minVal
                data *= 255/(maxVal - minVal)
                data = np.clip(data, 0, 255).astype(np.uint8)

                if palette is not None:
                    tmpImg = Image.fromarray(data, mode='P')
                    tmpImg.putpalette(palette)
                    imagesWL.append(tmpImg)
                else:
                    imagesWL.append(Image.fromarray(data))
        else:
            imagesWL.append(img)

    return imagesWL

def MosaicImageData(images: List[Any], rows: Optional[int] = None, cols: Optional[int] = None) -> np.ndarray:
    """
    Create a tiled mosaic of images.

    Create a mosaic image from a list of input images. Rows and cols
    can be provided or automatically calculated.
    
    Args:
        images: Images in PIL.Image format or numpy arrays.
        rows: Number of rows. Automatically calculated if None.
        cols: Number of cols. Automatically calculated if None.

    Returns:
        Mosaic image as numpy array.
    """

    shape = np.array(images[0]).shape
    numImages = len(images)

    if rows is None and cols is not None:
        rows = np.ceil(numImages/cols).astype(int)
    elif rows is not None and cols is None:
        cols = np.ceil(numImages/rows).astype(int)
    elif rows is None and cols is None:
        targetAspectRatio = 16/9
        imageAspectRatio = shape[1]/shape[0]

        cols = np.ceil(np.sqrt(numImages/targetAspectRatio/imageAspectRatio)).astype(int)
        rows = np.ceil(numImages/cols).astype(int)

        # Prevent images that are too tall
        if (shape[1]*cols) / (shape[0]*rows) < 0.67:
            # print(f'Mosaic is shape: ({shape[0]*rows}, {shape[1]*cols}) ({rows} rows x {cols} cols), with an aspect ratio of {(shape[1]*cols) / (shape[0]*rows)}, lower than threshold -- Overriding mosaic to have one less row')
            rows = np.max((rows-1, 1))
            cols = np.ceil(numImages / rows).astype(int)

    # Prevent blank tile with single row/col mosaics
    if (cols == 1) and (rows != numImages):
        rows = numImages

    if (rows == 1) and (cols != numImages):
        cols = numImages

    if rows*cols < numImages:
        print('Error: {rows} rows x {cols} cols is insufficient for {numImages} images')
        return None

    height = shape[0]*rows
    width  = shape[1]*cols
    mosaicArray = np.zeros((height, width) + shape[2:], dtype=np.array(images[0]).dtype)

    # print(f'Mosaic is shape: {mosaicArray.shape} ({rows} rows x {cols} cols)')

    for row in range(rows):
        for col in range(cols):
            idx = row*cols + col
            if idx >= len(images):
                continue

            mosaicArray[row*shape[0]:(row+1)*shape[0], col*shape[1]:(col+1)*shape[1], ...] = images[idx]

    return mosaicArray

def MosaicImages(images: List[Image.Image], heads: List[Any], mosaic_less_than: int = 6) -> List[Image.Image]:
    """
    Combine multiple images into a mosaic.

    Combine images into a grid mosaic sorted by slice or contrast.
    Also mosaic if there are only a small number of images total.
    
    Args:
        images: Images in PIL.Image format.
        heads: ImageHeaders for each image.
        mosaic_less_than: Create mosaic if fewer than this many images.
    
    Returns:
        Images after mosaicing, if applicable.
    """

    slices    = [head.slice    for head in heads]
    contrasts = [head.contrast for head in heads]

    # Create a list where each element contains all images for a given mosaic cell
    imagesSplit = []
    imagesMosaic = []

    # Option 1: Mosaic across slices
    if np.unique(slices).size > 1:
        for slice in np.unique(slices):
            imagesSplit.append([img for img, sli in zip(images, slices) if sli == slice])

        if np.unique([len(imgs) for imgs in imagesSplit]).size > 1:
            print('  ERROR: Failed to create mosaic because not all slices have the same number of images -- skipping mosaic!')
            imagesSplit = []
        else:
            print(f'  Creating a mosaic of {len(imagesSplit[0])} images with {np.unique(slices).size} slices in each')

    # Option 2: Mosaic across contrasts
    elif np.unique(contrasts).size > 1:
        for contrast in np.unique(contrasts):
            imagesSplit.append([img for img, con in zip(images, contrasts) if con == contrast])

        if np.unique([len(imgs) for imgs in imagesSplit]).size > 1:
            print('  ERROR: Failed to create mosaic because not all contrasts have the same number of images -- skipping mosaic!')
            imagesSplit = []
        else:
            print(f'  Creating a mosaic of {len(imagesSplit[0])} images with {np.unique(contrasts).size} contrasts in each')

    # Option 3: Mosaic across a small number of images
    elif (len(images) > 1) and (len(images) < mosaic_less_than):
        print(f'  Creating a mosaic of {len(images)} images (number of images in series is <{mosaic_less_than})')
        imagesSplit = images.copy()

        # Make sure all images have the same mode
        modes = set([img.mode for img in imagesSplit])
        if len(modes) > 1:
            print('  Warning: Series has images with mixed modes of type: ' + ', '.join(modes) + ' -- converting to P type')
            if 'P' not in modes:
                raise Exception('Unhandled case of mixed modes without a P type')

            for i, img in enumerate(imagesSplit):
                if img.mode != 'P':
                    imagesSplit[i] = imagesSplit[i].convert('P')

        # Create (single-frame) mosaic
        imgMode = imagesSplit[0].mode
        tmpImg = Image.fromarray(MosaicImageData(imagesSplit), mode=imgMode)
        
        if imgMode == 'P':
            palette = imagesSplit[0].getpalette()
            tmpImg.putpalette(palette)
        imagesMosaic = [tmpImg]
        imagesSplit = []

    if imagesSplit:
        # Loop over time dimension
        imagesMosaic = []
        for idx in range(len(imagesSplit[0])):
            imgMode = imagesSplit[0][idx].mode
            tmpImg = Image.fromarray(MosaicImageData([img[idx] for img in imagesSplit]), mode=imgMode)
            if imgMode == 'P':
                palette = imagesSplit[0][0].getpalette()
                tmpImg.putpalette(palette)
            imagesMosaic.append(tmpImg)

    if imagesMosaic:
        return imagesMosaic
    else:
        return images

def main(args: argparse.Namespace) -> None:
    """
    Outer main function that allows suppression of stdout
    """
    ctx = contextlib.redirect_stdout(io.StringIO()) if getattr(args, 'quiet', False) else contextlib.nullcontext()
    with ctx:
        _main_inner(args)

def _main_inner(args: argparse.Namespace) -> None:
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

            # Skip processing of all series other the one defined
            if args.series and group != args.series:
                continue

            print("Reading images from '/" + args.in_group + "/" + group + "'")

            (images, rois, heads, metas) = ReadMrdImageSeries(dset, group)
            print("  Read in %s images of shape %s" % (len(images), images[0].size[::-1]))

            images = ApplyWindowColormapROI(images, rois, heads, metas, args.rescale)

            if not args.no_mosaic_slices:
                images = MosaicImages(images, heads, args.mosaic_less_than)

            # Add SequenceDescriptionAdditional to filename, if present
            seqDescription = ''
            if metas:
                if 'SequenceDescriptionAdditional' in metas[0].keys():
                    seqDescription = '_' + metas[0]['SequenceDescriptionAdditional']
                elif 'GADGETRON_SeqDescription' in metas[0].keys():
                    if isinstance(metas[0]['GADGETRON_SeqDescription'], str):
                        seqDescription = '_' + metas[0]['GADGETRON_SeqDescription'].lstrip('_')
                    else:
                        seqDescription = '_' + '_'.join([s.lstrip('_') for s in metas[0]['GADGETRON_SeqDescription']])

            fileType = args.filetype.lstrip('.').lower()

            # GIF only supports 256 colors; switch to PNG if any RGB image exceeds that
            if fileType == 'gif' and any(img.mode == 'RGB' for img in images):
                for img in images:
                    if img.mode == 'RGB' and img.getcolors(maxcolors=256) is None:
                        print("  RGB image has >256 colors -- switching to .png to avoid dithering")
                        fileType = 'png'
                        break

            # Add MetaAttributes to GIF/PNG image (first MRD image only)
            saveargs = {}
            if metas:
                if fileType == 'png':
                    metadata = PngImagePlugin.PngInfo()
                    metadata.add_text("MetaAttributes", metas[0].serialize())
                    saveargs = {'pnginfo': metadata}
                elif fileType == 'gif':
                    saveargs = {'comment': metas[0].serialize().encode('utf-8')}

            # Make valid file name
            outFileName = os.path.splitext(os.path.basename(args.filename))[0] + '_' + args.in_group + '_' + group + seqDescription + '.' + fileType
            outFileName = "".join(c for c in outFileName if c.isalnum() or c in (' ','.','_')).rstrip()
            outFileName = outFileName.replace(" ", "_")
            outFilePath = os.path.join(os.path.dirname(args.filename), outFileName)

            print("  Writing image: %s " % (outFilePath))
            if len(images) > 1:
                images[0].save(outFilePath, save_all=True, append_images=images[1:], loop=0, duration=1000/args.fps, **saveargs)
            else:
                images[0].save(outFilePath, save_all=True, append_images=images[1:], **saveargs)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MRD image file to animated GIF',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename',                                      help='Input file')
    parser.add_argument('-g', '--in-group',                              help='Input data group')
    parser.add_argument('-r', '--rescale',          type=int,            help='Rescale factor (integer) for output images')
    parser.add_argument(      '--fps',              type=int,            help='Frame rate for animated images')
    parser.add_argument(      '--no-mosaic-slices', action='store_true', help='Do not mosaic images along slice dimension')
    parser.add_argument(      '--mosaic-less-than', type=int,            help='Mosaic images with less than this number of images in series')
    parser.add_argument(      '--filetype',         type=str,            help='File type for output images (gif or png)')
    parser.add_argument('-s', '--series',           type=str,            help='Process only this single series (e.g. image_0)')
    parser.add_argument('-q', '--quiet',            action='store_true', help='Suppress all stdout output')

    parser.set_defaults(**defaults)

    args = parser.parse_args()

    main(args)
