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
    'ref_filename':     '',
    'ref_in_group':     '',
    'ref_diff_scale':   1.0,
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

def GetMetaValueFromCandidates(meta: Dict[str, Any], *keys: str) -> Optional[Any]:
    """
    Return the first metadata value found for a list of candidate keys.

    Args:
        meta: MetaAttributes dictionary-like object.
        *keys: Ordered candidate keys to query.

    Returns:
        First non-None value, or None if no keys are present.
    """

    for key in keys:
        value = meta.get(key)
        if value is not None:
            return value

    return None

def ComputeWindowRanges(images: List[Any], metas: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    """
    Compute series-wide window/level range for an image series.

    Args:
        images: Images in PIL.Image format or numpy arrays.
        metas: MetaAttributes for each image.

    Returns:
        Tuple of (minVals, maxVals), each a per-image list.
        If all images have explicit metadata window/level values, use those per
        image values.  Otherwise, fall back to one series-wide range for all images.
    """

    if len(images) == 0:
        return ([], [])

    # Window/level defaults for all images in series
    seriesMaxVal = np.median([np.percentile(np.array(img), 95) for img in images])
    seriesMinVal = np.median([np.percentile(np.array(img),  5) for img in images])

    # Special case for "sparse" images, usually just text
    if seriesMaxVal == seriesMinVal:
        seriesMaxVal = np.median([np.max(np.array(img)) for img in images])
        seriesMinVal = np.median([np.min(np.array(img)) for img in images])

    # Extract window/level from each image's metadata
    metaRanges = []
    for meta in metas:
        windowCenter = GetMetaValueFromCandidates(meta, 'WindowCenter', 'GADGETRON_WindowCenter')
        windowWidth  = GetMetaValueFromCandidates(meta, 'WindowWidth',  'GADGETRON_WindowWidth')

        # In Gadgetron data, GADGETRON_WindowWidth does not take into account GADGETRON_ScaleRatio (contrary to DICOM interpretation)
        if (meta.get('GADGETRON_ScaleRatio') is not None) and (meta.get('GADGETRON_ScaleOffset') is not None):
            if (meta.get('GADGETRON_WindowWidth') is not None) and (meta.get('WindowWidth') is None):
                windowWidth = str(float(windowWidth) / float(meta.get('GADGETRON_ScaleRatio')))

        if (windowCenter is None) or (windowWidth is None):
            metaRanges.append(None)
        else:
            metaRanges.append((
                float(windowCenter) - float(windowWidth)/2,
                float(windowCenter) + float(windowWidth)/2,
            ))

    nonNoneRanges = [r for r in metaRanges if r is not None]

    # Case 1: No metadata ranges -> use series-wide range for all images.
    if len(nonNoneRanges) == 0:
        return ([seriesMinVal] * len(images), [seriesMaxVal] * len(images))

    # Case 2: Metadata ranges for all images -> use per-image metadata ranges.
    if len(metaRanges) == len(images) and all(r is not None for r in metaRanges):
        perImageRanges = [r for r in metaRanges if r is not None]
        if len(set(perImageRanges)) > 1:
            print('  Using per-image MetaAttribute window/level values')
        minVals, maxVals = zip(*perImageRanges)
        return (list(minVals), list(maxVals))

    # Case 3: Metadata ranges for only some images -> warn and fill missing
    # with the first metadata range.
    firstMetaRange = nonNoneRanges[0]
    print('  Warning: MetaAttribute window/level is present for only some images; using first MetaAttribute window/level for images without MetaAttribute window/level')
    filledRanges = [r if r is not None else firstMetaRange for r in metaRanges]
    minVals, maxVals = zip(*filledRanges)
    return (list(minVals), list(maxVals))

def ApplyRescaleSlopeIntercept(images: List[Image.Image], metas: List[Dict[str, Any]]) -> List[Image.Image]:
    """
    Apply per-image RescaleSlope/RescaleIntercept from MetaAttributes.

    For each image, if both RescaleSlope and RescaleIntercept are present in
    MetaAttributes, compute:
        val = raw * RescaleSlope + RescaleIntercept

    If these aren't present, apply Gadgetron ScaleOffset and ScaleRatio:
        val = (raw - GADGETRON_ScaleOffset) / GADGETRON_ScaleRatio 

    Args:
        images: Input images in PIL.Image format.
        metas: MetaAttributes for each image.

    Returns:
        Images with scaling applied only when both parameters are present.
        Images without applicable parameters are returned unchanged.
    """

    imagesScaled = []
    for imgIdx, (img, meta) in enumerate(zip(images, metas)):
        RescaleSlope     = meta.get('RescaleSlope')
        RescaleIntercept = meta.get('RescaleIntercept')

        if (RescaleSlope is None) or (RescaleIntercept is None):
            # Fall back to Gadgetron attribs
            ScaleRatio   = meta.get('GADGETRON_ScaleRatio')
            ScaleOffset  = meta.get('GADGETRON_ScaleOffset')

            if (ScaleRatio is not None) and (ScaleOffset is not None):
                RescaleSlope = 1.0 / float(ScaleRatio)
                RescaleIntercept = -1.0 * float(ScaleOffset) / float(ScaleRatio)

        if (RescaleSlope is None) or (RescaleIntercept is None):
            imagesScaled.append(img)
            continue

        if not np.isscalar(RescaleSlope):
            raise TypeError("RescaleSlope must be a scalar value, got %s for image index %d" % (type(RescaleSlope).__name__, imgIdx))
        if not np.isscalar(RescaleIntercept):
            raise TypeError("RescaleIntercept must be a scalar value, got %s for image index %d" % (type(RescaleIntercept).__name__, imgIdx))

        try:
            data = np.array(img).astype(np.float32)
            data = data*float(RescaleSlope) + float(RescaleIntercept)
            imagesScaled.append(Image.fromarray(data))

        except (TypeError, ValueError):
            imagesScaled.append(img)

    return imagesScaled

def ApplyWindowLevel(images: List[Image.Image], minVals: List[float], maxVals: List[float]) -> List[Image.Image]:
    """
    Apply window/level scaling and clip to display range.

    Args:
        images: Input images in PIL.Image format.
        minVals: Per-image lower window bounds.
        maxVals: Per-image upper window bounds.

    Returns:
        Window-leveled images in PIL.Image format with values in [0, 255].
    """

    imagesWL = []
    for img, minVal, maxVal in zip(images, minVals, maxVals):
        dataWL = np.array(img).astype(float) - minVal
        if maxVal != minVal:
            dataWL *= 255/(maxVal - minVal)
        dataWL = np.clip(dataWL, 0, 255).astype(np.uint8)
        imagesWL.append(Image.fromarray(dataWL))

    return imagesWL

def ApplyColormapROI(images: List[Image.Image], rois: List[List[Tuple]], heads: List[Any], metas: List[Dict[str, Any]], rescale: float) -> List[Image.Image]:
    """
    Apply colormaps and ROIs to images if applicable.

    Colormaps are loaded from correspondingly named .npy files and applied as
    a palette. ROIs are drawn into the image pixel data.
    
    Args:
        images: Images in PIL.Image format.
        rois: ROI data for each image (tuples of x, y, rgb, thickness).
        heads: ImageHeaders for each image.
        metas: MetaAttributes for each image.
        rescale: Rescale image dimensions by this factor.

    Returns:
        Images after colormap and ROI processing.
    """

    hasRois = any([len(x) > 0 for x in rois])

    imagesWL = []
    for img, roi, meta in zip(images, rois, metas):
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
                data = np.array(img)
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
                data = np.array(img, dtype=np.uint8)

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
            print("Input group not specified -- selecting most recent: %s" % list(dset.keys())[-1])
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

    # Determine the most appropriate dataset (group) in the reference file
    refInGroup = args.ref_in_group
    if args.ref_filename:
        try:
            with h5py.File(args.ref_filename, 'r') as dsetRefMeta:
                refGroups = list(dsetRefMeta.keys())

            print("Reference file %s contains %d groups:" % (args.ref_filename, len(refGroups)))
            print(" ", "\n  ".join(refGroups))

            if not refInGroup:
                print("Reference group not specified -- selecting most recent: %s" % refGroups[-1])
                refInGroup = refGroups[-1]
        except Exception as ex:
            print("Could not inspect reference file groups: %s" % ex)

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

            # Apply RescaleSlope and RescaleIntercept, if applicable
            images = ApplyRescaleSlopeIntercept(images, metas)

            # Keep original image data for diff
            imagesRaw = [np.array(img).astype(np.float32) for img in images]

            # Compute window/level using MetaAttributes if present, otherwise 5/95th percentile of pixel values
            minVals, maxVals = ComputeWindowRanges(images, metas)
            images = ApplyWindowLevel(images, minVals, maxVals)

            images = ApplyColormapROI(images, rois, heads, metas, args.rescale)

            is_diff = False
            # If applicable, load in a reference dataset for comparison/diff
            try:
                with ismrmrd.Dataset(args.ref_filename, refInGroup, create_if_needed=False, **modeargs) as dsetRef:
                    (imagesRef, roisRef, headsRef, metasRef) = ReadMrdImageSeries(dsetRef, group)
                print("  Read in %s reference images of shape %s" % (len(imagesRef), imagesRef[0].size[::-1]))

                # Apply RescaleSlope and RescaleIntercept, if applicable
                imagesRef = ApplyRescaleSlopeIntercept(imagesRef, metasRef)

                if len(imagesRef) != len(images):
                    print("  Warning: Number of reference images (%d) does not match number of source images (%d)" % (len(imagesRef), len(images)))
                    continue

                imagesRefRaw = [np.array(img).astype(np.float32) for img in imagesRef]

                # Apply the same source-derived window/level to the reference images
                # so displayed source/reference images are directly comparable.
                imagesRef = ApplyWindowLevel(imagesRef, minVals, maxVals)
                imagesRef = ApplyColormapROI(imagesRef, roisRef, headsRef, metasRef, args.rescale)

                # Create a vertically stacked combination of (image, reference image, difference)
                imagesCombined = []
                for img, imgRef, imgRaw, imgRefRaw, imgMinVal, imgMaxVal in zip(images, imagesRef, imagesRaw, imagesRefRaw, minVals, maxVals):
                    imgMode = img.mode

                    if len(imgRaw.shape) <= 2 or len(imgRefRaw.shape) <= 2:
                        # Calculate diff first in raw space for non-RGB images, then apply the same
                        # windowing used for display so grayscale diff intensity is consistent.
                        rawDiff = np.abs(imgRaw - imgRefRaw)
                        rawDiff = np.array(ApplyWindowLevel([Image.fromarray(rawDiff.astype(np.float32))], [imgMinVal], [imgMaxVal])[0], dtype=np.float32)
                        diffImg = rawDiff
                    else:
                        diffImg = np.zeros_like(imgRaw)

                    if imgMode == 'RGB':
                        # For images with ROIs, the ROI is burned into the image and converted to RGB.
                        # Calculate the diff in display space so that the diff can capture differences
                        # in both the underlying image and the ROI overlay.  For native RGB images,
                        # this should be identical to the raw diff
                        displayDiff = np.abs(np.array(img).astype(np.float32) - np.array(imgRef).astype(np.float32))

                        # Keep the strongest response from either the display-space
                        # diff or the windowed raw diff at each pixel.
                        diffImg = np.maximum(displayDiff.astype(np.float32), np.atleast_3d(diffImg).astype(np.float32))
                    else:
                        # Apply optional scaling to improve visibility of scalar images
                        diffImg = diffImg*args.ref_diff_scale

                    # Clip diff images to 0-255, which has been set previously by ApplyWindowColormapROI for other images
                    diffImg = np.clip(diffImg, 0, 255).astype(np.array(img).dtype)

                    tmpImg = Image.fromarray(np.vstack([img, imgRef, diffImg]), mode=imgMode)
                    if imgMode == 'P':
                        palette = img.getpalette()
                        tmpImg.putpalette(palette)

                    imagesCombined.append(tmpImg)

                images = imagesCombined
                is_diff = True
            except:
                pass

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
            diffSuffix = '_diff' if is_diff else ''
            outFileName = os.path.splitext(os.path.basename(args.filename))[0] + '_' + args.in_group + '_' + group + seqDescription + diffSuffix + '.' + fileType
            outFileName = "".join(c for c in outFileName if c.isalnum() or c in (' ','.','-','_')).rstrip()
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
    parser.add_argument(      '--ref-filename',     type=str,            help='Reference file to compare against')
    parser.add_argument(      '--ref-in-group',     type=str,            help='Data group in reference file')
    parser.add_argument(      '--ref-diff-scale',   type=float,          help='Scaling factor for difference image')
    parser.add_argument('-s', '--series',           type=str,            help='Process only this single series (e.g. image_0)')
    parser.add_argument('-q', '--quiet',            action='store_true', help='Suppress all stdout output')

    parser.set_defaults(**defaults)

    args = parser.parse_args()

    main(args)
