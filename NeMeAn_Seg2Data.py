# Created on Monday November 12 14:23:00 2022
# @author: Andres
# Takes sematically segmented images of axon/myelin,
# postprocesses the segmentation, labels individual axons and
# associated myelin, and calculates the morphometrics of those
# fibers.

import os
from pathlib import Path
from tqdm import tqdm
import numpy as np

from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000 # to allow loading large images and not trigger the error " Image size exceeds limit of ### pixels, could be decompression bomb DOS attack."

import ray
import zarr
from skimage.util.shape import view_as_windows
import gc
import math
import skimage.morphology as sm
from skimage.measure import regionprops
import mahotas as mh



def _getAMFLabels(isAxon, isMyelin, crop2bbox=True):
    """
    Uses watershed algorithm to associate myelin with individual axons from the interpolated contours between the two masks.
    :param isAxon: Array, binary mask for which pixels are axons. Shape (H,W).
    :param isMyelin: Array, binary mask for which pixels are myelin. Shape (H,W).
    :param crop2bbox: Bool, whether to crop out the background along edge of masks during compuation to save time. Default is True.
    :return: Integer, number of axons in mask
             Array, integer mask with background at 0 and each axon given a label [1-num_axons]
             Array, integer mask with background at 0 and each myelin given the same label as associated axon
             Array, integer mask for total fiber labeled [1-num_axons].
    """
    
    # Crop masks to minimal bounding box to hopefully speed up computation
    if crop2bbox:
        originalShape = isAxon.shape
        tempIsFiber = isAxon|isMyelin
        tempFiberPropList = regionprops(tempIsFiber.astype(int))
        crop_R0, crop_C0, crop_R1, crop_C1 = tempFiberPropList[0].bbox
        
        isAxon = isAxon[crop_R0:crop_R1, crop_C0:crop_C1]
        isMyelin = isMyelin[crop_R0:crop_R1, crop_C0:crop_C1]
    
    
    # Set perimeter of fiber as height 1, axon as height 0, and interpolate hights to build countours for watershed
    isFiber = isAxon|isMyelin
    distA = np.sqrt(mh.distance(~isAxon))
    distB = np.sqrt(mh.distance(isFiber))
    split = 0.45
    distInterp = distA/(distA+distB)
    dist = np.zeros(isAxon.shape)
    dist[distInterp<=split] = distA[distInterp<=split]
    dist[distInterp>split] = dist.max()+distB.max()-distB[distInterp>split]
    dist[~isFiber] = dist.max()*10

    # Label individual axons and associated myelin
    labeledAxons, num_axons = mh.label(isAxon)
    labeledMyelin = mh.cwatershed(dist, labeledAxons)
    labeledMyelin[~isMyelin] = 0
    
    if crop2bbox:
        uncropped_labeledAxons = np.zeros(originalShape, dtype=labeledAxons.dtype)
        uncropped_labeledAxons[crop_R0:crop_R1, crop_C0:crop_C1] = labeledAxons
        labeledAxons = uncropped_labeledAxons
        
        uncropped_labeledMyelin = np.zeros(originalShape, dtype=labeledMyelin.dtype)
        uncropped_labeledMyelin[crop_R0:crop_R1, crop_C0:crop_C1] = labeledMyelin
        labeledMyelin = uncropped_labeledMyelin
    
    labeledFibers = labeledAxons+labeledMyelin
    return num_axons, labeledAxons, labeledMyelin, labeledFibers


def _calcDist(centroid1, centroid2):
    """ Calculates the Euclidean distance between two 2d coordinates """
    xDist = centroid2[0]-centroid1[0]
    yDist = centroid2[1]-centroid1[1]
    dist = math.sqrt(xDist**2+yDist**2)
    return dist

def _get_bbox_and_padShape(array):
    """ Calculates the bounding box of a 2d-array and padding needed to recreate ndarray """
    # bbox format used by scikit-image.regionprops: (min_row, min_col, max_row, max_col)
    r,c = array.shape
    cols = np.any(array, axis=0)
    rows = np.any(array, axis=1)
    # Get min and max row and column values needed to crop
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    # Try expanding bounding box a tiny bit to allow a small border at edges
    for _ in range(2):
        if rmin-1 >= 0:
            rmin = rmin-1
        if cmin-1 >= 0:
            cmin = cmin-1
        if rmax+1 <= r:
            rmax = rmax+1
        if cmax+1 <= c:
            cmax = cmax+1
    # Generate tuple for bbox cropping and tuple needed to reconstruct original array using np.pad()
    bbox = (rmin, cmin, rmax+1, cmax+1)
    padShape = ((rmin,r-rmax-1),(cmin,c-cmax-1))
    return bbox, padShape

def _getCellPackingDensity(labeledFibers, fiberPropList, windowShape=(400,400), imgLinearRes=0.125,
                           edgeCleanMethod='fractional', edgeSize=1, fascicleMask=None, circularWindow=False, return_density=True):
    """
    Computes the cell-wise fiber density and fiber packing by using a window centered on each cell.
    :param labeledFibers: Array, background labeled 0 and each fiber labeled [1-num_axons]. Shape (H,W).
    :param fiberPropList: List, each element describes one labeled fiber (skimage.measure.regionprops). Shape (num_axons).
    :param windowShape: Tuple, 2 integer elements describing the height and width of the window to be used for calculating fiber density and packing. Default is (400,400).
    :param imgLinearRes: Float, for the um/px resolution of the image. Default is 0.125.
    :param edgeCleanMethod: One of {'fractional', 'top_left_cut', 'all_cut', 'none'}
        Default is 'fractional'.
        - 'fractional': for cells cut by the window, calculates what fraction of the cell is inside the window and uses that for fiber density calculation (only counts portion of cells inside window for packing).
        - 'top_left_cut': ignore cells cut by the top and left edges of the window for fiber density and fiber packing calculations (only counts portion of non-ignored cells inside window for packing).
        - 'all_cut': ignore all cells cut by the edges of the window for fiber density and fiber packing calculations.
        - 'none': count all cells cut by the edges of the window for fiber density and fiber packing calculations (only counts portion of cells inside window for packing).
    :param edgeSize: Positive integer, how many pixels away from window edge to not be considered cut by window (only used with 'top_left_cut' and 'all_cut'). Default is 1.
    :param fascicleMask: Bool Array (or None), background labeled False and fascicle labeled True. Shape (H,W). If provided, restricts fiber packing/density window measurements to only the window within fascicles. Default is None.
    :param circularWindow: Boolean, for whether to use a circular window (with diameter equal to shortest value in windowShape) instead of a rectangle.
    :param return_density: Boolean, for whether to perform the calculation for density and return it.
    :return: 2 Arrays, the first containing the fiber density measured around each labeled cell, the second containing the fiber packing measured around each labeled cell. Each with shape (num_axons).
    """
        
    numFibers = len(fiberPropList) #labeledFibers.max()
    h, w = labeledFibers.shape
    
    # Check if window larger than image
    if (windowShape[0]>h) or (windowShape[1]>w):
        raise Exception('Window is larger than image')
    
    # Setup window bounds (if window size isn't an odd number [can't be centered on a single pixel], then "center" is the top/left of center pixels)
    if windowShape[0]%2 == 0:
        uy = ly = int(windowShape[0]/2)
        uy = uy-1 # subtract one to "center" on top pixel when 2 pixels at vertical center
        ly = ly+1 # add 1 so that slices end at ly
    else:
        uy = int((windowShape[0]-1)/2)
        ly = int((windowShape[0]-1)/2)
    if windowShape[1]%2 == 0:
        lx = rx = int(windowShape[1]/2)
        lx = lx-1 # subtract one to "center" on left pixel when 2 pixels at horizontal center
        rx = rx+1 # add 1 so that slixes end at rx
    else:
        lx = int((windowShape[1]-1)/2)
        rx = int((windowShape[1]-1)/2)
    
    # Pad image(s) so we don't need to worry about windows going beyond image boundaries
    labeledFibers = np.pad(labeledFibers, ((ly,uy),(lx,rx)))
    if fascicleMask is not None:
        fascicleMask = np.pad(fascicleMask, ((ly,uy),(lx,rx))).astype(bool)
    
    if circularWindow:
        disk = sm.disk(np.min(windowShape)/2-0.5, dtype=bool)
    
    fiberDensityArray = np.zeros(numFibers)
    fiberPackingArray = np.zeros(numFibers)
    
    if edgeCleanMethod == 'fractional':
        # fiberPropLabelList = [fiberPropList[i].label for i in range(numFibers)]
        
        # Calculate the fractional size of one pixel for each fiber
        fiberPixelProportionList = [1/fiberPropList[i].area for i in range(numFibers)]
        offsetArray = np.concatenate(([0], fiberPixelProportionList))
        g = lambda x: offsetArray[x]
        # Map measurements back onto fibers
        fractionalSizeMap = g(labeledFibers)
    
    for f in tqdm(range(numFibers), total=numFibers, desc='Calculating for Cell', miniters=50):
        # get coordinates for center of fiber (remembering to offset coords due to padding +ly/+lx)
        y, x = fiberPropList[f].centroid
        y = int(round(y)+ly)
        x = int(round(x)+lx)
        
        # Center window on fiber
        y0 = int(y-uy)
        y1 = int(y+ly)
        x0 = int(x-lx)
        x1 = int(x+rx)
        
        window = np.copy(labeledFibers[y0:y1, x0:x1])
        
        # Adjust windowMask for ROI and window circularity
        windowMask = np.ones(windowShape, dtype=bool)
        # Check if a circular window is desired, if so 
        if circularWindow:
            windowMask[~disk] = False
        # Adjust for ROIs
        if fascicleMask is not None:
            windowedFascicleMask = fascicleMask[y0:y1, x0:x1].astype(bool)
            windowMask[~windowedFascicleMask] = False
        
        window[~windowMask] = 0
                
        # Measure metrics
        # Calculate fiber packing
        fiberArea = (window>0).sum()
        # Count pixels in window mask (accounts for ROI and circular windows)
        packingArea = windowMask.sum()
        fiberPackingArray[f] = fiberArea / packingArea # fiber area / packing area
        
        if return_density:
            # Calculate fiber density
            if edgeCleanMethod == 'fractional':
                # Measure what fraction of each cell is inside window (and circular structuring element if circular) and sum that, instead of just counting cells
                fiberCount = fractionalSizeMap[y0:y1, x0:x1][windowMask].sum()
            else:
                uniqueFibers = np.unique(window) # gets array of labels (including background) within window
                uniqueFibers = uniqueFibers[uniqueFibers!=0] # remove background from unique label list
                fiberCount = uniqueFibers.shape[0]
            densityArea = packingArea*((imgLinearRes/1000)**2) # convert pixel area to mm^2
            fiberDensityArray[f] = (fiberCount/1000) / densityArea # number of 1000 fibers per mm^2
    if return_density:
        return fiberDensityArray, fiberPackingArray
    else:
        return fiberPackingArray


def _getNNArea(labeledFibers, imgLinearRes=0.125, fascicleMask=None):
    """
    Uses watershed algorithm to associate myelin with individual axons from the interpolated contours between the two masks.
    :param labeledFibers: Array, background labeled 0 and each fiber labeled [1-num_axons]. Shape (H,W).
    :param imgLinearRes: Float, for the um/px resolution of the image. Default is 0.125.
    :param fascicleMask: Bool Array (or None), background labeled False and fascicle labeled True. Shape (H,W). If provided, restricts area measurements to only within fascicles. Default is None.
    :return: Array (shape: num_fibers), returns area (um^2) of image background closest to each cell. If fascicleMask is provided, area is restricted to only within fascicle mask.
    """
    num_fibers = labeledFibers.max()
    if fascicleMask is not None:
        # Crop masks to minimal fascicle bounding box to hopefully speed up computation
        tempFasciclePropList = regionprops(fascicleMask.astype(int))
        crop_R0, crop_C0, crop_R1, crop_C1 = tempFasciclePropList[0].bbox
        # Expand bounding box just a bit for extra-fasciular space balancing
        r_max, c_max = fascicleMask.shape
        for _ in range(2):
            if crop_R0-1 >= 0:
                crop_R0 = crop_R0-1
            if crop_C0-1 >= 0:
                crop_C0 = crop_C0-1
            if crop_R1+1 <= r_max:
                crop_R1 = crop_R1+1
            if crop_C1+1 <= c_max:
                crop_C1 = crop_C1+1
        
        labeledFibers = labeledFibers[crop_R0:crop_R1, crop_C0:crop_C1]
        fascicleMask = fascicleMask[crop_R0:crop_R1, crop_C0:crop_C1]
        isFiber = labeledFibers>0
        
        fascicleBG = fascicleMask & ~isFiber
        # Get distance map from edges of fibers and fascicle border
        distBG = np.sqrt(mh.distance(fascicleBG))
        
        # Treat extra-fascicular space as a labeled fiber to reduce bias at fascicle borders
        watershed_seed = np.copy(labeledFibers)
        extrafascicular_label = num_fibers+1
        watershed_seed[~fascicleMask] = extrafascicular_label
        # Assign each background pixel to nearest fiber (or the extra-fascicular space if closer) by using watershed algorithm
        labeledBackground = mh.cwatershed(distBG, watershed_seed)
        # Label extra-fascicular regions and the pixels assigned to it as 0
        labeledBackground[(~fascicleMask)|(labeledBackground==extrafascicular_label)] = 0
        
    else:
        isFiber = labeledFibers>0
        # Calculate each background pixel's distance from fibers
        distBG = np.sqrt(mh.distance(~isFiber))
        # Assign each background pixel to nearest fiber by using watershed algorithm
        labeledBackground = mh.cwatershed(distBG, labeledFibers)
    
    # # Exclude watershed seed regions from labels
    # labeledBackground[isFiber] = 0
    
    # Sum area assigned to each cell label
    # NNArea = (imgLinearRes**2)*np.array([np.count_nonzero(labeledBackground==cellLabel) for cellLabel in range(1,num_fibers+1)])
    labels, counts = np.unique(labeledBackground, return_counts=True)
    # # Exclude counts of extrafascicular space and pixels assigned to it, and convert area from pixels^2 to um^2
    # if labels[0]==0:
    #     NNArea = counts[1:num_fibers+1]*(imgLinearRes**2)
    # else:
    #     NNArea = counts[0:num_fibers]*(imgLinearRes**2)
    
    # Exclude counts of extrafascicular space and pixels assigned to it
    if labels[0]==0:
        labels = labels[1:]
        counts = counts[1:]
    # convert area from pixels^2 to um^2
    NNArea = np.zeros(num_fibers)
    NNArea[labels-1] = counts*(imgLinearRes**2)
    
    return NNArea



def getCellData(file_img_list, path_img, sampleName_list=[], imgLinearRes=0.125, maskLabels=[[255], [170], [84]], cellType='complete', detailedCombined=True,
                  ldStart=8, ldStop=15, save_path=None,
                  edgeMask_type=None, edgeMaskLabels=[], edgeMask_img_list=[], edgeDist_Name='edgeDistance',
                  windowShapes_list=[(400,400)], edgeCleanMethod='fractional', edgeSize=1, circularWindow=False,
                  fascicleMask_img_list=None, excludeMask_img_list=None):
    """
    Computes metrics for each cell in each image and combines cell data from multiple images of the same sample.
    :param file_img_list: List of image file names.
    :param path_img: String, path to folder where the segmentation images are stored and where the cleaned images will be saved.
    :param sampleName_list: List of strings (same length as file_img_list). Each string is used as a label for the sample at the same index.
    :param imgLinearRes: Float, for the um/px resolution of the image. Default is 0.125.
    :param maskLabels: list of 3 sublists of integers, first sublist contains labels for axon, second sublist contains labels for myelin, third sublist contains labels for collapsed myelin
    :param cellType: String, one of {'complete', 'collapsed', 'combined'}. Defines if labeled cells include only myelinated fibers ('complete'), only degenerated fibers missing an axon ('collapsed'), or both ('combined'). Default is 'complete'.
    :param detailedCombined: Boolean, whether to also calculate structural metrics for each cell type individually. Default is True.
    :param ldStart: Integer, when calculating local density coefficient and exponent, ignore cells before the (ldStart)th-nearest neighbor. Minimum 1 (0th-nearest neighbor is self and must be excluded)
    :param ldStop: Integer or 'MAX', when calculating local density coefficient and exponent, ignore cells after the (ldStop)th-nearest neighbor. 'MAX' includes all cells after ldStart. Minimum ldStart+1 (must be larger than ldStart)
    :param save_path: String, path to folder where generated files will be saved. If not provided, will save to path_img.
    :param edgeMask_type: String, one of {'mask', 'label', None}. This determines how feature information is read for calculating each cell's distance from that feature.
                          'mask' is used when a separate binary mask is being provided for the feature. 'label' is used when the feature is labeled in images in file_img_list. None is used when no distance is to be calculated. Default is None.
    :param edgeMaskLabels: List of integers, defining the labels from which to calculate the distance of each cell.
    :param edgeMask_img_list: List of binary edge mask file names (must have same size as file_img_list). If not empty, calculate the distance of each cell from edges of true labels in mask.
    :param edgeDist_Name: String, name to label edgeMask distance data.
    
    :param windowShapes_list: List of 2-element Tuples, 2 integer elements describing the height and width of the window to be used for calculating fiber density and packing. Default is (400,400).
    :param edgeCleanMethod: String, one of {'fractional', 'top_left_cut', 'all_cut', 'none'}
        Default is 'fractional'.
        - 'fractional': for cells cut by the window, calculates what fraction of the cell is inside the window and uses that for fiber density calculation (only counts portion of cells inside window for packing).
        - 'top_left_cut': ignore cells cut by the top and left edges of the window for fiber density and fiber packing calculations (only counts portion of non-ignored cells inside window for packing).
        - 'all_cut': ignore all cells cut by the edges of the window for fiber density and fiber packing calculations.
        - 'none': count all cells cut by the edges of the window for fiber density and fiber packing calculations (only counts portion of cells inside window for packing).
    :param edgeSize: Positive integer, how many pixels away from window edge to not be considered cut by window (only used with 'top_left_cut' and 'all_cut'). Default is 1.
    :param circularWindow: Boolean, for whether to use a circular window (with diameter equal to shortest value in windowShape) instead of a rectangle.
    :param fascicleMask_img_list: List of image file names for fascicle masks [Bool Arrays, background labeled False and fascicle labeled True. Same shape as associated segmented images]. If provided, restricts fiber packing/density window measurements to only the window within fascicles. Default is None.
    :param excludeMask_img_list: List  of image file names for exclude masks [Bool Arrays, background labeled False and regions to exclude from fascicles labeled True. Same shape as associated segmented images]. If provided with fascicle mask, restricts fiber packing/density window measurements to only the window within fascicles but outside exclusion zones.
                                 Usefull when portions of images have poor quality preventing accurate segmentation. Default is None.
    
    :return: Nothing
    
    :files generated: 3 zarr files are created for each sample, and 1 txt file shared by all samples
        - _cellData.zarr: Zarr array file with shape (num_fibers, num_measures). When using measurements of combined cell types, along the y-axis fibers labeled as normally myelinated are listed first, followed by collapsed cells.
        - _cellCounts.zarr: Zarr array file with the counts of normally myelinated and collapsed cells (for easy filtering of _cellData.zarr by cell type).
        - _cellLabels.zarr: Zarr array file with same shape as input sample image file (but only one "color" channel). Background is labeled as 0 and each cell is labeled with 1+ its index in the _cellData.zarr file. (for easy mapping of measurements back onto nerve images)
        - cellDataNames.txt: Text file where each line is the name for a measure saved in _cellData.zarr (with the same ordering).
    """
    chunkSize = (3000, 3000)
    
    # If no save_path given set to path_img
    if save_path is None:
        save_path = path_img
    # If save directory doesn't exist, create it
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Check cell type validity
    if (cellType != 'complete') and (cellType != 'collapsed') and (cellType != 'combined'):
        raise Exception('Invalid cellType provided. Expected one of {"complete", "collapsed", "combined"}, recieved: '+str(cellType))
    
    # Parse label lists: background has one label, but myelin and axon may have more than one (hence lists instead of ints) [this is the index value returned by axon_segmentation]
    # backgroundLabel = maskLabels[0]
    axonLabels = maskLabels[0]
    myelinLabels = maskLabels[1]
    if (cellType == 'collapsed') or (cellType == 'combined'):
        collapsedLabels = maskLabels[2]
    
    
    # Build list for storing names of cell metric data arrays
    dataname_list = ['AxonSize', 'MyelinSize', 'FiberSize', 'AxonCircularity', 'FiberCircularity', 'AxonAspectRatio', 'FiberAspectRatio', 
                     'gRatio', 'cRatio', 'CenterDistance', 'LocalDensityCoeff', 'LocalDensityExp'] \
                  + ['FiberDensity_'+str(min(windowShape))+'_'+cellType.title() for windowShape in windowShapes_list] \
                  + ['FiberPacking_'+str(min(windowShape))+'_'+cellType.title() for windowShape in windowShapes_list] \
                  + ['AxonPacking_'+str(min(windowShape)) for windowShape in windowShapes_list] \
                  + ['MyelinPacking_'+str(min(windowShape)) for windowShape in windowShapes_list] \
                  + ['GPacking_'+str(min(windowShape)) for windowShape in windowShapes_list]
    if fascicleMask_img_list is not None:
        dataname_list = dataname_list + ['FascicleNNArea', 'FiberNNRatio', 'AxonNNRatio', 'MyelinNNRatio']
    if (cellType == 'combined') and detailedCombined:
        dataname_list = dataname_list + ['LocalDensityCoeffDetailed', 'LocalDensityExpDetailed'] \
                                      + ['FiberDensity_'+str(min(windowShape))+'Detailed' for windowShape in windowShapes_list] \
                                      + ['FiberPacking_'+str(min(windowShape))+'Detailed' for windowShape in windowShapes_list]
    if (fascicleMask_img_list is not None) and (cellType == 'combined') and detailedCombined:
        dataname_list = dataname_list + ['FascicleNNAreaDetailed', 'FiberNNRatioDetailed', 'AxonNNRatioDetailed', 'MyelinNNRatioDetailed']
    
    useEdgeMask = edgeMask_type is not None #len(edgeMask_img_list)>0
    if useEdgeMask:
        dataname_list.append(edgeDist_Name)
    # Append Coordinates
    dataname_list = dataname_list+['CentroidR', 'CentroidC', 'BBoxR0', 'BBoxC0', 'BBoxR1', 'BBoxC1']
    
       
    # Save list of names of measured metrics
    output_dataNames_pathName = os.path.join(save_path, 'cellDataNames.txt')
    with open(output_dataNames_pathName, "w") as textfile:
        for dataname in dataname_list:
            textfile.write(dataname + "\n")
    
    
    for imgIdx, file_img in enumerate(file_img_list):
        cleanImage_pathName = os.path.join(path_img, file_img)

        print('')
        print('Processing Image: '+file_img)
        
        # Load segmentation image and ignore extra z-dimensions if in color (rgb channels should each be equal to grayscale label)
        prediction = mh.imread(cleanImage_pathName)
        if len(prediction.shape)>2:
            prediction = prediction[:,:,0]

        # Extract axon and myelin masks from predictions
        isAxon = np.isin(prediction, axonLabels)
        isMyelin = np.isin(prediction, myelinLabels)
        if (cellType == 'collapsed') or (cellType == 'combined'):
            isCollapsed = np.isin(prediction, collapsedLabels)

        ### Calculate individual cell metrics (eg. gRatio, axonCircularity, fiberCircularity, fascicleEdgeDistance)
        num_axons = 0
        num_myelin = 0
        num_collapsed = 0
        if (cellType == 'complete') or (cellType == 'combined'):
            print('Getting Axon/Myelin Labels')
            # Get labels for axon and associated myelin
            num_axons, labeledAxons, labeledMyelin, labeledComplete = _getAMFLabels(isAxon, isMyelin)
        if (cellType == 'collapsed') or (cellType == 'combined'):
            print('Getting Collapsed Labels')
            # Get labels for degenerated myelin fibers
            labeledCollapsed, num_collapsed = mh.label(isCollapsed)
            # Shift labels in case of compelete cells being first in combined lists
            shiftedLabeledCollapsed = np.copy(labeledCollapsed)
            shiftedLabeledCollapsed[labeledCollapsed>0] = shiftedLabeledCollapsed[labeledCollapsed>0]+num_axons
        num_fibers = num_axons+num_collapsed
        
        print('Calculating Regionprops')
        if cellType == 'collapsed':
            cellCountOutput = np.array([0, num_collapsed], dtype=int)
            labeledFibers = labeledCollapsed
        elif cellType == 'complete':
            cellCountOutput = np.array([num_axons, 0], dtype=int)
            labeledFibers = labeledComplete
            axonPropList = regionprops(labeledAxons)
            myelinPropList = regionprops(labeledMyelin)
        elif cellType == 'combined':
            # Shift label idx of collapsed cells to follow complete cells, then combine complete and collapsed fiber labels
            cellCountOutput = np.array([num_axons, num_collapsed], dtype=int)
            labeledFibers = np.copy(labeledComplete)
            labeledFibers[labeledCollapsed>0] = labeledCollapsed[labeledCollapsed>0]+num_axons
            axonPropList = regionprops(labeledAxons)
            myelinPropList = regionprops(labeledMyelin)
        print('Cell Counts (a, c): '+str(cellCountOutput))
        
        # Get properties for regions labeled >0
        fiberPropList = regionprops(labeledFibers)
        
        
        print('Calculating Fiber Size')
        # Get fiber size (um^2)
        fiberSizeArray = np.array([fiberPropList[i].area for i in range(num_fibers)]) * imgLinearRes**2
        
        
        print('Calculating Circularity')
        # Calculate circularity for each cell [4π*area/perimeter^2 [0,1]]
        circularity = lambda area, perimeter: 4*np.pi*(area)/perimeter**2
        axonCircularityArray = np.zeros((num_fibers))
        if (cellType == 'complete') or (cellType == 'combined'):
            axonCircularityArray[:num_axons] = np.array([circularity(axonPropList[i].area, axonPropList[i].perimeter) for i in range(num_axons)])
        fiberCircularityArray = np.array([circularity(fiberPropList[i].area, fiberPropList[i].perimeter) for i in range(num_fibers)])
        
        # Bound circularity range (sometimes perimeter measurement is less accurate and results in circularity >1 which should be impossible, this occurs mostly in cells <= 100 pixels in area)
        axonCircularityArray[axonCircularityArray>1] = 1
        fiberCircularityArray[fiberCircularityArray>1] = 1
        
        
        print('Calculating Aspect Ratio')
        # axis_minor_length/axis_major_length
        aspectRatio = lambda area, perimeter: 4*np.pi*(area)/perimeter**2
        axonAspectRatioArray = np.zeros((num_fibers))
        if (cellType == 'complete') or (cellType == 'combined'):
            axonAspectRatioArray[:num_axons] = np.array([axonPropList[i].axis_minor_length/axonPropList[i].axis_major_length for i in range(num_axons)])
        fiberAspectRatioArray = np.array([fiberPropList[i].axis_minor_length/fiberPropList[i].axis_major_length for i in range(num_fibers)])
        
        
        # Initialize axon specific calculations
        axonSizeArray = np.zeros((num_fibers))
        gRatioArray = np.zeros((num_fibers))
        centerDistArray = np.zeros((num_fibers))
        cDiffArray = np.zeros((num_fibers))
        cRatioArray = np.zeros((num_fibers))
        gRatioPerimArray = np.zeros((num_fibers))
        
        if ((cellType == 'complete') or (cellType == 'combined')) and (num_axons>0):
            print('Calculating Axon Size')
            # Get axon size (um^2)
            axonSizeArray[:num_axons] = [imgLinearRes**2 * axonPropList[i].area for i in range(num_axons)]
            
            print('Calculating g-Ratio [area]')
            # Calculate g-Ratio for each cell [req(axon)/req(fiber) where req is the equivalent radius of a circle with given area]
            gRatioArray[:num_axons] = [axonPropList[i].equivalent_diameter/fiberPropList[i].equivalent_diameter for i in range(num_axons)]
            
            print('Calculating g-Ratio [perimeter]')
            # Calculate g-Ratio for each cell [req(axon)/req(fiber) where req is the equivalent radius of a circle with given area]
            gRatioPerimArray[:num_axons] = np.array([axonPropList[i].perimeter/fiberPropList[i].perimeter for i in range(num_axons)])
            
            print('Calculating Distance Between Centers')
            # Calculate the distance between the center of myelin and the center of it's associated myelin for each cell [um]
            centerDistArray[:num_axons] = [imgLinearRes*_calcDist(axonPropList[i].centroid, myelinPropList[i].centroid) for i in range(num_axons)]
            
            print('Calculating Circularity Difference')
            cDiffArray[:num_axons] = axonCircularityArray[:num_axons]-fiberCircularityArray[:num_axons]
            
            print('Calculating c-Ratio')
            cRatioArray[:num_axons] = axonCircularityArray[:num_axons]/fiberCircularityArray[:num_axons]
        
        myelinSizeArray = fiberSizeArray-axonSizeArray
        
        print('Calculating Effective Local Density Coefficients')
        # Effective local density from https://doi.org/10.1038/srep04511
        # Check if there are enough cells for given start/stop
        if ldStop>num_fibers:
            # Not enough cells, so assign 0 to all
            localDensityCoeff_array = np.zeros((num_fibers))
            localDensityExp_array = np.zeros((num_fibers))
        else:
            # Calculate distance from each cell to all other cells [um]
            centroid_array = np.array([fiberPropList[i].centroid for i in range(num_fibers)])
            dist = lambda idx0, idx1: imgLinearRes*np.sqrt((centroid_array[idx1,0]-centroid_array[idx0,0])**2+(centroid_array[idx1,1]-centroid_array[idx0,1])**2)
            fiberIdxPairs_array = np.array(np.meshgrid(range(num_fibers), range(num_fibers)), dtype=int)
            cellDist_array = dist(fiberIdxPairs_array[0,:,:], fiberIdxPairs_array[1,:,:])
            # For each cell, sort distances from closest (itself) to furthest and then remove itself (to avoid 0 value distances in future steps)
            # can additionally ignore nth-nearest neighbors before ldStart and after ldStop
            ldStart = max(1, ldStart)
            if ldStop == 'MAX':
                cellDist_array_sorted_trimmed = np.sort(cellDist_array, axis=1)[:, ldStart:]
                ldStop = cellDist_array_sorted_trimmed.shape[1]
            else:
                ldStop = max(ldStart+1, ldStop)
                cellDist_array_sorted_trimmed = np.sort(cellDist_array, axis=1)[:, ldStart:ldStop+1]
            
            # Find coefficients for line of best fit on log-log plot of (distance to nth-nearest neighbor vs rank n)
            logX = np.log(np.array(range(ldStart, ldStop+1)))
            cellDist_array_sorted_trimmed_log = np.log(cellDist_array_sorted_trimmed)
            localDensityCoeff_lists = list(zip(*[np.polyfit(logX,cellDist_array_sorted_trimmed_log[cellIdx,:],deg=1) for cellIdx in range(num_fibers)]))
            # ln(y) = coeff0*ln(x) + coeff1
            # y = e^coeff1 * x^coeff0
            # coeff = e^coeff1
            # exp = coeff0
            localDensityCoeff_array = np.e**np.array(localDensityCoeff_lists[1])
            localDensityExp_array = np.array(localDensityCoeff_lists[0])
        
        
        # Check if fascicle masks are provided
        if fascicleMask_img_list is None:
            fascicleMask = None
        else:
            print('Loading Fascicle Mask')
            # If provided, load fascicle masks and ensure only 2 dimensions
            fascicleMask_pathName = os.path.join(path_img, fascicleMask_img_list[imgIdx])
            fascicleMask = mh.imread(fascicleMask_pathName).astype(bool)
            if len(fascicleMask.shape)>2:
                fascicleMask = fascicleMask[:,:,0]
            
            if excludeMask_img_list is not None:
                print('Loading Exclusion Mask(s)')
                # If provided load exclusion mask(s), ensure it's only 2 dimensions, and clear that region(s) from the fascicle mask
                if isinstance(excludeMask_img_list[imgIdx], list) or isinstance(excludeMask_img_list[imgIdx], tuple): # check if more than one mask to be excluded from this sample's fascicle mask
                    excludeMask_img_sublist = excludeMask_img_list[imgIdx]
                else:
                    excludeMask_img_sublist = [excludeMask_img_list[imgIdx]]
                for sublistIdx in range(len(excludeMask_img_sublist)): # loop through exclusion mask(s) for this sample
                    excludeMask_pathName = os.path.join(path_img, excludeMask_img_sublist[sublistIdx])
                    excludeMask = mh.imread(excludeMask_pathName).astype(bool)
                    if len(excludeMask.shape)>2:
                        excludeMask = excludeMask[:,:,0]
                    fascicleMask[excludeMask] = False
        
            print('Calculating Nearest-Neighbor Fascicle Area') # Assign each pixel of fascicle background (ignoring exclusion regions) to nearest cell and sum that area
            NNAreaArray = _getNNArea(labeledFibers, imgLinearRes=imgLinearRes, fascicleMask=fascicleMask)
        
        
        print('Calculating Cell-Wise Fiber Packing/Density') #cell-wise measurement might be flawed due to inherent underrepresentation of low density areas of fascicles, but is able to be used in multivariate analysis unlike pixel-wise measurements
        # Calculate fiber density and packing based on cell-centered windows
        fiberDensityArray_list = []
        fiberPackingArray_list = []
        for windowShape in windowShapes_list:
            print(' - Window Shape: '+str(windowShape))
            fiberDensityArray, fiberPackingArray = _getCellPackingDensity(labeledFibers, fiberPropList, windowShape=windowShape, imgLinearRes=imgLinearRes,
                                                                   edgeCleanMethod=edgeCleanMethod, edgeSize=1, fascicleMask=fascicleMask, circularWindow=circularWindow)
            fiberDensityArray_list.append(fiberDensityArray)
            fiberPackingArray_list.append(fiberPackingArray)
        
        
        # Axon/Myelin specific measures of density/packing
        if ((cellType == 'complete') or (cellType == 'combined')) and (num_axons>0):
            print('Calculating Cell-Wise Axon Packing') #cell-wise measurement might be flawed due to inherent underrepresentation of low density areas of fascicles, but is able to be used in multivariate analysis unlike pixel-wise measurements
            # Calculate axon density and packing based on cell-centered windows
            axonPackingArray_list = []
            for windowShape in windowShapes_list:
                print(' - Window Shape: '+str(windowShape))
                temp_axonPackingArray = _getCellPackingDensity(labeledAxons, axonPropList, windowShape=windowShape, imgLinearRes=imgLinearRes,
                                                              edgeCleanMethod=edgeCleanMethod, edgeSize=1, fascicleMask=fascicleMask, circularWindow=circularWindow, return_density=False)
                axonPackingArray_list.append(temp_axonPackingArray)
                
            # print('Calculating Cell-Wise Myelin Packing/Density') #cell-wise measurement might be flawed due to inherent underrepresentation of low density areas of fascicles, but is able to be used in multivariate analysis unlike pixel-wise measurements
            # # Calculate axon density and packing based on cell-centered windows
            # myelinPackingArray = _getCellPackingDensity(labeledMyelin, myelinPropList, windowShape=windowShape, imgLinearRes=imgLinearRes,
            #                                            edgeCleanMethod=edgeCleanMethod, edgeSize=1, fascicleMask=fascicleMask, circularWindow=circularWindow, return_density=False)
        else:
            axonPackingArray_list = [np.zeros_like(fiberPackingArray) for windowShape in windowShapes_list]
        
        
        print('Calculating Cell-Wise Myelin Packing')
        # mp = fp-ap
        myelinPackingArray_list = [fiberPackingArray_list[packingIdx]-axonPackingArray_list[packingIdx] for packingIdx in range(len(windowShapes_list))]
        
        print('Calculating Cell-Wise G-Packing')
        # gp = ap/fp
        gPackingArray_list = [axonPackingArray_list[packingIdx]/fiberPackingArray_list[packingIdx] for packingIdx in range(len(windowShapes_list))]
        
        
        
        if (cellType == 'combined') and detailedCombined:
            print('Calculating Detailed Data For Separated Cell Types:')
            
            
            print('Calculating Effective Local Density Coefficients For Separated Cell Types')
            detailedLocalDensityCoeff_array = np.zeros((num_fibers))
            detailedLocalDensityExp_array = np.zeros((num_fibers))
            # Repeat for complete cells only
            if ldStop<=num_axons:
                # Calculate distance from each cell to all other cells [um]
                centroid_array = np.array([fiberPropList[i].centroid for i in range(num_axons)])
                dist = lambda idx0, idx1: imgLinearRes*np.sqrt((centroid_array[idx1,0]-centroid_array[idx0,0])**2+(centroid_array[idx1,1]-centroid_array[idx0,1])**2)
                fiberIdxPairs_array = np.array(np.meshgrid(range(num_axons), range(num_axons)), dtype=int)
                cellDist_array = dist(fiberIdxPairs_array[0,:,:], fiberIdxPairs_array[1,:,:])
                # For each cell, sort distances from closest (itself) to furthest and then remove itself (to avoid 0 value distances in future steps)
                # can additionally ignore nth-nearest neighbors before ldStart and after ldStop
                ldStart = max(1, ldStart)
                if ldStop == 'MAX':
                    cellDist_array_sorted_trimmed = np.sort(cellDist_array, axis=1)[:, ldStart:]
                    ldStop = cellDist_array_sorted_trimmed.shape[1]
                else:
                    ldStop = max(ldStart+1, ldStop)
                    cellDist_array_sorted_trimmed = np.sort(cellDist_array, axis=1)[:, ldStart:ldStop+1]
                
                # Find coefficients for line of best fit on log-log plot of (distance to nth-nearest neighbor vs rank n)
                logX = np.log(np.array(range(ldStart, ldStop+1)))
                cellDist_array_sorted_trimmed_log = np.log(cellDist_array_sorted_trimmed)
                localDensityCoeff_lists = list(zip(*[np.polyfit(logX,cellDist_array_sorted_trimmed_log[cellIdx,:],deg=1) for cellIdx in range(num_axons)]))
                # ln(y) = coeff0*ln(x) + coeff1
                # y = e^coeff1 * x^coeff0
                # coeff = e^coeff1
                # exp = coeff0
                detailedLocalDensityCoeff_array[:num_axons] = np.e**np.array(localDensityCoeff_lists[1])
                detailedLocalDensityExp_array[:num_axons] = np.array(localDensityCoeff_lists[0])
            # Repeat for collapsed cells only
            if ldStop<=num_collapsed:
                # Calculate distance from each cell to all other cells [um]
                centroid_array = np.array([fiberPropList[i].centroid for i in range(num_collapsed)])
                dist = lambda idx0, idx1: imgLinearRes*np.sqrt((centroid_array[idx1,0]-centroid_array[idx0,0])**2+(centroid_array[idx1,1]-centroid_array[idx0,1])**2)
                fiberIdxPairs_array = np.array(np.meshgrid(range(num_collapsed), range(num_collapsed)), dtype=int)
                cellDist_array = dist(fiberIdxPairs_array[0,:,:], fiberIdxPairs_array[1,:,:])
                # For each cell, sort distances from closest (itself) to furthest and then remove itself (to avoid 0 value distances in future steps)
                # can additionally ignore nth-nearest neighbors before ldStart and after ldStop
                ldStart = max(1, ldStart)
                if ldStop == 'MAX':
                    cellDist_array_sorted_trimmed = np.sort(cellDist_array, axis=1)[:, ldStart:]
                    ldStop = cellDist_array_sorted_trimmed.shape[1]
                else:
                    ldStop = max(ldStart+1, ldStop)
                    cellDist_array_sorted_trimmed = np.sort(cellDist_array, axis=1)[:, ldStart:ldStop+1]
                
                # Find coefficients for line of best fit on log-log plot of (distance to nth-nearest neighbor vs rank n)
                logX = np.log(np.array(range(ldStart, ldStop+1)))
                cellDist_array_sorted_trimmed_log = np.log(cellDist_array_sorted_trimmed)
                localDensityCoeff_lists = list(zip(*[np.polyfit(logX,cellDist_array_sorted_trimmed_log[cellIdx,:],deg=1) for cellIdx in range(num_collapsed)]))
                # ln(y) = coeff0*ln(x) + coeff1
                # y = e^coeff1 * x^coeff0
                # coeff = e^coeff1
                # exp = coeff0
                detailedLocalDensityCoeff_array[num_axons:] = np.e**np.array(localDensityCoeff_lists[1])
                detailedLocalDensityExp_array[num_axons:] = np.array(localDensityCoeff_lists[0])
            
            
            print('Calculating Fiber Nearest-Neighbor Area For Separated Cell Types')
            detailedNNAreaArray = np.zeros((num_fibers))
            detailedNNAreaArray[:num_axons] = _getNNArea(labeledComplete, imgLinearRes=imgLinearRes, fascicleMask=fascicleMask)
            detailedNNAreaArray[num_axons:] = _getNNArea(shiftedLabeledCollapsed, imgLinearRes=imgLinearRes, fascicleMask=fascicleMask)
            
            
            print('Calculating Cell-Wise Fiber Packing/Density For Separated Cell Types')
            detailedDensityArray_list = []
            detailedPackingArray_list = []
            for windowShape in windowShapes_list:
                print(' - Window Shape: '+str(windowShape))
                detailedDensityArray = np.zeros((num_fibers))
                detailedPackingArray = np.zeros((num_fibers))
                
                # Get Density and Packing for complete cells
                detailedDensityArray[:num_axons], detailedPackingArray[:num_axons] = _getCellPackingDensity(labeledComplete, fiberPropList[:num_axons],
                                                                                     windowShape=windowShape, imgLinearRes=imgLinearRes,
                                                                                     edgeCleanMethod=edgeCleanMethod, edgeSize=1, fascicleMask=fascicleMask, circularWindow=circularWindow)
                # Use shiftedLabeledCollapsed to ensure that label values in array match label values in fiberPropsList
                detailedDensityArray[num_axons:], detailedPackingArray[num_axons:] = _getCellPackingDensity(shiftedLabeledCollapsed, fiberPropList[num_axons:],
                                                                                     windowShape=windowShape, imgLinearRes=imgLinearRes,
                                                                                     edgeCleanMethod=edgeCleanMethod, edgeSize=1, fascicleMask=fascicleMask, circularWindow=circularWindow)
                detailedDensityArray_list.append(detailedDensityArray)
                detailedPackingArray_list.append(detailedPackingArray)
        
        
        print('Getting Cell-Wise Coordinates')
        centroidR_array = [fiberPropList[i].centroid[0] for i in range(num_fibers)]
        centroidC_array = [fiberPropList[i].centroid[1] for i in range(num_fibers)]
        bboxR0_array = [fiberPropList[i].bbox[0] for i in range(num_fibers)]
        bboxC0_array = [fiberPropList[i].bbox[1] for i in range(num_fibers)]
        bboxR1_array = [fiberPropList[i].bbox[2] for i in range(num_fibers)]
        bboxC1_array = [fiberPropList[i].bbox[3] for i in range(num_fibers)]
        
        ## Compile data lists into list
        imgDataList_cellDataArray = [axonSizeArray, myelinSizeArray, fiberSizeArray, axonCircularityArray, fiberCircularityArray, axonAspectRatioArray, fiberAspectRatioArray,
                                     gRatioArray, cRatioArray, centerDistArray, localDensityCoeff_array, localDensityExp_array] \
                                    + fiberDensityArray_list + fiberPackingArray_list + axonPackingArray_list \
                                    + myelinPackingArray_list + gPackingArray_list
        if fascicleMask_img_list is not None:
            imgDataList_cellDataArray = imgDataList_cellDataArray + [NNAreaArray, fiberSizeArray/NNAreaArray, axonSizeArray/NNAreaArray, myelinSizeArray/NNAreaArray]
        if (cellType == 'combined') and detailedCombined:
            imgDataList_cellDataArray = imgDataList_cellDataArray + [detailedLocalDensityCoeff_array, detailedLocalDensityExp_array] \
                                                                  + detailedDensityArray_list + detailedPackingArray_list
        if (fascicleMask_img_list is not None) and (cellType == 'combined') and detailedCombined:
            imgDataList_cellDataArray = imgDataList_cellDataArray + [detailedNNAreaArray, fiberSizeArray/detailedNNAreaArray, axonSizeArray/detailedNNAreaArray, myelinSizeArray/detailedNNAreaArray]
        
        ### Use edge mask to get fiber metrics and append onto data list
        ## Load edge mask if provided
        if edgeMask_type == 'mask':
            edgeMask_pathName = os.path.join(path_img, edgeMask_img_list[imgIdx])
            print('Using Mask To Get Edge Distance')
            isEdge = np.logical_not(mh.imread(edgeMask_pathName).astype(bool))
            if len(isEdge.shape)>2:
                isEdge = isEdge[:,:,0]
        elif edgeMask_type == 'label':
            print('Using Label To Get Edge Distance')
            if len(edgeMaskLabels) == 0:
                raise Exception('Error: edgeMaskLabels is empty list, but edgeMask_type is label')
            isEdge = np.isin(prediction, edgeMaskLabels, invert=True)
                    
        ## Get distance map from edges
        if useEdgeMask:
            edgeDist = imgLinearRes*np.sqrt(mh.distance(isEdge)) #mh.distance returns the Squared Euclidean Distance; so square root it and scale by image resolution to get actual distance (um)
            # Find each cell's mean distance from edge
            edgeDistanceArray = np.zeros(num_fibers)
            for labelIdx in range(num_fibers):
                wy0, wx0, wy1, wx1 = fiberPropList[labelIdx].bbox
                windowedEdgeDist = edgeDist[wy0:wy1, wx0:wx1]
                edgeDistanceArray[labelIdx] = windowedEdgeDist[fiberPropList[labelIdx].image].mean()
            imgDataList_cellDataArray.append(edgeDistanceArray)
        
        ## Append cell cordinates
        imgDataList_cellDataArray = imgDataList_cellDataArray + [centroidR_array, centroidC_array, bboxR0_array, bboxC0_array, bboxR1_array, bboxC1_array]
        
        
        # if saveData:
        print('Saving Data To Output Arrays')
        # output_name = sampleName+'_'+str(cellCountOutput)
        
        # Get sample name for image
        if len(sampleName_list) > 0:
            sampleName = sampleName_list[imgIdx]
        else:
            sampleName = file_img.replace('.','_').split('_')[0]
            
        # Create paths to where to save
        output_labels_pathName = os.path.join(save_path, sampleName+'_cellLabels.zarr')
        output_data_pathName = os.path.join(save_path, sampleName+'_cellData.zarr')
        output_counts_pathName = os.path.join(save_path, sampleName+'_cellCounts.zarr')
        
        # Concentrate data for saving as zarr array with shape (num_cells, num_measures)
        # [axis order set to align with numpy and scikit-learn package functions, thus avoiding having to transpose large datasets]
        output_shape = (num_fibers, len(imgDataList_cellDataArray))
        output_cellDataArray = np.zeros(output_shape)
        for output_dataIdx, cellDataArray in enumerate(imgDataList_cellDataArray):
            output_cellDataArray[:, output_dataIdx] = cellDataArray
        
        # Actually save data
        zarr.convenience.save(output_labels_pathName, labeledFibers)
        zarr.convenience.save(output_data_pathName, output_cellDataArray)
        zarr.convenience.save(output_counts_pathName, cellCountOutput)
        
        print('Collecting Garbage')
        gc.collect()




@ray.remote
def _arraySum_chunked(coordi, chunkSize, paddedFibers, window):
    """
    Remote code for parallel calculation of convolution for a chunk of the sample for density and packing measurements.
    :param coordi: Coordinate, produced by np.ndindex(), for which chunk to convolve over.
    :param chunkSize: 2 Integer Array, descibes the shape of chunks for parallel processing.
    :param paddedFibers: Float Array to be convolutionally summed.
    :param window: Array, convolutional kernel used for summation in density and packing measurements.
    
    :return: Nothing
    """
    
    zzWindows = view_as_windows(paddedFibers, window.shape, step=1)
    
    start = (coordi*chunkSize).astype(int)
    stop = (np.minimum(zzWindows.shape[:2], (coordi+1)*chunkSize)).astype(int)
    
    # Check if window is a full kernel
    if np.all(window):
        outputChunk = np.sum(zzWindows[start[0]:stop[0], start[1]:stop[1], :,:], axis=(2,3))
    else:
        size = stop-start
        mask = np.broadcast_to(window, (size[0], size[1], window.shape[0], window.shape[1]))
        outputChunk = np.sum(zzWindows[start[0]:stop[0], start[1]:stop[1], :,:], axis=(2,3), where=mask)
    return outputChunk



def _getPaddedSum(Mask, window, chunkSize, crop2bbox=False, desc=""):
    """
    Parallelization handler for calculating convolution for density and packing measurements.
    :param Mask: Array to be convolutionally summed for density/packing measurements.
    :param window: Array, convolutional kernel used for summation in density and packing measurements.
    :param crop2bbox: Boolean, whether to crop the borders of Mask to non-zero regions, and thus reduce computation time. Default is False.
    :param desc: String, description to be displayed in tqdm progress bar. Default is "".
    
    :return: Nothing
    """
    
    # If cropping, remove blank background borders to reduce computation
    if crop2bbox:
        originalShape = Mask.shape
        tempMaskPropList = regionprops((Mask>0).astype(int))
        crop_R0, crop_C0, crop_R1, crop_C1 = tempMaskPropList[0].bbox
        Mask = Mask[crop_R0:crop_R1, crop_C0:crop_C1]
    
        
    # Add padding to labeled image (so that convolution result is same size as mask)
    padding = (np.array(window.shape)-1)/2
    startPad = (padding+0.5).astype(int)
    endPad = (padding).astype(int)
    paddedMask = np.pad(Mask, ((startPad[0],endPad[0]),(startPad[1],endPad[1]))).astype(float) # convert to float, in case mask is a boolean because a boolean mask takes 6 times longer to sum for some reason
    
    chunkSize_ref = ray.put(chunkSize)
    paddedMask_ref = ray.put(paddedMask)
    window_ref = ray.put(window)
    numChunks = np.ceil(np.array(Mask.shape)/chunkSize).astype(int)
    
    
    ## Calculate area of windows covered by mask
    # print('- - Summing mask within each window in parallel')
    # Use a sliding window to sum labeled pixels
    refList = [_arraySum_chunked.remote(np.array(coordi), chunkSize_ref, paddedMask_ref, window_ref) for coordi in np.ndindex(tuple(numChunks))]
    
    # Get summedMap from results in shared storage
    summedMap = np.zeros(Mask.shape)
    sumTQDM = tqdm(enumerate(np.ndindex(tuple(numChunks))), total=len(refList), desc=desc)
    for i, coordi in sumTQDM:
        coordi = np.array(coordi)
        startCh = (coordi*chunkSize).astype(int)
        stopCh = (np.minimum(summedMap.shape, (coordi+1)*chunkSize)).astype(int)
        
        summedMap[startCh[0]:stopCh[0], startCh[1]:stopCh[1]] = ray.get(refList[i])
    
    # If cropped, restore output to correct position in array of original shape
    if crop2bbox:
        uncropped_summedMap = np.zeros(originalShape, dtype=summedMap.dtype)
        uncropped_summedMap[crop_R0:crop_R1, crop_C0:crop_C1] = summedMap
        summedMap = uncropped_summedMap
    
    # Clean up references to objects in Ray shared storage so that Ray can clean up memory
    del chunkSize_ref
    # del viewShape_ref
    del window_ref
    del paddedMask_ref
    del refList
    
    return summedMap




def _getPixelDensity(labeledFibers, fiberPropList, onlyFascicleArea=True, fascicleAreaMap=None, window=np.ones((400,400)), imgLinearRes=0.125, chunkSize=(3000,3000), desc=""):
    """
    Computes the pixel-wise fiber density and fiber packing by using a window centered on each pixel.
    :param labeledFibers: Array, background labeled 0 and each fiber labeled [1-num_axons]. Shape (H,W).
    :param fiberPropList: List, each element describes one labeled fiber (skimage.measure.regionprops). Shape (num_axons).
    :param onlyFascicleArea: Boolean, whether to restrict pixel window measurements to only the window within fascicles.
    :param fascicleAreaMap: Array (same shape as labeledFibers), convolutional measurement of area of windows within fascicles.
    :param window: Array, convolutional kernel used for summation in density and packing measurements. Default is a full array of ones with shape (400,400). Must be same as window used for fascicleAreaMap calculation for measurements to be accurate.
    :param imgLinearRes: Float, for the um/px resolution of the image. Default is 0.125.
    :param chunkSize: 2 Integer Tuple, descibes the shape of chunks for parallel processing.
    :param desc: String, description to be displayed in tqdm progress bar. Default is "".
    
    :return: 2 Arrays, the first containing the fiber density measured around each pixel, the sexond containing the fiber packing measured around each pixel. Each with shape (H,W).
    """
    
    if onlyFascicleArea:
        nonZero = fascicleAreaMap>0
    
    ## Calculate label density of window centered on each pixel (1000 labels [eg. fibers/axon/myelin] per mm^2 within window)
    # Calculate fractional size of each pixel within a fiber
    # print('- Calculating fractional size of each pixel within a label')
    fiberPropAreaList = [1/fiberPropList[i].area for i in range(len(fiberPropList))]
    # Map measurement back onto image (adding 0 at front to account for background)
    offsetArray = np.concatenate(([0], fiberPropAreaList))
    g = lambda x: offsetArray[x]
    fractionalFiberMap = g(labeledFibers)
    
    # print('- Calculating Density by summing fractional area of labels in each window in parallel')
    fiberCountMap = _getPaddedSum(fractionalFiberMap, window, chunkSize, desc=desc)
    
    if onlyFascicleArea:
        # Calculate fiber density where area is measured only within fascicle mask
        fiberDensityMap = np.zeros(labeledFibers.shape)
        fiberDensityMap[nonZero] = (fiberCountMap[nonZero]/1000) / (fascicleAreaMap[nonZero]*(imgLinearRes/1000)**2)  # number of 1000 fibers per mm^2 (area only counted if inside fascicle)
    else:
        fiberDensityMap = (fiberCountMap/1000) / (window.sum()*(imgLinearRes/1000)**2)  # number of 1000 fibers per mm^2
    
    return fiberDensityMap



def _getPixelPacking(labeledFibers, onlyFascicleArea=True, fascicleAreaMap=None, window=np.ones((400,400)), imgLinearRes=0.125, chunkSize=(3000,3000), desc=""):
    """
    Computes the pixel-wise fiber density and fiber packing by using a window centered on each pixel.
    :param labeledFibers: Array, background labeled 0 and each fiber labeled [1-num_axons]. Shape (H,W).
    :param onlyFascicleArea: Boolean, whether to restrict pixel window measurements to only the window within fascicles.
    :param fascicleAreaMap: Array (same shape as labeledFibers), convolutional measurement of area of windows within fascicles.
    :param window: Array, convolutional kernel used for summation in density and packing measurements. Default is a full array of ones with shape (400,400). Must be same as window used for fascicleAreaMap calculation for measurements to be accurate.
    :param imgLinearRes: Float, for the um/px resolution of the image. Default is 0.125.
    :param chunkSize: 2 Integer Tuple, descibes the shape of chunks for parallel processing.
    :param desc: String, description to be displayed in tqdm progress bar. Default is "".
    
    :return: 2 Arrays, the first containing the fiber density measured around each pixel, the sexond containing the fiber packing measured around each pixel. Each with shape (H,W).
    """
    ## Calculate label packing of window centered on each pixel (labeled area/window area)
    # print('- Calculating Packing by summing area of labels in each window in parallel')
    fiberAreaMap = _getPaddedSum(labeledFibers>0, window, chunkSize, desc=desc)
    
    if onlyFascicleArea:
        # Calculate fiber packing where area is measured only within fascicle mask
        nonZero = fascicleAreaMap>0
        fiberPackingMap = np.zeros(labeledFibers.shape)
        fiberPackingMap[nonZero] = fiberAreaMap[nonZero] / fascicleAreaMap[nonZero]  # area only counted if inside fascicle
    else:
        fiberPackingMap = fiberAreaMap/window.sum()
    
    return fiberPackingMap



def _getPixelNNArea(labeledFibers, fiberPropList, axonPropList, imgLinearRes=0.125, fascicleMask=None):
    """
    Uses watershed algorithm to associate fascicular background pixels with nearest individual fibers/axons (a Voronoi diagram).
    Then calculates ratio of each fiber/axon area to its Voronoi area.
    :param labeledFibers: Array, background labeled 0 and each fiber labeled [1-num_axons]. Shape (H,W).
    :param fiberPropList: List, each element describes one labeled fiber (skimage.measure.regionprops). Shape (num_axons).
    :param axonPropList: List, each element describes one labeled axon (skimage.measure.regionprops). Shape (num_axons).
    :param imgLinearRes: Float, for the um/px resolution of the image. Default is 0.125.
    :param fascicleMask: Bool Array (or None), background labeled False and fascicle labeled True. Shape (H,W). If provided, restricts area measurements to only within fascicles. Default is None.
    
    :return: 3 Arrays each of shape (H,W).
        - The first is the Voronoi diagram where each pixel of a Voronoi cell is assigned a value equal to the area (um^2) of that Voronoi cell.
        - The second is the same but the assigned value is instead the fiber area divided by the associated Voronoi cell area.
        - The third is the same but the assigned value is instead the axon area divided by the associated Voronoi cell area.
    """
    num_fibers = labeledFibers.max()
    isFiber = labeledFibers>0
    
    fascicleBG = fascicleMask & ~isFiber
    # Get distance map from edges of fibers and fascicle border
    distBG = np.sqrt(mh.distance(fascicleBG))
    
    # Treat extra-fascicular space as a labeled fiber to reduce bias at fascicle borders
    watershed_seed = np.copy(labeledFibers)
    extrafascicular_label = num_fibers+1
    watershed_seed[~fascicleMask] = extrafascicular_label
    # Assign each background pixel to nearest fiber (or the extra-fascicular space if closer) by using watershed algorithm
    labeledFascicle = mh.cwatershed(distBG, watershed_seed)
    # Label extra-fascicular regions and the pixels assigned to it as 0
    labeledFascicle[(~fascicleMask)|(labeledFascicle==extrafascicular_label)] = 0
    
    # # Exclude watershed seed regions from labels
    # labeledFascicle[isFiber] = 0
    
    # Sum area assigned to each cell label
    # NNArea = np.array([np.count_nonzero(labeledFascicle==cellLabel) for cellLabel in range(1,num_fibers+1)])
    labels, counts = np.unique(labeledFascicle, return_counts=True) # may miss some labels if we don't count fiber area as part of the measure
    # Exclude counts of extrafascicular space and pixels assigned to it
    NNArea = np.zeros(num_fibers+1)
    NNArea[labels] = counts
    # if labels[0]==0:
    #     NNArea = counts[1:num_fibers+1]
    # else:
    #     NNArea = counts[0:num_fibers]
    
    # Get ratio of fascicle area to fiber area
    fiberNNRatio = [fiberPropList[cellLabel].area/NNArea[cellLabel+1] for cellLabel in range(num_fibers)]
    axonNNRatio = [axonPropList[cellLabel].area/NNArea[cellLabel+1] for cellLabel in range(num_fibers)]
    
    # Convert area from pixels^2 to um^2
    NNArea = NNArea*(imgLinearRes**2)
    
    # Set areas outside fascicle mask [0] and the pixels assigned to it [num_fibers+1] to 0
    # NNArea = np.concatenate(([0],NNArea))
    NNArea[0] = 0
    fiberNNRatio = np.concatenate(([0],fiberNNRatio))
    axonNNRatio = np.concatenate(([0],axonNNRatio))
    
    # Map measurement back onto fascicle
    g_NNArea = lambda x: NNArea[x]
    NNAreaMap = g_NNArea(labeledFascicle)
    g_fNNRatio = lambda x: fiberNNRatio[x]
    fiberNNRatioMap = g_fNNRatio(labeledFascicle)
    g_aNNRatio = lambda x: axonNNRatio[x]
    axonNNRatioMap = g_aNNRatio(labeledFascicle)
    
    return NNAreaMap, fiberNNRatioMap, axonNNRatioMap



def _getPixelSampleData(path_img, file_img, imgIdx, sampleName_list, dataname_list, axonLabels, myelinLabels, isFascicle, excludeMask_img_list, windowShapes_list,
                        chunkSize, onlyFascicleArea, circularWindow, imgLinearRes, getFiberD, getFiberP, getAxonP, getMyelinP, getGP, getNNratios,
                        useEdgeMask, edgeMask_type, edgeMaskLabels, edgeMask_img_list, save_path, res=1e-100):
    """
    Computes metrics for each pixel in image.
    :param path_img: String, path to folder where the segmentation images are stored and where the cleaned images will be saved.
    :param file_img: String of image file name.
    :param imgIdx: Integer for index of image in list of sample names.
    :param sampleName_list: List of strings. Each string is used as a label for the sample at the same index.
    :param dataname_list: List of strings. Each string is a label for a measurement being made with the same ordering as the data is stored in the pixel array.
    :param axonLabels: List of integers used to label axons in semantically segmented image.
    :param myelinLabels: List of integers used to label axons in semantically segmented image.
    :param isFascicle: Bool Array, background labeled False and fascicle labeled True. Shape (H,W).
    :param excludeMask_img_list: List of tuples, each tuple element being an exclusion mask file name (list must have same length as file_img_list, tuples can have any length).
    :param windowShapes_list: List of 2-element Tuples, 2 integer elements describing the height and width of the window to be used for calculating fiber density and packing.
    
    :param chunkSize: 2 Integer Tuple, descibes the shape of chunks for parallel processing.
    :param onlyFascicleArea: Boolean, whether to restrict measurements to fascicular area denoted in isFascicle.
    :param circularWindow: Boolean, for whether to use a circular window (with diameter equal to shortest value in windowShape) instead of a rectangle.
    :param imgLinearRes: Float, for the um/px resolution of the image.
    :param getFiberD: Boolean, whether to calculate and return fiber density.
    :param getFiberP: Boolean, whether to calculate and return fiber packing.
    :param getAxonP: Boolean, whether to calculate and return axon packing.
    :param getMyelinP: Boolean, whether to calculate and return myelin packing.
    :param getGP: Boolean, whether to calculate and return g-packing. G-packing is area of axons in a window divided by area of fibers in the same window. Only checked if both getFiberP and getAxonP are True.
    :param getNNratios: Boolean, whether to calculate and return nearest-neighbor area and ratios.
    
    :param useEdgeMask: Boolean, whether to calculate distance of each pixel from edge of masked object.
    :param edgeMask_type: String, one of {'mask', 'label', None}. This determines how feature information is read for calculating each cell's distance from that feature.
                          'mask' is used when a separate binary mask is being provided for the feature. 'label' is used when the feature is labeled in images in file_img_list. None is used when no distance is to be calculated. Default is None.
    :param edgeMask_img_list: List of binary edge mask file names (must have same size as file_img_list). If not empty, calculate the distance of each cell from edges of true labels in mask.
    :param edgeMaskLabels: List of integers, defining the labels from which to calculate the distance each cell is.
    :param save_path: String, path to folder where generated files will be saved. If not provided, will save to path_img.
    :param res: Float, very small value to avoid devide by zero errors when calculating g-packing.
    
    :return: Nothing
    
    :files generated: 2 zarr files are created for each sample, and 1 txt file shared by all samples
        - _pixelData.zarr: Zarr array file with shape (num_roi_pixels, num_measures).
        - _pixelMap.zarr: Zarr array file with shape (num_measures, image_height, image_width).
        - pixelDataNames.txt: Text file where each line is the name for a measure saved in the zarr files (with the same ordering).
    """
    
    
    cleanImage_pathName = os.path.join(path_img, file_img)

    print('')
    print('Processing Image: '+file_img)
    
    # Get sample name for image
    if len(sampleName_list) > 0:
        sampleName = sampleName_list[imgIdx]
    else:
        sampleName = file_img.replace('.','_').split('_')[0]

    # Load segmentation image and ignore extra z-dimensions if in color (rgb channels should each be equal to grayscale label)
    print('Loading Segmentation')
    prediction = mh.imread(cleanImage_pathName)
    if len(prediction.shape)>2:
        prediction = prediction[:,:,0]
    
    # Setup initial zarr array for saving data
    output_map_pathName = os.path.join(save_path, sampleName_list[imgIdx]+'_pixelMap.zarr')
    zarr_pixelMapArray = zarr.open(output_map_pathName, mode='w', shape=(1, prediction.shape[0], prediction.shape[1]), chunks=(1, 1000, 1000), dtype=float)
    
    # Extract axon and myelin masks from predictions
    isAxon = np.isin(prediction, axonLabels)
    isMyelin = np.isin(prediction, myelinLabels)
    
    print('Calculating Distance From Fascicle Edge')
    ## Get distance map from edges of fascicle
    # if excludeMask_img_list is not None:
    #     fascicleDistMap = imgLinearRes*np.sqrt(mh.distance(isTrueFascicle)) #mh.distance returns the Squared Euclidean Distance; so square root it and scale by image resolution to get actual distance (um)
    # else:
    fascicleDistMap = imgLinearRes*np.sqrt(mh.distance(isFascicle)) #mh.distance returns the Squared Euclidean Distance; so square root it and scale by image resolution to get actual distance (um)
    print('- Saving Fascicle Distance Map To Output Array')
    zarr_pixelMapArray[0,:,:] = fascicleDistMap
    # don't delete fascicleDistMap until after saving to DataArray (potentially incorporating exclusion regions) below
    
    
    if excludeMask_img_list is not None:
        print('Loading ROI Exclusions')
        # # Save copy of true fascicle mask for edge distance calculations
        # isTrueFascicle = np.copy(isFascicle)
        # # If provided load exclusion mask(s), ensure it's only 2 dimensions, and clear that region(s) from the fascicle mask
        if isinstance(excludeMask_img_list[imgIdx], list) or isinstance(excludeMask_img_list[imgIdx], tuple): # check if more than one mask to be excluded from this sample's fascicle mask
            excludeMask_img_sublist = excludeMask_img_list[imgIdx]
        else:
            excludeMask_img_sublist = [excludeMask_img_list[imgIdx]]
        for sublistIdx in range(len(excludeMask_img_sublist)): # loop through exclusion mask(s) for this sample
            excludeMask_pathName = os.path.join(path_img, excludeMask_img_sublist[sublistIdx])
            excludeMask = mh.imread(excludeMask_pathName).astype(bool)
            if len(excludeMask.shape)>2:
                excludeMask = excludeMask[:,:,0]
            isFascicle[excludeMask] = False
    
    # Setup DataArray to store unmapped data for multivariate analysis
    output_data_pathName = os.path.join(save_path, sampleName_list[imgIdx]+'_pixelData.zarr')
    zarr_pixelDataArray = zarr.open(output_data_pathName, mode='w', shape=(isFascicle.sum(), 1), chunks=(1000000, 1), dtype=float)
    
    zarr_pixelDataArray[:, 0] = fascicleDistMap[isFascicle]
    del fascicleDistMap
    
    
    
    # Crop image to isFascicle bbox to reduce computations
    # things to crop: isAxon, isMyelin, isFascicle
    bbox, padShape = _get_bbox_and_padShape(isFascicle)
    isAxon = isAxon[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    isMyelin = isMyelin[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    isFascicle = isFascicle[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    print('Working Area: '+str(bbox[2]-bbox[0])+'x'+str(bbox[3]-bbox[1]))
    
    ### Calculate pixel-wise metrics (eg. density and packing of axon, myelin, and fibers)
    print('Getting Labels')
    # Get labels for axon and associated myelin
    num_axons, labeledAxons, labeledMyelin, labeledFibers = _getAMFLabels(isAxon, isMyelin)
    
    
    print('Calculating Regionprops')
    # Get properties for regions labeled >0
    axonPropList = regionprops(labeledAxons)
    fiberPropList = regionprops(labeledFibers)
    
    
    ### Use edge mask to get fiber metrics and append onto data list
    if useEdgeMask:
        ## Load edge mask if provided
        if edgeMask_type == 'mask':
            edgeMask_pathName = os.path.join(path_img, edgeMask_img_list[imgIdx])
            print('Using Provided Edge Mask To Get Edge Distance')
            isEdge = np.logical_not(mh.imread(edgeMask_pathName).astype(bool))
            if len(isEdge.shape)>2:
                isEdge = isEdge[:,:,0]
        elif edgeMask_type == 'label':
            print('Using Provided Edge Label To Get Edge Distance')
            if len(edgeMaskLabels) == 0:
                raise Exception('Error: edgeMaskLabels is empty list, but edgeMask_type is label')
            isEdge = np.isin(prediction, edgeMaskLabels, invert=True)
    
        ## Get distance map from edges
        edgeDistMap = imgLinearRes*np.sqrt(mh.distance(isEdge)) #mh.distance returns the Squared Euclidean Distance; so square root it and scale by image resolution to get actual distance (um)
        print('- Saving Edge Distance Map To Output Array')
        zarr_pixelMapArray.append(np.expand_dims(edgeDistMap, axis=0), axis=0)
        zarr_pixelDataArray.append(np.expand_dims(edgeDistMap[bbox[0]:bbox[2],bbox[1]:bbox[3]][isFascicle], axis=1), axis=1)
        del edgeDistMap
    
    
    # Loop through given window sizes, calculating density and packing
    windowShapeTQDM = tqdm(windowShapes_list, total=len(windowShapes_list))
    for windowShape in windowShapeTQDM:
        windowShapeTQDM.set_description("Calculating Data using window shape %s" % str(windowShape))
    
        # Check if window larger than image
        if (windowShape[0]>prediction.shape[0]) or (windowShape[1]>prediction.shape[1]):
            raise Exception('Window is larger than image')
        if circularWindow:
            window = sm.disk(np.min(windowShape)/2-0.5, dtype=bool)
        else:
            window = np.ones(windowShape, dtype=bool)
    
        if onlyFascicleArea:
            # Calculate area of windows covered by fascicle
            # print('Calculating area of fascicle in each window in parallel')
            desc = "Calculating Fascicle Area Map"
            fascicleAreaMap = _getPaddedSum(isFascicle, window, chunkSize, desc=desc)
        else:
            fascicleAreaMap = None
        
        ## Get packing/density maps
        if getFiberD:
            # print('Calculating Pixel-Wise Fiber Density')
            # Calculate fiber density based on pixel-centered windows
            desc = "Calculating Fiber Density Map"
            fiberDensityMap = _getPixelDensity(labeledFibers, fiberPropList, onlyFascicleArea=onlyFascicleArea, fascicleAreaMap=fascicleAreaMap,
                                              window=window, imgLinearRes=imgLinearRes, chunkSize=chunkSize, desc=desc)
            # if saveData:
            zarr_pixelMapArray.append(np.expand_dims(np.pad(fiberDensityMap,padShape), axis=0), axis=0)
            zarr_pixelDataArray.append(np.expand_dims(fiberDensityMap[isFascicle], axis=1), axis=1)
            del fiberDensityMap
        
        if getFiberP:
            # print('Calculating Pixel-Wise Fiber Packing')
            # Calculate fiber packing based on pixel-centered windows
            desc = "Calculating Fiber Packing Map"
            fiberPackingMap = _getPixelPacking(labeledFibers, onlyFascicleArea=onlyFascicleArea, fascicleAreaMap=fascicleAreaMap,
                                               window=window, imgLinearRes=imgLinearRes, chunkSize=chunkSize, desc=desc)
            # if saveData:
            zarr_pixelMapArray.append(np.expand_dims(np.pad(fiberPackingMap,padShape), axis=0), axis=0)
            zarr_pixelDataArray.append(np.expand_dims(fiberPackingMap[isFascicle], axis=1), axis=1)
            
        if getAxonP:
            # print('Calculating Pixel-Wise Axon Packing')
            # Calculate axon packing based on pixel-centered windows
            desc = "Calculating Axon Packing Map"
            axonPackingMap = _getPixelPacking(labeledAxons, onlyFascicleArea=onlyFascicleArea, fascicleAreaMap=fascicleAreaMap,
                                              window=window, imgLinearRes=imgLinearRes, chunkSize=chunkSize, desc=desc)
            # if saveData:
            zarr_pixelMapArray.append(np.expand_dims(np.pad(axonPackingMap,padShape), axis=0), axis=0)
            zarr_pixelDataArray.append(np.expand_dims(axonPackingMap[isFascicle], axis=1), axis=1)
            
        if getMyelinP:
            # print('Calculating Pixel-Wise Myelin Packing')
            # Calculate myelin packing based on pixel-centered windows
            if getFiberP and getAxonP:
                myelinPackingMap = fiberPackingMap-axonPackingMap
            else:
                # Perform Calculation the hard way
                desc = "Calculating Myelin Packing Map"
                myelinPackingMap = _getPixelPacking(labeledMyelin, onlyFascicleArea=onlyFascicleArea, fascicleAreaMap=fascicleAreaMap,
                                                    window=window, imgLinearRes=imgLinearRes, chunkSize=chunkSize, desc=desc)
            # if saveData:
            zarr_pixelMapArray.append(np.expand_dims(np.pad(myelinPackingMap,padShape), axis=0), axis=0)
            zarr_pixelDataArray.append(np.expand_dims(myelinPackingMap[isFascicle], axis=1), axis=1)
            del myelinPackingMap
        
        if getGP:
            gpTQDM = tqdm([0], total=1, desc = 'Calculating G-Packing Map')
            for _ in gpTQDM:
                # Calculate g-packing based on pixel-centered windows
                gPackingMap = np.zeros(fiberPackingMap.shape)
                if getFiberP and getAxonP:
                    fiberAreaMap = fiberPackingMap
                    axonAreaMap = axonPackingMap
                else:
                    # Perform Calculation the hard way
                    # print('- Calculating Fiber Areas by summing area of labels in each window in parallel')
                    desc = "Calculating Fiber Area Map for G-Packing Map"
                    fiberAreaMap = _getPaddedSum(labeledFibers>0, window, chunkSize, desc=desc)
                    # print('- Calculating Axon Areas by summing area of labels in each window in parallel')
                    desc = "Calculating Axon Area Map for G-Packing Map"
                    axonAreaMap = _getPaddedSum(labeledAxons>0, window, chunkSize, desc=desc)
                
                # Check if fiberAreaMap has zeros within fascicle(s)
                hasZeros = (fiberAreaMap[isFascicle]==0).any()
                if hasZeros:
                    gPackingMap[isFascicle] = axonAreaMap[isFascicle]/(res+fiberAreaMap[isFascicle])
                else:
                    gPackingMap[isFascicle] = axonAreaMap[isFascicle]/fiberAreaMap[isFascicle]
                
                # Save data
                zarr_pixelMapArray.append(np.expand_dims(np.pad(gPackingMap,padShape), axis=0), axis=0)
                zarr_pixelDataArray.append(np.expand_dims(gPackingMap[isFascicle], axis=1), axis=1)
                del gPackingMap
        
        # Now that we don't need them, clean up fiber and axon packing maps if generated
        if getFiberP:
            del fiberPackingMap
        if getAxonP:
            del axonPackingMap
        
    if getNNratios:
        print('Calculating Nearest-Neighbor Fascicle Area and Ratios')
        fascicleNNAreaMap, fiberNNRatioMap, axonNNRatioMap = _getPixelNNArea(labeledFibers, fiberPropList, axonPropList, imgLinearRes=imgLinearRes, fascicleMask=isFascicle)
        myelinNNRatioMap = fiberNNRatioMap - axonNNRatioMap
        print('- Saving NN-Data To Output Array')
        zarr_pixelMapArray.append(np.expand_dims(np.pad(fascicleNNAreaMap,padShape), axis=0), axis=0)
        zarr_pixelMapArray.append(np.expand_dims(np.pad(fiberNNRatioMap,padShape), axis=0), axis=0)
        zarr_pixelMapArray.append(np.expand_dims(np.pad(axonNNRatioMap,padShape), axis=0), axis=0)
        zarr_pixelMapArray.append(np.expand_dims(np.pad(myelinNNRatioMap,padShape), axis=0), axis=0)
        zarr_pixelDataArray.append(np.expand_dims(fascicleNNAreaMap[isFascicle], axis=1), axis=1)
        zarr_pixelDataArray.append(np.expand_dims(fiberNNRatioMap[isFascicle], axis=1), axis=1)
        zarr_pixelDataArray.append(np.expand_dims(axonNNRatioMap[isFascicle], axis=1), axis=1)
        zarr_pixelDataArray.append(np.expand_dims(myelinNNRatioMap[isFascicle], axis=1), axis=1)
        del fascicleNNAreaMap
        del fiberNNRatioMap
        del axonNNRatioMap
        del myelinNNRatioMap
    
    
    
    


def getPixelData(file_img_list, path_img, sampleName_list=[], imgLinearRes=0.125, maskLabels=[[255], [170]], windowShapes_list=[(400,400)],
                 save_path=None,
                 fascicleMask_img_list=[], onlyFascicleArea=True, excludeMask_img_list=None, circularWindow=False,
                 edgeMask_type=None, edgeMaskLabels=[], edgeMask_img_list=[], edgeDist_Name='edgeDistance',
                 num_cpus=10, chunkSize=(3000,3000), getFiberD=True, getFiberP=True, getAxonP=True, getMyelinP=True, getGP=True, getNNratios=True, res=1e-100):
    """
    Computes metrics for each pixel in each image and combines pixel data from multiple images of the same sample.
    :param file_img_list: List of sublists. Each sublist contains image file names of images from the same sample.
    :param path_img: String, path to folder where the segmentation images are stored and where the cleaned images will be saved.
    :param sampleName_list: List of strings (same length as file_img_list). Each string is used as a label for the image at the same index.
    :param imgLinearRes: Float, for the um/px resolution of the image. Default is 0.125.
    :param maskLabels: 3 item list, first item is an integer for the background label, second item is a list of integers for myelin labels, third item is a list of integers for axon labels (labels provided imgs)
    :param windowShapes_list: List of 2-element Tuples, 2 integer elements describing the height and width of the window to be used for calculating fiber density and packing. Default is (400,400).
    
    :param save_path: String, path to folder where generated files will be saved. If not provided, will save to path_img.
    
    :param fascicleMask_img_list: List of fascicle mask file names (must have same length as file_img_list).
    :param onlyFascicleArea: Boolean, whether to restrict measurements to fascicular area denoted in fascicleMAsk. Default is True.
    :param excludeMask_img_list: None or a List of tuples, each tuple element being an exclusion mask file name (list must have same length as file_img_list, tuples can have any length).
    :param circularWindow: Boolean, for whether to use a circular window (with diameter equal to shortest value in windowShape) instead of a rectangle.
    
    :param edgeMask_type: String, one of {'mask', 'label', None}. This determines how feature information is read for calculating each cell's distance from that feature.
                          'mask' is used when a separate binary mask is being provided for the feature. 'label' is used when the feature is labeled in images in file_img_list. None is used when no distance is to be calculated. Default is None.
    :param edgeMaskLabels: List of integers, defining the labels from which to calculate the distance each cell is.
    :param edgeMask_img_list: List of binary edge mask file names (must have same size as file_img_list). If not empty, calculate the distance of each cell from edges of true labels in mask.
    :param edgeDist_Name: String, name to label edgeMask distance data.
    
    :param num_cpus: Number of cpu's to use for parallel processing. Default is 10.
    :param chunkSize: 2 Integer Array, descibes the shape of chunks for parallel processing. Default is (3000,3000).
    
    :param getFiberD: Boolean, whether to calculate and return fiber density. Default is True.
    :param getFiberP: Boolean, whether to calculate and return fiber packing. Default is True.
    :param getAxonP: Boolean, whether to calculate and return axon packing. Default is True.
    :param getMyelinP: Boolean, whether to calculate and return myelin packing. Default is True.
    :param getGP: Boolean, whether to calculate and return g-packing. G-packing is area of axons in a window divided by area of fibers in the same window. Only checked if both getFiberP and getAxonP are True.
    :param getNNratios: Boolean, whether to calculate and return nearest-neighbor area and ratios. Default is True.
    :param res: Float, very small value to avoid devide by zero errors when calculating g-packing.
    
    :return: Nothing
    
    :files generated: 2 zarr files are created for each sample, and 1 txt file shared by all samples
        - _pixelData.zarr: Zarr array file with shape (num_roi_pixels, num_measures).
        - _pixelMap.zarr: Zarr array file with shape (num_measures, image_height, image_width).
        - pixelDataNames.txt: Text file where each line is the name for a measure saved in the zarr files (with the same ordering).
    """
    
    # Shutdown any running ray, before init, incase of messy end of code
    print('Initializing parallel processing')
    ray.shutdown()
    ray.init(num_cpus=num_cpus)
    
    # If no save_path given set to path_img
    if save_path is None:
        save_path = path_img
    # If save directory doesn't exist, create it
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Generate a list of names for each sample if not provided
    if not (len(sampleName_list) == len(file_img_list)):
        sampleName_list = [file_img.replace('.','_').split('_')[0] for file_img in file_img_list]
    
    # Parse label lists: background has one label, but myelin and axon may have more than one (hence lists instead of ints) [this is the index value returned by axon_segmentation]
    # backgroundLabel = maskLabels[0]
    axonLabels = maskLabels[0]
    myelinLabels = maskLabels[1]
    
    # Ensure there are fascicle masks provided and matching number of samples
    if len(fascicleMask_img_list)==0:
        raise Exception('No fascicle mask info provided.')
    if len(fascicleMask_img_list)!=len(file_img_list):
        raise Exception('Length of fascicle mask list not equal to length of sample list.')
        
    # Check if edge mask is to be used
    useEdgeMask = edgeMask_type is not None #len(edgeMask_img_list)>0
    
    # If either getFiberP or getAxonP are false, don't calculate g-Packing
    if getFiberP==False | getAxonP==False:
        getGP = False
    
    # Setup list for storing sample data [[sample1],...] where each sample is a list [sampleName, cellCount, axonSize, ...]
    samplesList_sampleDataList_pixelDataArray = []
    dataname_list = ['FascicleEdgeDistance']
    # skipMapping = 1 # first element of data list is a single value NOT a distribution
    
    if useEdgeMask:
        dataname_list.append(edgeDist_Name)
    
    for windowShape in windowShapes_list:
        if getFiberD:
            dataname_list.append('FiberDensity_'+str(min(windowShape)))
        if getFiberP:
            dataname_list.append('FiberPacking_'+str(min(windowShape)))
        if getAxonP:
            dataname_list.append('AxonPacking_'+str(min(windowShape)))
        if getMyelinP:
            dataname_list.append('MyelinPacking_'+str(min(windowShape)))
        if getGP:
            dataname_list.append('GPacking_'+str(min(windowShape)))
    
    if getNNratios:
        dataname_list = dataname_list + ['FascicleNNArea', 'FiberNNRatio', 'AxonNNRatio', 'MyelinNNRatio']
    
    
    # Save list of names of measured metrics
    # if saveData:
    output_dataNames_pathName = os.path.join(save_path, 'pixelDataNames.txt')
    with open(output_dataNames_pathName, "w") as textfile:
        for dataname in dataname_list:
            textfile.write(dataname + "\n")
    
    # For each sample, calculate data and save to zarr arrays
    # file_img_TQDM = tqdm(enumerate(file_img_list))
    for imgIdx, file_img in enumerate(file_img_list):
        # file_img_TQDM.set_description("Measuring %s" % file_img)
        # Load fascicle masks and ensure only 2 dimensions
        fascicleMask_pathName = os.path.join(path_img, fascicleMask_img_list[imgIdx])
        isFascicle = mh.imread(fascicleMask_pathName).astype(bool)
        if len(isFascicle.shape)>2:
            isFascicle = isFascicle[:,:,0]
        
        # sampleDataList_pixelDataArray = 
        _getPixelSampleData(path_img, file_img, imgIdx, sampleName_list, dataname_list, axonLabels, myelinLabels, isFascicle,
                                            excludeMask_img_list, windowShapes_list, chunkSize, onlyFascicleArea, circularWindow, imgLinearRes,
                                            getFiberD, getFiberP, getAxonP, getMyelinP, getGP, getNNratios,
                                            useEdgeMask, edgeMask_type, edgeMaskLabels, edgeMask_img_list, save_path, res=res)
        # # Create new sublist for the sample and append to list of samples
        # samplesList_sampleDataList_pixelDataArray.append(sampleDataList_pixelDataArray)
        
        print("Garbage Collecting")
        gc.collect()
    
    ray.shutdown()
    
