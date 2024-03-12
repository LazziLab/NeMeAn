# Created on Monday November 12 14:23:00 2022
# @author: Andres
# Takes morphometric dataset, trains models to fit to given
# sample values, and validates the regression on the data set
# aside. 


import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import zarr
import dill

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import Nystroem





def loadCellSamples(data_path, cellData_list, dataname_list_file, cellCounts_list=None):
    """
    Gets lists of samples' data and cell counts
    :param data_path: String, path to folder where the data and cell count zarr arrays and dataname txt file are saved.
    :param cellData_list: List of strings of names of cell data zarr arrays.
    :param dataname_list_file: String of name of txt file containing list of names of measured data.
    :param cellCounts_list: None or a List of strings of names of cell count zarr arrays. Default is None.
    :return: 3 Lists (each length = # samples) containing the respective cell data arrays, cell data measurement names, and cell counts for each sample.
    """
    
    samplesList_cellDataArray = []
    samplesList_cellCountsArray = []
    
    # Loop through samples
    for sampleIdx in range(len(cellData_list)):
        # Load data array (measurement, cell)
        cellData_pathName = os.path.join(data_path, cellData_list[sampleIdx])
        cellData_arrays = zarr.open(cellData_pathName, mode='r') #np.copy(zarr.load(cellData_pathName))
        if cellData_arrays is None:
            raise Exception("File not found at: %s" % cellData_pathName)
        
        # Load cell counts [#complete, #collapsed]
        if cellCounts_list is None:
            # Assume all cells are typical [i.e. not hypomylinated or degenerated]
            # cellData_arrays are size [num_cells, num_features]
            cellCounts = np.array([cellData_arrays.shape[0],0], dtype=int)
        else:
            cellCounts_pathName = os.path.join(data_path, cellCounts_list[sampleIdx])
            cellCounts = zarr.open(cellCounts_pathName, mode='r') #np.copy(zarr.load(cellCounts_pathName))
            if cellCounts is None:
                raise Exception("File not found at: %s" % cellCounts_pathName)
        samplesList_cellDataArray.append(cellData_arrays)
        samplesList_cellCountsArray.append(cellCounts)
    
    # Load data names
    dataNames_pathName = os.path.join(data_path, dataname_list_file)
    with open(dataNames_pathName, "r") as textfile:
        dataname_list = [line.rstrip() for line in textfile.readlines()]
    
    return samplesList_cellDataArray, dataname_list, samplesList_cellCountsArray


def loadPixelSamples(data_path, pixelData_list, dataname_list_file):
    """
    Gets lists of samples' pixel-wise measurement data
    :param data_path: String, path to folder where the data zarr arrays and dataname txt file are saved.
    :param pixelData_list: List of strings of names of pixel data zarr arrays.
    :param dataname_list_file: String of name of txt file containing list of names of measured data.
    :return: 2 Lists (each length = # samples) containing the respective pixel data arrays and pixel data measurement names for each sample.
    """
    
    samplesList_pixelDataArray = []
    
    # # Loop through samples
    # for sampleIdx in range(len(pixelData_list)):
    num_samples = len(pixelData_list)
    sampleTQDM = tqdm(range(num_samples), total=num_samples)
    for sampleIdx in sampleTQDM:
        sampleTQDM.set_description("Loading sample: %s" % pixelData_list[sampleIdx])
        # Load data array (measurement, cell)
        pixelData_pathName = os.path.join(data_path, pixelData_list[sampleIdx])
        pixelData_arrays = zarr.open(pixelData_pathName, mode='r') #np.copy(zarr.load(pixelData_pathName))
        if pixelData_arrays is None:
            raise Exception("File not found at: %s" % pixelData_pathName)
        
        samplesList_pixelDataArray.append(pixelData_arrays)
    
    # Load data names
    dataNames_pathName = os.path.join(data_path, dataname_list_file)
    with open(dataNames_pathName, "r") as textfile:
        dataname_list = [line.rstrip() for line in textfile.readlines()]
    
    return samplesList_pixelDataArray, dataname_list





def trainRegs(samplesList_dataArray, sampleName_list, sampleIdx_list, sampleRegTargetVal_list_list, sampleRegBool_list_list, featureIdx_list_list,
              targetValueName_list=None, feature_groupName_list=None,
              train_ratio=0.7, equalTrainingSampleSize=False, max_iter=5000, tol=1e-12,
              save_path='', saveValidationData=False):
    """
    Trains svm models to each fit a regression (1 for effect of implantation, and for effect of stimulation 1 for each stimulation target value).
    :param samplesList_dataArray: List (length = # samples) containing the respective pixel data arrays.
    :param sampleName_list: List of strings (same length as samplesList_dataArray). Each string is used as a label for the sample at the same index.
    :param sampleIdx_list: List of sample indices to be drawn from for fitting.
    :param sampleRegTargetVal_list_list: List of sublists of numbers (list length = # of regressions models to fit, sublist length = # samples); each number is a target value for a sample to which the model will try and fit.
    :param sampleRegBool_list_list: List of sublists of booleans (list length = # of regressions models to fit, sublist length = # samples); each boolean is whether to include a sample in the regression model training.
    :param featureIdx_list_list: List of sublists of integers; each sublist is a set of indices for features to use in model training.
    :param targetValueName_list: List of strings (length equal to sampleRegTargetVal_list_list) with names for each target value type.
    :param feature_groupName_list: List of strings (length equal to featureIdx_list_list) with names for each feature set.
    
    :param train_ratio: Float (between 0 and 1) for the fraction of each sample to use in training (the rest is used in validation). Default is 0.7 (meaning 30% is saved for validation)
    :param equalTrainingSampleSize: Boolean for whether to have each sample provide the same number points for training (results in only the smallest sample providing the full train_ratio percent of its data for training). Default is False.
    :param max_iter: Int for max number of iteration used by SGDRegressor. Default is 5000.
    :param tol: Float for the stopping criterion used by SGDRegressor. Default is 1e-12.

    :param save_path: String, path to folder where the output files will be saved. Default is the current directory.
    :param saveValidationData: Boolean, whether to save the output of the regressions applied to just the part of the data set aside for validation as an additional separate zarr file. Default if False.
    
    :return: Nothing

    :files generated: 1 or 2 zarr file(s) is created for each sample, and 1 txt file shared by all samples
        - _regData.zarr: Zarr array file with shape (num_measurements, num_models).
        - _valData.zarr: Zarr array file with shape (num_validation_measurements, num_models). Generated if saveValidationData is True.
        - regDataNames.txt: Text file where each line is the name for a regression output saved in _regData.zarr (with the same ordering).
    """
    # Check if feture sets are given names, and if not leaving empty strings
    if targetValueName_list is None:
        targetValueName_list = ["" for i in range(len(sampleRegTargetVal_list_list))]

    # Check if feture sets are given names, and if not leaving empty strings
    if feature_groupName_list is None:
        feature_groupName_list = ["" for i in range(len(featureIdx_list_list))]
    
    
    sampleSize_array = np.array([sampleDataArray.shape[0] for sampleDataArray in samplesList_dataArray])
    minSampleSize = int(np.min(sampleSize_array))
    minTrainSize = int(minSampleSize*train_ratio)
    minValidSize = int(minSampleSize*(1-train_ratio))
    
    # Set aside 1-train_ratio of each sample for validation testing later
    print('Splitting measurements into train and validation sets')
    train_bool_list = []
    valid_bool_list = []
    trainSize_array = np.zeros(sampleSize_array.shape, dtype=int)
    validSize_array = np.zeros(sampleSize_array.shape, dtype=int)
    for sampleIdx, sampleSize in enumerate(sampleSize_array):
        sampleSize = sampleSize_array[sampleIdx]
        train_bool = np.zeros(sampleSize, dtype=bool)
        if equalTrainingSampleSize:
            # select training(/validation) data such that every sample provides the same amount of data points (and the smallest sample provides train_ratio(1-train_ratio) of its data pointsto the training(/validation) set)
            random_idxs = np.random.permutation(np.arange(sampleSize))
            train_idxs = random_idxs[:minTrainSize]
            valid_idxs = random_idxs[-minValidSize:]
            train_bool[train_idxs] = True
            valid_bool = np.zeros(sampleSize, dtype=bool)
            valid_bool[valid_idxs] = True
        else:
            # select training(/validation) data such that every sample provides train_ratio(1-train_ratio) of its data points to the training(/validation) set
            train_idxs = np.random.permutation(np.arange(sampleSize))[:int(sampleSize*train_ratio)]
            train_bool[train_idxs] = True
            valid_bool = np.logical_not(train_bool)
        
        train_bool_list.append(train_bool)
        valid_bool_list.append(valid_bool)
        trainSize_array[sampleIdx] = train_bool.sum()
        validSize_array[sampleIdx] = valid_bool.sum()
    
    
    
    # If save directory doesn't exist, create it
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    num_target_vals = len(sampleRegTargetVal_list_list) # Number of target values to be fit to
    num_featureGroups = len(featureIdx_list_list) # Number of sets of features being used for regression fitting
    num_models_out = num_target_vals * num_featureGroups # Total number of models to be trained, one for each combination of target value and feature set
    
    ## Generate zarr arrays for storing regression predictions for each sample
    samplesList_modelDataArray = []
    samplesList_validDataArray = []
    for sampleIdx, coreSampleName in enumerate(sampleName_list):
        sampleZarrName = coreSampleName+'_regData.zarr'
        output_data_pathName = os.path.join(save_path, sampleZarrName)
        sample_num_data = sampleSize_array[sampleIdx]
        samplesList_modelDataArray.append(zarr.open(output_data_pathName, mode='w', shape=(sample_num_data, num_models_out), chunks=(100000000,1), dtype=float))
        
        if saveValidationData:
            sampleZarrName = coreSampleName+'_valData.zarr'
            output_data_pathName = os.path.join(save_path, sampleZarrName)
            sample_num_data = validSize_array[sampleIdx]
            samplesList_validDataArray.append(zarr.open(output_data_pathName, mode='w', shape=(sample_num_data, num_models_out), chunks=(100000000,1), dtype=float))
    
    ## Save list of names of regression metrics
    # List of model names
    modelName_List = [feature_groupName_list[featureGroupIdx]+'_'+targetValueName_list[targetIdx] for featureGroupIdx in range(num_featureGroups) for targetIdx in range(num_target_vals)]
    # If save directory doesn't exist, create it
    Path(save_path).mkdir(parents=True, exist_ok=True)
    output_dataNames_pathName = os.path.join(save_path, 'regDataNames.txt')
    with open(output_dataNames_pathName, "w") as textfile:
        for dataname in modelName_List:
            textfile.write(dataname + "\n")
    
        
    # Loop through feature sets
    sampleIdx_array = np.array(sampleIdx_list, dtype=int)
    for featureGroupIdx, featureIdx_list in enumerate(featureIdx_list_list):
        num_features_in = len(featureIdx_list)
        
        # RBF settings
        # gamma defaults to auto
        # auto: 1 / n_features
        # scale: 1 / (n_features * X.var())
        gamma = 1/num_features_in
        
        # Loop through training targets (e.g. surgery or stim[Shannon])
        for targetIdx in range(0,num_target_vals):
            print('Filtering data for training '+feature_groupName_list[featureGroupIdx]+'_'+targetValueName_list[targetIdx])
            
            # Nystroem approximation of RBF kernel
            model = Pipeline([('scale0', StandardScaler()),
                              ('nystroem', Nystroem(kernel='rbf', gamma=gamma)), #, gamma=None, random_state=None, n_components=100)),
                              ('regression', SGDRegressor(max_iter=max_iter, tol=tol)) ])
            
            # Select only the samples allowed by sampleRegBool_list_list for this set of target values
            sampleRegBool_array = np.array(sampleRegBool_list_list[targetIdx], dtype=bool)
            targetSampleIdx_array = sampleIdx_array[sampleRegBool_array]
            
            # Compile into train/validation datasets only samples and features selected for regression to fit (checking if source data is zarr array)
            if isinstance(samplesList_dataArray[0], zarr.core.Array):
                X_train = np.concatenate([samplesList_dataArray[sampleIdx].oindex[train_bool_list[sampleIdx],featureIdx_list] for sampleIdx in targetSampleIdx_array], axis=0)
                X_valid = np.concatenate([samplesList_dataArray[sampleIdx].oindex[valid_bool_list[sampleIdx],featureIdx_list] for sampleIdx in targetSampleIdx_array], axis=0)
            else:
                X_train = np.concatenate([samplesList_dataArray[sampleIdx][train_bool_list[sampleIdx],:][:,featureIdx_list] for sampleIdx in targetSampleIdx_array], axis=0)
                X_valid = np.concatenate([samplesList_dataArray[sampleIdx][valid_bool_list[sampleIdx],:][:,featureIdx_list] for sampleIdx in targetSampleIdx_array], axis=0)
            
            # Select data that the regression is attempting to predict (filtering for train/valid and selected samples)
            sampleRegTargetVal_array = np.array(sampleRegTargetVal_list_list[targetIdx]) # get sample-wise target values
            y_train = np.repeat(sampleRegTargetVal_array[sampleRegBool_array], trainSize_array[targetSampleIdx_array]) #filter for samples selected and expand to measurement-wise target values
            y_valid = np.repeat(sampleRegTargetVal_array[sampleRegBool_array], validSize_array[targetSampleIdx_array])
            
            # Finally fit the data to the target values
            print('Training '+feature_groupName_list[featureGroupIdx]+'_'+targetValueName_list[targetIdx])
            model.fit(X_train, y_train)
            
            # Evaluate Algorithm
            print('-score')
            model_score = model.score(X_valid, y_valid)
            # modelscore_list.append(model_score)
            print(model_score)
            # output model info
            # print('-iterations')
            # print(model.named_steps['regression'].n_iter_)
            # print('-coef')
            # print(model.named_steps['regression'].coef_)
            
            # Save Regresion
            print('-save')
            with open(os.path.join(save_path, feature_groupName_list[featureGroupIdx]+'_'+targetValueName_list[targetIdx]+'.pkl'), 'wb') as file:
                dill.dump([model], file)
            # model_list.append(model)
            
            
            # apply to entire dataset (samplesList_dataArray)
            for sampleIdx, sampleDataArray in tqdm(enumerate(samplesList_dataArray), total=len(samplesList_dataArray)):
                # select only the features used for training (checking if data is zarr format or not)
                if isinstance(sampleDataArray, zarr.core.Array):
                    sampleDataArray_feat = sampleDataArray.oindex[:,featureIdx_list]
                else:
                    sampleDataArray_feat = sampleDataArray[:,featureIdx_list]
                samplesList_modelDataArray[sampleIdx][:,featureGroupIdx*num_target_vals+targetIdx] = model.predict(sampleDataArray_feat)
            
            if saveValidationData:
                # apply to validation subset of data (samplesList_dataArray)
                for sampleIdx, sampleDataArray in tqdm(enumerate(samplesList_dataArray), total=len(samplesList_dataArray)):
                    # select only the features used for training (checking if data is zarr format or not)
                    if isinstance(sampleDataArray, zarr.core.Array):
                        sampleDataArray_validFeat = sampleDataArray.oindex[valid_bool_list[sampleIdx],featureIdx_list]
                    else:
                        sampleDataArray_validFeat = sampleDataArray[valid_bool_list[sampleIdx],:][:,featureIdx_list]
                    samplesList_validDataArray[sampleIdx][:,featureGroupIdx*num_target_vals+targetIdx] = model.predict(sampleDataArray_validFeat)
