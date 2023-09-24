# paths class

import os
print('generating paths...')

def generate_paths():
    # Define data source and destination details
    DatastoreFolderTargetName = 'model training'
    TrainingFileNameCNN = 'Room_75_1Min_Train_Jan_Jul.csv'
    TrainingFileNameMLP = 'room75_1H_train_jan_jul.csv'
    TargetFileName = 'room75_result_train_jan_jul.csv'

    # Get the directory of the current script
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Join it with the desired subdirectories
    DestinationFolder = os.path.join(script_directory, DatastoreFolderTargetName)

    # Define results folder details
    folderOfResults = 'test_architectures'
    folderOfResultsF = os.path.join(DatastoreFolderTargetName, folderOfResults)
    folderOfResults = os.path.join(DestinationFolder, folderOfResultsF)

    # Create the results folder
    os.makedirs(folderOfResults, exist_ok=True)

    # The Paths for input Training and Target File are created
    TrainingPathCNN = os.path.join(DestinationFolder, TrainingFileNameCNN)
    TrainingPathMLP = os.path.join(DestinationFolder, TrainingFileNameMLP)
    TargetPath = os.path.join(DestinationFolder, TargetFileName)

    # Return as a dictionary
    return {
        'TrainingPathCNN': TrainingPathCNN,
        'TrainingPathMLP': TrainingPathMLP,
        'TargetPath': TargetPath,
        'DestinationFolder': DestinationFolder,
        'folderOfResults': folderOfResults
    }
