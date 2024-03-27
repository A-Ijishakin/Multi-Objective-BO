import requests
import tempfile
import os
import numpy as np
import pandas as pd
from joblib import load,dump
from sklearn.preprocessing import StandardScaler
from matminer.featurizers.composition import ElementProperty
from pymatgen.core.composition import Composition

def get_datasets():
    """
        Download to a local temp folder the SIMD thermoelectric dataset
    """
    dataset_url = "https://github.com/KRICT-DATA/SIMD/raw/main/dataset/estm.xlsx"
    response = requests.get(dataset_url)
    tmpdir =  tempfile.gettempdir()
    filepath = os.path.join(tmpdir, 'estm.xlsx')

    if response.status_code == 200:
        with open(filepath, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded: {filepath}")
    else:
        ValueError(f'Failed to download file: {response.status_code}')
    
    return filepath


def featurize_and_targets(dataset, element_map="magpie",scale_path="target_norm.joblib"):
    """
    Featurizes the 'Formula' column of a dataset using the specified element map and includes temperature as a feature.
    Splits the dataset into a features DataFrame containing only the featurized vectors and temperature, 
    and a targets DataFrame containing 'seebeck_coefficient(μV/K)', 'electrical_conductivity(S/m)', 
    and 'thermal_conductivity(W/mK)'.
    
    Parameters:
        dataset (str): The path to the dataset file.
        element_map (str, optional): The name of the preset element mapping for featurization.
        scale_path (str, optional): File saving the target scales.
    
    Returns:
        pd.DataFrame: Features
        pd.DataFrame: Targets.
    """
    df = pd.read_excel(dataset)
    
    # Featurize the 'Formula' column
    df['composition'] = df['Formula'].apply(lambda x: Composition(x))
    ep_featurizer = ElementProperty.from_preset(preset_name=element_map)
    df = ep_featurizer.featurize_dataframe(df, col_id='composition', ignore_errors=True)

    target_columns = ['seebeck_coefficient(μV/K)', 'electrical_conductivity(S/m)', 'thermal_conductivity(W/mK)']
    targets = df[target_columns].copy()

    # Normalize the target columns
    scalers = {col: StandardScaler() for col in target_columns}
    for col, scaler in scalers.items():
        targets[col] = scaler.fit_transform(targets[[col]])    
    dump(scalers, scale_path)
    
    # Remove columns not needed
    non_feature_columns = ['Formula', 
                           'composition',
                           'power_factor(W/mK2)',
                           'ZT','reference'] + target_columns
    features = df.drop(columns=non_feature_columns)
    #features['Temperature'] = df['temperature(K)']  
    
    return features, targets


def featurize(composition,element_map="magpie",temperature=300.0):
    featurizer = ElementProperty.from_preset(preset_name=element_map)
    features = [temperature,]
    features += featurizer.featurize(Composition(composition))
    return features

def reverse_target_norm(predicted_targets, scalers_path='target_norm.joblib'):
    """
    Reverses the normalization for a list of predicted target values, converting them back to their original scale.
        """
    scalers = load(scalers_path)
    original_scale_predictions = np.array([
        scalers[list(scalers.keys())[i]].inverse_transform([[prediction]])[0][0] 
        for i, prediction in enumerate(predicted_targets)
    ])
    return original_scale_predictions
