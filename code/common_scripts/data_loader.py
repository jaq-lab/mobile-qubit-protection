"""
Data Loader Utilities

This module provides standardized functions for loading processed data
with backward compatibility to legacy file formats.
"""

import pickle
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Legacy file mappings for backward compatibility
LEGACY_FILE_MAP = {
    'fig1': {
        'high': '2point_high_75_mV.pkl',  # Default, can specify mV
        'low': '2point_low.pkl'
    },
    'fig2': {
        'high': 'stationary_high_dense.pkl',
        'low': 'stationary_low.pkl'
    },
    'fig4': {
        'high': 'shuttling_high.pkl',
        'low': 'shuttling_low.pkl'
    },
    'fig5': {
        'driven': 'driven_gen.pkl'
    }
}


def load_figure_data(
    figure_number: str,
    version: str = "latest",
    base_path: Union[str, Path] = "processed_data",
    field: Optional[str] = None,
    legacy_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> Any:
    """
    Load processed data for a specific figure.
    
    Parameters:
    -----------
    figure_number : str
        Figure identifier (e.g., 'fig1', 'fig2', 'fig4', 'fig5')
    version : str
        'latest' or specific filename
    base_path : str or Path
        Base path for processed data (new structure)
    field : str, optional
        For figures with multiple fields ('high', 'low')
    legacy_path : str or Path, optional
        Path to legacy processed_data directory for backward compatibility
    **kwargs : dict
        Additional parameters (e.g., mV=75 for fig1)
        
    Returns:
    --------
    any
        Loaded data
    """
    # Resolve base path
    if isinstance(base_path, str) and not Path(base_path).is_absolute():
        current_file = Path(__file__)
        base_path = current_file.parent.parent / base_path
    base_path = Path(base_path)
    
    # Try new structure first
    new_path = base_path / figure_number
    if new_path.exists():
        if version == "latest":
            # Try to get latest file
            from .data_saver import get_latest_filename
            latest_filename = get_latest_filename(figure_number, base_path)
            if latest_filename:
                filepath = new_path / latest_filename
            else:
                # Find most recent file
                pkl_files = list(new_path.glob("*.pkl"))
                if pkl_files:
                    filepath = max(pkl_files, key=os.path.getmtime)
                else:
                    filepath = None
        else:
            filepath = new_path / version
        
        if filepath and filepath.exists():
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            print(f"✓ Loaded {figure_number} from {filepath}")
            return data
    
    # Fall back to legacy structure
    if legacy_path is None:
        # Try default legacy path (old processed_data location)
        current_file = Path(__file__)
        legacy_path = current_file.parent.parent.parent / 'code' / 'processed_data'
    
    legacy_path = Path(legacy_path)
    
    if legacy_path.exists() and figure_number in LEGACY_FILE_MAP:
        if field and field in LEGACY_FILE_MAP[figure_number]:
            legacy_file = LEGACY_FILE_MAP[figure_number][field]
            # Handle special cases like mV parameter
            if figure_number == 'fig1' and field == 'high' and 'mV' in kwargs:
                legacy_file = f"2point_high_{kwargs['mV']}_mV.pkl"
            
            filepath = legacy_path / legacy_file
            if filepath.exists():
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                print(f"✓ Loaded {figure_number} from legacy path {filepath}")
                return data
        else:
            # Load all fields
            data = {}
            for fld, filename in LEGACY_FILE_MAP[figure_number].items():
                if figure_number == 'fig1' and fld == 'high' and 'mV' in kwargs:
                    filename = f"2point_high_{kwargs['mV']}_mV.pkl"
                filepath = legacy_path / filename
                if filepath.exists():
                    with open(filepath, 'rb') as f:
                        data[fld] = pickle.load(f)
            if data:
                return data
    
    raise FileNotFoundError(
        f"Could not find data for {figure_number}"
        + (f" (field={field})" if field else "")
        + f"\nSearched paths:\n  - {base_path / figure_number}\n  - {legacy_path}"
    )


def load_spatial_correlations_data(
    data_path: Optional[Union[str, Path]] = None,
    mV: int = 75,
    use_legacy: bool = True
) -> Dict:
    """
    Wrapper for backward compatibility with existing loader.
    
    This function maintains the same interface as the old loader in modules/loader.py
    but uses the new loading infrastructure.
    
    Parameters:
    -----------
    data_path : str or Path, optional
        Path to processed data directory. If None, uses default.
    mV : int
        Voltage setting for high-field two-point data (default: 75)
    use_legacy : bool
        Whether to use legacy path if new structure not found
        
    Returns:
    --------
    dict
        Dictionary with structure:
        {
            'shuttling': {'high': {...}, 'low': {...}},
            'stationary': {'high': {...}, 'low': {...}},
            'two_point': {'high': {...}, 'low': {...}}
        }
    """
    if data_path is None:
        data_path = "processed_data"
    
    data = {
        'shuttling': {'high': {}, 'low': {}},
        'stationary': {'high': {}, 'low': {}},
        'two_point': {'high': {}, 'low': {}},
    }
    
    # Load using new loader
    try:
        # Shuttling data (fig4)
        try:
            shuttling_data = load_figure_data('fig4', legacy_path=data_path)
            if isinstance(shuttling_data, dict):
                if 'high' in shuttling_data:
                    data['shuttling']['high'] = shuttling_data['high']
                if 'low' in shuttling_data:
                    data['shuttling']['low'] = shuttling_data['low']
                # If data is directly the structure, use it
                if 'Ramsey' in shuttling_data:
                    data['shuttling'] = shuttling_data
        except FileNotFoundError:
            pass
        
        # Stationary data (fig2)
        try:
            stationary_high = load_figure_data('fig2', field='high', legacy_path=data_path)
            if stationary_high:
                data['stationary']['high'] = stationary_high
        except FileNotFoundError:
            pass
        
        try:
            stationary_low = load_figure_data('fig2', field='low', legacy_path=data_path)
            if stationary_low:
                data['stationary']['low'] = stationary_low
        except FileNotFoundError:
            pass
        
        # Two-point data (fig1)
        try:
            two_point_high = load_figure_data('fig1', field='high', legacy_path=data_path, mV=mV)
            if two_point_high:
                data['two_point']['high'] = two_point_high
        except FileNotFoundError:
            pass
        
        try:
            two_point_low = load_figure_data('fig1', field='low', legacy_path=data_path)
            if two_point_low:
                data['two_point']['low'] = two_point_low
        except FileNotFoundError:
            pass
        
    except Exception as e:
        print(f"Warning: Error loading with new loader: {e}")
        if use_legacy:
            # Fall back to old loader
            try:
                import sys
                current_file = Path(__file__)
                old_modules_path = current_file.parent.parent.parent / 'code' / 'modules'
                if old_modules_path.exists():
                    sys.path.insert(0, str(old_modules_path))
                    from loader import load_spatial_correlations_data as load_legacy
                    data = load_legacy(str(data_path) if data_path else None, mV)
            except ImportError:
                print("Could not fall back to legacy loader")
    
    return data



# =============================================================================
# PART 1: Data_loading
# =============================================================================

import pickle
import numpy as np
import os

def load_spatial_correlations_data(data_path=None, mV=75):
    """
    Load shuttling and stationary data from processed_data files.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the processed_data directory. If None, uses relative path from current directory.
        
    Returns:
    --------
    dict : Dictionary containing all loaded data organized by type and field
    """
  
    # Ensure path ends with /
    if not data_path.endswith('/'):
        data_path += '/'
    
    print(f"Loading data from: {data_path}")
    
    # Initialize data structure
    data = {
        'shuttling': {'high': {}, 'low': {}},
        'stationary': {'high': {}, 'low': {}},
        'two_point': {'high': {}, 'low': {}},
    }
    
    # Load shuttling data
    print("Loading shuttling data...")
    shuttling_files = {
        'high': 'shuttling_high.pkl',
        'low': 'shuttling_low.pkl'
    }
    
    for field, filename in shuttling_files.items():
        try:
            with open(f'{data_path}{filename}', 'rb') as f:
                shuttling_data = pickle.load(f)
                data['shuttling'][field] = shuttling_data
                print(f"✓ Loaded {filename}")
        except FileNotFoundError:
            print(f"✗ File not found: {data_path}{filename}")
        except Exception as e:
            print(f"✗ Error loading {filename}: {e}")
    
     # Load stationary data (high density and low density)
    print("\nLoading stationary data...")
    stationary_files = {
        'high': 'stationary_high_dense.pkl',
        'low': 'stationary_low.pkl'
    }
    
    for field, filename in stationary_files.items():
        try:
            with open(f'{data_path}{filename}', 'rb') as f:
                stationary_data = pickle.load(f)
                data['stationary'][field] = stationary_data
                print(f"✓ Loaded {filename}")
        except FileNotFoundError:
            print(f"✗ File not found: {data_path}{filename}")
        except Exception as e:
            print(f"✗ Error loading {filename}: {e}")











 
    # Load two-point correlation data
    print("\nLoading two-point correlation data...")
    two_point_files = {
        'high': f'2point_high_{mV}_mV.pkl',
        'low': '2point_low.pkl'
    }
    
    for field, filename in two_point_files.items():
        try:
            with open(f'{data_path}{filename}', 'rb') as f:
                two_point_data = pickle.load(f)
                # Initialize the field if not exists
                if field not in data['two_point']:
                    data['two_point'][field] = {}
                
                # Handle different data structures
                if isinstance(two_point_data, dict):
                    # Check for chi1, chi12 structure
                    for chi in ["chi1", "chi12"]:
                        if chi in two_point_data:
                            data['two_point'][field][chi] = two_point_data[chi]
                    
                    # Fallback: if no chi1/chi12, but there's a "chi" key
                    if "chi" in two_point_data and not any(chi in two_point_data for chi in ["chi1", "chi12"]):
                        data['two_point'][field]["chi"] = two_point_data["chi"]
                    
                    # If the whole structure is the data itself
                    if not any(key in two_point_data for key in ["chi1", "chi12", "chi"]):
                        data['two_point'][field] = two_point_data
                else:
                    # If it's not a dict, store as is
                    data['two_point'][field] = two_point_data
                    
                print(f"✓ Loaded {filename}")
     
                
        except FileNotFoundError:
            print(f"✗ File not found: {data_path}{filename}")
        except Exception as e:
            print(f"✗ Error loading {filename}: {e}")
        
    return data

def extract_shuttling_parameters(data, field='high', experiment='Ramsey'):
    """
    Extract shuttling parameters for a specific field and experiment type.
    
    Parameters:
    -----------
    data : dict
        Data loaded from load_spatial_correlations_data()
    field : str
        'high' or 'low' field
    experiment : str
        'Ramsey', 'Echo', or 'CPMG-3'
        
    Returns:
    --------
    dict : Dictionary with extracted parameters, sorted by distance
    """
    # Check if we have shuttling data in the correct format
    shuttling_key = f'shuttling_{field}' if f'shuttling_{field}' in data else 'shuttling'
    
    if shuttling_key not in data:
        print(f"No shuttling data found. Available keys: {list(data.keys())}")
        return None
    
    shuttling_data = data[shuttling_key]
    
    # Handle different data structures
    if isinstance(shuttling_data, dict):
        if experiment in shuttling_data:
            experiment_data = shuttling_data[experiment]
        elif field in shuttling_data and experiment in shuttling_data[field]:
            experiment_data = shuttling_data[field][experiment]
        else:
            print(f"Experiment {experiment} not found in {field} field data")
            print(f"Available data structure: {shuttling_data.keys() if isinstance(shuttling_data, dict) else type(shuttling_data)}")
            return None
    else:
        # If shuttling_data is a list or other structure, assume it's the experiment data directly
        experiment_data = shuttling_data
    
    # Extract parameters - handle different key names
    try:
        if isinstance(experiment_data, list) and len(experiment_data) > 0:
            # Check what keys are available in the first element
            sample_keys = list(experiment_data[0].keys()) if isinstance(experiment_data[0], dict) else []
            
            # Try different possible key names
            distance_keys = ['dx', 'distance', 'd', 'shuttling_distance']
            time_keys = ['Decay time', 'T2', 'T2_time', 'decay_time']
            time_err_keys = ['Decay time_err', 'T2_err', 'dT2', 'decay_time_err']
            exp_keys = ['Exponent', 'alpha', 'n', 'exponent']
            exp_err_keys = ['Exponent_err', 'alpha_err', 'dn', 'exponent_err']
            
            # Find the correct key names
            distance_key = next((k for k in distance_keys if k in sample_keys), 'dx')
            time_key = next((k for k in time_keys if k in sample_keys), 'Decay time')
            time_err_key = next((k for k in time_err_keys if k in sample_keys), 'Decay time_err')
            exp_key = next((k for k in exp_keys if k in sample_keys), 'Exponent') 
            exp_err_key = next((k for k in exp_err_keys if k in sample_keys), 'Exponent_err')
            
            distances = np.array([exp[distance_key] for exp in experiment_data])
            T2_times = np.array([exp[time_key] for exp in experiment_data])
            T2_errors = np.array([exp.get(time_err_key, 0) for exp in experiment_data])
            exponents = np.array([exp[exp_key] for exp in experiment_data])
            exponent_errors = np.array([exp.get(exp_err_key, 0) for exp in experiment_data])
            frequencies = np.array([exp.get('freq', 0) for exp in experiment_data])
            frequency_errors = np.array([exp.get('dfreq', 0) for exp in experiment_data])
        else:
            print(f"Unexpected experiment_data structure: {type(experiment_data)}")
            return None
    except Exception as e:
        print(f"Error extracting parameters: {e}")
        print(f"Experiment data structure: {type(experiment_data)}")
        if isinstance(experiment_data, list) and len(experiment_data) > 0:
            print(f"First element: {experiment_data[0]}")
        return None
    
    # Sort by distance
    sort_idx = np.argsort(distances)
    
    params = {
        'distances': distances[sort_idx],
        'T2_times': T2_times[sort_idx],
        'T2_errors': T2_errors[sort_idx],
        'exponents': exponents[sort_idx],
        'exponent_errors': exponent_errors[sort_idx],
        'frequencies': frequencies[sort_idx],
        'frequency_errors': frequency_errors[sort_idx]
    }
    
    return params

def extract_stationary_parameters(data, field='high'):
    """
    Extract stationary parameters for a specific field.
    
    Parameters:
    -----------
    data : dict
        Data loaded from load_spatial_correlations_data()
    field : str
        'high' or 'low' field
        
    Returns:
    --------
    dict : Dictionary with extracted parameters, sorted by distance
    """
    stationary_data = list(data['stationary'][field])

    
 
    
    # Extract parameters
    distances = np.array([stat['d'] for stat in stationary_data])
    T2_times = np.array([stat['T2'] for stat in stationary_data])
    T2_errors = np.array([stat['dT'] for stat in stationary_data])
    exponents = np.array([stat['n'] for stat in stationary_data])
    exponent_errors = np.array([stat['dn'] for stat in stationary_data])
    alphas = np.array([stat.get('alpha', 0) for stat in stationary_data])
    alpha_errors = np.array([stat.get('dalpha', 0) for stat in stationary_data])
    
    # Sort by distance
    sort_idx = np.argsort(distances)
    
    params = {
        'distances': distances[sort_idx],
        'T2_times': T2_times[sort_idx],
        'T2_errors': T2_errors[sort_idx],
        'exponents': exponents[sort_idx],
        'exponent_errors': exponent_errors[sort_idx],
        'alphas': alphas[sort_idx],
        'alpha_errors': alpha_errors[sort_idx]
    }
    
    return params



def extract_two_point_data(data, field='low', chi_type='chi1'):
    """
    Extract two-point correlation data for a specific field and chi type.
    
    Parameters:
    -----------
    data : dict
        Data loaded from load_spatial_correlations_data()
    field : str
        'high' or 'low' field
    chi_type : str
        'chi1', 'chi12', or 'chi'
        
    Returns:
    --------
    dict : Dictionary with two-point parameters, sorted by distance
    """
   
    two_point_data = data['two_point'][field]
    chi_data = two_point_data[chi_type]
    # Handle different data structures

    # Extract parameters from the chi data
    if isinstance(chi_data, list):

        distances = np.array([point['x'] for point in chi_data])
        T2_times = np.array([point['T2'] for point in chi_data])
        T2_errors = np.array([point['dT'] for point in chi_data])
        exponents = np.array([point['n'] for point in chi_data])
        exponent_errors = np.array([point['dn'] for point in chi_data])
        frequencies = np.array([point.get('freq', 0) for point in chi_data])
        T12eff = np.array([point.get('T12eff', 0) for point in chi_data])
        alpha12eff = np.array([point.get('alpha12eff', 0) for point in chi_data])
        rnm = np.array([point.get('rnm', 0) for point in chi_data])
        rnm_errors = np.array([point.get('rnm_err', 0) for point in chi_data])
        
        # Skip the first index of all matrices
        if chi_type == "chi12":
            distances = distances[1:]
            T2_times = T2_times[1:]
            T2_errors = T2_errors[1:]
            exponents = exponents[1:]
            exponent_errors = exponent_errors[1:]
            frequencies = frequencies[1:]
            T12eff = T12eff[1:]
            alpha12eff = alpha12eff[1:]
            rnm = rnm[1:]
            rnm_errors = rnm_errors[1:]

        sort_idx = np.argsort(distances)
        
        params = {
            'distances': distances[sort_idx],
            'T2_times': T2_times[sort_idx],
            'T2_errors': T2_errors[sort_idx],
            'exponents': exponents[sort_idx],
            'exponent_errors': exponent_errors[sort_idx],
            'frequencies': frequencies[sort_idx],
            'T12eff': T12eff[sort_idx],
            'alpha12eff': alpha12eff[sort_idx],
            'rnm': rnm[sort_idx],
            'rnm_errors': rnm_errors[sort_idx]
        }
        return params
    
    else:
        print(f"Unexpected data structure for {chi_type}: {type(chi_data)}")
        return None

