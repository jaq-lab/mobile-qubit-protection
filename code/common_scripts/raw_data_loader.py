"""
Raw Experimental Data Loader

This module provides functions to load raw experimental data from UUIDs.
Data is typically stored in HDF5 format and accessed through core_tools.
"""

import sys
from pathlib import Path
from typing import Union, List, Optional

def _import_core_tools():
    """Try to import load_hdf5_uuid from core_tools, with fallback to experiments path."""
    current_file = Path(__file__)
    
    # Try direct import first
    try:
        from core_tools.data.ds.ds_hdf5 import load_hdf5_uuid
        return load_hdf5_uuid
    except ImportError:
        pass
    
    # Fallback: add experiments path and try again
    # Navigate: code_v2/common/ -> code_v2 -> data_analysis -> code -> experiments
    experiments_path = current_file.parent.parent.parent / 'code' / 'experiments'
    if experiments_path.exists():
        if str(experiments_path) not in sys.path:
            sys.path.insert(0, str(experiments_path))
        try:
            # Try importing the module directly by importing the parent package first
            # This ensures the package structure is recognized
            import core_tools.data.ds.ds_hdf5 as ds_hdf5_module
            return ds_hdf5_module.load_hdf5_uuid
        except (ImportError, AttributeError):
            try:
                # Fallback: try standard import
                from core_tools.data.ds.ds_hdf5 import load_hdf5_uuid
                return load_hdf5_uuid
            except ImportError:
                pass
    
    # Try alternative path (workspace root)
    workspace_root = current_file.parent.parent.parent.parent
    experiments_path_alt = workspace_root / 'data_analysis' / 'code' / 'experiments'
    if experiments_path_alt.exists():
        if str(experiments_path_alt) not in sys.path:
            sys.path.insert(0, str(experiments_path_alt))
        try:
            import core_tools.data.ds.ds_hdf5 as ds_hdf5_module
            return ds_hdf5_module.load_hdf5_uuid
        except (ImportError, AttributeError):
            try:
                from core_tools.data.ds.ds_hdf5 import load_hdf5_uuid
                return load_hdf5_uuid
            except ImportError:
                pass
    
    raise ImportError(
        "core_tools not found. Please install core_tools or ensure it's in your Python path.\n"
        "Tried paths:\n"
        f"  - Direct import from core_tools.data.ds.ds_hdf5\n"
        f"  - {experiments_path}\n"
        f"  - {experiments_path_alt}"
    )


def load_raw_data_by_uuid(
    uuid: Union[int, List[int], str],
    data_path: Optional[Union[str, Path]] = None,
    field: Optional[str] = None,
    experiment_type: Optional[str] = None
):
    """
    Load raw experimental data by UUID.
    
    Parameters:
    -----------
    uuid : int, str, or list
        UUID(s) of the experiment(s). Can be string with underscores (will be cleaned)
    data_path : str or Path, optional
        Path where HDF5 files are stored/cached. If None, uses default data/ directory
    field : str, optional
        'high' or 'low' field (for metadata/annotation purposes)
    experiment_type : str, optional
        'Ramsey', 'Echo', 'CPMG-3', etc. (for metadata/annotation purposes)
    
    Returns:
    --------
    dataset or list of datasets
        Single dataset if uuid is scalar, list of datasets if uuid is a list.
        Failed loads return None in the list.
    """
    load_hdf5_uuid = _import_core_tools()
    
    if data_path is None:
        # Default to data/ directory relative to workspace
        current_file = Path(__file__)
        workspace_root = current_file.parent.parent.parent.parent
        data_path = workspace_root / 'data_analysis' / 'data'
    else:
        data_path = Path(data_path)
    
    data_path.mkdir(parents=True, exist_ok=True)
    data_path_str = str(data_path)
    
    def _clean_uuid(u):
        """Clean UUID by removing underscores and converting to int."""
        if isinstance(u, str):
            u = int(u.replace('_', ''))
        return int(u)
    
    def _load_single(u):
        """Load a single UUID."""
        try:
            clean_uuid = _clean_uuid(u)
            expected_file = Path(data_path_str) / f"ds_{clean_uuid}.hdf5"
            
            # Check if file exists before trying to load
            if not expected_file.exists():
                print(f"Warning: HDF5 file not found for UUID {u}: {expected_file}")
                return None
            
            ds = load_hdf5_uuid(clean_uuid, data_path_str)
            return ds
        except FileNotFoundError as e:
            print(f"Warning: File not found for UUID {u}: {e}")
            return None
        except Exception as e:
            print(f"Warning: Failed to load UUID {u}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    if isinstance(uuid, (list, tuple)):
        datasets = []
        for u in uuid:
            datasets.append(_load_single(u))
        return datasets
    else:
        return _load_single(uuid)


def load_experiment_group(
    uuid_list: List[Union[int, str]],
    data_path: Optional[Union[str, Path]] = None,
    field: Optional[str] = None,
    experiment_type: Optional[str] = None
):
    """
    Load a group of related experiments (e.g., all Ramsey measurements at different distances).
    
    Parameters:
    -----------
    uuid_list : list of ints or strings
        List of UUIDs for related experiments
    data_path : str or Path, optional
        Path where HDF5 files are stored/cached
    field : str, optional
        'high' or 'low' field (for metadata/annotation)
    experiment_type : str, optional
        'Ramsey', 'Echo', 'CPMG-3', etc. (for metadata/annotation)
    
    Returns:
    --------
    list of datasets
        List of loaded datasets. Failed loads are None.
    """
    return load_raw_data_by_uuid(uuid_list, data_path, field, experiment_type)


def get_default_data_path() -> Path:
    """
    Get the default data path for raw experimental data.
    
    Returns:
    --------
    Path
        Default path to data directory
    """
    current_file = Path(__file__)
    workspace_root = current_file.parent.parent.parent.parent
    return workspace_root / 'data_analysis' / 'data'

