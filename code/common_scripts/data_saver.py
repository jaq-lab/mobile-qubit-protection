"""
Data Saver Utilities

This module provides standardized functions for saving processed data
with versioning, metadata, and organization by figure number.
"""

import pickle
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


def save_figure_data(
    data: Any,
    figure_number: str,
    filename: Optional[str] = None,
    metadata: Optional[Dict] = None,
    base_path: Union[str, Path] = "processed_data"
) -> Path:
    """
    Save processed data for a specific figure with versioning.
    
    Parameters:
    -----------
    data : any
        Data to save (will be pickled)
    figure_number : str
        Figure identifier (e.g., 'fig1', 'fig2', 'fig4', 'fig5')
    filename : str, optional
        Specific filename. If None, uses timestamp.
    metadata : dict, optional
        Additional metadata to save alongside data
    base_path : str or Path
        Base path for processed data (relative to code_v2 or absolute)
        
    Returns:
    --------
    Path
        Path to saved file
    """
    # Resolve base path
    if isinstance(base_path, str) and not Path(base_path).is_absolute():
        # Relative to code_v2 directory
        current_file = Path(__file__)
        base_path = current_file.parent.parent / base_path
    base_path = Path(base_path)
    
    output_dir = base_path / figure_number
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{figure_number}_data_{timestamp}.pkl"
    elif not filename.endswith('.pkl'):
        filename = f"{filename}.pkl"
    
    filepath = output_dir / filename
    
    # Save data
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    # Save metadata if provided
    if metadata:
        metadata_file = filepath.with_suffix('.json')
        # Add timestamp to metadata
        metadata_with_time = {
            'saved_at': datetime.now().isoformat(),
            'figure_number': figure_number,
            **metadata
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata_with_time, f, indent=2)
    
    # Create/update latest symlink or record
    latest_file = output_dir / f"{figure_number}_latest.pkl"
    if latest_file.exists() and latest_file.is_symlink():
        latest_file.unlink()
    try:
        # Try to create symlink
        latest_file.symlink_to(filepath.name)
    except (OSError, NotImplementedError):
        # On Windows or if symlinks not supported, create a text file with the latest filename
        latest_file = output_dir / f"{figure_number}_latest.txt"
        with open(latest_file, 'w') as f:
            f.write(filepath.name)
    
    print(f"✓ Saved {figure_number} data to {filepath}")
    if metadata:
        print(f"  Metadata saved to {metadata_file}")
    
    return filepath


def save_figure_data_legacy(
    data: Any,
    figure_number: str,
    legacy_filename: str,
    base_path: Union[str, Path] = "processed_data"
) -> Path:
    """
    Save data with legacy filename for backward compatibility.
    
    This saves to both the old location (for compatibility) and the new structure.
    
    Parameters:
    -----------
    data : any
        Data to save
    figure_number : str
        Figure identifier
    legacy_filename : str
        Original filename (e.g., 'shuttling_high.pkl')
    base_path : str or Path
        Base path for processed data
        
    Returns:
    --------
    Path
        Path to saved file in legacy location
    """
    # Resolve base path
    if isinstance(base_path, str) and not Path(base_path).is_absolute():
        current_file = Path(__file__)
        base_path = current_file.parent.parent / base_path
    base_path = Path(base_path)
    
    # Save to legacy location (root of processed_data)
    legacy_dir = base_path
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_filepath = legacy_dir / legacy_filename
    
    with open(legacy_filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✓ Saved {legacy_filename} (legacy format)")
    
    # Also save to new structure
    save_figure_data(data, figure_number, legacy_filename.replace('.pkl', ''), 
                     base_path=base_path)
    
    return legacy_filepath


def get_latest_filename(figure_number: str, base_path: Union[str, Path] = "processed_data") -> Optional[str]:
    """
    Get the filename of the latest saved data for a figure.
    
    Parameters:
    -----------
    figure_number : str
        Figure identifier
    base_path : str or Path
        Base path for processed data
        
    Returns:
    --------
    str or None
        Filename of latest data, or None if not found
    """
    if isinstance(base_path, str) and not Path(base_path).is_absolute():
        current_file = Path(__file__)
        base_path = current_file.parent.parent / base_path
    base_path = Path(base_path)
    
    output_dir = base_path / figure_number
    
    if not output_dir.exists():
        return None
    
    # Try symlink first
    latest_file = output_dir / f"{figure_number}_latest.pkl"
    if latest_file.exists() and latest_file.is_symlink():
        return latest_file.readlink()
    
    # Try text file
    latest_file = output_dir / f"{figure_number}_latest.txt"
    if latest_file.exists():
        with open(latest_file, 'r') as f:
            return f.read().strip()
    
    # Fall back to most recent file
    pkl_files = list(output_dir.glob("*.pkl"))
    if pkl_files:
        latest = max(pkl_files, key=os.path.getmtime)
        return latest.name
    
    return None

