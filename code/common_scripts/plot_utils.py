"""
Plotting Utilities

This module provides standardized plotting functions and styles for figure generation.
"""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union, Tuple, Dict


# Common color schemes
COLORS = {
    'high_field': 'tab:red',
    'low_field': 'tab:blue',
    'default': 'tab:blue',
    'ramsey': 'tab:orange',
    'echo': 'tab:green',
    'cpmg3': 'tab:purple'
}

# Common figure sizes (inches)
FIG_SIZES = {
    'single_column': (4.5, 4.5),
    'double_column': (9, 4.5),
    'full_page': (8.5, 11),
    'wide': (12, 6),
    'tall': (6, 8)
}


def save_figure(
    fig: plt.Figure,
    filename: str,
    figure_number: str,
    base_path: Union[str, Path] = "figures",
    **kwargs
) -> Path:
    """
    Save figure with consistent settings.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Output filename
    figure_number : str
        Figure identifier (e.g., 'fig1', 'fig2')
    base_path : str or Path
        Base path for figures (relative to code_v2 or absolute)
    **kwargs : dict
        Additional arguments to pass to fig.savefig()
        
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
    
    filepath = output_dir / filename
    
    # Default save settings
    save_kwargs = {
        'bbox_inches': 'tight',
        'dpi': 300,
        'format': 'pdf' if filename.endswith('.pdf') else None,
        **kwargs
    }
    
    fig.savefig(filepath, **save_kwargs)
    print(f"✓ Saved figure to {filepath}")
    return filepath


def setup_plot_style(style: str = 'seaborn-v0_8-paper'):
    """
    Set up consistent plotting style.
    
    Parameters:
    -----------
    style : str
        Matplotlib style name
    """
    try:
        plt.style.use(style)
    except OSError:
        # Fallback to default style if specified style not available
        plt.style.use('default')
        print(f"Warning: Style '{style}' not available, using default style")


def get_figure_size(size_name: str = 'single_column') -> Tuple[float, float]:
    """
    Get standard figure size.
    
    Parameters:
    -----------
    size_name : str
        Name of size preset
        
    Returns:
    --------
    tuple
        (width, height) in inches
    """
    return FIG_SIZES.get(size_name, FIG_SIZES['single_column'])


def apply_publication_style():
    """
    Apply publication-quality plotting style settings.
    """
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'lines.linewidth': 1.5,
        'axes.linewidth': 1,
        'grid.linewidth': 0.5,
        'axes.grid': True,
        'grid.alpha': 0.3
    })

