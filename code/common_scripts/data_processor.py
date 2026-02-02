"""
Data Processing Functions

This module provides functions to process raw experimental data:
- Extract amplitudes from oscillating signals
- Fit decay curves (stretched exponentials, etc.)
- Extract physical parameters (T2, exponents, frequencies)
- Advanced envelope extraction using Hilbert transform
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, hilbert
from scipy.ndimage import maximum_filter1d, minimum_filter1d
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings
import matplotlib.pyplot as plt


def exp_decay(x, A, T, n, B):
    """Stretched exponential with baseline: A * exp(-(x/T)^n) + B"""
    return A * np.exp(-(x / T)**n) + B


def exp_decay_nob(x, A, T, n):
    """Stretched exponential without baseline: A * exp(-(x/T)^n)"""
    return A * np.exp(-(x / T)**n)


def analyze_ramsey(
    dataset,
    n_pipulse: int,
    cv_time: float,
    cv_stop: float,
    measurement: str = 'm1_5',
    fit_b: bool = True
) -> Tuple:
    """
    Analyze a Ramsey/echo/CPMG measurement (original function signature).
    
    This is the function actually used in the shuttling_data notebook.
    
    Parameters:
    -----------
    dataset : core_tools dataset
        Raw experimental dataset
    n_pipulse : int
        Number of pi pulses (0 for Ramsey, 1 for Echo, etc.)
    cv_time : float
        Total cycle time in ns
    cv_stop : float
        Cycle stop time in ns
    measurement : str
        Measurement channel name (default: 'm1_5')
    fit_b : bool
        Whether to fit with baseline offset B
        
    Returns:
    --------
    tuple: (A, A_err, T, T_err, n, n_err, x_data, amplitudes, popt)
        Fitted parameters and data
    """
    if dataset is None:
        raise ValueError("Dataset is None")
    
    amplitudes = []
    frequencies = []
    measurement_obj = getattr(dataset, measurement)  # Get property (not called) to access .y() and .x() methods
    measurement_data = measurement_obj()  # Call to get numpy array
    
    for i in range(len(measurement_data)):
        data = measurement_data[i]
        time = measurement_obj.y() * 0.004  # Access .y() on property object, not on numpy array
        def sin_wave(x, amplitude, frequency, phase, offset):
            return amplitude * np.sin(2 * np.pi * frequency * x + phase) + offset

        amplitude_guess = min(0.5, max(0, (np.max(data) - np.min(data)) / 2))
        frequency_guess = 0.004  
        offset_guess = np.mean(data)

        try:
            popt, pcov = curve_fit(
                sin_wave, time, data, 
                p0=[amplitude_guess, frequency_guess, 0, offset_guess],
                bounds=([-0.5, 0, -np.inf, -np.inf], [0.5, np.inf, np.inf, np.inf])
            )
            amplitude = popt[0]
            frequency = popt[1]
            amplitude_err = np.sqrt(pcov[0, 0])
            frequency_err = np.sqrt(pcov[1, 1])
            frequencies.append(frequency)
            
            if amplitude_err > 0.1 * amplitude:
                amplitude = amplitude_guess
        except Exception:
            amplitude = amplitude_guess
            frequency = frequency_guess

        amplitudes.append(amplitude)

    amplitudes = abs(np.array(amplitudes))
    time_step = 2 * (1 + n_pipulse) * (cv_time - cv_stop)

    x_data = measurement_obj.x() * time_step  # Access .x() on property object, not on numpy array

    x_data2 =getattr(dataset, measurement).x() * time_step  # Access .x() on property 

    
    A_guess = max(amplitudes)
    T_guess = x_data[len(x_data)//2]
    n_guess = 1.5
    B_guess = 0.03
    
    fit_success = False
    T_guess_values = [T_guess]

    for T_guess in T_guess_values:
        if fit_b:
            try:
                popt, pcov = curve_fit(
                    exp_decay, x_data, amplitudes, 
                    p0=[A_guess, T_guess, n_guess, B_guess],
                    bounds=([0, 0, 1, 0], [0.6, np.inf, 6, 0.5])
                )
                fit_success = True
                B_fit = popt[3]
                break
            except Exception:
                continue
        else:
            A_guess = max(amplitudes)
            n_guess = 1.5

            fit_success = False
            # Try multiple T_guess values for better convergence (matching old analyze_ramsey_yuta behavior)
            T_guess_values = np.linspace(1000, 10000, 10)

            for T_guess in T_guess_values:
                try:
                    popt, pcov = curve_fit(
                        exp_decay_nob, x_data, amplitudes, 
                        p0=[A_guess, T_guess, n_guess],
                        bounds=([0, 0, 1], [np.inf, np.inf, 2])
                    )
                    fit_success = True
                    B_fit = 0
                    break
                except Exception:
                    continue

    if not fit_success:
        plt.figure(figsize=(10, 6))
        plt.scatter(x_data, amplitudes, label='Data')
        plt.xlabel('Time (ns)')
        plt.ylabel('Amplitude')
        plt.title('Amplitudes Plot (Fitting Failed)')
        plt.legend()
        plt.show()
        raise ValueError("Fitting failed for all initial decay time guesses. Amplitudes have been plotted.")

    A_fit, T_fit, n_fit = popt[:3]
    A_err, T_err, n_err = np.sqrt(np.diag(pcov))[:3]

    return A_fit, A_err, T_fit, T_err, n_fit, n_err, x_data, amplitudes, popt


def analyze_echo_lowfield(dataset, cv_stop: float, cv_time: float = 20.0) -> Optional[Tuple]:
    """
    Analyze echo measurement at low field (uses m1_3 measurement).
    
    Parameters:
    -----------
    dataset : core_tools dataset
        Raw experimental dataset
    cv_stop : float
        Cycle stop time in ns
    cv_time : float
        Cycle time in ns (default: 20.0)
        
    Returns:
    --------
    tuple: (A, A_err, T, T_err, n, n_err, time_points, amplitudes, popt)
        Or None if analysis fails
    """
    if not hasattr(dataset, 'm1_3') or not callable(dataset.m1_3):
        print(f"Error: Dataset for cv_stop={cv_stop} does not have m1_3 method.")
        return None
    if not hasattr(dataset.m1_3, 'x') or not callable(dataset.m1_3.x):
        print(f"Error: Dataset m1_3 for cv_stop={cv_stop} does not have x method.")
        return None

    try:
        m1_3_data = dataset.m1_3()
        if m1_3_data is None or m1_3_data.shape[1] < 2:
            print(f"Error: m1_3 data is invalid for cv_stop={cv_stop}.")
            return None
        amplitudes = m1_3_data[:, 1] - m1_3_data[:, 0]

        time_points_raw = dataset.m1_3.x()
        if time_points_raw is None:
            print(f"Error: m1_3.x data is invalid for cv_stop={cv_stop}.")
            return None
        time_points = time_points_raw * (4 * (cv_time - cv_stop))

        if len(time_points) != len(amplitudes):
            print(f"Error: Mismatch between time points ({len(time_points)}) and amplitudes ({len(amplitudes)}) for cv_stop={cv_stop}.")
            return None
        if len(time_points) < 3:
            print(f"Error: Not enough data points ({len(time_points)}) for fitting for cv_stop={cv_stop}.")
            return None

    except Exception as e:
        print(f"Error accessing data for cv_stop={cv_stop}: {e}")
        return None

    def stretched_exp(t, A, T, n):
        epsilon = 1e-9
        return A * np.exp(-((t + epsilon) / T)**n)

    max_amp = max(amplitudes) if len(amplitudes) > 0 else 0.5
    mean_time = np.mean(time_points) if len(time_points) > 0 else 100.0
    initial_guesses = [
        [max_amp, mean_time, 1],
        [max_amp, mean_time/2, 2],
        [max_amp, mean_time*2, 0.5],
        [np.median(amplitudes), np.median(time_points), 1.5]
    ]

    best_fit = None
    min_ssr = float('inf')

    for p0 in initial_guesses:
        try:
            p0_adjusted = [max(1e-9, p) for p in p0]
            p0_adjusted[0] = max(1e-9, p0[0])

            popt, pcov = curve_fit(
                stretched_exp, time_points, amplitudes,
                p0=p0_adjusted,
                bounds=([0, 1e-9, 1e-9], [np.inf, np.inf, 4]),
                maxfev=5000
            )

            residuals = amplitudes - stretched_exp(time_points, *popt)
            ssr = np.sum(residuals**2)

            if ssr < min_ssr:
                if np.all(np.diag(pcov) > 0):
                    perr = np.sqrt(np.diag(pcov))
                    if not np.any(np.isnan(perr)) and np.all(perr < popt * 5):
                        min_ssr = ssr
                        best_fit = (popt, perr)
        except (RuntimeError, ValueError):
            continue
            
    if best_fit is None:
        print(f"All fitting attempts failed to converge or produce valid errors for cv_stop={cv_stop}")
        return None

    popt, perr = best_fit
    return popt[0], perr[0], popt[1], perr[1], popt[2], perr[2], time_points, amplitudes, popt


def analyze_ramsey_measurement(
    dataset,
    n_pipulse: int = 0,
    cv_time: float = 20.0,
    cv_stop: float = 0.0,
    measurement: str = 'm1_5',
    plot: bool = False
) -> Dict:
    """
    Analyze a single Ramsey measurement dataset.
    
    This extracts the amplitude decay from oscillating signals and fits it 
    to a stretched exponential.
    
    Parameters:
    -----------
    dataset : core_tools dataset
        Raw experimental dataset
    n_pipulse : int
        Number of pi pulses (0 for Ramsey, 1 for Echo, etc.)
    cv_time : float
        Total cycle time in ns
    cv_stop : float
        Cycle stop time in ns (related to shuttling distance)
    measurement : str
        Measurement channel name (default: 'm1_5')
    plot : bool
        Whether to create diagnostic plots
        
    Returns:
    --------
    dict with keys:
        - 'A': fitted amplitude
        - 'A_err': amplitude error
        - 'T': T2 decay time (ns)
        - 'T_err': T2 error (ns)
        - 'n': stretched exponent
        - 'n_err': exponent error
        - 'time_points': time axis (ns)
        - 'amplitudes': extracted amplitudes
        - 'fit_params': full fit parameters array
        - 'success': whether fit was successful
    """
    if dataset is None:
        return None
    
    try:
        # Extract measurement data
        measurement_func = getattr(dataset, measurement)
        amplitudes = []
        
        for i in range(len(measurement_func())):
            # Extract phase and data
            data = measurement_func()[i]
            phase = measurement_func().y() * 0.004
            
            # Fit sinusoidal oscillation to extract amplitude
            def sin_wave(x, amplitude, frequency, phase, offset):
                return amplitude * np.sin(2 * np.pi * frequency * x + phase) + offset

            amplitude_guess = min(0.5, max(0, (np.max(data) - np.min(data)) / 2))
            frequency_guess = 0.004  
            offset_guess = np.mean(data)

            try:
                popt, pcov = curve_fit(
                    sin_wave, phase, data, 
                    p0=[amplitude_guess, frequency_guess, 0, offset_guess],
                    bounds=([-0.5, 0, -np.inf, -np.inf], 
                           [0.5, np.inf, np.inf, np.inf])
                )
                amplitude = popt[0]
                amplitude_err = np.sqrt(pcov[0, 0])
                
                # If error is too large, use guess instead
                if amplitude_err > 0.1 * abs(amplitude):
                    amplitude = amplitude_guess
            except Exception:
                amplitude = amplitude_guess

            amplitudes.append(amplitude)

        # Fit stretched exponential to amplitude decay
        def exp_decay(x, A, T, n):
            """Stretched exponential: A * exp(-(x/T)^n)"""
            return A * np.exp(-(x / T)**n) 

        amplitudes = abs(np.array(amplitudes))
        time_step = 2 * (1 + n_pipulse) * (cv_time - cv_stop)
        time_points = measurement_func().x() * time_step

        # Fit with multiple initial guesses for robustness
        A_guess = max(amplitudes) if len(amplitudes) > 0 else 0.5
        n_guess = 1.5
        
        fit_success = False
        T_guess_values = np.linspace(1000, 10000, 10)
        best_popt = None
        best_pcov = None
        
        for T_guess in T_guess_values:
            try:
                popt, pcov = curve_fit(
                    exp_decay, time_points, amplitudes, 
                    p0=[A_guess, T_guess, n_guess],
                    bounds=([0, 0, 1], [np.inf, np.inf, 2]),
                    method='lm'  # Levenberg-Marquardt
                )
                fit_success = True
                best_popt = popt
                best_pcov = pcov
                break
            except Exception:
                continue
        
        if not fit_success:
            warnings.warn("Fitting failed for all initial guesses")
            return {
                'success': False,
                'time_points': time_points,
                'amplitudes': amplitudes
            }
        
        A_fit, T_fit, n_fit = best_popt
        try:
            A_err, T_err, n_err = np.sqrt(np.diag(best_pcov))
        except Exception:
            A_err = T_err = n_err = 0.0
        
        result = {
            'A': A_fit,
            'A_err': A_err,
            'T': T_fit,  # T2 time
            'T_err': T_err,
            'n': n_fit,  # Exponent
            'n_err': n_err,
            'time_points': time_points,
            'amplitudes': amplitudes,
            'fit_params': best_popt,
            'success': True
        }
        
        if plot:
            # Create diagnostic plot
            import matplotlib.pyplot as plt
            x_fit = np.linspace(0, max(time_points), 100)
            y_fit = exp_decay(x_fit, A_fit, T_fit, n_fit)
            
            plt.figure(figsize=(10, 6))
            plt.errorbar(time_points, amplitudes, yerr=A_err, fmt='o', label='Data')
            plt.plot(x_fit, y_fit, 'r-', label=f'Fit: T2={T_fit:.1f}ns, n={n_fit:.2f}')
            plt.xlabel('Time (ns)')
            plt.ylabel('Amplitude')
            plt.title('Ramsey Decay Fit')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        
        return result
        
    except Exception as e:
        warnings.warn(f"Error analyzing Ramsey measurement: {e}")
        return None


def process_shuttling_dataset(
    uuid_list: List[Union[int, str]],
    cv_time: float = 20.0,
    cv_stop: Union[float, List[float]] = 0.0,
    experiment_type: str = 'Ramsey',
    field: str = 'high',
    data_path: Optional[str] = None,
    measurement: str = 'm1_5',
    distance_formula: Optional[callable] = None
) -> List[Dict]:
    """
    Process a complete shuttling dataset (multiple measurements at different distances).
    
    Parameters:
    -----------
    uuid_list : list of ints or strings
        UUIDs for measurements at different shuttling distances
    cv_time : float
        Cycle time in ns
    cv_stop : float or list of floats
        Cycle stop times (corresponding to shuttling distances).
        If float, same value used for all measurements.
    experiment_type : str
        'Ramsey', 'Echo', or 'CPMG-3'
    field : str
        'high' or 'low'
    data_path : str or Path, optional
        Path to raw data
    measurement : str
        Measurement channel name
    distance_formula : callable, optional
        Function to calculate distance from cv_time and cv_stop.
        If None, uses default: (cv_time - cv_stop) * 0.06 * 180
        
    Returns:
    --------
    list of dicts
        Processed results, one per distance. Each dict contains the analysis results
        plus metadata (experiment_type, field, cv_stop, distance).
    """
    from .raw_data_loader import load_experiment_group
    
    datasets = load_experiment_group(uuid_list, data_path, field, experiment_type)
    
    # Map experiment type to number of pi pulses
    n_pipulse_map = {'Ramsey': 0, 'Echo': 1, 'CPMG-3': 3}
    n_pipulse = n_pipulse_map.get(experiment_type, 0)
    
    # Handle cv_stop: single value or list
    if isinstance(cv_stop, (list, tuple, np.ndarray)):
        cv_stops = list(cv_stop)
        if len(cv_stops) != len(datasets):
            warnings.warn(
                f"cv_stop list length ({len(cv_stops)}) doesn't match "
                f"number of datasets ({len(datasets)}). Using first value for all."
            )
            cv_stops = [cv_stops[0]] * len(datasets)
    else:
        cv_stops = [float(cv_stop)] * len(datasets)
    
    # Default distance formula
    if distance_formula is None:
        def distance_formula(ct, cs):
            return (ct - cs) * 0.06 * 180  # Convert to nm
    
    results = []
    
    for ds, stop in zip(datasets, cv_stops):
        if ds is None:
            results.append(None)
            continue
        
        try:
            result = analyze_ramsey_measurement(
                ds, n_pipulse, cv_time, stop, measurement
            )
            
            if result is not None and result.get('success', False):
                # Add metadata
                result['experiment_type'] = experiment_type
                result['field'] = field
                result['cv_stop'] = stop
                result['cv_time'] = cv_time
                result['distance'] = distance_formula(cv_time, stop)
                results.append(result)
            else:
                results.append(None)
                
        except Exception as e:
            warnings.warn(f"Error processing dataset with cv_stop={stop}: {e}")
            results.append(None)
    
    return results


def format_shuttling_data_for_saving(processed_results: List[Dict]) -> Dict:
    """
    Format processed shuttling data into a structure suitable for saving.
    
    Parameters:
    -----------
    processed_results : list of dicts
        Results from process_shuttling_dataset()
        
    Returns:
    --------
    dict
        Formatted data with keys: distances, T2_times, T2_errors, exponents, exponent_errors
    """
    # Filter out None results
    valid_results = [r for r in processed_results if r is not None and r.get('success', False)]
    
    if not valid_results:
        return {
            'distances': np.array([]),
            'T2_times': np.array([]),
            'T2_errors': np.array([]),
            'exponents': np.array([]),
            'exponent_errors': np.array([]),
            'frequencies': np.array([]),
            'frequency_errors': np.array([])
        }
    
    # Extract arrays
    distances = np.array([r.get('distance', 0) for r in valid_results])
    T2_times = np.array([r.get('T', 0) for r in valid_results])
    T2_errors = np.array([r.get('T_err', 0) for r in valid_results])
    exponents = np.array([r.get('n', 1) for r in valid_results])
    exponent_errors = np.array([r.get('n_err', 0) for r in valid_results])
    
    # Sort by distance
    sort_idx = np.argsort(distances)
    
    return {
        'distances': distances[sort_idx],
        'T2_times': T2_times[sort_idx],
        'T2_errors': T2_errors[sort_idx],
        'exponents': exponents[sort_idx],
        'exponent_errors': exponent_errors[sort_idx],
        'frequencies': np.zeros_like(distances[sort_idx]),  # Placeholder
        'frequency_errors': np.zeros_like(distances[sort_idx])  # Placeholder
    }


# ============================================================================
# Advanced Envelope Extraction Functions (from fitting.py)
# ============================================================================

def find_frequencies(data, wait_times):
    """
    Extract the dominant frequency from oscillating data using FFT.
    
    Parameters:
    -----------
    data : np.ndarray
        Signal data (oscillating measurements)
    wait_times : np.ndarray
        Time points corresponding to the data
        
    Returns:
    --------
    freq : float
        Dominant frequency in the data (in units of 1/wait_times)
    """
    peaks_max, _ = find_peaks(data, distance=10)
    peaks_min, _ = find_peaks(-data, distance=10)
    
    # Combine maxima and minima points for envelope
    envelope_times_max = wait_times[peaks_max]
    envelope_data_max = data[peaks_max]
    envelope_times_min = wait_times[peaks_min]
    envelope_data_min = data[peaks_min]
    
    # Calculate amplitudes from mean
    envelope_data_max_amp = np.abs(envelope_data_max - np.mean(data))
    envelope_data_min_amp = np.abs(envelope_data_min - np.mean(data))
    
    # Combine all points for fitting
    envelope_times = np.concatenate([envelope_times_max, envelope_times_min])
    envelope_amps = np.concatenate([envelope_data_max_amp, envelope_data_min_amp])
    
    # Sort by time
    sort_idx = np.argsort(envelope_times)
    envelope_times = envelope_times[sort_idx]
    envelope_amps = envelope_amps[sort_idx]
    
    # Calculate frequency using FFT
    data_fft = np.fft.fft(data - np.mean(data))
    fft_freqs = np.fft.fftfreq(len(wait_times), wait_times[1] - wait_times[0])
    main_freq_idx = np.argmax(np.abs(data_fft[1:len(data_fft)//2])) + 1
    freq = np.abs(fft_freqs[main_freq_idx])
    return freq


def envelope_func(t, A,tau, n):
        """Target envelope function: A * exp(-(t/tau)**n)"""
        # Basic implementation without extensive clipping/epsilon
        with np.errstate(divide='ignore', invalid='ignore'): # Suppress warnings
            base = t / tau
            base = np.maximum(base, 0) # Avoid issues with negative base if t or tau were negative
            exponent_arg = -(base**n)
        # Handle potential NaNs resulting from 0**negative_n if t=0
        exponent_arg = np.nan_to_num(exponent_arg, nan=-np.inf)
        return 1/2*A * np.exp(exponent_arg) # Clip to avoid exp 


def full_envelope_func(t, A, tau, n, phi0, f):



    with np.errstate(divide='ignore', invalid='ignore'): # Suppress warnings
        base = t / tau
        base = np.maximum(base, 0) # Avoid issues with negative base if t or tau were negative
        exponent_arg = -(base**n)
        # Handle potential NaNs resulting from 0**negative_n if t=0
    exponent_arg = np.nan_to_num(exponent_arg, nan=-np.inf)
    return A/2* np.cos(2*np.pi*f*t + phi0)*np.exp(exponent_arg)
    

def get_full_fit(data, wait_times, guess, window_size = 100):
    window_size = 100 # Adjust this based on visual inspection of drift vs oscillation
    drift = pd.Series(data).rolling(window=window_size, center=True, min_periods=5).mean().to_numpy()
    y_detrended = data - drift   
    p0 = [guess["A"],guess["T2"], guess["n"], guess["phi0"], guess["f"]] # Use detrended envelope for A guess
    bounds = ([0, 1000, 1, -np.pi, 0], [1,15000, 2, np.pi, 1]) # Parameters should be positive
    popt, pcov = curve_fit(full_envelope_func, wait_times, y_detrended, p0=p0, bounds=bounds, maxfev=100000)
    A, tau, n, phi0, f = popt
    A_err, tau_err, n_err, phi0_err, f_err = np.sqrt(np.diag(pcov))
    return A, A_err, tau, tau_err, n, n_err, wait_times, y_detrended, None, phi0, f, f_err, None



def get_fit_hilbert_drift(data, wait_times, guess, window_size = 100, use_hilbert=True):


    # Define the model function
    window_size = window_size # Adjust this based on visual inspection of drift vs oscillation
    drift = pd.Series(data).rolling(window=window_size, center=True, min_periods=5).mean().to_numpy()
    y_detrended = data - drift
    if use_hilbert:
        envelope = np.abs(hilbert(y_detrended))
    else:
        # Alternative envelope extraction: use moving maximum filter
        envelope = maximum_filter1d(y_detrended, size=40, mode='reflect')
        envelope2 = minimum_filter1d(y_detrended, size=40, mode='reflect')
        envelope = (envelope - envelope2)/2.


    p0 = [1,guess["T2"], guess["n"]] # Use detrended envelope for A guess
    bounds = ([0, 1000, 1], [1,15000, 2]) # Parameters should be positive
    popt, pcov = curve_fit(envelope_func, wait_times, envelope, p0=p0, bounds=bounds, maxfev=100000)
    A, tau, n = popt
    A_err, tau_err, n_err = np.sqrt(np.diag(pcov))

    def phase_fit(t, phi, f):
        return A/2* np.cos(2*np.pi*f*t + phi)

    # Fit the phase freq
    guess_phase = np.arccos(y_detrended[0]) # Initial guess for phase
    if y_detrended[1] - y_detrended[0] > 0:
        guess_phase = -guess_phase
    p0_phase = [guess_phase, guess["freq"]] # Initial guess for phase
    bounds_phase = [(-np.pi, 0), (np.pi,1)]
    popt_phase, pcov_phase = curve_fit(phase_fit, wait_times, y_detrended, p0=p0_phase, bounds=bounds_phase, maxfev=100000)
    phase = popt_phase[0]
    freq = popt_phase[1]    
    freq_err = np.sqrt(np.diag(pcov_phase))[1]

    B = np.mean(drift)



    return A, A_err, tau, tau_err, n, n_err, wait_times, envelope,y_detrended,  phase, freq, freq_err, B


def fit_rnm(tau, chi_dc_data, T2_xn, T2_xm, alpha_xn, alpha_xm, 
            initial_guess=None, bounds=None, S0_guess=0.0, return_S0=True):
    """
    Fit the DC correlation formula to experimental data.
    
    Formula:
    χ_DC(x_n,x_m,τ) ≈ (τ/2/T_{2,x_n})^α_{x_n} + (τ/2/T_{2,x_m})^α_{x_m} 
                      + 2r_{n,m}(τ/2/√(T_{2,x_n}T_{2,x_m}))^((α_{x_n}+α_{x_m})/2)
    
    Parameters:
    -----------
    tau : array-like
        Evolution times (τ values)
    chi_dc_data : array-like
        Experimental χ_DC data to fit
    T2_xn : float
        T2 time for position x_n
    T2_xm : float
        T2 time for position x_m  
    alpha_xn : float
        Exponent α for position x_n
    alpha_xm : float
        Exponent α for position x_m
    initial_guess : float, optional
        Initial guess for r_{n,m}. Default is 0.5
    bounds : tuple, optional
        Bounds for r_{n,m} and S0 as ((r_lower, r_upper), (S0_lower, S0_upper)). Default is ((-1, 1), (0, 1))
    S0_guess : float, optional
        Initial guess for S0 decay parameter. Default is 0.0
    return_S0 : bool, optional
        If True, return S0 and S0_err. If False, return only r_nm, r_nm_err, chi_dc_fit. Default is True
        
    Returns:
    --------
    If return_S0=True:
        r_nm : float
            Fitted correlation coefficient r_{n,m}
        r_nm_err : float
            Standard error of r_{n,m}
        S0 : float
            Fitted additional decay parameter
        S0_err : float
            Standard error of S0
        chi_dc_fit : array-like
            Fitted χ_DC values
    
    If return_S0=False:
        r_nm : float
            Fitted correlation coefficient r_{n,m}
        r_nm_err : float
            Standard error of r_{n,m}
        chi_dc_fit : array-like
            Fitted χ_DC values
    """
    # Convert inputs to numpy arrays
    tau = np.asarray(tau)
    chi_dc_data = np.asarray(chi_dc_data)
    
    # Set default initial guess and bounds
    if initial_guess is None:
        initial_guess = 0.5
    if bounds is None:
        # curve_fit expects bounds as (lower_bounds, upper_bounds)
        bounds = ([-1, 1])  # r_nm: [-1, 1], S0: [0, 1]
    elif len(bounds) == 2 and not isinstance(bounds[0], tuple):
        # Handle old format: bounds = [lower, upper] for r_nm only
        bounds = ([bounds[0]], [bounds[1]])  # Add default bounds for S0
    
    def chi_dc_model(tau, r_nm):
        """
        DC correlation model function
        
        χ_DC(τ) = (τ/2/T2_xn)^α_xn + (τ/2/T2_xm)^α_xm 
                  + 2*r_nm*(τ/2/√(T2_xn*T2_xm))^((α_xn+α_xm)/2)
        """
        tau_half = tau / 2.0
        
        # First term: (τ/2/T_{2,x_n})^α_{x_n}
        term1 = (tau_half / T2_xn) ** alpha_xn
        
        # Second term: (τ/2/T_{2,x_m})^α_{x_m}
        term2 = (tau_half / T2_xm) ** alpha_xm
        
        # Third term: 2*r_{n,m}*(τ/2/√(T_{2,x_n}*T_{2,x_m}))^((α_{x_n}+α_{x_m})/2)
        sqrt_T2_product = np.sqrt(T2_xn * T2_xm)
        avg_alpha = (alpha_xn + alpha_xm) / 2.0
        term3 = 2.0 * r_nm * (tau_half / sqrt_T2_product) ** avg_alpha
        return np.exp(-term1 - term2 - term3)
    
    try:
        # Perform the fit
        popt, pcov = curve_fit(chi_dc_model, tau, chi_dc_data, 
                              p0=[initial_guess], 
                              bounds=bounds,
                              maxfev=10000)
        
        r_nm = popt[0]

        r_nm_err = np.sqrt(np.diag(pcov))[0]

        # Calculate fitted values
        chi_dc_fit = chi_dc_model(tau, r_nm)
        
     
        return r_nm, r_nm_err, chi_dc_fit
        
    except Exception as e:
        print(f"Fitting failed: {e}")
        if return_S0:
            return np.nan, np.nan, np.full_like(tau, np.nan)
        else:
            return np.nan, np.nan, np.full_like(tau, np.nan)
