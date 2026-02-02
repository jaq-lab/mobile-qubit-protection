
## FAST PSD
import numpy as np
import matplotlib.pyplot as plt
import time
import numba # Required for acceleration

@numba.njit(fastmath=True)
def calculate_correlation_uniform_direct_sum_numba(measurements_01, max_lag):
    """
    Calculates C(k) = <I[i+k]I[i]> - <I[i]>^2 for 0/1 measurements (I)
    with uniform time steps using direct summation, accelerated by Numba.
    Assumes measurements_01 is a NumPy array. Returns only correlation values.

    Args:
        measurements_01 (np.ndarray): Sequence of 0 or 1 outcomes.
        max_lag (int): Maximum lag index (k) to compute. Must be < n.

    Returns:
        np.ndarray: Correlation values C(k) for k=0 to max_lag.
    """
    I = measurements_01
    n = len(I)

    # Calculate mean and mean squared (Numba handles this)
    mean_I = np.mean(I)
    mean_I_squared = mean_I**2
    # Initialize result array
    correlation = np.full(max_lag + 1, np.nan, dtype=np.float64)

    # Loop through each lag k from 0 to max_lag
    for k in range(max_lag + 1):
        num_pairs = n - k
        # Ensure there are pairs to sum (should always be true for k<=max_lag<n)
        if num_pairs <= 0:
            continue

        # Direct summation of products I[i] * I[i+k]
        sum_prod = 0.0
        for i in range(num_pairs): # loop i from 0 to n-k-1
            sum_prod += I[i] * I[i + k]

        # Calculate average product and correlation for lag k
        avg_prod = sum_prod / num_pairs
        correlation[k] = avg_prod - mean_I_squared

    return correlation

# --- Wrapper function to handle inputs and time lags ---
def calculate_correlation_uniform_direct(measurements_01, dt, max_lag=None):
    """
    User-facing function for direct summation correlation on uniform data.
    Handles input checks and calls the Numba-accelerated core function.

    Args:
        measurements_01 (array-like): Sequence of 0 or 1 measurement outcomes.
        dt (float): The uniform time step between measurements.
        max_lag (int, optional): Maximum lag index (k) to compute.
                                 If None, computes for all possible lags (up to N-1).

    Returns:
        tuple: (time_lags, correlation)
            - time_lags (np.ndarray): The time lags t = k*dt.
            - correlation (np.ndarray): The calculated correlation C(t=k*dt).
    """
    # Ensure input is a NumPy array (needed for Numba function)
    I = np.asarray(measurements_01)
    n = len(I)

    if n == 0:
        return np.array([]), np.array([])

    # Determine max_lag if not provided or if out of bounds
    if max_lag is None:
        max_lag = n - 1
    elif max_lag >= n:
        print(f"Warning: max_lag ({max_lag}) >= n ({n}). Reducing max_lag to {n-1}.")
        max_lag = n - 1
    if max_lag < 0:
         print(f"Warning: max_lag ({max_lag}) < 0. Calculating only lag 0.")
         max_lag = 0 # Or return error/empty

    # Call the Numba compiled function
    # Ensure I has a type Numba understands (e.g., float64 or int64)
    corr_values = calculate_correlation_uniform_direct_sum_numba(
        I.astype(np.float64), # Cast to float64 for Numba sum
        max_lag
    )

    # Calculate corresponding time lags
    lags_k = np.arange(max_lag + 1)
    time_lags = lags_k * dt

    return time_lags, corr_values


def moving_average(data, window_size):
    """
    Computes the moving average of a 1D array using a simple box filter.
    Args:
        data (np.ndarray): Input 1D array.
        window_size (int): Size of the moving average window.
    Returns:
        np.ndarray: Moving average of the input data.
    """
    if window_size <= 0:
        raise ValueError("Window size must be positive.")
    if len(data) < window_size:
        raise ValueError("Data length must be greater than window size.")

    cumsum = np.cumsum(data)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size


def calculate_chi_with_background(t, T_vals, alpha_vals, correlation_matrix, T0, alpha0):
    """
    Calculate chi(t) with background decay and spatial correlations.
    
    Parameters:
    -----------
    t : float
        Time value
    T_vals : array
        T2 decay times for each position
    alpha_vals : array
        Alpha exponents for each position
    correlation_matrix : 2D array
        Spatial correlation matrix
    T0 : float
        Background decay time
    alpha0 : float
        Background exponent
        
    Returns:
    --------
    float : chi(t) value
    """
    N = len(T_vals)
    
    # Background decay term
    chi_background = (t / T0)**alpha0
    
    # Spatially correlated term
    chi_spatial = 0
    for i in range(N):
        for j in range(N):
            term_i = (t / (T_vals[i] * (N)))**(alpha_vals[i] / 2)
            term_j = (t / (T_vals[j] * (N)))**(alpha_vals[j] / 2)
            chi_spatial += correlation_matrix[i, j] * term_i * term_j
    
    return chi_background + chi_spatial


def calculate_chi_with_background_corrected(T_vals, alpha_vals, N, correlation_matrix, T0, alpha0, dt_factor=0.1):
    """
    Calculate t_star and alpha_eff for the chi function with background and correlations.
    
    CORRECTED: t_star is when chi(t) = 1 (coherence = exp(-chi) = 1/e)
    IMPORTANT: t_star = T_eff (they are the same!)
    
    Parameters:
    -----------
    T_vals : array
        T2 decay times for each position
    alpha_vals : array
        Alpha exponents for each position  
    N : int
        Number of points to use
    correlation_matrix : 2D array
        Spatial correlation matrix
    T0 : float
        Background decay time
    alpha0 : float
        Background exponent
    dt_factor : float
        Step size for numerical differentiation
        
    Returns:
    --------
    tuple : (t_star, alpha_eff) where t_star = T_eff
    """
    from scipy.optimize import root_scalar
    
    # Define chi function for this system
    def chi_func(t):
        return calculate_chi_with_background(t, T_vals, alpha_vals, correlation_matrix, T0, alpha0)
    
    # Find t_star where chi(t) = 1 (coherence = 1/e)
    def chi_minus_one(t):
        return chi_func(t) - 1.0
    
    # Estimate reasonable bounds
    T_mean = np.mean(T_vals)
    t_low = T_mean / 100
    t_high = T_mean * 10
    
    try:
        sol = root_scalar(chi_minus_one, bracket=[t_low, t_high], method='brentq')
        t_star = sol.root
    except:
        # Fallback: use approximate estimate
        t_star = T_mean
    
    # Calculate effective alpha using numerical differentiation around t_star
    dt = dt_factor * t_star
    t1, t2 = t_star/2 - dt, t_star/2 + dt
    t1 = max(t1, dt)  # Ensure positive
    t2 = max(t2, 2*dt)
    
    chi1 = chi_func(t1)
    chi2 = chi_func(t2)
    
    if chi1 > 0 and chi2 > 0 and t1 > 0 and t2 > 0:
        # alpha_eff = d(ln(chi))/d(ln(t))
        alpha_eff = np.log(chi2/chi1) / np.log(t2/t1)
    else:
        alpha_eff = np.nan
    
    # t_star = T_eff (they are the same when chi(t_star) = 1)
    return t_star, alpha_eff

