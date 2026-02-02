"""
Floquet system with spatio-temporal noise using FFT-based generation.

This module provides a high-performance implementation that pre-generates
a 2D spatio-temporal noise field using FFT, which is much faster than
updating noise step-by-step.
"""

import numpy as np
from scipy.special import jv
from scipy.linalg import expm
from scipy.ndimage import gaussian_filter
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator
from multiprocessing import Pool

# Pauli matrices
sz = np.array([[1, 0], [0, -1]], dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)


def simulate_correlated_field(nx, nt, xc, tc, Lx, Lt, sigma=1.0, seed=None):
    """
    Generate a 2D spatio-temporal correlated noise field using FFT.
    
    This generates a field xi(x, t) with correlation lengths xc (space) and tc (time).
    The field has exponential correlation: <xi(x,t) xi(x',t')> ~ exp(-|x-x'|/xc - |t-t'|/tc)
    
    Parameters
    ----------
    nx, nt : int
        Number of grid points in x and t
    xc, tc : float
        Correlation lengths in space and time
    Lx, Lt : float
        Physical domain size in space and time
    sigma : float, optional
        Standard deviation of the noise (default: 1.0)
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    field : array, shape (nx, nt)
        The correlated noise field
    x_grid : array, shape (nx,)
        Spatial grid points
    t_grid : array, shape (nt,)
        Time grid points
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 1. Generate White Noise
    white_noise = np.random.normal(0, 1, (nx, nt))
    
    # 2. Transform to Frequency Domain
    noise_fft = np.fft.fft2(white_noise)
    
    # 3. Construct Frequency Grid
    # Frequencies kx and kt
    kx = np.fft.fftfreq(nx, d=Lx/nx) * 2 * np.pi
    kt = np.fft.fftfreq(nt, d=Lt/nt) * 2 * np.pi
    KX, KT = np.meshgrid(kx, kt, indexing='ij')
    
    # 4. Define the Spectral Filter (Sqrt of PSD)
    # For exponential correlation exp(-|r|/L), the PSD is Lorentzian: 1 / (1 + (kL)^2)
    # The filter is the sqrt of the PSD
    # For separable 2D case with correlation lengths xc and tc:
    # PSD(kx, kt) = 1 / ((1 + (kx*xc)^2) * (1 + (kt*tc)^2))
    # Filter = sqrt(PSD) = 1 / (sqrt(1 + (kx*xc)^2) * sqrt(1 + (kt*tc)^2))
    
    # Add small epsilon to avoid division by zero at k=0
    eps = 1e-10
    filter_x = 1.0 / np.sqrt(1 + (KX * xc)**2 + eps)
    filter_t = 1.0 / np.sqrt(1 + (KT * tc)**2 + eps)
    spectral_filter = filter_x * filter_t
    
    # 5. Apply Filter
    shaped_fft = noise_fft * spectral_filter
    
    # 6. Inverse FFT to get real space field
    # Take the real part (imaginary part is numerical error)
    field = np.real(np.fft.ifft2(shaped_fft))
    
    # Normalize variance to sigma
    # The FFT approach generates a field with unit variance (approximately)
    # We normalize to get the desired standard deviation
    current_std = np.std(field)
    if current_std > 1e-10:
        field = field / current_std * sigma
    else:
        # Fallback: if field is too small, use simple Gaussian
        field = np.random.normal(0, sigma, (nx, nt))
    
    # Create grid arrays
    x_grid = np.linspace(-Lx/2, Lx/2, nx)
    t_grid = np.linspace(0, Lt, nt)
    
    return field, x_grid, t_grid


class SpatioTemporalNoise:
    """
    Wrapper for spatio-temporal noise field with interpolation.
    
    This class pre-generates a 2D noise field and provides fast interpolation
    to get noise values at any (x, t) point.
    """
    
    def __init__(self, nx, nt, xc, tc, Lx, Lt, sigma=1.0, seed=None):
        """
        Initialize spatio-temporal noise field.
        
        Parameters
        ----------
        nx, nt : int
            Number of grid points in x and t
        xc, tc : float
            Correlation lengths in space and time
        Lx, Lt : float
            Physical domain size in space and time
        sigma : float, optional
            Standard deviation of the noise (default: 1.0)
        seed : int, optional
            Random seed for reproducibility (None = random each time)
        """
        self.nx = nx
        self.nt = nt
        self.xc = xc
        self.tc = tc
        self.Lx = Lx
        self.Lt = Lt
        self.sigma = sigma
        self.seed = seed
        self._field = None
        self._x_grid = None
        self._t_grid = None
        self._interpolator = None
        self._generate_field()
    
    def _generate_field(self, seed=None):
        """Generate a new noise field (call this for each trajectory)."""
        use_seed = seed if seed is not None else self.seed
        self._field, self._x_grid, self._t_grid = simulate_correlated_field(
            self.nx, self.nt, self.xc, self.tc, 
            self.Lx, self.Lt, self.sigma, use_seed
        )
        
        # Create interpolator for fast access
        self._interpolator = RegularGridInterpolator(
            (self._x_grid, self._t_grid),
            self._field,
            method='linear',
            bounds_error=False,
            fill_value=None  # Will use nearest value instead of zero
        )
    
    def reset(self, seed=None):
        """Generate a new noise field (for new trajectory)."""
        # Always generate a new random seed if not explicitly provided
        # This ensures each trajectory gets a different noise realization
        if seed is None:
            seed = np.random.randint(0, 2**31)
        self._generate_field(seed)
    
    def get_statistics(self):
        """Get noise field statistics for diagnostics."""
        if self._field is None:
            return None
        return {
            'mean': np.mean(self._field),
            'std': np.std(self._field),
            'min': np.min(self._field),
            'max': np.max(self._field),
            'x_range': (self._x_grid[0], self._x_grid[-1]),
            't_range': (self._t_grid[0], self._t_grid[-1])
        }
    
    def get_xi(self, x, t):
        """
        Get noise value at position x and time t.
        
        Parameters
        ----------
        x : float or array
            Spatial position(s)
        t : float or array
            Time(s)
            
        Returns
        -------
        xi : float or array
            Noise value(s) at (x, t)
        """
        # Handle both scalar and array inputs
        x = np.asarray(x)
        t = np.asarray(t)
        
        if x.shape != t.shape:
            # Broadcast if needed
            x, t = np.broadcast_arrays(x, t)
        
        # Reshape for interpolator (expects (N, 2) array)
        points = np.column_stack([x.ravel(), t.ravel()])
        
        # Check bounds and use nearest value if outside
        x_min, x_max = self._x_grid[0], self._x_grid[-1]
        t_min, t_max = self._t_grid[0], self._t_grid[-1]
        
        # Clamp to bounds to avoid extrapolation issues
        x_clamped = np.clip(points[:, 0], x_min, x_max)
        t_clamped = np.clip(points[:, 1], t_min, t_max)
        points_clamped = np.column_stack([x_clamped, t_clamped])
        
        xi = self._interpolator(points_clamped)
        
        # Reshape back to original shape
        return xi.reshape(x.shape)


class floquet_system:
    """
    Floquet-driven qubit system with spatio-temporal noise.
    
    This class simulates a qubit subject to:
    - Detuning: det
    - Rabi drive: Vr * cos(wr * t + phi0)
    - Conveyor modulation: dom * cos(wc * t)
    - Spatio-temporal noise: xi(x, t) with correlation lengths xc (space) and tc (time)
    """
    
    def __init__(self, det, Vr, wc, dom, wr, Ldot, x0, phi0=0, 
                 spatio_temporal_noise=None, state=np.array([1, 0], dtype=complex)):
        """
        Initialize Floquet system.
        
        Parameters
        ----------
        det : float
            Qubit detuning
        Vr : float
            Rabi drive amplitude
        wc : float
            Conveyor frequency
        dom : float
            Modulation amplitude
        wr : float
            Rabi drive frequency
        Ldot : float
            Spatial noise correlation length (for backward compatibility)
        x0 : float
            Shuttling amplitude
        phi0 : float, optional
            Initial phase of Rabi drive (default: 0)
        spatio_temporal_noise : SpatioTemporalNoise, optional
            Pre-generated spatio-temporal noise field (default: None)
        state : array, optional
            Initial qubit state (default: [1, 0])
        """
        self.det = det
        self.Vr = Vr
        self.wc = wc
        self.dom = dom
        self.wr = wr
        self.phi0 = phi0
        self.state = state
        self.Ldot = Ldot
        self.x0 = x0
        self.spatio_temporal_noise = spatio_temporal_noise
    
    def get_noise_profile(self, times):
        """
        Get noise profile along trajectory x(t) = -x0 * cos(wc * t).
        
        Applies Gaussian smoothing with sigma=Ldot to account for finite wavefunction size.
        
        Parameters
        ----------
        times : array
            Time points to evaluate noise at
            
        Returns
        -------
        xis : array
            Noise values at the given times (Gaussian smoothed)
        """
        if self.spatio_temporal_noise is None:
            return np.zeros(len(times), dtype=float)
        
        # Calculate spatial positions from times
        xs = -self.x0 * np.cos(self.wc * times)
        
        # Get noise values from spatio-temporal field
        xis = self.spatio_temporal_noise.get_xi(xs, times)
        
        # Apply Gaussian smoothing to account for finite wavefunction size (Ldot)
        # This smooths the noise along the trajectory
        if len(xis) > 1 and self.Ldot > 0:
            # Convert Ldot to index units for gaussian_filter
            # Approximate: dx ~ |xs[1] - xs[0]| for small dt
            if len(xis) > 1:
                dx_approx = np.mean(np.abs(np.diff(xs)))
                if dx_approx > 0:
                    sigma_pixels = self.Ldot / dx_approx
                    xis = gaussian_filter(xis, sigma=sigma_pixels)
        
        return xis
    
    def get_noise_integral(self):
        """
        Compute time-averaged noise integral for effective model.
        
        This computes: (1/T) ∫₀^T ξ(-x0 cos(wc t), t) dt
        where T = 2π/wc is the period.
        
        Returns
        -------
        float
            Time-averaged noise value
        """
        if self.spatio_temporal_noise is None:
            return 0.0
        
        T = 2 * np.pi / self.wc
        
        # Sample over one period
        n_samples = 1000
        times = np.linspace(0, T, n_samples)
        xis = self.get_noise_profile(times)
        
        # Integrate using trapezoidal rule
        return np.trapz(xis, times) / T
    
    def H_RWA(self, t, xi):
        """
        RWA Hamiltonian.
        
        H = (det - wr*sin(phi0) - dom*cos(wc*t) + xi) * sz/2 + Vr/2 * sx
        
        Note: The original code used: H = (det - wr*sin(phi0) - dom*cos(wc*t) + xi) * sz/2 + Vr/2 * sx
        But check if the sign of dom term is correct.
        """
        # Original form from LZSM_oldest: H_RWA = (det - wr*sin(phi0) - dom*cos(wc*t) + xi) * sz/2 + Vr/2 * sx
        detuning = self.det - self.wr * np.sin(self.phi0) - self.dom * np.cos(self.wc * t) + xi
        return detuning * sz / 2 + self.Vr / 2 * sx
    
    def H_eff(self, xi):
        """
        Effective (time-averaged) Hamiltonian.
        
        H_eff = (det + xi_T) * sz/2 + Vr * J_k(dom/wc) / 2 * sx
        where k = det/wr and xi_T is the time-averaged noise.
        """
        assert self.det % self.wr == 0, f"det ({self.det}) must be a multiple of wr ({self.wr})"
        k = int(self.det / self.wr)
        bessel_factor = jv(k, self.dom / self.wc)
        return (self.det + xi) * sz / 2 + self.Vr * bessel_factor / 2 * sx
    
    def get_trajectory(self, t_T, dt_T, model='RWA'):
        """
        Compute single trajectory.
        
        Parameters
        ----------
        t_T : float
            Total time in units of periods T = 2π/wc
        dt_T : float
            Time step in units of periods
        model : str, optional
            Model to use: 'RWA' or 'effective' (default: 'RWA')
            
        Returns
        -------
        times : array
            Time points
        trajectory : array, shape (n, 2)
            Qubit state trajectory
        xis : array
            Noise values along trajectory
        """
        # Generate new noise field for this trajectory
        if self.spatio_temporal_noise is not None:
            self.spatio_temporal_noise.reset()
        
        T = 2 * np.pi / self.wc
        t = t_T * T
        dt = dt_T * T
        n = int(t / dt)
        nT = int(T / dt)
        
        times = np.linspace(0, t, n)
        
        # Get noise profile for RWA model
        if model == 'RWA':
            # Get noise directly along the full trajectory
            xis = self.get_noise_profile(times)
        elif model == 'effective':
            # Get time-averaged noise
            xi = self.get_noise_integral()
            xis = np.full(n, xi)  # Constant for effective model
        else:
            raise ValueError(f"Unknown model: {model}")
        
        # Initialize trajectory
        trajectory = np.zeros((2, n), dtype=complex)
        trajectory[:, 0] = self.state
        
        # Evolve trajectory
        for i in range(1, n):
            if model == 'RWA':
                H = self.H_RWA(times[i-1], xis[i-1])
            elif model == 'effective':
                H = self.H_eff(xis[i-1])
            
            trajectory[:, i] = expm(-1j * H * dt) @ trajectory[:, i-1]
        
        return times, trajectory.T, xis
    
    def _run_single_trajectory(self, args):
        """
        Helper method for parallel execution.
        Must be a method (not nested function) to be picklable.
        """
        t_T, dt_T, model = args
        return self.get_trajectory(t_T, dt_T, model)
    
    def avg_trajectories(self, t_T, dt_T, Ntraj, model='RWA'):
        """
        Average over multiple trajectories with parallelization.
        
        Parameters
        ----------
        t_T : float
            Total time in units of periods
        dt_T : float
            Time step in units of periods
        Ntraj : int
            Number of trajectories to average over
        model : str, optional
            Model to use: 'RWA' or 'effective' (default: 'RWA')
            
        Returns
        -------
        times : array
            Time points
        probs : array, shape (n, 2)
            Averaged probabilities
        xis : array
            Noise values (from first trajectory)
        """
        # Try parallel execution first
        try:
            args_list = [(t_T, dt_T, model) for _ in range(Ntraj)]
            
            with Pool() as pool:
                results = pool.map(self._run_single_trajectory, args_list)
            
            # Extract results
            times = results[0][0]  # All trajectories have same times
            trajectories = [r[1] for r in results]
            xis = results[0][2]  # Use xis from first trajectory
            
        except Exception as e:
            # Fallback to sequential execution if parallelization fails
            print(f"Parallelization failed: {e}. Falling back to sequential execution.")
            trajectories = []
            for i in range(Ntraj):
                times, traj, xis = self.get_trajectory(t_T, dt_T, model)
                trajectories.append(traj)
        
        probs = np.abs(trajectories)**2
        return times, np.average(probs, axis=0), xis

