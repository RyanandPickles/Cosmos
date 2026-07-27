import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def get_zadoff_chu_sequence(N, root):
    """
    Generate a length-N Zadoff-Chu sequence.
    
    Parameters:
        N (int): Length of the sequence (must be prime relative to root).
        root (int): Root index of the sequence.
    
    Returns:
        np.ndarray: Zadoff-Chu sequence.
    """
    n = np.arange(N)
    return np.exp(-1j * np.pi * root * n * (n + 1) / N)

def get_subcarrier_indices(num_subcarriers, pilot_spacing, num_nulls_dc=1, num_nulls_lower=0, num_nulls_upper=0):
    """
    Returns the indices for data, pilots, and null subcarriers.
    
    Parameters:
    num_subcarriers   : Total FFT size (N_fft)
    pilot_spacing     : Integer spacing (e.g., 2 = every other usable is a pilot)
    num_nulls_dc      : Number of nulls at the center frequency
    num_nulls_lower   : Number of guard subcarriers at the start of the vector
    num_nulls_upper   : Number of guard subcarriers at the end of the vector
    
    Returns:
    data_idx          : Array of indices for data symbols
    pilot_idx         : Array of indices for pilot symbols
    null_idx          : Array of all null indices (lower + DC + upper)
    """
    n_fft = num_subcarriers
    
    # 1. Define Null Indices
    lower_nulls = np.arange(0, num_nulls_lower)
    upper_nulls = np.arange(n_fft - num_nulls_upper, n_fft)
    
    # Center the DC nulls in the middle of the spectrum
    dc_center = n_fft // 2
    dc_start = dc_center - (num_nulls_dc // 2)
    dc_nulls = np.arange(dc_start, dc_start + num_nulls_dc)
    
    # Combine and ensure unique indices
    null_idx = np.unique(np.concatenate([lower_nulls, dc_nulls, upper_nulls]))
    
    # 2. Identify Usable Indices
    # Usable indices are any indices not occupied by nulls
    all_indices = np.arange(n_fft)
    usable_mask = ~np.isin(all_indices, null_idx)
    usable_indices = all_indices[usable_mask]
    
    # 3. Partition Usable Indices into Pilots and Data
    # Pilot spacing applies to the sequence of usable subcarriers
    pilot_idx = usable_indices[::pilot_spacing]
    
    # Data indices are usable indices that were not selected as pilots
    data_mask = ~np.isin(usable_indices, pilot_idx)
    data_idx = usable_indices[data_mask]
    
    return data_idx, pilot_idx, null_idx

def estimate_cfo(received_signal, K, N, Ts):
    """
    Estimate coarse CFO from the STF portion of a received signal.

    Parameters:
    - received_signal : np.ndarray
        The received complex baseband samples containing the STF.
    - K : int
        Total number of repeated short symbols in STF.
    - N : int
        Length of each repeated sequence in samples (period).
    - Ts : float
        Sampling period in seconds.

    Returns:
    - f_cfo : float
        Estimated coarse carrier frequency offset (Hz).
    """
    # 1. Extract the segment of the signal we will use for autocorrelation.
    # We correlate samples at index 'n' with samples at index 'n + N'.
    # Therefore, the first part goes from 0 to (K-1)*N.
    # The delayed part goes from N to K*N.
    r_first = received_signal[0 : (K - 1) * N]
    r_delayed = received_signal[N : K * N]

    # 2. Perform the delay-and-correlate summation (Autocorrelation).
    # This averages the phase rotation across all available STF repetitions.
    # P = sum( r*(n) * r(n + N) )
    # correlation_sum = np.sum(np.conj(r_first) * r_delayed)
    correlation_sum = np.vdot(r_first, r_delayed)

    # 3. Extract the phase angle from the complex correlation result.
    # np.angle returns the result in radians (-pi to pi).
    theta_hat = np.angle(correlation_sum)

    # 4. Convert the phase rotation into a frequency value in Hz.
    # Formula: Delta_f = theta / (2 * pi * N * Ts)
    f_cfo = theta_hat / (2 * np.pi * N * Ts)

    return f_cfo

def symbol_synchronization_moe(received_signal, samples_per_symbol_input, interpolation_factor, plot=False):
    """
    Performs maximum output energy symbol synchronization on a received signal.

    This method finds the optimal sampling instant within a symbol period
    by upsampling the signal and then searching for the phase that maximizes
    the energy of the sampled symbols. It assumes the input signal is
    already passed through a matched filter (or equivalent processing)
    such that maximizing output energy directly relates to optimal sampling.

    Args:
        received_signal (np.ndarray): The input received signal (1D array),
                                      typically after matched filtering. Can be complex.
        samples_per_symbol_input (int): The number of samples per symbol in the
                                        `received_signal`.
        interpolation_factor (int): The factor by which to upsample the signal
                                    to allow for finer phase resolution in the search.
                                    Must be a positive integer. If 1, no upsampling
                                    is performed, and synchronization occurs on existing samples.

    Returns:
        tuple: A tuple containing:
            - synchronized_symbols (np.ndarray): The signal sampled at the
                                                  optimal symbol synchronization phase.
            - optimal_phase_offset (int): The index of the optimal phase offset
                                           within the interpolated symbol period (0 to new_samples_per_symbol - 1).
    Raises:
        ValueError: If interpolation_factor is not a positive integer,
                    or if samples_per_symbol_input is not positive,
                    or if the signal is too short for processing.
    """
    # Input validation
    if not isinstance(samples_per_symbol_input, int) or samples_per_symbol_input <= 0:
        raise ValueError("samples_per_symbol_input must be a positive integer.")
    
    # Allow interpolation_factor to be 1
    if not isinstance(interpolation_factor, int) or interpolation_factor < 1:
        raise ValueError("interpolation_factor must be a positive integer (>= 1).")
    
    received_signal = np.array(received_signal)
    if len(received_signal) < samples_per_symbol_input:
        raise ValueError(f"Received signal length ({len(received_signal)}) is too short "
                         f"to contain even one symbol based on samples_per_symbol_input ({samples_per_symbol_input}).")

    # 1. Upsample the received signal (or use as is if interpolation_factor is 1)
    # This creates a higher resolution version of the signal, allowing for more precise
    # determination of the optimal sampling point.
    upsampled_signal = upsample_signal(received_signal, interpolation_factor)

    # Calculate the new number of samples per symbol in the upsampled signal.
    # This is the interval between consecutive symbol samples in the upsampled domain.
    new_samples_per_symbol = samples_per_symbol_input * interpolation_factor

    # Ensure the upsampled signal is long enough to find at least one full symbol
    if len(upsampled_signal) < new_samples_per_symbol:
        raise ValueError("Upsampled signal is too short to extract even one symbol after interpolation.")

    max_energy = -1.0  # Initialize with a very small energy value
    optimal_phase_offset = 0 # Initialize optimal phase
    
    # List to store energies for plotting
    energy_vs_offset = []

    # 2. Search for the optimal phase offset by maximizing output energy
    # We iterate through all possible starting phase offsets within one full upsampled symbol period.
    # This covers all possible fine timing shifts.
    for phase_offset in range(new_samples_per_symbol):
        # Extract potential symbol samples for the current phase offset.
        # We start at `phase_offset` and then take every `new_samples_per_symbol`-th sample.
        current_sampled_symbols = upsampled_signal[phase_offset::new_samples_per_symbol]

        # Calculate the energy of these sampled symbols.
        # Energy is the sum of the squared magnitudes of the complex samples.
        current_energy = np.sum(np.abs(current_sampled_symbols)**2)
        
        # Store the energy for plotting
        energy_vs_offset.append(current_energy)

        # If the current energy is greater than the maximum found so far, update the optimum.
        if current_energy > max_energy:
            max_energy = current_energy
            optimal_phase_offset = phase_offset

    # 3. Extract the synchronized symbols using the determined optimal phase offset
    # These are the actual symbol estimates that will be passed on for further decoding.
    synchronized_symbols = upsampled_signal[optimal_phase_offset::new_samples_per_symbol]

    if False:
        print(f"Symbol Synchronization Report:")
        print(f"  Input samples per symbol: {samples_per_symbol_input}")
        print(f"  Interpolation factor: {interpolation_factor}")
        print(f"  Optimal timing offset found (index in upsampled signal): {optimal_phase_offset}")
        print(f"  Number of synchronized symbols extracted: {len(synchronized_symbols)}")

    # 4. Plotting the energy vs. phase offset
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(new_samples_per_symbol), energy_vs_offset, marker='o', linestyle='-')
        plt.axvline(optimal_phase_offset, color='r', linestyle='--', label=f'Optimal Timing Offset: {optimal_phase_offset}')
        plt.title('Output Energy vs. Timing Offset')
        plt.xlabel('Timing Offset')
        plt.ylabel('Total Output Energy')
        plt.grid(True)
        plt.legend()
        plt.show()

    return synchronized_symbols # , optimal_phase_offset

def block_correlation(x, N):
    """
    Computes the correlation between N-sample segments and their immediate next
    N-sample segments in sequence x.

    Inputs:
        x - input signal (numpy array or list) of length M
        N - segment length to correlate

    Output:
        correlations - numpy array of correlation values between each N-pair segment
    """

    x = np.array(x).flatten()  # Ensure x is a flat NumPy array
    M = len(x)

    # Ensure we have enough samples to compare at least one full pair
    if 2 * N > M:
        raise ValueError('Input sequence is too short for even one N-to-N comparison.')

    # Number of full N-to-N segment comparisons possible
    num_segments = (M - N) // N

    correlations = np.zeros(num_segments)  # Preallocate

    for k in range(num_segments):
        idx1 = k * N
        idx2 = idx1 + N

        segment1 = x[idx1 : idx1 + N]
        segment2 = x[idx2 : idx2 + N]

        # Compute normalized correlation (cosine similarity)
        # Handle cases where norm might be zero to avoid division by zero
        norm_segment1 = np.linalg.norm(segment1)
        norm_segment2 = np.linalg.norm(segment2)

        if norm_segment1 == 0 or norm_segment2 == 0:
            correlations[k] = 0  # Or handle as appropriate for your application
        else:
            correlations[k] = np.dot(segment1, segment2) / (norm_segment1 * norm_segment2)

    return correlations

def gen_rand_bits(N):
    bits = np.random.randint(0, 2, size=(N, 1))
    return bits

def get_qam_constellation(M, Es=1):
    """
    Generate an M-QAM constellation with Gray coding.
    
    Parameters:
    M (int): Modulation order (must be a perfect square).
    Es (float): Symbol energy normalization factor (default is 1).
    
    Returns:
    np.ndarray: Array of M-QAM constellation points following Gray coding.
    """
    m = int(np.sqrt(M))
    if m ** 2 != M:
        raise ValueError("M must be a perfect square.")

    # Function to convert binary to Gray code
    def binary_to_gray(n):
        return n ^ (n >> 1)

    # Generate Gray-coded indices for both axes
    gray_indices = np.array([binary_to_gray(i) for i in range(m)])

    # Normalize Gray-coded indices to constellation points
    re_gray = 2 * gray_indices - (m - 1)  # Shift and scale
    im_gray = 2 * gray_indices - (m - 1)

    # Create the constellation using Gray-coded indices
    const = np.array([complex(re, -im) for im in im_gray for re in re_gray])  # Flip imag for correct quadrant

    # Normalize to unit average energy Es
    const = const / np.sqrt(np.mean(np.abs(const) ** 2)) * np.sqrt(Es)

    return const

def get_qam_constellation_old(M,Es=1):
    """
    Generate M-QAM constellation points.
    
    Parameters:
    M (int): Modulation order (must be a perfect square).
    
    Returns:
    np.ndarray: Array of M-QAM constellation points.
    """
    m = int(np.sqrt(M))
    if m ** 2 != M:
        raise ValueError("M must be a perfect square.")
    re = np.arange(-m + 1, m, 2)
    im = np.arange(-m + 1, m, 2)
    const = np.array([x + 1j * y for x in re for y in im])
    const = const / np.sqrt(np.mean(np.power(np.abs(const),2))) * np.sqrt(Es)
    return const

def gen_rand_qam_symbols(N,M=4):
    if not (M and (M & (M - 1)) == 0):
        raise ValueError("M must be a power of 2")
    
    m = int(np.sqrt(M))
    if m ** 2 != M:
        raise ValueError("M must be a square number for square QAM constellations")
    
    # random symbols
    # real_part = np.random.randint(0, m, N) * 2 - (m - 1)
    # imag_part = np.random.randint(0, m, N) * 2 - (m - 1)
    # symbols = real_part + 1j * imag_part
    # symbols /= np.sqrt((2 / 3) * (M - 1))  # Normalize average power to 1

    # get constellation
    const = get_qam_constellation(M,Es=1)

    # draw random symbols from constellation
    idx = np.random.randint(0,M,N)
    symbols = const[idx]

    return symbols, const

def get_rrc_pulse(beta, span, sps):
    """
    Generate a Root Raised Cosine (RRC) pulse shape, handling beta=0.
    """
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    
    # Handle the beta = 0 case (Pure Sinc Pulse)
    if beta == 0:
        pulse = np.sinc(t)
    else:
        pulse = np.zeros_like(t, dtype=float)
        
        # Avoid division by zero in the main formula by using a mask
        # Case 1: t = 0
        idx_zero = np.isclose(t, 0)
        pulse[idx_zero] = 1 - beta + (4 * beta / np.pi)
        
        # Case 2: t = +/- 1/(4*beta)
        idx_special = np.isclose(np.abs(t), 1 / (4 * beta))
        if np.any(idx_special):
            val = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + 
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
            pulse[idx_special] = val
            
        # Case 3: Everywhere else
        idx_rest = ~idx_zero & ~idx_special
        tr = t[idx_rest]
        num = np.sin(np.pi * tr * (1 - beta)) + 4 * beta * tr * np.cos(np.pi * tr * (1 + beta))
        den = np.pi * tr * (1 - (4 * beta * tr) ** 2)
        pulse[idx_rest] = num / den

    # Unit energy normalization
    pulse /= np.sqrt(np.sum(np.abs(pulse)**2))
    return pulse

def get_rc_pulse(beta, span, sps):
    """
    Generate a Raised Cosine (RC) pulse shape.
    
    Parameters:
        beta (float): Roll-off factor (0 to 1).
        span (int): Number of symbol durations the filter spans.
        sps (int): Samples per symbol.
    
    Returns:
        np.ndarray: RC pulse shape.
    """
    T = 1  # Symbol duration
    t = np.arange(-span * T / 2, span * T / 2 + 1 / sps, 1 / sps)
    pulse = np.zeros_like(t)

    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            pulse[i] = 1.0
        elif beta != 0 and np.isclose(abs(ti), T / (2 * beta)):
            # L'Hôpital's rule at t = ±T/(2β)
            pulse[i] = (np.pi / 4) * np.sinc(1 / (2 * beta))
        else:
            numerator = np.sin(np.pi * ti / T)
            sinc_part = numerator / (np.pi * ti / T)
            cos_part = np.cos(np.pi * beta * ti / T)
            denom = 1 - (2 * beta * ti / T) ** 2
            pulse[i] = sinc_part * cos_part / denom

    # Normalize to unit energy
    # pulse /= np.sqrt(np.sum(pulse**2))
    return pulse

def create_pulse_train(symbols,sps):
    symbols = np.array(symbols)
    pulse_train = np.zeros((sps*len(symbols)),dtype=symbols.dtype)
    pulse_train[::sps] = symbols
    return pulse_train

def frame_sync_stf_ltf(received_signal, training_seq, num_samples_post=0, num_samples_pre=0):
    """
    Perform frame synchronization via correlation.

    Parameters:
    - received_signal : np.ndarray
        Complex baseband received signal (1D array).
    - training_seq : np.ndarray
        Known complex training sequence used for correlation.
    - num_samples_post : int
        Number of samples to include after the detected sequence start.
    - num_samples_pre : int
        Number of samples to include before the detected sequence start.
    
    Returns:
    - synced_segment : np.ndarray
        Portion of the received signal from (start - K) to (start + N)
    - start_index : int
        Estimated start index of the training sequence within received_signal
    """
    if num_samples_post <= 0:
        num_samples_post = len(training_seq)

    # Cross-correlation (complex baseband)
    corr = signal.correlate(received_signal, training_seq.conj(), mode='valid')

    # Find peak of the magnitude of the correlation
    peak_index = np.argmax(np.abs(corr))

    # Determine the start of the training sequence in the received signal
    start = peak_index

    # Compute start and end indices for slicing
    start_idx = max(0, start - num_samples_pre)
    end_idx = start + num_samples_post

    # Guard against out-of-bounds access
    if end_idx > len(received_signal):
        raise ValueError("Requested slice exceeds received signal length.")

    # Extract the desired segment
    synced_segment = received_signal[start_idx:end_idx]

    # Channel estimate
    h = corr[peak_index]

    plt.figure()
    plt.plot(np.abs(corr))
    plt.show()

    return synced_segment, h

def repetition_correlator(x,N,buffer=0,plot=False):
    """
    Computes sliding-window normalized correlations between every
    N-sample segment and the next N-sample segment.

    Parameters:
    - x : np.ndarray
        Input signal (1D array, real or complex)
    - N : int
        Segment length for correlation
    -buffer : int
        Length from the end of the correlation to ignore
    - plot : bool
        Whether to plot correlation values

    Returns:
    - correlations : np.ndarray (complex)
        Array of scalar correlation values (one per sliding window)
    - peak_index : int
        Index (starting sample) where the maximum correlation magnitude occurs
    """
    x = np.asarray(x).flatten()
    M = len(x)
    num_windows = M - 2 * N + 1

    if num_windows < 1:
        raise ValueError("Signal too short for even one sliding N-to-N comparison.")

    correlations = np.zeros(num_windows, dtype=np.complex128)

    for k in range(num_windows):
        seg1 = x[k : k + N]
        seg2 = x[k + N : k + 2 * N]
        correlations[k] = np.vdot(seg1, seg2) / np.sqrt(N) 

    correlations[-buffer:] = 0
    peak_index = np.argmax(np.abs(correlations))  # sample index where max magnitude occurs

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(np.abs(correlations), label='|correlation|')
        plt.plot(peak_index, np.abs(correlations[peak_index]), 'ro', label='Peak')
        plt.title('Sliding N-to-N Correlation')
        plt.xlabel('Sliding Window Start Index')
        plt.ylabel('Correlation Magnitude')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return correlations, peak_index

def custom_corr(x, N, plot=False):
    """
    Computes sliding-window normalized correlations between every
    N-sample segment and the next N-sample segment.

    Parameters:
    - x : np.ndarray
        Input signal (1D array, real or complex)
    - N : int
        Segment length for correlation
    - plot : bool
        Whether to plot correlation values

    Returns:
    - correlations : np.ndarray (complex)
        Array of scalar correlation values (one per sliding window)
    - peak_index : int
        Index (starting sample) where the maximum correlation magnitude occurs
    """
    x = np.asarray(x).flatten()
    M = len(x)
    num_windows = M - 2 * N + 1

    if num_windows < 1:
        raise ValueError("Signal too short for even one sliding N-to-N comparison.")

    correlations = np.zeros(num_windows, dtype=np.complex128)

    for k in range(num_windows):
        seg1 = x[k : k + N]
        seg2 = x[k + N : k + 2 * N]
        correlations[k] = np.vdot(seg1, seg2) / np.sqrt(N) 

    peak_index = np.argmax(np.abs(correlations))  # sample index where max magnitude occurs

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(np.abs(correlations), label='|correlation|')
        plt.plot(peak_index, np.abs(correlations[peak_index]), 'ro', label='Peak')
        plt.title('Sliding N-to-N Correlation')
        plt.xlabel('Sliding Window Start Index')
        plt.ylabel('Correlation Magnitude')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return correlations, peak_index

def synchronize(y,x,N=0):
    if N <= 0:
        N = len(x)
    L = np.max((len(y),len(x)))
    zz = np.correlate(x,y,mode='full') / len(x)
    zz[:L-len(y)-1+N:] = 0
    z = np.abs(zz)
    idx_max = np.argmax(z)
    idx = L - idx_max - 1
    xx = y[idx:idx+N]
    h = zz[idx_max]
    if len(xx) < N:
        print('!!! len(xx) LESS THAN N')
        print('len(xx): ', len(xx))
        print('len(x): ', len(x))
        print('len(y): ', len(y))
        print('N: ', N)
        print('L: ', L)
        print('idx_max: ',idx_max)
        print('synch index: ',idx)
    # plt.figure()
    # plt.plot(z)
    # plt.show()
    return xx, h

def zadoff_chu_sequence(N, root):
    """
    Generate a length-N Zadoff-Chu sequence.
    
    Parameters:
        N (int): Length of the sequence (must be prime relative to root).
        root (int): Root index of the sequence.
    
    Returns:
        np.ndarray: Zadoff-Chu sequence.
    """
    n = np.arange(N)
    return np.exp(-1j * np.pi * root * n * (n + 1) / N)

def demod_nearest(y,S):
    """
    Snap each point in y to the nearest point in S.
    
    Parameters:
    y (np.ndarray): Array of complex numbers to be demodulated.
    S (np.ndarray): Array of possible constellation points.
    
    Returns:
    np.ndarray: Array of demodulated points.
    """
    return np.array([S[np.argmin(np.abs(s - S))] for s in y])

def calc_symbol_error_rate(s1,s2):
    return np.mean(s1 != s2)
    
def bits_to_qam_symbols(bits, M):
    """
    Convert a bit sequence to a sequence of M-QAM symbols.
    
    Parameters:
    bits (np.ndarray): Input bit sequence.
    M (int): Modulation order (must be a perfect square).
    
    Returns:
    np.ndarray: Sequence of M-QAM symbols.
    """
    k = int(np.log2(M))
    bits = np.asarray(bits).flatten()  # Ensure it's a 1D array
    remainder = len(bits) % k

    # Pad bits with zeros if necessary
    if remainder != 0:
        bits = np.pad(bits, (0, k - remainder), mode='constant')

    constellation = get_qam_constellation(M,Es=1)
    bit_groups = bits.reshape(-1, k)
    decimal_values = int(np.array([int("".join(map(str, group)), 2) for group in bit_groups]))
    return constellation[decimal_values], remainder

def downsample_signal(x, M):
    """
    Downsamples a signal by an integer factor M using an antialiasing filter
    followed by decimation.

    This function effectively reduces the sampling rate of the input signal
    by a factor of M.

    Args:
        x (np.ndarray): The input signal (1D array). Can be real or complex.
        M (int): The downsampling factor. Must be a positive integer.

    Returns:
        np.ndarray: The downsampled signal.

    Raises:
        ValueError: If M is not a positive integer.
    """

    if not isinstance(M, int) or M <= 0:
        raise ValueError("Downsampling factor M must be a positive integer.")

    if M == 1:
        # No downsampling needed
        return np.array(x)

    # 1. Antialiasing Low-Pass Filter
    # Design a low-pass FIR filter.
    # The cutoff frequency should be 1/M of the original Nyquist frequency.
    # Since firwin's cutoff is normalized to 0.5 (Nyquist), the cutoff is 0.5 / M.

    # Number of taps for the FIR filter. A common choice is 6*M for reasonable performance.
    # More taps mean a sharper cutoff but also longer delay and higher computational cost.
    num_taps = 6 * M  # Can be adjusted based on desired filter performance
    cutoff_freq = 0.5 / M  # Normalized cutoff frequency (Nyquist = 0.5 of original Fs)

    # Create the filter coefficients
    b = signal.firwin(num_taps, cutoff_freq, pass_zero=True)

    # Apply the filter to the signal
    # lfilter handles complex inputs correctly.
    filtered_x = signal.lfilter(b, 1.0, x)

    # Account for filter delay by trimming the beginning
    # This helps align the downsampled output with the original signal's features,
    # though for decimation, it's primarily about getting correct samples,
    # and the absolute timing might be adjusted externally.
    delay = (num_taps - 1) // 2
    
    # Ensure we don't try to access out of bounds if the signal is very short
    if len(filtered_x) <= delay:
        return np.array([]) # Return empty array if no valid samples after delay and decimation

    # 2. Decimation (Select every M-th sample)
    # Start decimation from 'delay' index to account for filter group delay.
    downsampled_x = filtered_x[delay::M]

    return downsampled_x

def upsample_signal(x, M):
    """
    Upsamples a signal by an integer factor M using zero-insertion followed by
    an antialiasing low-pass filter.

    This function effectively increases the sampling rate of the input signal
    by a factor of M.

    Args:
        x (np.ndarray): The input signal (1D array). Can be real or complex.
        M (int): The upsampling factor. Must be a positive integer.

    Returns:
        np.ndarray: The upsampled and filtered signal.

    Raises:
        ValueError: If M is not a positive integer.
    """

    if not isinstance(M, int) or M <= 0:
        raise ValueError("Upsampling factor M must be a positive integer.")

    if M == 1:
        # No upsampling needed
        return np.array(x)

    # 1. Zero-insertion (Upsampling by M)
    # Create an array of zeros with the new length (len(x) * M)
    # and insert the original signal samples at every M-th position.
    upsampled_zeros = np.zeros(len(x) * M, dtype=x.dtype)
    upsampled_zeros[::M] = x

    # 2. Antialiasing Low-Pass Filter
    # Design a low-pass FIR filter.
    # The cutoff frequency should be 1/M of the new Nyquist frequency
    # (which is 0.5 when normalized to the new sampling rate).
    # So, the normalized cutoff is 1/M * 0.5 = 0.5 / M.

    # Number of taps for the FIR filter. A common choice is 6*M for reasonable performance.
    # More taps mean a sharper cutoff but also longer delay and higher computational cost.
    num_taps = 10 * M  # Can be adjusted based on desired filter performance
    cutoff_freq = 0.5 / M  # Normalized cutoff frequency (Nyquist = 0.5)

    # If the signal is complex, we need to handle the filter design carefully.
    # For a real-valued FIR filter, `firwin` is suitable for both real and complex signals
    # as `lfilter` handles complex inputs correctly.
    b = signal.firwin(num_taps, cutoff_freq, pass_zero=True)

    # Apply the filter to the upsampled signal.
    # lfilter performs forward and reverse filtering for zero-phase, but for upsampling
    # a simple forward filter is usually sufficient as phase distortion might be acceptable
    # or compensated later. For strict linear phase, one might consider filtfilt, but it
    # would make the signal real.
    filtered_x = signal.lfilter(b, 1.0, upsampled_zeros)

    # Due to filter delay, the start of the signal might be distorted.
    # We can trim the initial samples corresponding to half the filter order.
    # This assumes a symmetric FIR filter and aims for roughly zero phase distortion.
    # For non-causal applications, scipy.signal.filtfilt could be used, but it requires real signals.
    delay = (num_taps - 1) // 2
    # Ensure we don't return an empty array if x is too short or M is too large
    if len(filtered_x) > delay:
        return filtered_x[delay:]
    else:
        # Fallback for very short signals or very long filters
        return filtered_x

def cgauss_rv(mu,sigmasq,N):
    """
    Generate realizations of a complex Gaussian random variable.
    
    Parameters:
    mu (complex): Mean of the complex Gaussian random variable.
    sigmasq (float): Variance of the complex Gaussian random variable.
    N (int): Number of samples to generate.
    
    Returns:
    np.ndarray: An array of realizations of the complex Gaussian random variable.
    """
    real_part = np.random.normal(mu.real, np.sqrt(sigmasq / 2), N)
    imag_part = np.random.normal(mu.imag, np.sqrt(sigmasq / 2), N)
    return real_part + 1j * imag_part

def get_token(idx):
    token = f"SuperCoolTokenForIan_{idx}_force"
    return token