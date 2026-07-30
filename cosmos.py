import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from helpers import *
from digicomm import *

class PlutoTransmitter:
    def __init__(self):
        self.sample_rate = 1e6
        self.carrier_frequency = 915e6
        self.rf_bandwidth = self.sample_rate * 2

        self.tx_gain_min = -89.75
        self.tx_gain_max = 0
        self.tx_gain_resolution = 0.25

        self.num_stf_repeat = 64
        self.num_stf_symbols_per_sequence = 19
        self.stf_root = 11
        self.num_ltf_repeat = 2
        self.num_ltf_symbols_per_sequence = 937
        self.ltf_root = 11
        self.num_stf_ltf_zero_symbols = 100 # between STF and LTF

        self.pilot_modulation_order = 4
        self.num_pilot_symbols = 100
        self.pilot_rng_seed = 416
        self.num_zero_pad_symbols = 100 # between LTF and pilots

        self.sps = 10
        self.pulse_shape_beta = 1
        self.pulse_shape_span = 23
    
    def set_sdr(self,sdr):
        sdr.tx_destroy_buffer()
        sdr.rx_destroy_buffer()
        self.sdr = sdr
        self.set_carrier_frequency(int(self.carrier_frequency))
        self.set_sample_rate(int(self.sample_rate))
        self.set_rf_bandwidth(self.rf_bandwidth)
        self.set_power_level(80)
        self.set_cyclic_buffer(True)
        return
    
    def set_carrier_frequency(self,value):
        FREQ_MIN = 900e6
        FREQ_MAX = 930e6
        value = clamp(value,FREQ_MIN,FREQ_MAX)
        self.carrier_frequency = int(value)
        self.sdr.tx_lo = int(value)

    def set_channel(self, channel):
        start_freq = 900e6
        end_freq = 930e6
        num_channels = 9
        step = (end_freq - start_freq) / (num_channels - 1)
        freq = start_freq + ((channel-1) * step) # channel uses 1-indexing
        self.set_carrier_frequency(freq)
        return

    def set_power_level(self,level):
        GAIN_MIN = self.tx_gain_min
        GAIN_MAX = self.tx_gain_max
        GAIN_RES = self.tx_gain_resolution
        tx_gain_dB = map_level(level,GAIN_MIN,GAIN_MAX,GAIN_RES)
        self.set_transmit_gain(tx_gain_dB)

    def set_transmit_gain(self,value):
        self.sdr.tx_hardwaregain_chan0 = value

    def set_rf_bandwidth(self,value):
        RF_BW_MIN = 200e3
        RF_BW_MAX = 56e6
        value = clamp(value,RF_BW_MIN,RF_BW_MAX)
        self.sdr.tx_rf_bandwidth = int(value)

    def set_sample_rate(self,value):
        SAMPLE_RATE_MIN = 600e3
        SAMPLE_RATE_MAX = 61e6
        value = clamp(value,SAMPLE_RATE_MIN,SAMPLE_RATE_MAX)
        self.sample_rate = int(value)
        self.sdr.sample_rate = int(value)
    
    def set_cyclic_buffer(self,value):
        self.sdr.tx_cyclic_buffer = bool(value)

    def set_stf(self,sequence_length,repetitions,root):
        self.num_stf_symbols_per_sequence = sequence_length
        self.num_stf_repeat = repetitions
        self.stf_root = root
        return
    
    def set_ltf(self,sequence_length,repetitions,root):
        self.num_ltf_symbols_per_sequence = sequence_length
        self.num_ltf_repeat = repetitions
        self.ltf_root = root
        return

    def scale_transmit_signal(self,signal):
        scale = 1.0
        max_val = np.max(np.abs([np.real(signal), np.imag(signal)]))
        signal_scaled = signal / max_val * scale * (2**14 - 1) 
        return signal_scaled
 
    def create_pulse_train(self,symbols,sps):
        symbols = np.array(symbols)
        pulse_train = np.zeros((sps*len(symbols)),dtype=symbols.dtype)
        pulse_train[::sps] = symbols
        return pulse_train

    def get_pulse_shape(self):
        beta = self.pulse_shape_beta
        span = int(self.pulse_shape_span)
        sps = self.sps
        pulse_shape = get_rrc_pulse(beta,span,sps)
        return pulse_shape

    def pulse_shape_symbols(self,symbols):
        pulse_shape = self.get_pulse_shape()
        self.pulse_shape_length = len(pulse_shape)
        pulse_train = self.create_pulse_train(symbols,self.sps)
        signal = np.convolve(pulse_train,pulse_shape)
        return signal
    
    def generate_preamble_symbols(self):
        # Short Training Field (STF)
        length = self.num_stf_symbols_per_sequence
        root = self.stf_root
        self.stf_sequence = get_zadoff_chu_sequence(length,root)
        self.stf_symbols = np.tile(self.stf_sequence,self.num_stf_repeat)
        self.num_stf_symbols = self.num_stf_repeat * self.num_stf_symbols_per_sequence

        # Long Training Field (LTF)
        length = self.num_ltf_symbols_per_sequence
        root = self.ltf_root
        self.ltf_sequence = get_zadoff_chu_sequence(length,root)
        self.ltf_symbols = np.tile(self.ltf_sequence,self.num_ltf_repeat)
        self.num_ltf_symbols = self.num_ltf_repeat * self.num_ltf_symbols_per_sequence

        # Zero padding between STF and LTF
        stf_ltf_zero_symbols = np.zeros((self.num_stf_ltf_zero_symbols,))

        # STF + Zeros + LTF
        stf_ltf_symbols = np.concatenate((self.stf_symbols, stf_ltf_zero_symbols, self.ltf_symbols))
        self.num_stf_ltf_symbols = self.num_stf_symbols + self.num_ltf_symbols + self.num_stf_ltf_zero_symbols

        # Zero pad between STF+LTF and pilots
        num_zeros = self.num_zero_pad_symbols
        zero_pad = np.zeros((num_zeros,))
        # self.transmit_symbols_zero_pad_length = num_zeros

        # Pilots
        M = self.pilot_modulation_order
        seed = self.pilot_rng_seed
        idx = get_random_integers(self.num_pilot_symbols,M,seed)
        constellation = get_qam_constellation(M,Es=1)
        self.pilot_symbols = constellation[idx]

        # Create full preamble
        self.preamble_symbols = np.concatenate((stf_ltf_symbols, zero_pad, self.pilot_symbols))
        self.num_preamble_symbols = self.num_stf_ltf_symbols + self.num_zero_pad_symbols + self.num_pilot_symbols
        self.stf_ltf_symbols = stf_ltf_symbols
        return self.preamble_symbols

    def transmit(self,symbols):
        preamble_symbols = self.generate_preamble_symbols()
        self.desired_transmit_symbols = symbols
        self.desired_transmit_symbols_real = not np.iscomplexobj(symbols)
        self.num_transmit_symbols = len(symbols)
        if self.desired_transmit_symbols_real:
            symbols = symbols + 1j * symbols
        symbols = np.concatenate((preamble_symbols, symbols))
        signal = self.pulse_shape_symbols(symbols)
        signal_scaled = self.scale_transmit_signal(signal)
        self.sdr.tx(signal_scaled)
        print('Transmitting...')
        return
    
    def stop_transmission(self):
        self.sdr.tx_destroy_buffer()
        print('Transmission stopped.')
        return
    
class PlutoReceiver:
    def __init__(self):
        self.sample_rate = 1e6
        self.carrier_frequency = 915e6
        self.rf_bandwidth = self.sample_rate * 2
        self.gain_control_mode = 'manual'
        self.rx_buffer_size = 100e3

        self.rx_gain_min = 0
        self.rx_gain_max = 74.5
        self.rx_gain_resolution = 0.25

        self.num_stf_repeat = 64
        self.num_stf_symbols_per_sequence = 19
        self.stf_root = 11
        self.num_ltf_repeat = 2
        self.num_ltf_symbols_per_sequence = 937
        self.ltf_root = 11
        self.num_stf_ltf_zero_symbols = 100 # between STF and LTF

        self.pilot_modulation_order = 4
        self.num_pilot_symbols = 100
        self.pilot_rng_seed = 416
        self.num_zero_pad_symbols = 100 # between LTF and pilots

        self.sps = 10
        self.pulse_shape_beta = 1
        self.pulse_shape_span = 23

        self.num_transmit_symbols = [] # need to specify this
        self.desired_transmit_symbols_real = True # change this to True for COSMOS

    def set_sdr(self,sdr):
        sdr.tx_destroy_buffer()
        sdr.rx_destroy_buffer()
        self.sdr = sdr
        self.set_sample_rate(self.sample_rate)
        self.set_buffer_size(self.rx_buffer_size)
        self.set_gain_control_mode(self.gain_control_mode)
        self.set_rf_bandwidth(self.rf_bandwidth)
        self.set_carrier_frequency(int(self.carrier_frequency))
        # sdr.rx_lo = int(self.carrier_frequency)
        # sdr.sample_rate = int(self.sample_rate)
        # sdr.rx_gain = 40
        # sdr.rx_buffer_size = 100e3
        return
    
    def set_carrier_frequency(self,value):
        FREQ_MIN = 900e6
        FREQ_MAX = 930e6
        value = clamp(value,FREQ_MIN,FREQ_MAX)
        self.carrier_frequency = int(value)
        # print('Setting RX carrier frequency: ', str(int(value)))
        self.sdr.rx_lo = int(value)
        return

    def set_channel(self, channel):
        start_freq = 900e6
        end_freq = 930e6
        num_channels = 9
        step = (end_freq - start_freq) / (num_channels - 1)
        freq = start_freq + ((channel-1) * step) # channel uses 1-indexing
        self.set_carrier_frequency(freq)
        return

    def set_gain_level(self,level):
        GAIN_MIN = self.rx_gain_min
        GAIN_MAX = self.rx_gain_max
        GAIN_RES = self.rx_gain_resolution
        rx_gain_dB = map_level(level,GAIN_MIN,GAIN_MAX,GAIN_RES)
        # print('Setting RX gain: ', str(int(rx_gain_dB)))
        self.set_receive_gain(rx_gain_dB)
        return

    def set_receive_gain(self,value):
        self.set_gain_control_mode('manual')
        self.sdr.rx_hardwaregain_chan0 = value
        return

    def set_gain_control_mode(self,value):
        self.sdr.gain_control_mode_chan0 = value
        return

    def set_rf_bandwidth(self,value):
        RF_BW_MIN = 200e3
        RF_BW_MAX = 56e6
        value = clamp(value,RF_BW_MIN,RF_BW_MAX)
        self.sdr.rx_rf_bandwidth = int(value)
        return

    def set_sample_rate(self,value):
        SAMPLE_RATE_MIN = 600e3
        SAMPLE_RATE_MAX = 61e6
        value = clamp(value,SAMPLE_RATE_MIN,SAMPLE_RATE_MAX)
        self.sample_rate = int(value)
        self.sdr.sample_rate = int(value)
        self.set_rf_bandwidth(2*value)
        return
    
    def set_buffer_size(self,value):
        BUFFER_SIZE_MIN = 1e3 # not hardware constrained
        BUFFER_SIZE_MAX = 5e6 # not hardware constrained
        value = clamp(value,BUFFER_SIZE_MIN,BUFFER_SIZE_MAX)
        self.sdr.rx_buffer_size = int(value)
        return
    
    def set_stf(self,sequence_length,repetitions,root):
        self.num_stf_symbols_per_sequence = sequence_length
        self.num_stf_repeat = repetitions
        self.stf_root = root
        return
    
    def set_ltf(self,sequence_length,repetitions,root):
        self.num_ltf_symbols_per_sequence = sequence_length
        self.num_ltf_repeat = repetitions
        self.ltf_root = root
        return
     
    def get_pulse_shape(self):
        rolloff = self.pulse_shape_beta
        span = int(self.pulse_shape_span)
        sps = self.sps
        pulse_shape = get_rrc_pulse(rolloff,span,sps)
        return pulse_shape

    def get_matched_filter(self):
        pulse_shape = self.get_pulse_shape()
        return np.conj(pulse_shape[::-1])

    def matched_filter(self,signal):
        filter = self.get_matched_filter()
        signal = np.convolve(signal,filter)
        return signal
    
    def fetch_rx_buffer(self):
        rx_signal = self.sdr.rx() # capture raw samples from Pluto
        max_val = np.max(np.abs([np.real(rx_signal), np.imag(rx_signal)]))
        if max_val < (0.1 * 2048):
            print('Max value: ', str(max_val))
            print('Warning: Severe quantization noise (X). Consider increasing the receive gain.')
        return rx_signal
    
    def symbol_synchronization(self,signal,interp=8,debug=False):
        # symbols = signal[::self.sps] # no symbol synch
        symbols = symbol_synchronization_moe(signal,self.sps,interp,plot=debug)
        return symbols
    
    def frame_synchronization_schmidl_cox(self,symbols,debug=False):
        num_stf_symbols = self.num_stf_symbols_per_sequence * self.num_stf_repeat
        num_ltf_symbols = self.num_ltf_symbols_per_sequence * self.num_ltf_repeat
        buffer = num_ltf_symbols + self.num_zero_pad_symbols + self.num_pilot_symbols + self.num_transmit_symbols
        _, idx_peak = repetition_correlator(symbols,self.num_ltf_symbols_per_sequence,buffer=buffer,plot=debug)
        num_samples_pre = num_stf_symbols + self.num_stf_ltf_zero_symbols
        total_len_frame = num_stf_symbols + num_ltf_symbols + self.num_stf_ltf_zero_symbols + self.num_zero_pad_symbols + self.num_pilot_symbols + self.num_transmit_symbols
        num_samples_post = total_len_frame - num_samples_pre
        idx_start = idx_peak - num_samples_pre
        idx_stop = idx_peak + num_samples_post
        symbols = symbols[idx_start:idx_stop]
        return symbols

    def frame_synchronization(self,symbols,debug=False):
        symbols = self.frame_synchronization_schmidl_cox(symbols,debug=debug)
        return symbols
    
    def parse_frame(self,symbols):    
        # STF + LTF
        num_stf_symbols = self.num_stf_symbols_per_sequence * self.num_stf_repeat
        num_ltf_symbols = self.num_ltf_symbols_per_sequence * self.num_ltf_repeat
        num_stf_ltf_symbols = num_stf_symbols + num_ltf_symbols + self.num_stf_ltf_zero_symbols

        idx_start = 0
        idx_stop = idx_start + num_stf_ltf_symbols
        stf_ltf_symbols = symbols[idx_start:idx_stop]

        # STF
        self.rx_stf = stf_ltf_symbols[0:num_stf_symbols]

        # LTF
        idx_start = num_stf_symbols + self.num_stf_ltf_zero_symbols
        idx_stop = idx_start + num_ltf_symbols
        self.rx_ltf = stf_ltf_symbols[idx_start:idx_stop]

        # pilots
        idx_start = num_stf_ltf_symbols + self.num_zero_pad_symbols
        idx_stop = idx_start + self.num_pilot_symbols + self.num_transmit_symbols
        pilots_plus_data_symbols = symbols[idx_start:idx_stop]
        self.rx_pilots = pilots_plus_data_symbols[0:self.num_pilot_symbols]

        # data symbols
        idx_start = self.num_pilot_symbols
        idx_stop = idx_start + self.num_transmit_symbols
        self.rx_data_symbols = pilots_plus_data_symbols[idx_start:idx_stop]

        if False:
            plt.figure(figsize=(6, 6))
            plt.scatter(np.real(self.rx_pilots),np.imag(self.rx_pilots), color='red', label='Received Pilot Symbols')
            plt.scatter(np.real(self.rx_data_symbols),np.imag(self.rx_data_symbols), color='blue', label='Received Data Symbols')
            plt.title('Parse Frame')
            plt.xlabel('Real Component')
            plt.ylabel('Imaginary Component')
            plt.grid(True)
            plt.legend()
            plt.show()
        
        return
    
    def frequency_synchronization_moose(self,symbols,debug=False):
        # Extract STF and LTF
        stf = self.rx_stf
        ltf = self.rx_ltf

        # Bounds on unambiguous CFO estimation
        T = self.sps / self.sample_rate
        cfo_max_coarse = 1 / (2 * self.num_stf_symbols_per_sequence * T)
        cfo_max_fine = 1 / (2 * self.num_ltf_symbols_per_sequence * T)
        
        # Coarse CFO estimation using STF
        cfo_est_coarse = estimate_cfo(stf,self.num_stf_repeat,self.num_stf_symbols_per_sequence,T)

        # Apply CFO correction to LTF
        t = np.arange(len(ltf)) / self.sample_rate # time vector
        ltf = ltf * np.exp(-2.0j*np.pi*cfo_est_coarse*t*self.sps)

        # Fine CFO estimation using LTF
        cfo_est_fine = estimate_cfo(ltf,self.num_ltf_repeat,self.num_ltf_symbols_per_sequence,T)
        
        # Debugging
        if debug:
            print(f"Max Unambiguous CFO (coarse): {cfo_max_coarse:.2f} Hz")
            print(f"Estimated CFO (coarse): {cfo_est_coarse:.2f} Hz")
            print(f"Max Unambiguous CFO (fine): {cfo_max_fine:.2f} Hz")
            print(f"Estimated CFO (fine): {cfo_est_fine:.2f} Hz")

        # Apply total CFO correction to RX symbols
        t = np.arange(len(symbols)) / self.sample_rate # time vector
        symbols = symbols * np.exp(-2.0j*np.pi*(cfo_est_coarse+cfo_est_fine)*t*self.sps)
        return symbols

    def frequency_synchronization(self,symbols,debug=False):
        symbols = self.frequency_synchronization_moose(symbols,debug=debug)
        return symbols
    
    def channel_estimation_single_tap(self,debug=False):
        tx_pilots = self.generate_pilot_symbols()
        rx_pilots = self.rx_pilots
        h = np.mean(rx_pilots/tx_pilots)
        if debug:
            plt.figure(figsize=(6, 6))
            plt.scatter(np.real(rx_pilots),np.imag(rx_pilots), color='red', label='Received Pilot Symbols')
            plt.scatter(np.real(tx_pilots),np.imag(tx_pilots), color='blue', label='Transmitted Pilot Symbols')
            plt.title('Channel Estimate: ' + str(h))
            plt.xlabel('Real Component')
            plt.ylabel('Imaginary Component')
            plt.grid(True)
            plt.legend()
            plt.show()
        return h
    
    def channel_equalization_single_tap(self,symbols,h,debug=False):
        symbols /= h
        return symbols

    def channel_equalization(self,symbols,debug=False):
        h = self.channel_estimation_single_tap(debug)
        symbols = self.channel_equalization_single_tap(symbols,h,debug)
        return symbols

    def receive(self):
        self.sdr.rx_destroy_buffer()
        rx_signal = self.fetch_rx_buffer()
        rx_signal -= np.mean(rx_signal) # remove DC component from RX signal
        rx_signal = self.matched_filter(rx_signal)
        symbols = self.symbol_synchronization(rx_signal,interp=8,debug=False)
        symbols = self.frame_synchronization(symbols,debug=False)
        self.parse_frame(symbols) # to populate STF, LTF
        symbols = self.frequency_synchronization(symbols,debug=False)
        self.parse_frame(symbols) # to populate pilots, data
        symbols = self.channel_equalization(self.rx_data_symbols,debug=False)
        # symbols = self.unshuffle(symbols)
        if self.desired_transmit_symbols_real:
            symbols = np.real(symbols)
        # symbols = self.symbol_detection(symbols,debug=True)
        return symbols

    def generate_pilot_symbols(self):
        M = self.pilot_modulation_order
        seed = self.pilot_rng_seed
        idx = get_random_integers(self.num_pilot_symbols,M,seed)
        constellation = get_qam_constellation(M,Es=1)
        symbols = constellation[idx]
        # self.pilot_symbols = symbols
        return symbols

