from scipy.fft import fftfreq
import numpy as np
from constants import bands

def diferential_entropy(fft_coefs, N, sf, band): # Find closest indices of band in frequency vector
    # get band boundaries
    low, high = band

    # get frequency axis
    freqs = fftfreq(N, 1/sf)

    # get indexes for the given frequnecy band
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    
    P = np.sum(np.abs(fft_coefs[idx_band])**2) / N

    # return 0.5 * np.log2(P*2*np.pi*np.e/N)
    return 0.5 * np.log(P*2*np.pi*np.e/N)


def multi_diferential_entropy(fft_coefs, N, sf, band): # Find closest indices of band in frequency vector
    # get band boundaries
    low, high = band

    # get frequency axis
    freqs = fftfreq(N, 1/sf)

    # get indexes for the given frequnecy band
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    
    P = np.sum(np.abs((fft_coefs[:, idx_band]))**2, axis=1) / N

    return 0.5 * np.log(P*2*np.pi*np.e/N)

def psd_band(fft_coefs, N, sf, band): # Find closest indices of band in frequency vector
    # get band boundaries
    low, high = band

    # get frequency axis
    freqs = fftfreq(N, 1/sf)

    # get indexes for the given frequnecy band
    idx_band = np.logical_and(freqs >= low, freqs < high)

    
    return np.sum(10*np.log10(np.abs(fft_coefs[idx_band])**2))/ N

def multi_psd_band(fft_coefs, N, sf, band):
    # get band boundaries
    low, high = band

    # get frequency axis
    freqs = fftfreq(N, 1/sf)

    # get indexes for the given frequnecy band
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # calculate the psd
    psd = np.abs(fft_coefs[:, idx_band])**2


    return np.sum(psd, axis=1) / (high-low)    

    
