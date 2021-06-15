# Echo server program
import socket
import json
import numpy as np
import joblib
import sys

from scipy.fft import fftfreq, fft
from scipy import signal, kaiser
from bottleneck import move_mean    
from scipy.signal.windows import hann
import time

model = joblib.load('best_model_with_scaling.sav')
# scaler_svc = joblib.load('scaler_svc.sav')


def band_pass_coefs(lowcut, highcut, sf, order):
    nyq = 0.5 * sf
    low = lowcut / nyq
    high = highcut / nyq

    return signal.butter(order, [low, high], btype='band')

def multi_psd_band(fft_coefs, N, sf, band):
    # get band boundaries
    low, high = band

    # get frequency axis
    freqs = fftfreq(N, 1/sf)

    # get indexes for the given frequnecy band
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # calculate the psd
    psd = np.abs(fft_coefs[:, idx_band])**2

    # return scaled and normalized psd, since we are summing across
    # a frequency band, we need to average it by the N fft bins
    return np.sum(psd, axis=1) / (high-low)

win_size = 800
overlap_size = 400
sf = 200
bands = [
    (1, 4),  # 'delta'
    (4, 8),  # 'theta'
    (8, 12),  # 'alpha'
    (12, 30),  # 'beta'
    (30, 50),  # 'gamma'
]

b, a = band_pass_coefs(lowcut=1, highcut=50, sf=sf, order=5)
win = hann(win_size)

def calculateFeatures(signals):
    
    signals_filt = signal.filtfilt(b, a, signals)
    signals_filt_fft = fft(win*signals_filt)
    psd_band_features = np.array(
        [multi_psd_band(signals_filt_fft, N=win_size, sf=sf, band=band) for band in bands]).T.reshape(1, -1)
        
    return psd_band_features




HOST = ''                 # Symbolic name meaning all available interfaces
PORT = 5204              # Arbitrary non-privileged port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
conn, addr = s.accept()
print('Connected by', addr)
buffer = []
messageCount = 0
bufferInitialized = False
trimBuffer = False
exceptionCount = 0
predictionCount = 0
while True:
    
    
    data = conn.recv(102400)
    # print('Received data...')
    # print("----- Message start -----")
    # print(data)
    # print("----- Message end -----")
    if not data:
        break
    try:
        time_signals = json.loads(data)
    except json.decoder.JSONDecodeError:
        print('Exception while json processing:')
        # exceptionCount += 1
        print(data)
        break    

        # if (exceptionCount > 2):
        #     break
        # else:
        #     continue
    

    # exceptionCount = 0

    if not bufferInitialized:
        buffer = np.array(time_signals)
        bufferInitialized = True
    else:
        if trimBuffer:
            buffer = buffer[:, overlap_size:]
            trimBuffer = False

        buffer = np.concatenate((buffer, time_signals), axis=1)

    # print(buffer)
    
    messageCount += 1

    if messageCount >= win_size/8:
        # reset message count, this will reset the buffer
        messageCount = overlap_size/8
        trimBuffer = True

        assert buffer.shape == (8, win_size), f"Buffer has unexpected dimensions {buffer.shape}"
        
        start_time = time.perf_counter()

        # make prediction        
        features = calculateFeatures(buffer[:6])

        # features_scaled = scaler_svc.transform(features) 
        
        prediction = model.predict(features)
        
        print("--- {:.3f} ms ---".format((time.perf_counter() - start_time) * 1000.0))
        predictionCount += 1

        jsonResponse = {
            "message": int(prediction[0])
        }

        conn.send(json.dumps(jsonResponse).encode('utf-8'))
        
        print('sending data')
        
    
conn.close()
# optionally put a loop here so that you start
# listening again after the connection closes
