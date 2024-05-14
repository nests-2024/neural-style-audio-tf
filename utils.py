import librosa
import numpy as np

N_FFT = 2048

def read_audio_spectum(filename, nfft=N_FFT):
    x, sr = librosa.load(filename)
    S = librosa.stft(x, n_fft=nfft)
    p = np.angle(S)

    S = np.log1p(np.abs(S[:, :430]))
    return S, sr


def spectrum_to_audio(spectrum, nfft=N_FFT):
    p = 2 * np.pi * np.random.random_sample(spectrum.shape) - np.pi
    for i in range(128):
        S = spectrum * np.exp(1j * p)
        wav = librosa.istft(S)
        p = np.angle(librosa.stft(wav, n_fft=N_FFT))
    return wav
