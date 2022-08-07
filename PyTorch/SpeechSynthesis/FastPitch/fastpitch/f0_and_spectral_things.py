import torch
import librosa
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from scipy.stats import linregress

def estimate_pitch(wav, mel_len, method='pyin', normalize_mean=None,
                   normalize_std=None, n_formants=1):

    if type(normalize_mean) is float or type(normalize_mean) is list:
        normalize_mean = torch.tensor(normalize_mean)

    if type(normalize_std) is float or type(normalize_std) is list:
        normalize_std = torch.tensor(normalize_std)

    if method == 'pyin':

        snd, sr = librosa.load(wav)
        pitch_mel, voiced_flag, voiced_probs = librosa.pyin(
	        # snd, fmin=40, fmax=600, frame_length=1024)
            snd, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'), frame_length=1024)
        assert np.abs(mel_len - pitch_mel.shape[0]) <= 1.0
        pitch_mel = np.where(np.isnan(pitch_mel), 0.0, pitch_mel)
        pitch_mel = torch.from_numpy(pitch_mel).unsqueeze(0)
        pitch_mel = F.pad(pitch_mel, (0, mel_len - pitch_mel.size(1)))

        if n_formants > 1:
            raise NotImplementedError

    else:
        raise ValueError

    pitch_mel = pitch_mel.float()

    if normalize_mean is not None:
        assert normalize_std is not None
        pitch_mel = normalize_pitch(pitch_mel, normalize_mean, normalize_std)

    return pitch_mel

def normalize_pitch(pitch, mean, std):
    zeros = (pitch == 0.0)
    pitch -= mean[:, None] # set to 0
    pitch /= std[:, None] # scale down
    pitch[zeros] = 0.0
    return pitch

def mean_delta_f0(pitch):
    # print(pitch)
    # mean = torch.mean(pitch)
    # print(mean)
    # mean = mean.numpy()
    # print(mean)
    pitch = pitch.numpy()
    # print(pitch)
    mean = np.true_divide(pitch.sum(1),(pitch!=0).sum(1))
    # print(mean)
    def func(a,b):
        return a-b
    vfunc = np.vectorize(func)
    delta = vfunc(pitch, mean)
    # print(torch.from_numpy(delta).shape)  
    return torch.from_numpy(mean), torch.from_numpy(delta)

# form merlin, original trying see desktop
def interpolate_f0(pitch_mel_array):
    frames = len(pitch_mel_array)
    last_value = 0.0
    pitch_mel = pitch_mel_array
    for i in range(frames):
        if pitch_mel_array[i] == 0.0:
            j = i+1
            for j in range(i+1, frames):
                if pitch_mel_array[j] > 0:
                    break 
            if j < frames - 1:
                if last_value > 0.0:
                    step = (pitch_mel_array[j] - pitch_mel_array[i-1]) / float(j - i)
                    for k in range(i, j):
                        pitch_mel[k] = pitch_mel[i-1] + step * (k - i + 1)
                else:
                    for k in range(i, j):
                        pitch_mel[k] = pitch_mel_array[j]
            else:
                for k in range(i, j):
                    pitch_mel[k] = last_value
        else:
            pitch_mel[i] = pitch_mel_array[i]
            last_value = pitch_mel_array[i]
    return pitch_mel


def f0_slope(pitch):
    pitch = pitch.numpy()[0]
    # print(pitch.size)
    x = range(len(pitch))
    y = pitch
    fit = np.polyfit(x, y, 1)
    # fit_fn = np.poly1d(fit)
    slope, intercept = fit
    delta= []
    # print(pitch)
    for i in range(len(pitch)):    
        y = slope * i + intercept
        delta_slope = pitch[i] - y
        delta.append(delta_slope)
    # print("slope: ",s," intercept: ",i)
    # for i in range(len(y)):
    #     plt.plot(x[i], y[i], markersize=3, marker='o')
    # # plt.xlabel('Period (T)')
    # # plt.ylabel('Semimajor Axis (a)')
    # # plt.xscale('log')
    # # plt.yscale('log')
    # # plt.title('Logarithmic scale of T vs a')
    # plt.plot(x, fit_fn(x))
    # plt.show()
    f0_slope = torch.stack((torch.from_numpy(np.array([slope])), torch.from_numpy(np.array([intercept]))), 1)[0]
    delta_slope = torch.from_numpy(np.array([delta]))
    # print(delta_slope.shape)
    return f0_slope, delta_slope

# pitch = torch.load("C:/Users/wx_Ca\OneDrive - University of Edinburgh/Desktop/Dissertation/baseline/baseline_pitch_pt/LJ016-0117.pt")
# pitch = pitch.numpy()[0]
# pitch = interpolate_f0(pitch)
# pitch = torch.from_numpy(pitch).unsqueeze(0)
# # print(mean_delta_f0(pitch))
# slope, delta = f0_slope(pitch)
# # print(slope.shape)
# print(slope.shape)


# def spectral_tilt():
#     pass

# def h_n_r():
#     pass

# pitch = torch.load("C:/Users/wx_Ca\OneDrive - University of Edinburgh/Desktop/Dissertation/baseline/baseline_pitch_pt/LJ016-0117.pt")
# print(pitch)


def f0_range(pitch):
    pitch = pitch.numpy()[0]
    # uniques = np.unique(pitch)
    # print(len(pitch))
    # print(len(uniques))
    min = np.percentile(pitch, 5)
    max = np.percentile(pitch, 95)
    range_f0 = torch.from_numpy(np.array([max]) - np.array([min]) )
    # print(pitch)
    # print(len(pitch))
    return range_f0

# pitch = torch.load("C:/Users/wx_Ca\OneDrive - University of Edinburgh/Desktop/Dissertation/baseline/baseline_pitch_pt/LJ016-0117.pt")
# pitch = pitch.numpy()[0]
# pitch = interpolate_f0(pitch)
# pitch = torch.from_numpy(pitch).unsqueeze(0)
# print(f0_range(pitch))