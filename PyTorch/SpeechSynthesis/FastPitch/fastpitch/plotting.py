import librosa
import numpy as np
import os
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# criterion = nn.MSELoss()
# loss_base_gt = torch.sqrt(criterion(f0s[0], f0s[1]))
# loss_base_add_O = torch.sqrt(criterion(f0s[1], f0s[2]))
# loss_base_emb_O = torch.sqrt(criterion(f0s[1], f0s[3]))
# print(loss_base_gt, loss_base_add_O, loss_base_emb_O)

# --pitch-mean 214.72203 
# --pitch-std 65.72038
def normalize_pitch(pitch, mean, std):
    zeros = (pitch == 0.0)
    pitch -= mean[:, None] # set to 0
    pitch /= std[:, None] # scale down
    pitch[zeros] = 0.0
    return pitch

def interpolate(pitch_mel_array):
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

add_path = 'C:/Users/wx_Ca/OneDrive - University of Edinburgh/Desktop/slope_add_0.45_-5_3'
for dirpath, dirnames, filenames in os.walk(add_path):
    f0s = []
    times = []
    means = []
    slopes = []
    for filename in filenames:
        wav = os.path.join(dirpath, filename)
        print(wav)
        data, sr = librosa.load(wav, sr=8000, mono=True)
        print(data.shape)
        f0, vid, vpd = librosa.pyin(data, sr=8000, fmin=20, fmax=600, frame_length=1024)
        # print(f0.shape)
        f0 = np.nan_to_num(f0)
        f0 = np.array(f0)
        f0 = interpolate(f0)
        f0s.append(f0)
        mean = np.true_divide(f0.sum(0),(f0!=0).sum(0))
        # print(mean)
        means.append(mean)
        x = range(len(f0))
        y = f0
        fit = np.polyfit(x, y, 1)
        slope, intercept = fit
        print(slope)
        slopes.append(slope)
        time = librosa.times_like(f0)
        times.append(time)

######################################
expected = []
for i in range(81):
    e_slope = -0.5 + i*0.01
    expected.append(e_slope)
print(expected)
print(len(slopes))

fig, ax = plt.subplots()
ax.set(title='f0 slope controlling')
ax.set_ylabel('f0 slope got')
ax.set_xlabel('f0 slope asked for')
ax.set_xlim(-1, 3)
ax.set_ylim(-1, 3)
ax.plot(expected, slopes, color='blue', label='add-first')
# ax.plot(expected, slope, color='green', label='emb-first higher-controlled')
ax.plot(expected, expected, color='grey', label='expected  f0')
# ax.plot([214.72203 for i in range(600)], color = 'lightgrey', linestyle = "dotted")
ax.legend(loc='upper left')
plt.show()

# emb_path = 'C:/Users/wx_Ca/OneDrive - University of Edinburgh/Desktop/mean/mean_controlling_emb/-2.9-5.8'
# for dirpath, dirnames, filenames in os.walk(emb_path):
#     e_f0s = []
#     imes = []
#     e_means = []
#     for filename in filenames:
#         wav = os.path.join(dirpath, filename)
#         print(wav)
#         data, sr = librosa.load(wav, sr=8000, mono=True)
#         print(data.shape)
#         e_f0, vid, vpd = librosa.pyin(data, sr=8000, fmin=20, fmax=600, frame_length=1024)
#         print(e_f0.shape)
#         e_f0 = np.nan_to_num(e_f0)
#         e_f0 = np.array(e_f0)
#         e_f0 = interpolate(e_f0)
#         e_mean = np.true_divide(e_f0.sum(0),(e_f0!=0).sum(0))
#         print(e_mean)
#         e_means.append(e_mean)
#         e_f0s.append(e_f0)
#         time = librosa.times_like(e_f0)
#         times.append(time)

# expected = []
# n_expected=[]
# for i in range(59):
#     pitch = 65.72038*i*0.1 + 186.1783
#     expected.append(pitch)
# for i in range(30):
#     pitch = 65.72038*(-i)*0.1 + 186.1783
#     n_expected.append(pitch)
# print(expected)
# print(n_expected)

# print(len(means))
# print(len(e_means))

# p_m = means[0:59]
# print(p_m, len(p_m))
# n_m = means[59:89]
# n_m.insert(0, 222.10661373435622)
# print(n_m, len(n_m))

# e_p_m = e_means[0:59]
# print(e_p_m, len(e_p_m))
# e_n_m = e_means[59:89]
# e_n_m.insert(0, 213.78685099479188) # to be changed
# print(e_n_m, len(e_n_m))

# fig, ax = plt.subplots()
# ax.set(title='mean f0 controlling')
# ax.set_ylabel('mean f0 got (Hz)')
# ax.set_xlabel('mean f0 asked for (Hz)')
# ax.set_xlim(0, 600)
# ax.set_ylim(0, 600)
# ax.plot(expected, p_m, color='blue', label='add-first higher-controlled')
# ax.plot(n_expected, n_m, color='tomato', label='add-first lower-controlled')
# ax.plot(expected, e_p_m, color='green', label='emb-first higher-controlled')
# ax.plot(n_expected, e_n_m, color='pink', label='emb-first lower-controlled')
# ax.plot(expected, expected, color='grey', label='expected mean f0')
# ax.plot(n_expected, n_expected, color='grey')
# ax.vlines(214.72203, 20, 600, colors = 'lightgrey', linestyles = "dotted", label='global mean')
# ax.plot([214.72203 for i in range(600)], color = 'lightgrey', linestyle = "dotted")
# ax.legend(loc='upper left')
# plt.show()

################################## contours ##################################

# fig, ax = plt.subplots()
# ax.set(title='PYIN f0 estimation ')
# ax.set_ylim(100, 450)
# ax.plot(times[0], f0s[0], label='groundtruth', color='black', linewidth=2)
# ax.plot(times[1], f0s[1], label='baseline', color='grey', linewidth=2)
# ax.plot(times[2], f0s[2], label='O_A', color='red', linewidth=2)
# ax.plot(times[3], f0s[3], label='O_E', color='blue', linewidth=2)
# ax.plot(times[4], f0s[4], label='P_A', color='yellow', linewidth=2)
# ax.plot(times[5], f0s[5], label='P_E', color='green', linewidth=2)
# plt.show()


############################### drawing hnr histogram distribution ###############################
# hnr_file = 'C:/Users/wx_Ca\OneDrive - University of Edinburgh/Desktop/voice lab/hnr.txt'
# hnr_list = []
# with open(hnr_file, 'r') as f:
#     for line in f:
#         line = line.split()[1]
#         hnr_list.append(line)
# print(len(hnr_list), min(hnr_list), max(hnr_list)) # 13100 0.7398815521997723 9.871653761881996
# hnrList = [round(float(x)) for x in hnr_list]
# plt.hist(hnrList, bins=range(0, 10 + 1, 1))
# plt.show()
##################################################################################################