import librosa
import numpy as np
import os
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

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

path = 'C:/Users/wx_Ca/OneDrive - University of Edinburgh/Desktop/plotting/mean_controlling'
for dirpath, dirnames, filenames in os.walk(path):
    f0s = []
    times = []
    means = []
    for filename in filenames:
        wav = os.path.join(dirpath, filename)
        print(wav)
        data, sr = librosa.load(wav, sr=8000, mono=True)
        print(data.shape)
        f0, vid, vpd = librosa.pyin(data, sr=8000, fmin=40, fmax=600, frame_length=1024)
        print(f0.shape)
        f0 = np.nan_to_num(f0)
        f0 = np.array(f0)
        f0 = interpolate(f0)
        mean = np.true_divide(f0.sum(0),(f0!=0).sum(0))
        print(mean)
        means.append(mean)
        f0s.append(f0)
        time = librosa.times_like(f0)
        times.append(time)

# criterion = nn.MSELoss()
# loss_base_gt = torch.sqrt(criterion(f0s[0], f0s[1]))
# loss_base_add_O = torch.sqrt(criterion(f0s[1], f0s[2]))
# loss_base_emb_O = torch.sqrt(criterion(f0s[1], f0s[3]))
# print(loss_base_gt, loss_base_add_O, loss_base_emb_O)

expected = []
for i in range(101):
    pitch = 65.72038*i*0.1 + 186.1783
    expected.append(pitch)
print(means)
print(len(means))
fig, ax = plt.subplots()
ax.set(title='mean f0 controlling')
ax.set_ylabel('what we got in Hz')
ax.set_xlabel('what we asked in Hz')
ax.plot(range(101), means, label='controlling')
ax.plot(range(101), expected, label='expected')
# ax.plot(range(51), [186.1783629 for i in range(51)], label='add_first')
# ax.plot(range(51), [196.8518806 for i in range(51)], label='baseline')
# ax.plot(range(51), [200.641918 for i in range(51)], label='groundtruth')
ax.legend(loc='upper right')
plt.show()


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