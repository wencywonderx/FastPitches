import librosa
import numpy as np
import os
import math
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

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

# path = 'C:/Users/wx_Ca/OneDrive - University of Edinburgh/Desktop/plotting'
# for dirpath, dirnames, filenames in os.walk(path):
#     f0s = []
#     times = []
#     for filename in filenames:
#         print(filename)
#         wav = os.path.join(dirpath, filename)
#         # wav = "C:/Users/wx_Ca/OneDrive - University of Edinburgh/Desktop/plotting/001_base.wav"
#         data, sr = librosa.load(wav, sr=8000, mono=True)
#         print(data.shape)
#         f0, vid, vpd = librosa.pyin(data, sr=8000, fmin=40, fmax=600, frame_length=1024)
#         # print(f0.shape)
#         # print(f0)
#         f0 = np.nan_to_num(f0)
#         # print(f0)
#         f0 = np.array(f0)
#         f0 = interpolate(f0)
#         length = len(f0)
#         seg = math.floor(length/10)
#         # print(seg)
#         f0_seg = []
#         for i, value in enumerate(f0):
#             if i==seg*1 or i==seg*2 or i==seg*3 or i==seg*4 or i==seg*5 or i==seg*6 or i==seg*7 or i==seg*8 or i==seg*9 or i==seg*10:
#                 print(i)
#                 f0_seg.append(value)
#         print(f0_seg)
#         f0s.append(f0_seg)
        # time = librosa.times_like(f0)
        # times.append(time)


# fig, ax = plt.subplots()
# ax.set(title='PYIN f0 estimation ')
# ax.set_ylim(150, 400)
# x = range(1, 11)

# model = make_interp_spline(x, f0s[0])
# xs = np.linspace(1,10,500)
# ys = model(xs)
# ax.plot(xs, ys, label='baseline', color='cyan', linewidth=2)
# # ax.plot(xs, make_interp_spline(x, f0s[1]), label='O_A', color='red', linewidth=2)
# # ax.plot(xs, make_interp_spline(x, f0s[2]), label='O_E', color='blue', linewidth=2)
# # ax.plot(xs, make_interp_spline(x, f0s[3]), label='P_A', color='yellow', linewidth=2)
# # ax.plot(xs, make_interp_spline(x, f0s[4]), label='P_E', color='green', linewidth=2)

# ax.legend(loc='upper right')
# plt.show()


##################################################### drawing hnr histogram distribution ###########################################
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
#####################################################################################################################################