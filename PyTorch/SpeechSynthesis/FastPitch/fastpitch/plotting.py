import parselmouth

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set() # Use seaborn's default style to make attractive graphs
plt.rcParams['figure.dpi'] = 100 # Show nicely large images in this notebook

snd = parselmouth.Sound("C:/Users/wx_Ca/OneDrive - University of Edinburgh/Desktop/Dissertation/pitch-prom-focus1-1.wav")

plt.figure()
plt.plot(snd.xs(), snd.values.T)
plt.xlim([snd.xmin, snd.xmax])
plt.xlabel("time [s]")
plt.ylabel("amplitude")
plt.show() # or plt.savefig("sound.png"), or plt.savefig("sound.pdf")

snd_part = snd.extract_part(from_time=0.9, preserve_times=True)

plt.figure()
plt.plot(snd_part.xs(), snd_part.values.T, linewidth=0.5)
plt.xlim([snd_part.xmin, snd_part.xmax])
plt.xlabel("time [s]")
plt.ylabel("amplitude")
plt.show()

def draw_spectrogram(spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")

# def draw_intensity(intensity):
#     plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
#     plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
#     plt.grid(False)
#     plt.ylim(0)
#     plt.ylabel("intensity [dB]")


# intensity = snd.to_intensity()
# spectrogram = snd.to_spectrogram()
# plt.figure()
# draw_spectrogram(spectrogram)
# plt.twinx()
# draw_intensity(intensity)
# plt.xlim([snd.xmin, snd.xmax])
# plt.show()


def draw_pitch(pitch):
    # Extract selected pitch contour, and
    # replace unvoiced samples by NaN to not plot
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values==0] = np.nan
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
    plt.grid(False)
    plt.ylim(0, pitch.ceiling)
    plt.ylabel("fundamental frequency [Hz]")

pitch = snd.to_pitch()

# If desired, pre-emphasize the sound fragment before calculating the spectrogram
pre_emphasized_snd = snd.copy()
pre_emphasized_snd.pre_emphasize()
spectrogram = pre_emphasized_snd.to_spectrogram(window_length=0.03, maximum_frequency=8000)

plt.figure()
draw_spectrogram(spectrogram)
plt.twinx()
draw_pitch(pitch)
plt.xlim([snd.xmin, snd.xmax])
plt.show()

# import pandas as pd

# def facet_util(data, **kwargs):
#     digit, speaker_id = data[['digit', 'speaker_id']].iloc[0]
#     sound = parselmouth.Sound("audio/{}_{}.wav".format(digit, speaker_id))
#     draw_spectrogram(sound.to_spectrogram())
#     plt.twinx()
#     draw_pitch(sound.to_pitch())
#     # If not the rightmost column, then clear the right side axis
#     if digit != 5:
#         plt.ylabel("")
#         plt.yticks([])

# results = pd.read_csv("other/digit_list.csv")

# grid = sns.FacetGrid(results, row='speaker_id', col='digit')
# grid.map_dataframe(facet_util)
# grid.set_titles(col_template="{col_name}", row_template="{row_name}")
# grid.set_axis_labels("time [s]", "frequency [Hz]")
# grid.set(facecolor='white', xlim=(0, None))
# plt.show()