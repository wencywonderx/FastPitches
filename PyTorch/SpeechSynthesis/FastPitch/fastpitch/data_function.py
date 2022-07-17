# *****************************************************************************
#  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import functools
import re
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from scipy.stats import betabinom

import common.layers as layers
from common.text.text_processing import TextProcessing
from common.utils import load_wav_to_torch, load_filepaths_and_text, to_gpu
from fastpitch.pitch_things import interpolate_f0, estimate_pitch, normalize_pitch, mean_delta_f0

class BetaBinomialInterpolator:
    """Interpolates alignment prior matrices to save computation.

    Calculating beta-binomial priors is costly. Instead cache popular sizes
    and use img interpolation to get priors faster.
    """
    def __init__(self, round_mel_len_to=100, round_text_len_to=20):
        self.round_mel_len_to = round_mel_len_to
        self.round_text_len_to = round_text_len_to
        self.bank = functools.lru_cache(beta_binomial_prior_distribution)

    def round(self, val, to):
        return max(1, int(np.round((val + 1) / to))) * to

    def __call__(self, w, h):
        bw = self.round(w, to=self.round_mel_len_to)
        bh = self.round(h, to=self.round_text_len_to)
        ret = ndimage.zoom(self.bank(bw, bh).T, zoom=(w / bw, h / bh), order=1)
        assert ret.shape[0] == w, ret.shape
        assert ret.shape[1] == h, ret.shape
        return ret

def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling=1.0):
    P = phoneme_count
    M = mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M+1):
        a, b = scaling * i, scaling * (M + 1 - i)
        rv = betabinom(P, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return torch.tensor(np.array(mel_text_probs))


class TTSDataset(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self,
                 dataset_path,
                 audiopaths_and_text,
                 text_cleaners,
                 n_mel_channels,
                 symbol_set='english_basic', # Input characters (define your own in $FP/common/text/symbols.py)
                 p_arpabet=1.0,
                 n_speakers=1,
                 load_mel_from_disk=True,
                 load_pitch_from_disk=True,
                 pitch_mean=214.72203,  # LJSpeech defaults
                 pitch_std=65.72038,
                 max_wav_value=None,
                 sampling_rate=None,
                 filter_length=None,
                 hop_length=None,
                 win_length=None,
                 mel_fmin=None,
                 mel_fmax=None,
                 prepend_space_to_text=False,
                 append_space_to_text=False,
                 pitch_online_dir=None,
                 betabinomial_online_dir=None,
                 use_betabinomial_interpolator=True,
                 pitch_online_method='pyin',
                 interpolate_f0 = False, #-------------------------C
                 mean_and_delta_f0 = False, #----------------------C
                 f0_slope = False, #--------------------------C
                 **ignored):

        # Expect a list of filenames
        if type(audiopaths_and_text) is str:
            audiopaths_and_text = [audiopaths_and_text]

        self.dataset_path = dataset_path # loads audio,text pairs   
        self.audiopaths_and_text = load_filepaths_and_text( 
            audiopaths_and_text, dataset_path,
            has_speakers=(n_speakers > 1))
        self.load_mel_from_disk = load_mel_from_disk
        if not load_mel_from_disk:  # extracting mel during training
            self.max_wav_value = max_wav_value
            self.sampling_rate = sampling_rate
            self.stft = layers.TacotronSTFT(
                filter_length, hop_length, win_length,
                n_mel_channels, sampling_rate, mel_fmin, mel_fmax)
        self.load_pitch_from_disk = load_pitch_from_disk

        self.prepend_space_to_text = prepend_space_to_text
        self.append_space_to_text = append_space_to_text

        assert p_arpabet == 0.0 or p_arpabet == 1.0, (
            'Only 0.0 and 1.0 p_arpabet is currently supported. '
            'Variable probability breaks caching of betabinomial matrices.')

        self.tp = TextProcessing(symbol_set, text_cleaners, p_arpabet=p_arpabet)
        self.n_speakers = n_speakers
        self.pitch_tmp_dir = pitch_online_dir
        self.f0_method = pitch_online_method
        self.betabinomial_tmp_dir = betabinomial_online_dir
        self.use_betabinomial_interpolator = use_betabinomial_interpolator
        self.interpolate_f0 = interpolate_f0
        self.mean_and_delta_f0 = mean_and_delta_f0

        if use_betabinomial_interpolator:
            self.betabinomial_interpolator = BetaBinomialInterpolator() 

        expected_columns = (2 + int(load_pitch_from_disk) + (n_speakers > 1)) # check data

        assert not (load_pitch_from_disk and self.pitch_tmp_dir is not None)

        if len(self.audiopaths_and_text[0]) < expected_columns:
            raise ValueError(f'Expected {expected_columns} columns in audiopaths file. '
                             'The format is <mel_or_wav>|[<pitch>|]<text>[|<speaker_id>]')

        if len(self.audiopaths_and_text[0]) > expected_columns:
            print('WARNING: Audiopaths file has more columns than expected')

        to_tensor = lambda x: torch.Tensor([x]) if type(x) is float else x
        self.pitch_mean = to_tensor(pitch_mean)
        self.pitch_std = to_tensor(pitch_std)

    def __getitem__(self, index):
        # Separate filename and text
        if self.n_speakers > 1:
            audiopath, *extra, text, speaker = self.audiopaths_and_text[index]
            speaker = int(speaker)
        else:
            audiopath, *extra, text = self.audiopaths_and_text[index]
            speaker = None

        mel = self.get_mel(audiopath) # (n_mel_channel, mel_len)
        text = self.get_text(text) # (text_len)
        if self.mean_and_delta_f0:
            pitch, mean_f0, delta_f0 = self.get_pitch(index, mel.size(-1), self.interpolate_f0, self.mean_and_delta_f0)
        else:
            pitch = self.get_pitch(index, mel.size(-1), self.interpolate_f0, self.mean_and_delta_f0) # (num_formants, mel_len)
            mean_f0 = None
            delta_f0 = None
        # mean_f0, delta_f0 = self.get_mean_and_f0(pitch) 
        energy = torch.norm(mel.float(), dim=0, p=2) # (mel_len) ----------------------------------------Q: why energy norm mel_len?
        attn_prior = self.get_prior(index, mel.shape[1], text.shape[0])
        assert pitch.size(-1) == mel.size(-1) # (mel_len, text_len)

        # No higher formants?
        if len(pitch.size()) == 1:
            pitch = pitch[None, :] # None is for formants (how to add formants?)

        return (text, mel, len(text), pitch, energy, speaker, attn_prior,
                audiopath, mean_f0, delta_f0) # final return for TTSDataset -----------------------------------------------C

    def __len__(self):
        return len(self.audiopaths_and_text)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm,
                                                 requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.load(filename)  # load the extracted mel from file in SCRATCH
            # assert melspec.size(0) == self.stft.n_mel_channels, (
            #     'Mel dimension mismatch: given {}, expected {}'.format(
            #         melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, text):
        text = self.tp.encode_text(text) # normalizes text and converts them to sequences of one-hot vectors
        space = [self.tp.encode_text("A A")[1]]

        if self.prepend_space_to_text:
            text = space + text

        if self.append_space_to_text:
            text = text + space

        return torch.LongTensor(text) # can make it predict word here, now is one-hot vector for each phone and concatenate(?)

    def get_prior(self, index, mel_len, text_len):

        if self.use_betabinomial_interpolator:
            return torch.from_numpy(self.betabinomial_interpolator(mel_len,
                                                                   text_len)) # duration tensor calling

        if self.betabinomial_tmp_dir is not None:
            audiopath, *_ = self.audiopaths_and_text[index]
            fname = Path(audiopath).relative_to(self.dataset_path) if self.dataset_path else Path(audiopath)
            fname = fname.with_suffix('.pt')
            cached_fpath = Path(self.betabinomial_tmp_dir, fname)

            if cached_fpath.is_file():
                return torch.load(cached_fpath)

        attn_prior = beta_binomial_prior_distribution(text_len, mel_len)

        if self.betabinomial_tmp_dir is not None:
            cached_fpath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(attn_prior, cached_fpath)

        return attn_prior

    def get_pitch(self, index, mel_len=None, interpolate = False, mean_delta = False):
        audiopath, *fields = self.audiopaths_and_text[index]

        if self.n_speakers > 1:
            spk = int(fields[-1])
        else:
            spk = 0

        if self.load_pitch_from_disk:
            # print("\n pitch loaded from disk")
            pitchpath = fields[0]
            pitch = torch.load(pitchpath)
            # print("\n pitch tensor loaded from disk \n", pitch)
            if self.interpolate_f0:
                # print("\n interpolating f0")
                pitch = pitch.numpy()[0]
                # print("\n converted to pitch array \n", pitch)
                pitch = interpolate_f0(pitch)
                # print("\n --------------------interpolated pitch array \n", pitch)
                pitch = torch.from_numpy(pitch).unsqueeze(0)
                # print("\n convert to pitch tensor\n", pitch)
            if self.pitch_mean is not None:
                assert self.pitch_std is not None
                pitch = normalize_pitch(pitch, self.pitch_mean, self.pitch_std)                
            if self.mean_and_delta_f0:
                # print("\n extracting mean and delta f0")
                mean_f0, delta_f0 = mean_delta_f0(pitch)
                # print("\n mean and delta calculated \n", mean_f0, delta_f0)
                return pitch, mean_f0, delta_f0 
            return pitch

        if self.pitch_tmp_dir is not None: # a temperary directory to load pitch file after the frst epoch calculated. to speed up
            fname = Path(audiopath).relative_to(self.dataset_path) if self.dataset_path else Path(audiopath)
            fname_method = fname.with_suffix('.pt')
            cached_fpath = Path(self.pitch_tmp_dir, fname_method)
            if cached_fpath.is_file():
                return torch.load(cached_fpath)

        # No luck so far - calculate
        wav = audiopath
        if not wav.endswith('.wav'):
            wav = re.sub('/mels/', '/wavs/', wav)
            wav = re.sub('.pt$', '.wav', wav)

        pitch_mel = estimate_pitch(wav, mel_len, self.f0_method,
                                   self.pitch_mean, self.pitch_std)

        if self.pitch_tmp_dir is not None and not cached_fpath.is_file():
            cached_fpath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pitch_mel, cached_fpath)

        return pitch_mel
    

class TTSCollate: #padding, make it rectangular, because tensor cannot accept different length
    """Zero-pads model inputs and targets based on number of frames per step"""

    def __call__(self, batch):
        """Collate training batch from normalized text and mel-spec"""
        # Right zero-pad all one-hot text sequences to max input length

        # batch = (text, mel, len(text), pitch, energy, speaker, attn_prior, audiopath, mean_f0, delta_f0, f0_slope)

        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]  # num of phones
        # put samples in order and take the longest sample length
        # print("\n this is input_lengths:", input_lengths)

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]  # first one in TTSDataset
            text_padded[i, :text.size(0)] = text
        # print("\n this is text_padded:", text_padded.size, text_padded)

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])

        # Include mel padded and gate padded -------------------------------------------------Q:gate?
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch)) #----------------------------------------Q
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1] # the second one
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1) # mel_length ---------------------------------------------------Q

        # print("\n this is output_lengths:", output_lengths)
        # print("\n this is mel_padded:", mel_padded.size, mel_padded)
        # print("n\ this is mean_f0:", mean_f0)

        n_formants = batch[0][3].shape[0]
        
        pitch_padded = torch.zeros(mel_padded.size(0), n_formants,
                                   mel_padded.size(2), dtype=batch[0][3].dtype) # (batch_size, n_formants, mel_length)
        energy_padded = torch.zeros_like(pitch_padded[:, 0, :]) # take the energy of f0
        # ----------------------------added by me------------------------------
        delta_f0_padded = torch.zeros(mel_padded.size(0), n_formants,
                                   mel_padded.size(2), dtype=batch[0][3].dtype)
        mean_f0 = torch.zeros_like(input_lengths)
        # ----------------------------------------------------------------------
        for i in range(len(ids_sorted_decreasing)):
            pitch = batch[ids_sorted_decreasing[i]][3]
            energy = batch[ids_sorted_decreasing[i]][4] 
            pitch_padded[i, :, :pitch.shape[1]] = pitch
            energy_padded[i, :energy.shape[0]] = energy
            #--------------added by me------------------------
            if batch[0][9] is not None and batch[0][8] is not None:                
                delta_f0 = batch[ids_sorted_decreasing[i]][9]
                delta_f0_padded[i, :, :delta_f0.shape[1]] = delta_f0
                mean_f0[i] = batch[ids_sorted_decreasing[i]][8]
                print("padded mean f0: ", mean_f0)
            else:
                delta_f0 = None,
                delta_f0_padded = None,
                mean_f0 = None
            #-------------------------------------------------

        # print("n\ this is pitch_padded:", pitch_padded.size, pitch_padded)
        # print("n\ this is energy_padded:", energy_padded.size, energy_padded)
        # print("n\ this is delta_f0_padded:", delta_f0_padded.size, delta_f0_padded)

        if batch[0][5] is not None:
            speaker = torch.zeros_like(input_lengths)
            for i in range(len(ids_sorted_decreasing)):
                speaker[i] = batch[ids_sorted_decreasing[i]][5]
        else:
            speaker = None
        # print("n\ this is speaker:", speaker)

        attn_prior_padded = torch.zeros(len(batch), max_target_len,
                                        max_input_len)
        attn_prior_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            prior = batch[ids_sorted_decreasing[i]][6]
            attn_prior_padded[i, :prior.size(0), :prior.size(1)] = prior
        # print("n\ this is attn_prior_padded:", attn_prior_padded.size, attn_prior_padded)

        # Count number of items - characters in text 
        len_x = [x[2] for x in batch] #------------------------------------------------------------------1
        len_x = torch.Tensor(len_x) # same number as input lengths
        # print("n\ this is len_x:", len_x)

        audiopaths = [batch[i][7] for i in ids_sorted_decreasing]
        # (text, mel, len(text), pitch, energy, speaker, attn_prior, audiopath, mean_f0, delta_f0)
        # print("n\ this is audiopaths:", audiopaths)

        return (text_padded, input_lengths, mel_padded, output_lengths, len_x,
                pitch_padded, energy_padded, speaker, attn_prior_padded,
                audiopaths, mean_f0, delta_f0_padded) # change also in prepare_data.py and model.py(245)
                # original batch:
                # mel : [n_mel_channel, mel_len]
                # text : [text_len]
                # pitch : [num_formants, mel_len] 
                # energy : [mel_len]
                # prior : [mel_len, text_len] 
                # input length is text length, output length is mel length from TactronSTFT(short time fourier transform)
                # input_lengths and output_lens are for unpadding


def batch_to_gpu(batch):
    (text_padded, input_lengths, mel_padded, output_lengths, len_x,
     pitch_padded, energy_padded, speaker, attn_prior, audiopaths, mean_f0, delta_f0_padded) = batch

    text_padded = to_gpu(text_padded).long()
    input_lengths = to_gpu(input_lengths).long()
    mel_padded = to_gpu(mel_padded).float()
    output_lengths = to_gpu(output_lengths).long()
    pitch_padded = to_gpu(pitch_padded).float()
    energy_padded = to_gpu(energy_padded).float()
    attn_prior = to_gpu(attn_prior).float()
    if speaker is not None:
        speaker = to_gpu(speaker).long()
    if delta_f0 is not None and mean is not None:
        mean = to_gpu(mean_f0).long()
        delta_f0 = to_gpu(delta_f0_padded).float()

    # Alignments act as both inputs and targets - pass shallow copies ------------------------------------Q
    x = [text_padded, input_lengths, mel_padded, output_lengths,
         pitch_padded, energy_padded, speaker, attn_prior, audiopaths, mean, delta_f0]
    y = [mel_padded, input_lengths, output_lengths]
    len_x = torch.sum(output_lengths) # still input length ------------------------------------------------2
    return (x, y, len_x)
