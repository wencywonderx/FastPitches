from fastpitch.prepare_dataset import TTSDataset


test = TTSDataset(
            "U:\home\wencywonder\FastPitches\PyTorch\SpeechSynthesis\FastPitch\fastpitch\test_folder",
            'U:\home\wencywonder\FastPitches\PyTorch\SpeechSynthesis\FastPitch\filelists\test_folder\test_file.txt',
            text_cleaners=['english_cleaners_v2'],
            n_mel_channels=80,
            p_arpabet=0.0,
            n_speakers=1,
            load_mel_from_disk=False,
            load_pitch_from_disk=True,
            pitch_mean=None,
            pitch_std=None,
            max_wav_value=32768.0,
            sampling_rate=22050,
            filter_length=1024,
            hop_length=256,
            win_length=1024,
            mel_fmin=0.0,
            mel_fmax=8000.0,
            betabinomial_online_dir=None,
            pitch_online_dir=None,
            pitch_online_method='pyin',
            interpolate=True,
            mean_delta=True)

print(test, test.size)