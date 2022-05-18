"""
Adithya Bhaskar, 2022.
This file houses the functions that help to load, process and write back
data that are to be used for training/inference.
"""

from config import *
from utils.globals import *

import os
import numpy as np
import librosa

def get_spectrograms_for_file(file_path):
	"""
	Returns the log mel specrogram's first and last n_frames frames for each window
	having no silence frames. 
	Implementation is based on 
	https://github.com/HarryVolek/PyTorch_Speaker_Verification/blob/master/data_preprocess.py.
	"""
	min_length = (num_frames*hop + window)*sr
	# Load the audio
	y, _ = librosa.load(file_path, sr=sr)
	# Split the audio into non-silent intervals. 
	# Reference implementation takes top_db (thresh for silence) to be 30, but librosa
	# default is 60.
	intervals = librosa.effects.split(y, top_db=30)
	extracted = []
	for i in range(intervals.shape[0]):
		begin = intervals[i][0]
		end = intervals[i][1]
		if end - begin <= min_length:
			continue
		# Extract relevant portion of wav
		yp = y[begin:end]
		# Perform STFT
		stft = librosa.stft(y=yp, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
		# Squared magnitude of stft - abs necessary because complex
		sqmag = np.abs(stft) ** 2
		# Get mel basis
		M = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
		# Extract log mel spectrogram
		logmel = np.log10(np.dot(M, sqmag) + epsilon)
		# Return the first and last n_frames frames
		extracted.append(logmel[:, :num_frames])
		extracted.append(logmel[:, -num_frames:])
	return extracted

def get_spectrograms_for_speaker(speaker_dir):
	"""
	Given a directory with a speaker's utterances, returns the concatenated list
	of extracted log mel features from them *after* converting it into a numpy array.
	"""
	extracted = []
	for fname in os.listdir(speaker_dir):
		if fname.endswith(".WAV.wav"):
			extracted += get_spectrograms_for_file(os.path.join(speaker_dir, fname))
	return np.array(extracted)

def save_spectrograms(splits=["TRAIN", "TEST"]):
	"""
	Call only once. Goes through each speaker dir and saves the generated spectrograms
	under LOGMEL_ROOT/{split}
	"""
	for split in splits:
		split_data_dir = os.path.join(TIMIT_ROOT, split)
		split_logmel_dir = os.path.join(LOGMEL_ROOT, split)
		for DR in os.listdir(split_data_dir):
			DR_dir = os.path.join(split_data_dir, DR)
			for speaker in os.listdir(DR_dir):
				extracted = get_spectrograms_for_speaker(os.path.join(DR_dir, speaker))
				out_file = os.path.join(split_logmel_dir, "{}.npy".format(speaker))
				np.save(open(out_file, 'wb+'), extracted)

def load_data(splits=["TRAIN", "TEST"], min_samples=4):
	"""
	Loads the dataset -- removes all speakers with < min_samples examples.
	"""
	data = {}
	for split in splits:
		part = []
		ldir = os.path.join(LOGMEL_ROOT, split)
		for fname in os.listdir(ldir):
			if not fname.endswith(".npy"):
				continue
			narray = np.load(open(os.path.join(ldir, fname), "rb"))
			if narray.shape[0] < min_samples:
				continue
			part.append(narray)
			data[split] = part
	return data

if __name__ == '__main__':
	pass