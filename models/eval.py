"""
Adithya Bhaskar, 2022.
This file contains functions to evaluate the model's performance
on datasets passed to the respective functions.
"""

from config import *
from utils.globals import *

import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

def get_similarity_metrics(embedder, examples, n_same=3200, \
	n_different=4800, seed=None):
	"""
	Returns two arrays containing the sorted similarity scores of
	n_same pairs of utterances from the same speaker and n_different
	pairs of utterances from different speakers.
	Scores lie in [-1, 1]. A score closer to 1 implies a better match.
	"""
	if n_same % BATCH_SIZE != 0 or n_different % BATCH_SIZE != 0:
		print("For now we require 2*n_same and 2*n_not_same to be divisible by BATCH_SIZE")
		return [], []
	if BATCH_SIZE % 2 != 0:
		print("Odd batch sizes are not supported for this function")
		return [], []
	embedder.eval()
	same_scores = []
	different_scores = []
	n_examples_per_batch = int(BATCH_SIZE / 2)
	n_same_batches = int(2*n_same / BATCH_SIZE)
	n_different_batches = int(2*n_different / BATCH_SIZE)
	n_speakers = len(examples)

	if seed is not None:
		random.seed(seed)

	for b in range(n_same_batches):
		if b != 0 and b % 20 == 0:
			print("{} / {} batches done in the same portion".format(b, n_same_batches))
		utts = []
		for i in range(n_examples_per_batch):
			# Select a random speaker
			speaker = random.randint(0, n_speakers-1)
			# Get two different utterances for this speaker
			utt1 = random.randint(0, examples[speaker].shape[0]-1)
			utt2 = random.randint(0, examples[speaker].shape[0]-1)
			while utt2 == utt1:
				utt2 = random.randint(0, examples[speaker].shape[0]-1)
			utts.append(torch.from_numpy(examples[speaker][utt1]))
			utts.append(torch.from_numpy(examples[speaker][utt2]))
		utts = torch.stack(utts)
		utts = torch.permute(utts, (0,2,1)).to(device)
		with torch.no_grad():
			reps = embedder(utts)
		for i in range(n_examples_per_batch):
			similarity = torch.dot(reps[2*i], reps[2*i+1])
			same_scores.append(similarity)

	for b in range(n_different_batches):
		if b != 0 and b % 20 == 0:
			print("{} / {} batches done in the different portion".format(b, n_different_batches))
		utts = []
		for i in range(n_examples_per_batch):
			# Select two random speakers
			speaker1 = random.randint(0, n_speakers-1)
			speaker2 = random.randint(0, n_speakers-1)
			while speaker2 == speaker1:
				speaker2 = random.randint(0, n_speakers-1)
			# Get an utterance from each
			utt1 = random.randint(0, examples[speaker1].shape[0]-1)
			utt2 = random.randint(0, examples[speaker2].shape[0]-1)
			utts.append(torch.from_numpy(examples[speaker1][utt1]))
			utts.append(torch.from_numpy(examples[speaker2][utt2]))
		utts = torch.stack(utts)
		utts = torch.permute(utts, (0,2,1)).to(device)
		with torch.no_grad():
			reps = embedder(utts)
		for i in range(n_examples_per_batch):
			similarity = torch.dot(reps[2*i], reps[2*i+1])
			different_scores.append(similarity)

	return sorted(same_scores), sorted(different_scores)

def get_false_accept_and_reject_rates(same_scores, different_scores, \
	start=0, stop=1, step=0.01):
	"""
	Given the sorted scores for pairs of utterances from same and different
	speakers, calculates the False Acceptance Rate (FAR) and False Rejection
	Rate (FRR) for various threshold values of acceptance, and returns them.
	"""
	false_accept_rates = []
	false_reject_rates = []
	threshs = []
	thresh = start
	same_idx = 0
	different_idx = 0
	while thresh <= stop:
		while same_idx < len(same_scores) and same_scores[same_idx] < thresh:
			same_idx += 1
		while different_idx < len(different_scores) and different_scores[different_idx] < thresh:
			different_idx += 1
		num_same_rejected = same_idx
		num_different_accepted = len(different_scores) - different_idx
		false_accept_rate = num_different_accepted / len(different_scores)
		false_reject_rate = num_same_rejected / len(same_scores)
		false_accept_rates.append(false_accept_rate)
		false_reject_rates.append(false_reject_rate)
		threshs.append(thresh)
		thresh += step
	return false_accept_rates, false_reject_rates, threshs

def get_thresh_and_eer(fars, frrs, threshs):
	"""
	Given the FARs and FRRs for a sequence of thresholds, calculates the
	Equal Error Rate and the threshold at which it is achieved.
	"""
	n = len(threshs)
	if n == 0:
		return 0, 1
	min_idx = 0
	for i in range(1, n):
		if abs(fars[i]-frrs[i]) < abs(fars[min_idx]-frrs[min_idx]):
			min_idx = i
	return threshs[min_idx], (fars[min_idx]+frrs[min_idx])/2

def plot_far_and_frr(fars, frrs, threshs, addendum=None, save_path=None):
	"""
	Plot the FAR and FRR v/s the threshold values passed, and optionally
	save the plot to the specified path.
	"""
	if addendum is None:
		title = "False Acceptance and Rejection Rate v/s Threshold"
	else:
		title = "False Acceptance and Rejection Rate v/s Threshold - {}".\
			format(addendum)
	plt.title(title)
	plt.plot(threshs, fars, label="False Acceptance Rate")
	plt.plot(threshs, frrs, label="False Rejection Rate")
	plt.xlabel("Threshold Value")
	plt.ylabel("Rate")
	plt.legend()
	if save_path is None:
		plt.show()
	else:
		plt.savefig(save_path)
  
def eval_on_dataset(embedder, dataset, save_path=None, addendum=None):
    """
    Evaluates the embedder on the given dataset.
    Returns the threshold and EER value.
    Optionally, saves the FAR, FRR v/s Threshold plots.
    Also optionally, an addendum that is appended to the plot titles
    can be passed.
    """
    same, different = get_similarity_metrics(embedder, dataset)
    fars, frrs, threshs = get_false_accept_and_reject_rates(same, different)
    thresh, eer = get_thresh_and_eer(fars, frrs, threshs)
    if save_path is not None:
        plot_far_and_frr(fars, frrs, threshs, addendum, save_path)
    return thresh, eer        

if __name__ == '__main__':
    pass