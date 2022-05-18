"""
Adithya Bhaskar, 2022.
This file defines the datasets, dataloaders and models, and exposes
functions to train them.
"""

from config import *
from utils.globals import *

import re
import os
import time
import datetime
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

class SpeakerVerificationDataset(Dataset):
	def __init__(self, logmels, n_speakers=n_speakers, \
	  n_samples_per_speaker=n_utterances_per_speaker, total_examples=80000):
		"""
		total_examples is the number of examples drawn per epoch
		"""
		self.logmels = logmels
		self.n_total_speakers = len(self.logmels)
		self.n_speakers = n_speakers
		self.n_samples_per_speaker = n_samples_per_speaker
		self.total_examples = total_examples

	def __len__(self):
		return self.total_examples

	def __getitem__(self, idx):
		"""
		Simply ignore idx and return a random sample.
		"""
		# First, select n different random speakers
		# Use the commented code when number of speakers is more
		# speakers = np.random.permutation(self.n_total_speakers)\
		# [:self.n_speakers]
		speakers = []
		while len(speakers) < self.n_speakers:
			speaker = random.randint(0, self.n_total_speakers-1)
			if speaker not in speakers:
				speakers.append(speaker)
		data = []
		for speaker in speakers:
			# We may have as low as 8-10 (up to 28) examples per speaker, and we want to choose
			# 4-10 of them. A permutation likely avoids the otherwise many tries.
			utter_idxs = np.random.permutation( \
				self.logmels[speaker].shape[0])[:self.n_samples_per_speaker]
			utterances = torch.from_numpy( \
				self.logmels[speaker][utter_idxs, :, :])
			data.append(utterances)
		item = torch.stack(data)
		# Currently have (speaker, utterance, mel, frames)
		# Reorder to (speaker, utterance, frames, mel)
		return torch.permute(item, (0,1,3,2))

class SpeakerEmbedder(nn.Module):
	"""
	The input to this model is of shape (batch_size*N*M, frames, mel)
	"""
	def __init__(self):
		super(SpeakerEmbedder, self).__init__()
		self.LSTMs = nn.LSTM(input_size=n_mels, hidden_size=n_hidden, \
							num_layers=num_layers, batch_first=True)
		self.FC = nn.Linear(n_hidden, n_projection)

	def forward(self, x):
		LSTMs_out, _ = self.LSTMs(x)
		# Current shape is (batch_size*N*M, n_timesteps, n_hidden)
		# Need only the last time step
		last_out = LSTMs_out[:, LSTMs_out.size(1)-1]
		# Now the shape is (batch_size*N*M, n_hidden)
		FC_out = self.FC(last_out)
		# Normalize each "row"
		FC_out = FC_out / torch.linalg.norm(FC_out, axis=1).unsqueeze(axis=1)
		return FC_out

class LossModule(nn.Module):
	# Values taken from 
	# https://github.com/HarryVolek/PyTorch_Speaker_Verification/blob/master/speech_embedder_net.py
	def __init__(self):
		super(LossModule, self).__init__()
		self.w = nn.Parameter(torch.tensor(10.0), requires_grad=True)
		self.b = nn.Parameter(torch.tensor(-5.0), requires_grad=True)

	def forward(self, embeddings):
		# The input should be in the shape (batch_size, N, M, n_projection)
		# First get the centroids
		centroids = torch.mean(embeddings, dim=2)
		N = embeddings.shape[1]
		M = embeddings.shape[2]
		S = torch.zeros(BATCH_SIZE, N, M, N)
		loss = 0
		# Good old loops, here we come...
		for b in range(BATCH_SIZE):
			for j in range(N):
				for i in range(M):
					for k in range(N):
						if j == k:
							# In this case recompute centroid to not 
							# include current example
							centroid = (M*centroids[b,k] - \
								embeddings[b,j,i]) / (M-1)
						else:
							centroid = centroids[b,k]
						S[b,j,i,k] = self.w*torch.dot( \
							embeddings[b,j,i], centroid) + self.b
						if j == k:
							loss -= S[b,j,i,k]
		expsum = torch.sum(torch.exp(S), axis=-1)
		loss += torch.sum(torch.log(expsum))
		return loss

def get_models_and_optimizers(only_models=False):
	"""
	Get the models and optionally, the optimizers.
	The latter are needed for training.
	"""
	embedder = SpeakerEmbedder()
	lossmodule = LossModule()
	if torch.cuda.is_available():
		embedder.cuda()
		lossmodule.cuda()
	if only_models:
		return embedder, lossmodule
	embedder_optimizer = AdamW(embedder.parameters(), \
		lr=embedder_lr, eps=epsilon)
	lossmodule_optimizer = AdamW(lossmodule.parameters(), \
		lr=1e-3, eps=epsilon)
	return embedder, lossmodule, embedder_optimizer, lossmodule_optimizer

def load_embedder_from_checkpt(embedder, checkpt_nr=3):
	"""
	Load model states from specific checkpoints. 3 works best on
	the test set, while 10 is preferred on the train set.
	"""
	embedder_path = os.path.join(CHECKPT_DIR, "checkpt-embedder-{}.pt".\
		format(checkpt_nr))
	embedder.load_state_dict(torch.load(embedder_path, map_location=device))
 
def get_dataloader(data, total_examples=80000):
	"""
	Returns a dataloader for the data passed in.
	"""
	dataset = SpeakerVerificationDataset(data, total_examples=total_examples)
	return DataLoader(dataset, batch_size=BATCH_SIZE)

def get_max_checkpt(checkpt_dir):
	max_checkpt = 0
	for filename in os.listdir(checkpt_dir):
		if re.match(r"checkpt-embedder-([0-9]+).pt", filename):
			checkpt_num = int(filename.split('.')[-2].split('-')[-1])
			if checkpt_num > max_checkpt:
				max_checkpt = checkpt_num
	return max_checkpt

def load_latest_checkpt(embedder, lossmodule, embedder_optimizer, \
	lossmodule_optimizer, checkpt_dir=CHECKPT_DIR):
	if force_restart_training:
		return 0
	mx_checkpt = get_max_checkpt(checkpt_dir)
	if mx_checkpt > 0:
		embedder_path = os.path.join(checkpt_dir, "checkpt-embedder-{}.pt".format(mx_checkpt))
		lossmodule_path = os.path.join(checkpt_dir, "checkpt-lossmodule-{}.pt".format(mx_checkpt))
		embedder.load_state_dict(torch.load(embedder_path, map_location=device))
		lossmodule.load_state_dict(torch.load(lossmodule_path, map_location=device))
		if load_opts:
			embedder_opt_path = os.path.join(checkpt_dir, "checkpt-eopt-{}.pt".format(mx_checkpt))
			lossmodule_opt_path = os.path.join(checkpt_dir, "checkpt-lopt-{}.pt".format(mx_checkpt))  
			embedder_optimizer.load_state_dict(torch.load(embedder_opt_path, map_location=device))
			lossmodule_optimizer.load_state_dict(torch.load(lossmodule_opt_path, map_location=device))
	return mx_checkpt

def format_time(elapsed):
	elapsed_rounded = int(round(elapsed))
	return str(datetime.timedelta(seconds=elapsed_rounded))

def train_models(embedder, lossmodule, embedder_optimizer, \
	lossmodule_optimizer, train_dataloader):
	start_epoch = load_latest_checkpt(embedder, lossmodule, \
		embedder_optimizer, lossmodule_optimizer)
	for epoch in range(start_epoch, NUM_EPOCHS):
		print("============ Epoch {} / {} ============".\
			format(epoch+1, NUM_EPOCHS))
		print("Training phase")
		epoch_loss = 0.0
		embedder.train()
		lossmodule.train()
		epoch_start = time.time()
		if (epoch+1) % halve_after_every == 0:
			for param_group in embedder_optimizer.param_groups:
				param_group['lr'] /= 2
			for param_group in lossmodule_optimizer.param_groups:
				param_group['lr'] /= 2
		for step, batch in enumerate(train_dataloader):
			batch = batch.to(device)
			if step % 40 == 0 and step != 0:
				elapsed = format_time(time.time() - epoch_start)
				print("Batch {} of {}. Elapsed {}".format(step, \
					len(train_dataloader), elapsed))
			N = batch.shape[1]
			M = batch.shape[2]
			embedder_in = batch.reshape(BATCH_SIZE*N*M, batch.shape[3], \
				batch.shape[4])
			embedder.zero_grad()
			lossmodule.zero_grad()
			embeddings = embedder(embedder_in)
			embeddings = embeddings.reshape(BATCH_SIZE, N, M, n_projection)
			loss = lossmodule(embeddings)
			loss.backward()
			epoch_loss += loss.detach()
			clip_grad_norm_(embedder.parameters(), 3.0)
			clip_grad_norm_(lossmodule.parameters(), 1.0)
			embedder_optimizer.step()
			lossmodule_optimizer.step()
		epoch_loss /= len(train_dataloader) * BATCH_SIZE
		print("Epoch finished. Average training loss: {}".format(epoch_loss))

		if save:
			embedder_path = os.path.join(CHECKPT_DIR, "checkpt-embedder-{}.pt".\
				format(epoch+1))
			lossmodule_path = os.path.join(CHECKPT_DIR, "checkpt-lossmodule-{}.pt".\
				format(epoch+1))
			embedder_opt_path = os.path.join(CHECKPT_DIR, "checkpt-eopt-{}.pt".format(epoch+1))
			lossmodule_opt_path = os.path.join(CHECKPT_DIR, "checkpt-lopt-{}.pt".format(epoch+1))
			torch.save(embedder.state_dict(), embedder_path)
			torch.save(lossmodule.state_dict(), lossmodule_path)
			torch.save(embedder_optimizer.state_dict(), embedder_opt_path)
			torch.save(lossmodule_optimizer.state_dict(), lossmodule_opt_path)

def run_utts_through_model(embedder, utts):
    """
    Given a list of utterances in the form of mel spectrograms, converts them
    to a stacked torch array, then runs them through the model and finally
    converts the resulting output back to a list of vectors and returns them.
    """
    utts = torch.stack(utts)
    utts = torch.permute(utts, (0,2,1))
    reps = embedder(utts)
    reps_list = [reps[i] for i in range(reps.shape[0])]
    return reps_list

if __name__ == '__main__':
    pass