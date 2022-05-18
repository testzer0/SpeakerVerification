"""
Adithya Bhaskar, 2022.
This file contains the configuration parameters for the training and 
usage of the models.
"""

TIMIT_ROOT = "data/TIMIT/data"  # The directory with the TIMIT data
LOGMEL_ROOT = "data/TIMIT/data/logmels/"
                                # The directory with the logmels
CHECKPT_DIR = "checkpoints"

num_frames = 180                # Number of frames after preprocessing
hop = 0.01                      # Hop length in s
window = 0.025                  # Window size in s
n_fft = 512                     # Length of windowed signal after padding
sr = 16000                      # Sampling rate
win_length = int(window * sr)   # Window length
hop_length = int(hop * sr)      # Hop length
n_mels = 40                     # Number of Mel bands
epsilon = 1e-8                  # Small amount to add to avoid taking log of 0

embedder_lr = 1e-3              # Learning rate for embedder
lossmodule_lr = 1e-3            # Learning rate for lossmodule
n_hidden = 768                  # Dimensionality of LSTM outputs
n_projection = 256              # Dimensionality after projection
num_layers = 3                  # Number of LSTM layers
n_speakers = 3                  # Number of speakers per batch
n_utterances_per_speaker = 10   # Number of utterances per speaker each batch

BATCH_SIZE = 16                 # Batch size
NUM_EPOCHS = 10                 # Number of epochs

force_restart_training = False  # Force training to restart from epoch 0
save = True                     # Whether to save model parameters
load_opts = True                # Load optimizer states along with model param values
halve_after_every = 12          # Number of epochs after which to halve learning rate       
                                # For now we only train for 10 epochs so unused.
                                
if __name__ == '__main__':
    pass