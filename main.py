"""
Adithya Bhaskar, 2022.
This file has the main driver functions which interface with user inputs and
call/drive various components of the model.
"""

from config import *
from utils.globals import *

from utils.data import get_spectrograms_for_file, \
    save_spectrograms, load_data
from models.models import get_models_and_optimizers, \
    load_embedder_from_checkpt, get_dataloader, train_models, \
    run_utts_through_model
from models.eval import eval_on_dataset

import argparse
import os
import numpy as np
import torch

data = None
embedder = None

def preprocess():
    """
    Preprocess the data to extract logmels, and then save them
    as numpy arrays.
    """
    save_spectrograms()
    
def train():
    """
    Load data, wrap it up in a dataloader and train the model.
    """
    global data, embedder
    data = load_data()
    train_dataloader = get_dataloader(data['TRAIN'])
    embedder, lossmodule, embedder_optimizer, lossmodule_optimizer = \
        get_models_and_optimizers()
    train_models(embedder, lossmodule, embedder_optimizer, \
        lossmodule_optimizer, train_dataloader)
    
def evaluate(checkpt, split, plot):
    """
    Evaluate the model loaded from checkpointt numbered checkpt on the split
    `split`, and optionally store the generated plot to `plot`.
    """
    global data, embedder
    if data is None:
        data = load_data()
    if embedder is None:
        embedder, _ = get_models_and_optimizers(only_models=True)
    load_embedder_from_checkpt(embedder, checkpt)
    addendum = "Checkpoint {}: {}".format(checkpt, split[0]+split.lower()[1:])
    thresh, eer = eval_on_dataset(embedder, data[split], save_path=plot, \
        addendum=addendum)
    print("EER (Equal Error Rate) = {}, achieved at a threshold of {}".format( \
        eer, thresh))
    
def similarity(path1, path2, checkpt, thresh):
    """
    Given the path to two audio paths and a threshold, computes
    their similarity score and decides whether both depict the same
    person.
    """
    global embedder
    if embedder is None:
        embedder, _ = get_models_and_optimizers(only_models=True)
    load_embedder_from_checkpt(embedder, checkpt)
    max_utts_per_speaker = 8
    utts1 = get_spectrograms_for_file(path1)
    if len(utts1) > max_utts_per_speaker:
        utts1 = utts1[:max_utts_per_speaker]
    utts2 = get_spectrograms_for_file(path2)
    if len(utts2) > max_utts_per_speaker:
        utts2 = utts2[:max_utts_per_speaker]
    nutts1 = len(utts1)
    nutts2 = len(utts2)
    utts = utts1 + utts2
    utts = [torch.from_numpy(utt) for utt in utts]
    reps = run_utts_through_model(embedder, utts)
    reps1, reps2 = reps[:nutts1], reps[nutts1:]
    avg_similarity = 0
    for i in range(nutts1):
        for j in range(nutts2):
            avg_similarity += torch.dot(reps1[i], reps2[j])
    avg_similarity /= nutts1 * nutts2
    if avg_similarity > thresh:
        print("The two audio files contain the same speaker", end=" ")
    else:
        print("The two audio files contain different speakers", end=" ")
    print("[Avg. Similarity Score = {}].".format(avg_similarity))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", help="Preprocess data and store it "
                        "under LOGMEL_DIR (see config.py)", action="store_true")
    parser.add_argument("--train", help="Train the model on the preprocessed "
                        "dataset pointed to by LOGMEL_DIR", action="store_true")
    parser.add_argument("--eval", help="Evaluate the model, by default of "
                        "checkpoint 3 on the test set. To change the checkpoint "
                        "number or split, use the --checkpt and --split "
                        "parameters", action="store_true")
    parser.add_argument("--checkpt", help="Load model states from the checkpoint "
                        "number passed here. Default is 3.", type=int)
    parser.add_argument("--split", help="The split to evaluate on. Must be one "
                         " of train/test. Default is test.", type=str)
    parser.add_argument("--thresh", help="Threshold for similarity. Default 0.75", \
                        type=float)
    parser.add_argument("--similarity", help="Calculate similarity score and "
                        "decision on two audio files. The paths to these files "
                        "must be passed via the path1 and path2 arguments", \
                        action="store_true")
    parser.add_argument("--path1", help="The path to the first audio file", type=str)
    parser.add_argument("--path2", help="The path to the second audio file", type=str)    
    parser.add_argument("--plot", help="If specified, the plot generated during evaluation "
                        "will be saved to this location", type=str)
    args = parser.parse_args()
    
    usage = True
    checkpt = 3
    split = "TEST"
    thresh = 0.75
    plot = None
    if args.checkpt is not None:
        checkpt = args.checkpt
    if args.split == "train":
        split = "TRAIN"
    if args.thresh is not None:
        thresh = args.thresh
    if args.plot is not None:
        plot = args.plot
    
    if args.preprocess:
        usage = False
        preprocess()
    if args.train:
        usage = False
        train()
    if args.eval:
        usage = False
        evaluate(checkpt, split, plot)
    if args.similarity:
        usage = False
        if args.path1 is None or args.path2 is None:
            print("Both paths must be provided!")
            exit(-1)
        similarity(args.path1, args.path2, checkpt, thresh)
        
    if usage:
        parser.print_help()