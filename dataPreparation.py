# -*- coding: utf-8 -*-

import numpy as np
import h5py as h5py
import math

def get_data():
    X_train_all = np.load("../X_train_all.npy")
    Y_train_all = np.load("../Y_train_all.npy")
    X_valid_all = np.load("../X_valid_all.npy")
    Y_valid_all = np.load("../Y_valid_all.npy")
    X_test  = np.load("../X_test.npy")
    Y_test  = np.load("../Y_test.npy")
    snr_mod_pairs_test=np.load("../snr_mod_pairs_test.npy")
    mods = np.load("../mods.npy").tolist()
    snrs = np.load("../snrs.npy").tolist()
    nsps = np.load("../nsps.npy").tolist()

    return X_train_all, Y_train_all, X_valid_all, Y_valid_all,X_test, Y_test, snrs, mods,snr_mod_pairs_test

#get_data()

def get_no_augment_data():
    X_train_all = np.load("../X_train_no_augm.npy")
    Y_train_all = np.load("../Y_train_no_augm.npy")
    X_valid_all = np.load("../X_valid_no_augm.npy")
    Y_valid_all = np.load("../Y_valid_no_augm.npy")
    X_test  = np.load("../X_test.npy")
    Y_test  = np.load("../Y_test.npy")
    snr_mod_pairs_test=np.load("../snr_mod_pairs_test.npy")
    mods = np.load("../mods.npy").tolist()
    snrs = np.load("../snrs.npy").tolist()
    nsps = np.load("../nsps.npy").tolist()

    return X_train_all, Y_train_all, X_valid_all, Y_valid_all,X_test, Y_test, snrs, mods,snr_mod_pairs_test

def get_all_snr_data():
    X_train_all = np.load("../X_train_all_snr.npy")
    Y_train_all = np.load("../Y_train_all_snr.npy")
    X_valid_all = np.load("../X_valid_all_snr.npy")
    Y_valid_all = np.load("../Y_valid_all_snr.npy")
    X_test  = np.load("../X_test.npy")
    Y_test  = np.load("../Y_test.npy")
    snr_mod_pairs_test=np.load("../snr_mod_pairs_test.npy")
    mods = np.load("../mods.npy").tolist()
    snrs = np.load("../snrs.npy").tolist()
    nsps = np.load("../nsps.npy").tolist()

    return X_train_all, Y_train_all, X_valid_all, Y_valid_all,X_test, Y_test, snrs, mods,snr_mod_pairs_test

def get_mix_data():
    X_train_all = np.load("../X_train_mix.npy")
    Y_train_all = np.load("../Y_train_mix.npy")
    X_valid_all = np.load("../X_valid_mix.npy")
    Y_valid_all = np.load("../Y_valid_mix.npy")
    X_test  = np.load("../X_test.npy")
    Y_test  = np.load("../Y_test.npy")
    snr_mod_pairs_test=np.load("../snr_mod_pairs_test.npy")
    mods = np.load("../mods.npy").tolist()
    snrs = np.load("../snrs.npy").tolist()
    nsps = np.load("../nsps.npy").tolist()

    return X_train_all, Y_train_all, X_valid_all, Y_valid_all,X_test, Y_test, snrs, mods,snr_mod_pairs_test
