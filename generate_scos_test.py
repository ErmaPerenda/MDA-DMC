import numpy as np 

import itertools
import h5py as h5py

import matplotlib as mpl
mpl.use('Agg')


from numpy import linalg as LA
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
import math
import scipy.io
from pandas.plotting import register_matplotlib_converters
from matplotlib import cm
import tensorflow as tf
from tensorflow import keras

import os
import random
from scipy.ndimage import zoom
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.manifold import TSNE

register_matplotlib_converters()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
#tf.compat.v1.enable_eager_execution()
#load data set and prepare X_train, X_test, Y_train, Y_test
#dataset is given and presents dictionary whith keys whose format is tuple ('modulation_name',snr_db)
#each key has assigned matrix (1000,2,128) where is 1000 number of observations, 2 is I and Q ()
#first element is real part while the second is imaginary part, and 128 
#is number of samples
train=True
N_samples=1024 #need to fix this why it 65
N=800

#dataset_name='../dataset1024.mat'
#dataset_name='../dataset1024_new.mat'
#dataset_name='../dataset1024_rayleigh_up.mat'
#dataset_name='../dataset1024_rician_up.mat'
#dataset_name='../dataset1024_rayleigh_fs_1000.mat'
#dataset_name='../dataset1024_rician_fs_200.mat'
#dataset_name='../dataset1024_noenergynormiq.mat'
#dataset_name='../dataset1024_noenergynormiq_rayleigh8.mat'
#dataset_name='../dataset1024_noenergynormiq_rici8.mat'
#dataset_name='../dataset1024_noenergynormiq_awgn8.mat'
#dataset_name='../dataset1024_iqnoenergy_riciLsTest.mat'
#dataset_name='../dataset1024_iqnoenergy_rayLsTest.mat'
#dataset_name='../dataset1024_iqnoenergy_awgnLsTest.mat'
#dataset_name='../dataset1024_iqnoenergy_awgnAlpha55.mat'
#dataset_name='../dataset1024_iqnoenergy_rayAlpha55.mat'
#dataset_name='../dataset1024_iqnoenergy_riciAlpha55_200sa.mat'
#dataset_name='../dataset1024_iqnoenergy_rayOther_200sa.mat'
#dataset_name='../dataset1024_iqnoenergy_rici_Lstest_200sa.mat'
#dataset_name='../dataset1024_iqnoenergy_awgn_600fs_200sa.mat'

#dataset_names=['../../dataset1024_iqenergy_awgn_sps4_new_augm.mat','../../dataset1024_iqenergy_awgn_sps4_new_augm.mat']
#dataset_names=['../../dataset1024_iqenergy_awgn_sps4_new_augm_with_rician.mat','../../dataset1024_iqenergy_awgn_sps4_new_augm_with_rician.mat']

#dataset_names=['../../dataset1024_iqenergy_awgn_sps4_new_augm_with_rician.mat','../../dataset1024_iqenergy_awgn_sps4_rayleigh.mat']
#dataset_names=['../../dataset1024_iqenergy_awgn_sps4_new_augm_with_rician.mat','../../dataset1024_iqenergy_awgn_sps4_rician.mat']

#dataset_names=['../../big_dataset1024_iqenergy_awgn_sps4.mat','../../big_dataset1024_iqenergy_awgn_sps4.mat']

#dataset_names=['../../big_dataset1024_iqenergy_awgn_sps4.mat','../../big_dataset1024_iqenergy_rayleigh_sps4.mat']
#dataset_names=['../../big_dataset1024_iqenergy_awgn_sps4.mat','../../big_dataset1024_iqenergy_rician_sps4.mat']
#dataset_names=['../../big_dataset1024_iqenergy_awgn_sps4.mat','../../big_dataset1024_iqenergy_clockoffset_sps4.mat']

#dataset_names=['../../../dataset1024_iqenergy_awgn_sps4_new_augm_with_rician.mat','../../../dataset1024_iqenergy_awgn_sps4_new_augm_with_rician.mat']
#dataset_names=['../../dataset1024_iqenergy_awgn_sps4_new_augm_with_rician.mat','../../dataset1024_iqenergy_awgn_sps4_rayleigh.mat']
 
dataset_names=['big_sco.mat','big_sco.mat']#,'../../../../baseline_dataset1024_iqenergy_hardware_sco.mat']


X=np.array([])
Y=np.array([])
            

X_train=np.array([])
amp_train=np.array([])
phase_train=np.array([])
Y_train=np.array([])
snr_mod_pairs_train=np.array([])

X_test=np.array([])
X_test_rec=np.array([])
amp_test=np.array([])
phase_test=np.array([])
Y_test=np.array([])
snr_mod_pairs_test=np.array([])


X_valid=np.array([])
amp_valid=np.array([])
phase_valid=np.array([])
Y_valid=np.array([])
snr_mod_pairs_valid=np.array([])

snrs=[] #unique values of snr values
mods=[] #unique values of modulation formats
number_transmissions=[]
snr_mod_pairs=np.array([])
power_ratios=[]
n_ffts=[]
nsps=[]
bws=[]
scos=[]

max_number_transmissions=5



split_factor_train=0.8 #split datasets to training and testing data
split_factor_valid=0.1
def gen_data():
    global X, Y, X_test, X_train, Y_train,Y_test, Y_valid, X_valid, snrs, mods, snr_mod_pairs, split_factor
    global snr_mod_pairs_test, snr_mod_pairs_train,Train, datasets_name, labels_name
    global power_ratios, train, bws, n_ffts, nsps, nsps_target, n_fft_target
    global snr_mod_pairs_valid, N_samples,scos
    print ("\n"*2,"*"*10,"Train/test dataset split - Start","*"*10,"\n"*2)
    print ("Doing")
    ctrl_ds=0
    
    for dataset_name in dataset_names:
        print("dataset is ",dataset_name)
        ctrl_ds=ctrl_ds+1
        
        f = h5py.File(dataset_name, 'r')
        ds=f['ds']

        for key in ds.keys():
            #print(key)
            pom=key.split('rer')
            mod=str(pom[0])

            #if str(mod)=="PSK8" or str(mod)=="QAM16" or str(mod)=="QAM64":
            #    print("it is ",mod)
            #    continue
            snr=int(pom[1].replace('neg','-'))
            l=pom[2]
            freq=str(pom[3])
            m=pom[4]
            nt=int(pom[5])
            sco=int(pom[6].replace('neg','-'))

            sps=round(float(l)/float(m),3)
            # ['APSK128', 'APSK16', 'APSK256', 'APSK32', 'APSK32', 'BFM', 'BPSK', 'CPFSK', 
            #'DSBAM', 'GFSK', 'OQPSK', 'PAM4', 'PSK8', 'QAM128', 'QAM16', 'QAM256', 'QAM32', 
            #'QAM32', 'QPSK', 'SSBAM'])


            #if mod.find('BPSK')<0 and mod.find('PSK8')<0 and mod.find('PAM4')<0 and mod.find('QPSK')<0 and mod.find('BFM')<0  and mod.find('FSK')<0 and mod.find('BAM')<0:
            #   continue


            #if snr<5:
            # continue
            #if str(sps)!=str(4.000):
            #   continue


            #if str(sps)!=str(2.000) and str(sps)!=str(4.000) and str(sps)!=str(8.000) and str(sps)!=str(16.000) and str(sps)!=str(32.000):
            #   continue
            #else:
            #   continue

            print ("it is ",sps)
            if(snr not in snrs):
                snrs.append(snr)

            if(mod not in mods):
                mods.append(mod)

            if (sps not in nsps):
                nsps.append(sps)

            if sco not in scos:
                scos.append(sco)



            values=np.array(ds.get(key))
            shape=values.shape
            #print(shape)
            total_len_a=shape[2]
            values_pom=np.transpose(values,(2,1,0))
            #print("after tran ",values.shape)
            values=values_pom[:,0:2,0:N_samples]
            values_weak=values_pom[:,6:8,0:N_samples]
            values_strong=values_pom[:,8:10,0:N_samples]
            values_rician_weak=values_pom[:,10:12,0:N_samples]
            values_rician_strong=values_pom[:,12:14,0:N_samples]

            #print("after tran ",values.shape)
            # if dataset_name.find("../dataset1024_iqnoenergy_rici_Lstest_200sa.mat")<0:
            #   #print("no it is rici")
            #   split_factor_train=0.05
            #   split_factor_valid=0.05
            # else:
            #   #print("it is rici")
            #   split_factor_train=0.25
            #   split_factor_valid=0.25

            if ctrl_ds ==2:
                split_factor_train=0.05
                split_factor_valid=0.05
            else:
                split_factor_train=0.05 #split datasets to training and testing data
                split_factor_valid=0.05
                #total_len_a=100
            
            
            train_len=int(round(split_factor_train*total_len_a))
            valid_len=int(round(split_factor_valid*total_len_a))
            test_len=int(total_len_a-(train_len+valid_len))

            if train_len>total_len_a:
                print("Split factor cannot be higher than 1")
                exit()

            b=np.full((total_len_a,1),mod)
            c=np.full((total_len_a,5),[mod,snr,sps,nt,sco])
            

            indices=np.arange(total_len_a)
            np.random.seed(10000)
            np.random.shuffle(indices)

            if ctrl_ds== 2:# and snr==18:
                print("it is 2")
                if X_test.size == 0:
                    X_test=np.array(values[indices[train_len:train_len+test_len]])
                    Y_test=np.array(b[indices[train_len:train_len+test_len]])
                    snr_mod_pairs_test=np.array(c[indices[train_len:train_len+test_len]])
                else:
                    X_test=np.vstack((X_test, values[indices[train_len:train_len+test_len]]))
                    Y_test=np.append(Y_test,b[indices[train_len:train_len+test_len]],axis=0)
                    snr_mod_pairs_test=np.append(snr_mod_pairs_test, c[indices[train_len:train_len+test_len]],axis=0)



            if ctrl_ds==1:# and snr==18: # in [-2,2,6,10,12,16,20]):
                print("it is ",snr)
                if X.size == 0:
                    X=np.array(values)
                    Y=np.array(b)
                    snr_mod_pairs=np.array(c)

                    X_train=np.array(values[indices[0:train_len]])
                    Y_train=np.array(b[indices[0:train_len]])
                    snr_mod_pairs_train=np.array(c[indices[0:train_len]])

                    X_valid=np.array(values[indices[train_len+test_len:total_len_a]])
                    Y_valid=np.array(b[indices[train_len+test_len:total_len_a]])
                    snr_mod_pairs_valid=np.array(c[indices[train_len+test_len:total_len_a]])

                else:
                    train=False
                    test_tf=True
                    if (train==True):
                        X_train=np.vstack((X_train,values[indices[0:train_len]]))
                        Y_train=np.append(Y_train,b[indices[0:train_len]],axis=0)
                        snr_mod_pairs_train=np.append(snr_mod_pairs_train, c[indices[0:train_len]],axis=0)

                        
                        X_valid=np.vstack((X_valid, values[indices[train_len+test_len:total_len_a]]))
                        Y_valid=np.append(Y_valid,b[indices[train_len+test_len:total_len_a]],axis=0)
                        snr_mod_pairs_valid=np.append(snr_mod_pairs_valid, c[indices[train_len+test_len:total_len_a]],axis=0)

                          
        f.close()

    snrs.sort()
    mods.sort()
    nsps.sort()
    scos.sort()

    indices=np.arange(len(X_train))
    np.random.seed(100000)
    np.random.shuffle(indices)
    
    X_train=np.array(X_train[indices])
    Y_train=np.array(Y_train[indices])
    snr_mod_pairs_train=np.array(snr_mod_pairs_train[indices])
    #print(X_train)
    #print(Y_train)

    
    indices=np.arange(len(X_test))
    np.random.seed(100000)
    np.random.shuffle(indices)
    
    X_test=np.array(X_test[indices])
    Y_test=np.array(Y_test[indices])
    snr_mod_pairs_test=np.array(snr_mod_pairs_test[indices])

    indices=np.arange(len(X_valid))
    np.random.seed(100000)
    np.random.shuffle(indices)
    
    X_valid=np.array(X_valid[indices])
    Y_valid=np.array(Y_valid[indices])
    snr_mod_pairs_valid=np.array(snr_mod_pairs_valid[indices])


    
    print("SNR values are ",snrs)
    print("Modulation formats are ",mods)
    print("nsps are ",nsps)
    print("scos are ",scos)

    print("\n\nComplete datasets have shapes such:")
    print("Input dataset: ",X.shape)
    print("Output dataset: ", Y.shape)
    print("SNR Modulation pairs: ",snr_mod_pairs.shape)

    print("\n\nTrain datasets have shapes such:")
    print("Train Input datasets: ",X_train.shape)
    print("Train Output datasets: ", Y_train.shape)
    print("Train SNR Modulation pairs: ", snr_mod_pairs_train.shape)

    print("\n\nTest datasets have shapes such: ")
    print("Test Input datasets: ",X_test.shape)
    print("Test Output datasets: ",Y_test.shape)
    print("Test SNR Modualtion pairs: ",snr_mod_pairs_test.shape)

    print("\n\nValid datasets have shapes such: ")
    print("Valid Input datasets: ",X_valid.shape)
    print("Valid Output datasets: ",Y_valid.shape)
    print("Valid SNR Modualtion pairs: ",snr_mod_pairs_valid.shape)

    print("\n"*2,"*"*10,"Train/test dataset split - Done","*"*10,"\n"*2)



def max_norm():
    global X_test,X_test_rec, X_train,X_valid,Train,amp_train,phase_train,amp_test,phase_test, amp_valid,phase_valid
    print("Normalizing1")
    print(X_train.shape)
    for i in range(X_test.shape[0]):
        sample=X_test[i]
        max_val = max(max(np.abs(sample[0,:])), max(np.abs(sample[1,:])))
        X_test[i] = sample/max_val
    print(X_test.shape)

    print("Normalizing2")



def to_one_hot(yy):
    #print (yy)
    yy=list(yy)
    #print(yy)
    yy1=np.zeros([len(yy),max(yy)+1])
    yy1[np.arange(len(yy)),yy]=1
    return yy1


def encode_labels():
    global Y_train,Y_test,Y_valid, snr_mod_pairs_test,snr_mod_pairs_train,mods
    print("\n"*2,"*"*10,"Label binary encoding - Start","*"*10,"\n"*2)
    print("Doing...")

    Y_test=to_one_hot(map(lambda x:mods.index(x),Y_test))
    
    print("\n","*"*10,"Label binary encoding - Done","*"*10,"\n"*2)

def transform_input():
    global X_test, X_train,X_valid,Train,amp_train,phase_train,amp_test,phase_test, amp_valid,phase_valid
    print ("\n"*2,"*"*10,"Input dataset transformation-Start","*"*10,"\n"*2)
    print ("Doing...")
    train=False
    transform=False
    if (train==True and transform==True):
        print("we do transform")
        for sample in X_train:

            for i in range(0,sample.shape[1]):
                
                sig_amp=LA.norm([sample[0][i],sample[1][i]])
                sig_phase=np.arctan2(sample[1][i],sample[0][i])/np.pi
                sample[0][i]=sig_amp
                sample[1][i]=sig_phase


    if (transform==True):   
        for sample in X_test:
            
            for i in range(0,sample.shape[1]):
                sig_amp=LA.norm([sample[0][i],sample[1][i]])
                sig_phase=np.arctan2(sample[1][i],sample[0][i])/np.pi
                sample[0][i]=sig_amp
                sample[1][i]=sig_phase

    if (train==True and transform == True):
        print("we do transform") 

        for sample in X_valid:
            
            for i in range(0,sample.shape[1]):
                sig_amp=LA.norm([sample[0][i],sample[1][i]])
                sig_phase=np.arctan2(sample[1][i],sample[0][i])/np.pi
                sample[0][i]=sig_amp
                sample[1][i]=sig_phase

    print("\n\nInput datasets after transformation have shapes such:")
    print("Train Input datasets: ",X_train.shape)
    print("Test Input datasets: ",X_test.shape)
    print("Valid Input datasets: ",X_valid.shape)


    X_test=np.transpose(X_test,(0,2,1))
    
    print("after transpose")
    print("Train Input datasets: ",X_train.shape)
    print("Test Input datasets: ",X_test.shape)

    X_test=np.reshape(X_test,(-1,N_samples,2,1))
    

    #print(amp_train)
    #print(phase_train)

    print("after reshape ")
    print("Train Input datasets: ",X_train.shape)
    print("Test Input datasets: ",X_test.shape)
    print("Valid Input datasets: ",X_valid.shape)




    print("\n"*2,"*"*10,"Input dataset transformation-Done","*"*10,"\n"*2)



################################Main Program##########################################
#1. step - load dataset, split to train and test datasets, encode labels as binary vectors
parse = True
if parse:
    gen_data()
    encode_labels()
    max_norm()
    transform_input()
    np.save("X_test.npy",X_test)
    np.save("Y_test.npy",Y_test)
    np.save("mods.npy",mods)
    np.save("snrs.npy",snrs)
    np.save("nsps.npy",nsps)
    np.save("scos.npy",scos)
    np.save("snr_mod_pairs_test.npy",snr_mod_pairs_test)
    