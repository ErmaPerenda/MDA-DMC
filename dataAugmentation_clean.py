# -*- coding: utf-8 -*-
import numpy as np
import h5py as h5py
import math
import random


train=True
N_samples=128
no_augm=1
thr_prob=0.5
seed_SNR=18


#dataset_name='../dataset1024.mat'
#dataset_name='../dataset1024_new.mat'
#dataset_name='../dataset1024_rayleigh_up.mat'
#dataset_name='../dataset1024_rician_up.mat'
#dataset_name='../dataset1024_rayleigh_fs_1000.mat'
#dataset_name='../dataset1024_rician_fs_200.mat'
#dataset_name='../dataset1024_noenergynormiq.mat'
#dataset_name='../dataset1024_noenergynormiq_rayleigh8.mat'
#dataset_name='../dataset1024_noenergynormiq_rici8.mat'
#dataset_names=['../dataset1024_noenergynormiq_awgn8.mat']
#dataset_name='../dataset1024_iqnoenergy_rayLsTest.mat'
#dataset_names=['../dataset128_noenergynormiq_awgn4_baseline.mat']
#dataset_names=['../dataset1024_noenergynormiq_awgn8.mat','../dataset1024_noenergynormiq_rayleigh8.mat']
#dataset_names=['../dataset128_noenergynormiq_awgn4_baseline.mat','../dataset128_noenergynormiq_awgn4_baseline.mat']
#dataset_names=['../../dataset1024_iqenergy_awgn_sps4_new_augm.mat','../../dataset1024_iqenergy_awgn_sps4_new_augm.mat']

#dataset_names=['big_awgn.mat','big_awgn.mat']#,'../../../dataset1024_iqenergy_awgn_sps4_new_augm_with_rician.mat']
#dataset_names=['../../baseline_dataset1024_iqenergy_awgn_sps4_1xaugm_only18.mat','../../dataset1024_iqenergy_awgn_sps4_new_augm_with_rician.mat']
dataset_names=['baseline_ds.mat','baseline_ds.mat']

X=np.array([])
Y=np.array([])

idx_train=[]
idx_valid=[]

X_train=np.array([])
X_train_new=np.array([])
X_train_all=np.array([])
X_train_src=np.array([])
X_train_weak=np.array([])
X_train_strong=np.array([])
X_train_medium=np.array([])


X_train_paired_1=np.array([])
X_train_1=np.array([])
X_train_2=np.array([])
X_train_paired_2=np.array([])
Y_train_paired=np.array([])
Y_train_paired_1=np.array([])

X_valid_1=np.array([])
X_valid_2=np.array([])
X_valid_paired_1=np.array([])
X_valid_paired_2=np.array([])
Y_valid_paired=np.array([])
Y_valid_paired_1=np.array([])


Y_train=np.array([])
snr_mod_pairs_train=np.array([])
Y_train_new=np.array([])
snr_mod_pairs_train_new=np.array([])
Y_train_all=np.array([])
snr_mod_pairs_train_all=np.array([])

X_anchor=np.array([])
Y_anchor=np.array([])
Y_negative=np.array([])
X_anchor_org=np.array([])
X_negative_org=np.array([])
X_negative=np.array([])
X_positive=np.array([])

X_anchor_valid=np.array([])
Y_anchor_valid=np.array([])
Y_negative_valid=np.array([])
X_anchor_org_valid=np.array([])
X_negative_org_valid=np.array([])
X_negative_valid=np.array([])
X_positive_valid=np.array([])


X_test=np.array([])
Y_test=np.array([])
snr_mod_pairs_test=np.array([])


X_valid=np.array([])
X_valid_weak=np.array([])
X_valid_strong=np.array([])
X_valid_medium=np.array([])
X_valid_new=np.array([])
X_valid_all=np.array([])
X_valid_src=np.array([])
Y_valid=np.array([])
Y_valid_new=np.array([])
Y_valid_all=np.array([])
snr_mod_pairs_valid=np.array([])
snr_mod_pairs_valid_new=np.array([])
snr_mod_pairs_valid_all=np.array([])

snrs=[] #unique values of snr values
mods=[] #unique values of modulation formats
number_transmissions=[]
snr_mod_pairs=np.array([])
power_ratios=[]
n_ffts=[]
nsps=[]
bws=[]


split_factor_train=0.8 #split datasets to training and testing data
split_factor_valid=0.1

def gen_data():
    global X, Y, X_test, X_train, Y_train,Y_test, Y_valid, X_valid, snrs, mods, snr_mod_pairs, split_factor
    global snr_mod_pairs_test, snr_mod_pairs_train,Train, datasets_name, labels_name
    global power_ratios, train, bws, n_ffts, nsps, nsps_target, n_fft_target
    global snr_mod_pairs_valid
    global N_samples,seed_SNR
    global X_train_weak, X_train_strong
    global X_valid_weak, X_valid_strong

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
            #   print("it is ",mod)
            #    continue


            snr=int(pom[1].replace('neg','-'))
            l=pom[2]
            freq=str(pom[3])
            m=pom[4]
            nt=int(pom[5])
            nt=0

            sps=round(float(l)/float(m),3)

            #if mod.find('BPSK')<0 and mod.find('QPSK')<0:
            #    continue

            #if (mod.find('BPSK')<0 and mod.find('BFM')<0 and mod.find('CPFSK')<0 and mod.find('DSBAM')<0
            #    and mod.find('SSBAM')<0 and mod.find('QAM16')<0 and mod.find('QAM64')<0
            #    and mod.find('PSK8')<0 and mod.find('PAM4')<0 and mod.find('QPSK')<0 and mod.find('GFSK')<0  and mod.find('FSK')<0 and mod.find('BAM')<0
            #    ):
            #    continue
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



            values=np.array(ds.get(key))
            shape=values.shape
            #print(shape)
            #print("values ", values[0])
            total_len_a=shape[2]
            values_pom=np.transpose(values,(2,1,0))
            #print("values ", values[0])
            
            #print("after tran ",values.shape)

            
            values=values_pom[:,0:2,0:N_samples]
            values_weak=values_pom[:,6:8,0:N_samples]
            values_strong=values_pom[:,8:10,0:N_samples]
            values_weak_rician=values_pom[:,10:12,0:N_samples]
            values_strong_rician=values_pom[:,12:14,0:N_samples]

            #cumulants_val=values_pom[:,3,1:12]
            #print("after tran ",values.shape)
            #print("after tran ",values_weak.shape)
            #print("after tran ",values_strong.shape)
            
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
                split_factor_train=0.8 #split datasets to training and testing data
                split_factor_valid=0.1
                total_len_a=100
            
            indices=np.arange(total_len_a)
            np.random.seed(10000)
            np.random.shuffle(indices)

            b=np.full((total_len_a,1),mod)
            c=np.full((total_len_a,4),[mod,snr,sps,nt])

            #indices=indices[0:330]
            #print("length of indices ", len(indices))

            

            train_len=int(round(split_factor_train*total_len_a))
            valid_len=int(round(split_factor_valid*total_len_a))
            test_len=int(total_len_a-(train_len+valid_len))

            #print("total len is ", total_len_a)
            #print ("train len ", train_len)
            #print ("test len ", test_len)
            #print("valid len ",valid_len)

            if train_len>total_len_a:
                print("Split factor cannot be higher than 1")
                exit()

            if ctrl_ds== 2:
                print("it is 2")
                if X_test.size == 0:
                    X_test=np.array(values[indices[train_len:train_len+test_len]])
                    Y_test=np.array(b[indices[train_len:train_len+test_len]])
                    snr_mod_pairs_test=np.array(c[indices[train_len:train_len+test_len]])

                    #cumulants_test=np.array(cumulants_val[indices[train_len:train_len+test_len]])
                    
                else:
                    #cumulants_test=np.vstack((cumulants_test, cumulants_val[indices[train_len:train_len+test_len]]))
                    
                    X_test=np.vstack((X_test, values[indices[train_len:train_len+test_len]]))
                    Y_test=np.append(Y_test,b[indices[train_len:train_len+test_len]],axis=0)
                    snr_mod_pairs_test=np.append(snr_mod_pairs_test, c[indices[train_len:train_len+test_len]],axis=0)



            if ctrl_ds==1 and (snr==seed_SNR):
                #print("it is 1")
                print("mod is ",mod, "snr ", snr)

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
                    train=True
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

    indices=np.arange(len(X_train))
    np.random.seed(100000)
    np.random.shuffle(indices)
    
    X_train=np.array(X_train[indices])
    Y_train=np.array(Y_train[indices])
    snr_mod_pairs_train=np.array(snr_mod_pairs_train[indices])
    

    
    indices=np.arange(len(X_test))
    np.random.seed(100000)
    np.random.shuffle(indices)
    
    X_test=np.array(X_test[indices])
    Y_test=np.array(Y_test[indices])
    snr_mod_pairs_test=np.array(snr_mod_pairs_test[indices])
    #cumulants_test=np.array(cumulants_test[indices])

    indices=np.arange(len(X_valid))
    np.random.seed(100000)
    np.random.shuffle(indices)
    
    X_valid=np.array(X_valid[indices])
    Y_valid=np.array(Y_valid[indices])
    snr_mod_pairs_valid=np.array(snr_mod_pairs_valid[indices])
    #cumulants_valid=np.array(cumulants_valid[indices])


    
    print("SNR values are ",snrs)
    print("Modulation formats are ",mods)
    print("nsps are ",nsps)

    print("\n\nComplete datasets have shapes such:")
    print("Input dataset: ",X.shape)
    print("Output dataset: ", Y.shape)
    print("SNR Modulation pairs: ",snr_mod_pairs.shape)

    print("\n\nTrain datasets have shapes such:")
    print("Train Input datasets: ",X_train.shape)
    print("Train Output datasets: ", Y_train.shape)
    print("Train SNR Modulation pairs: ", snr_mod_pairs_train.shape)
    #print("Train cumulants datasets: ", cumulants_train.shape)

    print("\n\nTest datasets have shapes such: ")
    print("Test Input datasets: ",X_test.shape)
    print("Test Output datasets: ",Y_test.shape)
    print("Test SNR Modualtion pairs: ",snr_mod_pairs_test.shape)
    #print("Test cumulants datasets: ", cumulants_test.shape)

    print("\n\nValid datasets have shapes such: ")
    print("Valid Input datasets: ",X_valid.shape)
    print("Valid Output datasets: ",Y_valid.shape)
    print("Valid SNR Modualtion pairs: ",snr_mod_pairs_valid.shape)
    #print("Valid cumulants datasets: ", cumulants_valid.shape)
    print("mods are ",mods)
    print("sps are ",nsps)

    print("\n"*2,"*"*10,"Train/test dataset split - Done","*"*10,"\n"*2)


def to_one_hot(yy):
    #print (yy)
    yy=list(yy)
    #print(yy)
    yy1=np.zeros([len(yy),max(yy)+1])
    yy1[np.arange(len(yy)),yy]=1
    return yy1

def max_norm():
    global X_test,X_test_rec, X_train,X_valid,Train,amp_train,phase_train,amp_test,phase_test, amp_valid,phase_valid
    print("Normalizing1")
    print(X_train.shape)
    for i in range(X_train.shape[0]):
        sample=X_train[i]
        max_val = max(max(np.abs(sample[0,:])), max(np.abs(sample[1,:])))
        X_train[i] = sample/max_val
    print(X_train.shape)

    print("Normalizing2")

    for i in range(X_valid.shape[0]):
        sample=X_valid[i]
        max_val = max(max(np.abs(sample[0,:])), max(np.abs(sample[1,:])))
        X_valid[i] = sample/max_val

    for i in range(X_test.shape[0]):
        sample=X_test[i]
        max_val = max(max(np.abs(sample[0,:])), max(np.abs(sample[1,:])))
        X_test[i] = sample/max_val


def encode_labels():
    global Y_train,Y_test,Y_valid, snr_mod_pairs_test,snr_mod_pairs_train,mods
    print("\n"*2,"*"*10,"Label binary encoding - Start","*"*10,"\n"*2)
    print("Doing...")

    # onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    # Y_train = onehot_encoder.fit_transform(Y_train[..., np.newaxis])
    # Y_test = onehot_encoder.fit_transform(Y_test[..., np.newaxis])
    # Y_valid = onehot_encoder.fit_transform(Y_valid[..., np.newaxis])
    Y_train=to_one_hot(map(lambda x:mods.index(x),Y_train))
    Y_valid=to_one_hot(map(lambda x:mods.index(x),Y_valid))
    Y_test=to_one_hot(map(lambda x:mods.index(x),Y_test))

    print("\n","*"*10,"Label binary encoding - Done","*"*10,"\n"*2)

def transform_input():
    global X_test, X_train,X_valid,Train,amp_train,phase_train,amp_test,phase_test, amp_valid,phase_valid
    global X_train_strong, X_train_weak, X_valid_weak, X_valid_strong
    print ("\n"*2,"*"*10,"Input dataset transformation-Start","*"*10,"\n"*2)
    print ("Doing...")
    
    print("\n\nInput datasets after transformation have shapes such:")
    print("Train Input datasets: ",X_train.shape)
    print("Test Input datasets: ",X_test.shape)
    print("Valid Input datasets: ",X_valid.shape)


    X_train=np.transpose(X_train,(0,2,1))
    X_valid=np.transpose(X_valid,(0,2,1))
    X_test=np.transpose(X_test,(0,2,1))
    
    print("after transpose")
    print("Train Input datasets: ",X_train.shape)
    print("Test Input datasets: ",X_test.shape)


    X_train=np.reshape(X_train,(-1,N_samples,2,1))
    X_valid=np.reshape(X_valid,(-1,N_samples,2,1))
    X_test=np.reshape(X_test,(-1,N_samples,2,1))


    #print(amp_train)
    #print(phase_train)

    print("after reshape ")
    print("Train Input datasets: ",X_train.shape)
    print("Test Input datasets: ",X_test.shape)
    print("Valid Input datasets: ",X_valid.shape)




    print("\n"*2,"*"*10,"Input dataset transformation-Done","*"*10,"\n"*2)


def time_plot(x,y):
    global N_samples

    plot_sample=x[:,:,0]
    plot_sample_2=y[:,:,0]
    print("sample shape is ",plot_sample.shape)
    i_1=plot_sample[:,0]
    q_1=plot_sample[:,1]

    i_2=plot_sample_2[:,0]
    q_2=plot_sample_2[:,1]

    t=np.arange(N_samples)

    plt.figure(figsize=(15,10))
    plt.plot(t,i_1,color='blue',label='no noise')
    plt.plot(t,i_2,color='red',label='noise')
    plt.legend(loc=4, ncol=1,prop=dict(weight='bold',size='12.5'),labelspacing=0.8,numpoints=1)
    plt.ylabel('In-phase')
    plt.xlabel('Time')
    plt.title('Source I/Q'); plt.xlabel('time lags');
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig("./images/i_time.png")

    plt.figure(figsize=(15,10))
    plt.plot(t,q_1,color='blue',label='no noise')
    plt.plot(t,q_2,color='red', label='noise')
    plt.legend(loc=4, ncol=1,prop=dict(weight='bold',size='12.5'),labelspacing=0.8,numpoints=1)
    plt.ylabel('Quadrature')
    plt.xlabel('Time')
    plt.title('Source I/Q'); plt.xlabel('time lags');
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig("./images/q_time.png")


def imagegrid(x,y):
    plt.figure(figsize=(15,10))
    plot_sample=x[:,:,0]
    print("sample shape is ",plot_sample.shape)

    for point in plot_sample:
        print("point shape ",point.shape)
        sig_i=point[0]
        sig_q=point[1]

        plt.plot(sig_i,sig_q,'o',color='blue')
    plt.ylabel('Quadrature')
    plt.xlabel('In-phase')
    plt.title('Source I/Q'); plt.xlabel('time lags');
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig("./images/source_amp_phase.png")
    plt.figure(figsize=(15,10))
    plot_sample=y[:,:,0]
    for point in plot_sample:
        sig_i=point[0]
        sig_q=point[1]
        plt.plot(sig_i,sig_q,'o',color='red')

    plt.ylabel('Quadrature')
    plt.xlabel('In-phase')
    plt.title('Reconstructed I/Q'); plt.xlabel('time lags');
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig("./images/rec_amp_phase.png")


def add_noise(x,level):
    global N_samples
    snr_db=level
    x_c=x[:,0,0]+1j*x[:,1,0]
    
    
    signal_power=(1/N_samples)*np.sum(np.power(np.abs(x_c),2))
    #print("signal power is ",signal_power)
    

    snr_wat=10**(snr_db/10)
    no=signal_power/snr_wat
    std_val=math.sqrt((no/2))
    
    gnoise=std_val*(np.random.randn(N_samples)+1j*np.random.randn(N_samples))
    #print('gnoise shape ',gnoise.shape)

    y=x_c+gnoise

    y_i=np.real(y)
    y_q=np.imag(y)

    y=np.zeros(shape=x.shape)
    y[:,0,0]=y_i
    y[:,1,0]=y_q
    
    
    max_val = max(max(np.abs(y[:,0,0])), max(np.abs(y[:,1,0])))
    #print("max val is ",max_val)
    y= y/max_val
    

    return y

def radial_shift(x,level):
    global N_samples

    #print("x shape is ",x.shape)
    y=np.zeros(shape=x.shape)
    #step=level/(200*10**3)
    step=level
    i=1
    for p in x:
        #print("pshape is ",p.shape)
        theta=step*i
        rotation=np.zeros(shape=(2,2))
        rotation[0][0] = math.cos(math.radians(theta))
        rotation[0][1] = -math.sin(math.radians(theta))
        rotation[1][0] = math.sin(math.radians(theta))
        rotation[1][1] = math.cos(math.radians(theta))

        p_new=np.matmul(rotation,p)
        #print("p new shape is ",p_new.shape)
        y[i-1]=p_new

        i=i+1

    #print("x new shape is ",y.shape)

    max_val = max(max(np.abs(y[:,0,0])), max(np.abs(y[:,1,0])))
    #print("max val is ",max_val)
    y= y/max_val

    #print("rotation done ",level)
    #time_plot(x,y)
    #imagegrid(x,y)
    return y 


def rotate(x,level):
    global N_samples

    theta=level

    #print("chosen value is ", theta)
    rotation=np.zeros(shape=(2,2))
    rotation[0][0] = math.cos(math.radians(theta))
    rotation[0][1] = -math.sin(math.radians(theta))
    rotation[1][0] = math.sin(math.radians(theta))
    rotation[1][1] = math.cos(math.radians(theta))

    #print("rotation is ",rotation)

    y=np.matmul(rotation,x)

    max_val = max(max(np.abs(y[:,0,0])), max(np.abs(y[:,1,0])))
    #print("max val is ",max_val)
    y= y/max_val

    #print("rotation done ",level)
    #time_plot(x,y)
    #imagegrid(x,y)
    return y 

def stretch_sample(x,dx):
    #print("iq imbalance ",level)
    global N_samples

    rotation=np.zeros(shape=(2,2))
    if dx>0:
        rotation[0][0] =dx
    else:
        rotation[0][0] =1

    rotation[0][1] = 0
    rotation[1][0] =0
    if dx<0:
        rotation[1][1] = np.abs(dx)
    else:
        rotation[1][1]=1

    #print("rotation is ",rotation)

    y=np.matmul(rotation,x) 

    max_val = max(max(np.abs(y[:,0,0])), max(np.abs(y[:,1,0])))
    #print("max val is ",max_val)
    y= y/max_val
    
    return y 


def data_augmentation_radial_shift():
    global X_train, X_valid, snr_mod_pairs_train, snr_mod_pairs_valid
    global no_augm, mods, snrs
    global X_train_all, X_train_src, X_valid_all, X_valid_src
    global Y_train_all, Y_valid_all
    global snr_mod_pairs_valid_all,snr_mod_pairs_train_all

    X_train_all_new=X_train_all
    X_train_src_new=X_train_src
    Y_train_all_new=Y_train_all
    snr_mod_pairs_train_all_new=snr_mod_pairs_train_all

    X_valid_all_new=X_valid_all
    X_valid_src_new=X_valid_src
    Y_valid_all_new=Y_valid_all
    snr_mod_pairs_valid_all_new=snr_mod_pairs_valid_all

    i=0
    print("we do data augmenation ")
    step_range=list(range(-400,401,20))
    #step_range_2=list(range(-200,-20,20))
    #step_range_3=list(range(400,200,-40))
    #step_range_5=list(range(200,20,-20))
    #step_range_4=list(range(-20,21,5))

    if 0 in step_range:
        step_range.remove(0)
    else:
        print("zero is not there")
    #step_range=step_range+step_range_4+step_range_3+step_range_5+step_range_2
    
    step_range=np.array(step_range)
    step_range=step_range/10.0
    step_range=list(step_range)
    step_range.sort()
    #snr_range=[20]
    print("step ranges are ",step_range)
    print("step len is ", len(step_range))
    
    
    for mod in mods:
        for snr in snrs:
            indices_src=[]
            indices_noise=[]
            indices_rotation=[]
            indices_stretch=[]
            i=0
            j=0
            for snr_mod in snr_mod_pairs_train_all:
                #print("snr mod is ", snr_mod)
                if (snr_mod[1] == str(snr) and  snr_mod[0]== mod):
                    if int(snr_mod[3])==0:
                        indices_src.append(i)
                    if int(snr_mod[3])==1 or int(snr_mod[3])==0:
                        indices_noise.append(i)
                    if int(snr_mod[3])==2:
                        indices_rotation.append(i)
                    if int(snr_mod[3])==3:
                        indices_stretch.append(i)
                    
                i=i+1

            print("Total number src data for ", mod, "and snr ", snr, " is ", len(indices_src))
            print("Total number noise data for ", mod, "and snr ", snr, " is ", len(indices_noise))
            print("Total number rotate data for ", mod, "and snr ", snr, " is ", len(indices_rotation))
            print("Total number stretch data for ", mod, "and snr ", snr, " is ", len(indices_stretch))
            

            m=3
            for step in step_range:
                i_src=[]
                i_noise=[]
                i_rotation=[]
                i_stretch=[]

                if len(indices_src)>0:
                    i_src=random.choices(indices_src,k=m)

                if len(indices_noise)>0 :
                    i_noise=random.choices(indices_noise,k=m)

                if len(indices_rotation)>0:
                    i_rotation=random.choices(indices_rotation,k=m)

                if len(indices_stretch)>0:
                    i_stretch=random.choices(indices_stretch,k=m)
                    
                i_src=[]
                if len(i_src)<0:
                    print("it is radial shift")
                    x_random=i_src[0:4]+i_noise[0:4]+i_rotation[0:4] #+i_stretch[0:4]
                    x_random_valid=i_src[4:5]+i_noise[4:5]+i_rotation[4:5]#+i_stretch[4]
                else:
                    x_random=i_noise[0:(m-1)]+i_rotation[0:(m-1)]+i_stretch[0:(m-1)]
                    x_random_valid=i_noise[(m-1):m]+i_rotation[(m-1):m]+i_stretch[(m-1):m]
                    #print("x random is ",x_random)

                for i in x_random:
                    snr_mod=snr_mod_pairs_train_all[i]
                    snr_mod[3]=4
                    sample=X_train_all[i]
                    sample_src=X_train_src[i]
                    weak_sample=radial_shift(sample,step)

                    X_train_all_new=np.vstack((X_train_all_new,[weak_sample]))
                    X_train_src_new=np.vstack((X_train_src_new,[sample_src]))
                    Y_train_all_new=np.append(Y_train_all_new,[Y_train_all[i]], axis=0)
                    snr_mod_pairs_train_all_new=np.append(snr_mod_pairs_train_all_new, [snr_mod],axis=0)

                for i in x_random_valid:
                    snr_mod=snr_mod_pairs_train_all[i]
                    snr_mod[3]=4
                    sample=X_train_all[i]
                    sample_src=X_train_src[i]
                    weak_sample=radial_shift(sample,step)

                    X_valid_all_new=np.vstack((X_valid_all_new,[weak_sample]))
                    X_valid_src_new=np.vstack((X_valid_src_new,[sample_src]))
                    Y_valid_all_new=np.append(Y_valid_all_new,[Y_train_all[i]], axis=0)
                    snr_mod_pairs_valid_all_new=np.append(snr_mod_pairs_valid_all_new, [snr_mod],axis=0)


    X_train_all=X_train_all_new
    Y_train_all=Y_train_all_new
    X_train_src=X_train_src_new
    snr_mod_pairs_train_all=snr_mod_pairs_train_all_new

    indices=np.arange(len(X_train_all))
    np.random.seed(150000)
    np.random.shuffle(indices)

    X_train_all=np.array(X_train_all[indices])
    X_train_src=np.array(X_train_src[indices])
    Y_train_all=np.array(Y_train_all[indices])
    snr_mod_pairs_train_all=np.array(snr_mod_pairs_train_all[indices])

    
    X_valid_all=X_valid_all_new
    Y_valid_all=Y_valid_all_new
    X_valid_src=X_valid_src_new
    snr_mod_pairs_valid_all=snr_mod_pairs_valid_all_new

    indices=np.arange(len(X_valid_all))
    np.random.seed(50000)
    np.random.shuffle(indices)
    
    X_valid_all=np.array(X_valid_all[indices])
    Y_valid_all=np.array(Y_valid_all[indices])
    X_valid_src=np.array(X_valid_src[indices])
    snr_mod_pairs_valid_all=np.array(snr_mod_pairs_valid_all[indices])

    print("\n\nTrain datasets have shapes such:")
    print("Train SNR Modulation pairs all: ", snr_mod_pairs_train_all.shape)
    print("Train X_train allw datasets: ", X_train_all.shape)

    print("Valid SNR Modualtion pairs all: ",snr_mod_pairs_valid_all.shape)
    print("Valid all datasets: ", X_valid_all.shape)

    print("\n"*2,"*"*10,"Train/test dataset split - Done","*"*10,"\n"*2)

def data_augmentation_stretch():
    global X_train, X_valid, snr_mod_pairs_train, snr_mod_pairs_valid
    global no_augm, mods, snrs,seed_SNR
    global X_train_all, X_train_src, X_valid_all, X_valid_src
    global Y_train_all, Y_valid_all
    global snr_mod_pairs_valid_all,snr_mod_pairs_train_all
    X_train_all_new=X_train_all
    X_train_src_new=X_train_src
    Y_train_all_new=Y_train_all
    snr_mod_pairs_train_all_new=snr_mod_pairs_train_all

    X_valid_all_new=X_valid_all
    X_valid_src_new=X_valid_src
    Y_valid_all_new=Y_valid_all
    snr_mod_pairs_valid_all_new=snr_mod_pairs_valid_all

    i=0
    print("we do data augmenation ")
    a_step=4
    dxs=list(range(-40,-10,a_step))
    dxs2=list(range(40,10,-a_step))
    #a_step=2

    #dxs3=list(range(-20,-10,a_step))
    #dxs4=list(range(20,10,-a_step))


    #i_0=dxs2.index(10)
    if a_step==4:
        dxs2.append(10+a_step/4.0)
        dxs2.append(-10-(a_step/4.0))
    else:
        dxs2.append(10+a_step/2.0)
        dxs2.append(-10-(a_step/2.0))

    
    dxs=dxs+dxs2#+dxs3+dxs4
    
    dxs_a=np.array(dxs)
    dxs_a=dxs_a/10.0
    dxs=list(dxs_a)
    dys=[0]
    dxs.sort()
    dys.sort()
    print("dxs are ",dxs)
    print("len dxs is ",len(dxs))
    
    
    for mod in mods:
        for snr in snrs:
            indices_src=[]
            indices_noise=[]
            indices_rotation=[]
            indices_radial_shift=[]
            i=0
            j=0
            for snr_mod in snr_mod_pairs_train_all:
                #print("snr mod is ", snr_mod)
                if (snr_mod[1] == str(snr) and  snr_mod[0]== mod):
                    if int(snr_mod[3])==0:
                        indices_src.append(i)
                    if int(snr_mod[3])==1 or int(snr_mod[3])==0:
                        indices_noise.append(i)
                    if int(snr_mod[3])==2:
                        indices_rotation.append(i)
                    if int(snr_mod[3])==4:
                        indices_radial_shift.append(i)
                    
                i=i+1

            print("Total number src data for ", mod, "and snr ", snr, " is ", len(indices_src))
            print("Total number noise data for ", mod, "and snr ", snr, " is ", len(indices_noise))
            print("Total number rotate data for ", mod, "and snr ", snr, " is ", len(indices_rotation))
            print("Total number radial shift data for ", mod, "and snr ", snr, " is ", len(indices_radial_shift))
            

            m=3
            for dx in dxs:
                for dy in dys:
                    i_src=[]
                    i_noise=[]
                    i_rotation=[]
                    i_radial_shift=[]
                    if len(indices_src)>0:
                        i_src=random.choices(indices_src,k=m)
                    if len(indices_noise)>0:
                        i_noise=random.choices(indices_noise,k=m)

                    if len(indices_rotation)>0:
                        i_rotation=random.choices(indices_rotation,k=m)

                    if len(indices_radial_shift)>0:
                        i_radial_shift=random.choices(indices_radial_shift,k=m)
                    
                    i_src=[]
                    if len(i_src)<0:
                        print("it is stretch")
                        x_random=i_src[0:4]+i_noise[0:4]+i_rotation[0:4]+i_radial_shift[0:4]
                        x_random_valid=i_src[4:5]+i_noise[4:5]+i_rotation[4:5]+i_radial_shift[4:5]
                    else:
                        x_random=i_noise[0:(m-1)]+i_rotation[0:(m-1)]+i_radial_shift[0:(m-1)]
                        x_random_valid=i_noise[(m-1):m]+i_rotation[(m-1):m]+i_radial_shift[(m-1):m]

                    

                    for i in x_random:
                        snr_mod=snr_mod_pairs_train_all[i]
                        snr_mod[3]=3
                        sample=X_train_all[i]
                        sample_src=X_train_src[i]
                        weak_sample=stretch_sample(sample,dx)

                        X_train_all_new=np.vstack((X_train_all_new,[weak_sample]))
                        X_train_src_new=np.vstack((X_train_src_new,[sample_src]))
                        Y_train_all_new=np.append(Y_train_all_new,[Y_train_all[i]], axis=0)
                        snr_mod_pairs_train_all_new=np.append(snr_mod_pairs_train_all_new, [snr_mod],axis=0)

                    for i in x_random_valid:
                        snr_mod=snr_mod_pairs_train_all[i]
                        snr_mod[3]=3
                        sample=X_train_all[i]
                        sample_src=X_train_src[i]
                        weak_sample=stretch_sample(sample,dx)

                        X_valid_all_new=np.vstack((X_valid_all_new,[weak_sample]))
                        X_valid_src_new=np.vstack((X_valid_src_new,[sample_src]))
                        Y_valid_all_new=np.append(Y_valid_all_new,[Y_train_all[i]], axis=0)
                        snr_mod_pairs_valid_all_new=np.append(snr_mod_pairs_valid_all_new, [snr_mod],axis=0)

    X_train_all=X_train_all_new
    Y_train_all=Y_train_all_new
    X_train_src=X_train_src_new
    snr_mod_pairs_train_all=snr_mod_pairs_train_all_new

    indices=np.arange(len(X_train_all))
    np.random.seed(150000)
    np.random.shuffle(indices)

    X_train_all=np.array(X_train_all[indices])
    X_train_src=np.array(X_train_src[indices])
    Y_train_all=np.array(Y_train_all[indices])
    snr_mod_pairs_train_all=np.array(snr_mod_pairs_train_all[indices])

    
    X_valid_all=X_valid_all_new
    Y_valid_all=Y_valid_all_new
    X_valid_src=X_valid_src_new
    snr_mod_pairs_valid_all=snr_mod_pairs_valid_all_new

    indices=np.arange(len(X_valid_all))
    np.random.seed(50000)
    np.random.shuffle(indices)
    
    X_valid_all=np.array(X_valid_all[indices])
    Y_valid_all=np.array(Y_valid_all[indices])
    X_valid_src=np.array(X_valid_src[indices])
    snr_mod_pairs_valid_all=np.array(snr_mod_pairs_valid_all[indices])

    print("\n\nTrain datasets have shapes such:")
    print("Train SNR Modulation pairs all: ", snr_mod_pairs_train_all.shape)
    print("Train X_train allw datasets: ", X_train_all.shape)

    print("Valid SNR Modualtion pairs all: ",snr_mod_pairs_valid_all.shape)
    print("Valid all datasets: ", X_valid_all.shape)

    print("\n"*2,"*"*10,"Train/test dataset split - Done","*"*10,"\n"*2)


def data_augmentation_rotation_first():
    global X_train, X_valid, snr_mod_pairs_train, snr_mod_pairs_valid
    global no_augm, mods, snrs,seed_SNR
    global X_train_all, X_train_src, X_valid_all, X_valid_src
    global Y_train_all, Y_valid_all
    global snr_mod_pairs_valid_all,snr_mod_pairs_train_all
    
    print("we do data augmenation ")
    angle_range=list(range(-180,180,10))
    angle_range.remove(0)
    #snr_range=[20]
    print("angle ranges are ",angle_range)
    print("angle len is ", len(angle_range))
    i=0

    for sample in X_train:
        snr_mod=snr_mod_pairs_train[i]
        snr_mod[3]=0
        if X_train_all.size==0:
            X_train_all=np.array([sample])
            X_train_src=np.array([sample])
            Y_train_all=np.array([Y_train[i]])
            snr_mod_pairs_train_all=np.array([snr_mod])
        else:
            p=1
            X_train_all=np.vstack((X_train_all,[sample]))
            X_train_src=np.vstack((X_train_src,[sample]))
            Y_train_all=np.append(Y_train_all,[Y_train[i]], axis=0)
            snr_mod_pairs_train_all=np.append(snr_mod_pairs_train_all, [snr_mod],axis=0)
        i=i+1
    
    i=0
    for sample in X_valid:
        if X_valid_all.size==0:
            X_valid_all=np.array([sample])
            X_valid_src=np.array([sample])
            Y_valid_all=np.array([Y_valid[i]])
            snr_mod_pairs_valid_all=np.array([snr_mod_pairs_valid[i]])
        else:
            p=1
            X_valid_all=np.vstack((X_valid_all,[sample]))
            X_valid_src=np.vstack((X_valid_src,[sample]))
            Y_valid_all=np.append(Y_valid_all,[Y_valid[i]], axis=0)
            snr_mod_pairs_valid_all=np.append(snr_mod_pairs_valid_all, [snr_mod_pairs_valid[i]],axis=0)
        i=i+1
    
    print("before rot")
    print("X train all size ",X_train_all.shape)
    print("X_valid all size ",X_valid_all.shape)
    for mod in mods:
        indices=[]
        i=0
        j=0
        for snr_mod in snr_mod_pairs_train:
            if (snr_mod[0]== mod):
                indices.append(i)
            i=i+1

        print("Total number data for ", mod," is ", len(indices))
        if len(indices) == 0:
            print("continue")
            continue 

        for angle in angle_range:
            x_random=random.choices(indices,k=2)
            for i in x_random:
                snr_mod=snr_mod_pairs_train[i]
                snr_mod[3]=1
                sample=X_train[i]
                sample_src=X_train[i]
                weak_sample=rotate(sample,angle)
                X_train_all=np.vstack((X_train_all,[weak_sample]))
                X_train_src=np.vstack((X_train_src,[sample_src]))
                Y_train_all=np.append(Y_train_all,[Y_train[i]], axis=0)
                snr_mod_pairs_train_all=np.append(snr_mod_pairs_train_all, [snr_mod],axis=0)
                

    for mod in mods:
        indices=[]
        i=0
        j=0
        for snr_mod in snr_mod_pairs_valid:
            if (snr_mod[0]== mod):
                indices.append(i)
            i=i+1

        print("Total number data for ", mod, " is ", len(indices))
        if len(indices) == 0:
            print("continue")
            continue 

        for angle in angle_range:
            x_random=random.choices(indices,k=1)
            for i in x_random:
                snr_mod=snr_mod_pairs_valid[i]
                snr_mod[3]=1
                sample=X_valid[i]
                sample_src=X_valid[i]
                weak_sample=rotate(sample,angle)

                X_valid_all=np.vstack((X_valid_all,[weak_sample]))
                X_valid_src=np.vstack((X_valid_src,[sample_src]))
                Y_valid_all=np.append(Y_valid_all,[Y_valid[i]], axis=0)
                snr_mod_pairs_valid_all=np.append(snr_mod_pairs_valid_all, [snr_mod],axis=0)
                

    indices=np.arange(len(X_train_all))
    np.random.seed(150000)
    np.random.shuffle(indices)

    X_train_all=np.array(X_train_all[indices])
    X_train_src=np.array(X_train_src[indices])
    Y_train_all=np.array(Y_train_all[indices])
    snr_mod_pairs_train_all=np.array(snr_mod_pairs_train_all[indices])

    
    indices=np.arange(len(X_valid_all))
    np.random.seed(50000)
    np.random.shuffle(indices)
    
    X_valid_all=np.array(X_valid_all[indices])
    Y_valid_all=np.array(Y_valid_all[indices])
    X_valid_src=np.array(X_valid_src[indices])
    snr_mod_pairs_valid_all=np.array(snr_mod_pairs_valid_all[indices])


    print("\n\nTrain datasets have shapes after rotation such:")
    print("Train SNR Modulation pairs all: ", snr_mod_pairs_train_all.shape)
    print("Train X_train allw datasets: ", X_train_all.shape)

    print("Valid SNR Modualtion pairs all: ",snr_mod_pairs_valid_all.shape)
    print("Valid all datasets: ", X_valid_all.shape)

    print("\n"*2,"*"*10,"Train/test dataset split - Done","*"*10,"\n"*2)

def data_augmentation_rotation_new():
    global X_train, X_valid, snr_mod_pairs_train, snr_mod_pairs_valid
    global no_augm, mods, snrs,seed_SNR
    global X_train_all, X_train_src, X_valid_all, X_valid_src
    global Y_train_all, Y_valid_all
    global snr_mod_pairs_valid_all,snr_mod_pairs_train_all
    X_train_all_new=X_train_all
    X_train_src_new=X_train_src
    Y_train_all_new=Y_train_all
    snr_mod_pairs_train_all_new=snr_mod_pairs_train_all

    X_valid_all_new=X_valid_all
    X_valid_src_new=X_valid_src
    Y_valid_all_new=Y_valid_all
    snr_mod_pairs_valid_all_new=snr_mod_pairs_valid_all

    i=0
    print("we do data augmenation ")
    angle_range=list(range(-180,180,10))
    #angle_range=list(range(-90,90,5))
    angle_range.remove(0)
    #snr_range=[20]
    print("angle ranges are ",angle_range)
    print("angle len is ", len(angle_range))

    for mod in mods:
        for snr in snrs:
            indices_src=[]
            indices_noise=[]
            indices_stretch=[]
            indices_radial_shift=[]
            i=0
            j=0
            for snr_mod in snr_mod_pairs_train_all:
                #print("snr mod is ", snr_mod)
                if (snr_mod[1] == str(snr) and  snr_mod[0]== mod):
                    if int(snr_mod[3])==0:
                        indices_src.append(i)
                    if int(snr_mod[3])==1:
                        indices_noise.append(i)
                    if int(snr_mod[3])==3:
                        indices_stretch.append(i)
                    if int(snr_mod[3])==4:
                        indices_radial_shift.append(i)
                    
                i=i+1

            print("Total number src data for ", mod, "and snr ", snr, " is ", len(indices_src))
            print("Total number noise data for ", mod, "and snr ", snr, " is ", len(indices_noise))
            print("Total number stretch data for ", mod, "and snr ", snr, " is ", len(indices_stretch))
            print("Total number radial shift data for ", mod, "and snr ", snr, " is ", len(indices_radial_shift))
            
            m=3
            for angle in angle_range:
                i_src=[]
                i_stretch=[]
                i_noise=[]
                i_radial_shift=[]
                if len(indices_src)>0:
                    i_src=random.choices(indices_src,k=m)

                if len(indices_noise)>0:
                    i_noise=random.choices(indices_noise,k=m)

                if len(indices_stretch)>0:
                    i_stretch=random.choices(indices_stretch,k=m)

                if len(indices_radial_shift)>0:
                    i_radial_shift=random.choices(indices_radial_shift,k=m)

                if int(snr)==seed_SNR:
                    i_noise=i_src
                    i_src=[]

                if len(i_src)>0:
                    #print("it is rotation")
                    x_random=i_src[0:(m-1)]+i_noise[0:(m-1)]#+i_stretch[0:4]+i_radial_shift[0:4]
                    x_random_valid=i_src[(m-1):m]+i_noise[(m-1):m] #+i_stretch[4]+i_radial_shift[4]
                else:
                    x_random=i_noise[0:(m-1)]+i_stretch[0:(m-1)]+i_radial_shift[0:(m-1)]
                    x_random_valid=i_noise[(m-1):m] +i_stretch[(m-1):m]+i_radial_shift[(m-1):m]
                
                for i in x_random:
                    snr_mod=snr_mod_pairs_train_all[i]
                    snr_mod[3]=2
                    sample=X_train_all[i]
                    sample_src=X_train_src[i]
                    weak_sample=rotate(sample,angle)

                    X_train_all_new=np.vstack((X_train_all_new,[weak_sample]))
                    X_train_src_new=np.vstack((X_train_src_new,[sample_src]))
                    Y_train_all_new=np.append(Y_train_all_new,[Y_train_all[i]], axis=0)
                    snr_mod_pairs_train_all_new=np.append(snr_mod_pairs_train_all_new, [snr_mod],axis=0)
                
                for i in x_random_valid:
                    snr_mod=snr_mod_pairs_train_all[i]
                    snr_mod[3]=2
                    sample=X_train_all[i]
                    sample_src=X_train_src[i]
                    weak_sample=rotate(sample,angle)

                    X_valid_all_new=np.vstack((X_valid_all_new,[weak_sample]))
                    X_valid_src_new=np.vstack((X_valid_src_new,[sample_src]))
                    Y_valid_all_new=np.append(Y_valid_all_new,[Y_train_all[i]], axis=0)
                    snr_mod_pairs_valid_all_new=np.append(snr_mod_pairs_valid_all_new, [snr_mod],axis=0)
                
                




    X_train_all=X_train_all_new
    Y_train_all=Y_train_all_new
    X_train_src=X_train_src_new
    snr_mod_pairs_train_all=snr_mod_pairs_train_all_new

    indices=np.arange(len(X_train_all))
    np.random.seed(150000)
    np.random.shuffle(indices)

    X_train_all=np.array(X_train_all[indices])
    X_train_src=np.array(X_train_src[indices])
    Y_train_all=np.array(Y_train_all[indices])
    snr_mod_pairs_train_all=np.array(snr_mod_pairs_train_all[indices])

    X_valid_all=X_valid_all_new
    Y_valid_all=Y_valid_all_new
    X_valid_src=X_valid_src_new
    snr_mod_pairs_valid_all=snr_mod_pairs_valid_all_new

    indices=np.arange(len(X_valid_all))
    np.random.seed(50000)
    np.random.shuffle(indices)
    
    X_valid_all=np.array(X_valid_all[indices])
    Y_valid_all=np.array(Y_valid_all[indices])
    X_valid_src=np.array(X_valid_src[indices])
    snr_mod_pairs_valid_all=np.array(snr_mod_pairs_valid_all[indices])

    print("\n\nTrain datasets have shapes such:")
    print("Train SNR Modulation pairs all: ", snr_mod_pairs_train_all.shape)
    print("Train X_train allw datasets: ", X_train_all.shape)

    print("Valid SNR Modualtion pairs all: ",snr_mod_pairs_valid_all.shape)
    print("Valid all datasets: ", X_valid_all.shape)

    print("\n"*2,"*"*10,"Train/test dataset split - Done","*"*10,"\n"*2)


def data_augmentation_addition_rician():
    global X_train, X_valid, snr_mod_pairs_train, snr_mod_pairs_valid
    global no_augm, mods, snrs,seed_SNR
    global X_train_all, X_train_src, X_valid_all, X_valid_src
    global Y_train_all, Y_valid_all
    global snr_mod_pairs_valid_all,snr_mod_pairs_train_all
    X_train_all_new=X_train_all
    X_train_src_new=X_train_src
    Y_train_all_new=Y_train_all
    snr_mod_pairs_train_all_new=snr_mod_pairs_train_all

    X_valid_all_new=X_valid_all
    X_valid_src_new=X_valid_src
    Y_valid_all_new=Y_valid_all
    snr_mod_pairs_valid_all_new=snr_mod_pairs_valid_all
    
    print("we do data augmenation ")
    for mod in mods:
        for snr in snrs:
            indices_rotation=[]

            i=0
            j=0
            for snr_mod in snr_mod_pairs_train_all:
                #print("snr mod is ", snr_mod)
                if (snr_mod[1] == str(snr) and  snr_mod[0]== mod):
                    if int(snr_mod[3])==2:
                        indices_rotation.append(i)
                    
                i=i+1

            print("Total number rotation data for ", mod, "and snr ", snr, " is ", len(indices_rotation))
            

            i_rotation=random.choices(indices_rotation,k=10)

            x_random=i_rotation[0:8]
            x_random_valid=i_rotation[8:10]

            for i in x_random:
                snr_mod=snr_mod_pairs_train_all[i]
                snr_mod[3]=5
                sample=X_train_all[i]
                sample_src=X_train_src[i]

                y=np.add(sample_src,sample)

                max_val = max(max(np.abs(y[:,0,0])), max(np.abs(y[:,1,0])))
                weak_sample= y/max_val

                X_train_all_new=np.vstack((X_train_all_new,[weak_sample]))
                X_train_src_new=np.vstack((X_train_src_new,[sample_src]))
                Y_train_all_new=np.append(Y_train_all_new,[Y_train_all[i]], axis=0)
                snr_mod_pairs_train_all_new=np.append(snr_mod_pairs_train_all_new, [snr_mod],axis=0)

            for i in x_random_valid:
                snr_mod=snr_mod_pairs_train_all[i]
                snr_mod[3]=5
                sample=X_train_all[i]
                sample_src=X_train_src[i]

                y=np.add(sample_src,sample)

                max_val = max(max(np.abs(y[:,0,0])), max(np.abs(y[:,1,0])))
                weak_sample= y/max_val

                X_valid_all_new=np.vstack((X_valid_all_new,[weak_sample]))
                X_valid_src_new=np.vstack((X_valid_src_new,[sample_src]))
                Y_valid_all_new=np.append(Y_valid_all_new,[Y_train_all[i]], axis=0)
                snr_mod_pairs_valid_all_new=np.append(snr_mod_pairs_valid_all_new, [snr_mod],axis=0)

    X_train_all=X_train_all_new
    Y_train_all=Y_train_all_new
    X_train_src=X_train_src_new
    snr_mod_pairs_train_all=snr_mod_pairs_train_all_new

    indices=np.arange(len(X_train_all))
    np.random.seed(150000)
    np.random.shuffle(indices)

    X_train_all=np.array(X_train_all[indices])
    X_train_src=np.array(X_train_src[indices])
    Y_train_all=np.array(Y_train_all[indices])
    snr_mod_pairs_train_all=np.array(snr_mod_pairs_train_all[indices])

    
    X_valid_all=X_valid_all_new
    Y_valid_all=Y_valid_all_new
    X_valid_src=X_valid_src_new
    snr_mod_pairs_valid_all=snr_mod_pairs_valid_all_new

    indices=np.arange(len(X_valid_all))
    np.random.seed(50000)
    np.random.shuffle(indices)
    
    X_valid_all=np.array(X_valid_all[indices])
    Y_valid_all=np.array(Y_valid_all[indices])
    X_valid_src=np.array(X_valid_src[indices])
    snr_mod_pairs_valid_all=np.array(snr_mod_pairs_valid_all[indices])

    print("\n\nTrain datasets have shapes such:")
    print("Train SNR Modulation pairs all: ", snr_mod_pairs_train_all.shape)
    print("Train X_train allw datasets: ", X_train_all.shape)

    print("Valid SNR Modualtion pairs all: ",snr_mod_pairs_valid_all.shape)
    print("Valid all datasets: ", X_valid_all.shape)

    print("\n"*2,"*"*10,"Train/test dataset split - Done","*"*10,"\n"*2)

def data_augmentation_second():
    global X_train, X_valid, snr_mod_pairs_train, snr_mod_pairs_valid
    global no_augm, mods, snrs,seed_SNR
    global X_train_all, X_train_src, X_valid_all, X_valid_src
    global Y_train_all, Y_valid_all
    global snr_mod_pairs_valid_all,snr_mod_pairs_train_all
    X_train_all_new=X_train_all
    X_train_src_new=X_train_src
    Y_train_all_new=Y_train_all
    snr_mod_pairs_train_all_new=snr_mod_pairs_train_all

    X_valid_all_new=X_valid_all
    X_valid_src_new=X_valid_src
    Y_valid_all_new=Y_valid_all
    snr_mod_pairs_valid_all_new=snr_mod_pairs_valid_all

    i=0
    print("we do data augmenation ")
    snr_range=list(range(-6,22,2))
    #snr_range=[20]
    print("snr ranges are ",snr_range)

    for sample in X_train_all:

        for k in range(no_augm):
            #print("k is ",k)
            for snr_val in snr_range:
                if snr_val==seed_SNR:
                    continue
                sample_src=X_train_src[i]
                weak_sample=add_noise(sample,snr_val)
                snr_mod=snr_mod_pairs_train_all[i]
                snr_mod[1]=snr_val

                X_train_all_new=np.vstack((X_train_all_new,[weak_sample]))
                X_train_src_new=np.vstack((X_train_src_new,[sample_src]))
                Y_train_all_new=np.append(Y_train_all_new,[Y_train_all[i]], axis=0)
                snr_mod_pairs_train_all_new=np.append(snr_mod_pairs_train_all_new, [snr_mod],axis=0)
        i=i+1

    i=0
    for sample in X_valid_all:

        for k in range(no_augm):
            #print("k is ",k)
            for snr_val in snr_range:
                if snr_val==seed_SNR:
                    continue
                sample_src=X_valid_src[i]
                weak_sample=add_noise(sample,snr_val)
                snr_mod=snr_mod_pairs_valid_all[i]
                snr_mod[1]=snr_val

                X_valid_all_new=np.vstack((X_valid_all_new,[weak_sample]))
                X_valid_src_new=np.vstack((X_valid_src_new,[sample_src]))
                Y_valid_all_new=np.append(Y_valid_all_new,[Y_valid_all[i]], axis=0)
                snr_mod_pairs_valid_all_new=np.append(snr_mod_pairs_valid_all_new, [snr_mod],axis=0)
        i=i+1



    

    X_train_all=X_train_all_new
    Y_train_all=Y_train_all_new
    X_train_src=X_train_src_new
    snr_mod_pairs_train_all=snr_mod_pairs_train_all_new

    indices=np.arange(len(X_train_all))
    np.random.seed(150000)
    np.random.shuffle(indices)

    X_train_all=np.array(X_train_all[indices])
    X_train_src=np.array(X_train_src[indices])
    Y_train_all=np.array(Y_train_all[indices])
    snr_mod_pairs_train_all=np.array(snr_mod_pairs_train_all[indices])

    X_valid_all=X_valid_all_new
    Y_valid_all=Y_valid_all_new
    X_valid_src=X_valid_src_new
    snr_mod_pairs_valid_all=snr_mod_pairs_valid_all_new

    
    indices=np.arange(len(X_valid_all))
    np.random.seed(50000)
    np.random.shuffle(indices)
    
    X_valid_all=np.array(X_valid_all[indices])
    Y_valid_all=np.array(Y_valid_all[indices])
    X_valid_src=np.array(X_valid_src[indices])
    snr_mod_pairs_valid_all=np.array(snr_mod_pairs_valid_all[indices])


    print("\n\nTrain datasets have shapes such:")
    print("Train SNR Modulation pairs all: ", snr_mod_pairs_train_all.shape)
    print("Train X_train allw datasets: ", X_train_all.shape)

    print("Valid SNR Modualtion pairs all: ",snr_mod_pairs_valid_all.shape)
    print("Valid all datasets: ", X_valid_all.shape)

    print("\n"*2,"*"*10,"Train/test dataset split - Done","*"*10,"\n"*2)

def data_augmentation():
    global X_train, X_valid, snr_mod_pairs_train, snr_mod_pairs_valid
    global no_augm,seed_SNR
    global X_train_all, X_train_src, X_valid_all, X_valid_src
    global Y_train_all, Y_valid_all
    global snr_mod_pairs_valid_all,snr_mod_pairs_train_all
    i=0
    print("we do data augmenation ")
    snr_range=list(range(-6,22,2))
    #snr_range=[20]
    print("snr ranges are ",snr_range)

    for sample in X_train:

        if X_train_all.size==0:
            X_train_all=np.array([sample])
            X_train_src=np.array([sample])
            Y_train_all=np.array([Y_train[i]])
            snr_mod_pairs_train_all=np.array([snr_mod_pairs_train[i]])
        else:
            p=1
            X_train_all=np.vstack((X_train_all,[sample]))
            X_train_src=np.vstack((X_train_src,[sample]))
            Y_train_all=np.append(Y_train_all,[Y_train[i]], axis=0)
            snr_mod_pairs_train_all=np.append(snr_mod_pairs_train_all, [snr_mod_pairs_train[i]],axis=0)

        for k in range(no_augm):
            #print("k is ",k)
            for snr_val in snr_range:
                if snr_val==seed_SNR and k==0:
                    continue
                weak_sample=add_noise(sample,snr_val)
                snr_mod=snr_mod_pairs_train[i]
                snr_mod[1]=snr_val
                if snr_val==seed_SNR:
                    snr_mod[3]=0
                else:
                    snr_mod[3]=1

                X_train_all=np.vstack((X_train_all,[weak_sample]))
                X_train_src=np.vstack((X_train_src,[sample]))
                Y_train_all=np.append(Y_train_all,[Y_train[i]], axis=0)
                snr_mod_pairs_train_all=np.append(snr_mod_pairs_train_all, [snr_mod],axis=0)
       
        i=i+1
    i=0
    for sample in X_valid:

        if X_valid_all.size==0:
            X_valid_all=np.array([sample])
            X_valid_src=np.array([sample])
            Y_valid_all=np.array([Y_valid[i]])
            snr_mod_pairs_valid_all=np.array([snr_mod_pairs_valid[i]])
        else:
            p=1
            X_valid_all=np.vstack((X_valid_all,[sample]))
            X_valid_src=np.vstack((X_valid_src,[sample]))
            Y_valid_all=np.append(Y_valid_all,[Y_valid[i]], axis=0)
            snr_mod_pairs_valid_all=np.append(snr_mod_pairs_valid_all, [snr_mod_pairs_valid[i]],axis=0)

        for k in range(no_augm):
            #print("k is ",k)
            for snr_val in snr_range:
                if snr_val==seed_SNR and k==0:
                    continue
                weak_sample=add_noise(sample,snr_val)
                snr_mod=snr_mod_pairs_valid[i]
                snr_mod[1]=snr_val
                if snr_val==seed_SNR:
                    snr_mod[3]=0
                else:
                    snr_mod[3]=1

                X_valid_all=np.vstack((X_valid_all,[weak_sample]))
                X_valid_src=np.vstack((X_valid_src,[sample]))
                Y_valid_all=np.append(Y_valid_all,[Y_valid[i]], axis=0)
                snr_mod_pairs_valid_all=np.append(snr_mod_pairs_valid_all, [snr_mod],axis=0)

        i=i+1

    

    indices=np.arange(len(X_train_all))
    np.random.seed(150000)
    np.random.shuffle(indices)

    X_train_all=np.array(X_train_all[indices])
    X_train_src=np.array(X_train_src[indices])
    Y_train_all=np.array(Y_train_all[indices])
    snr_mod_pairs_train_all=np.array(snr_mod_pairs_train_all[indices])

    
    indices=np.arange(len(X_valid_all))
    np.random.seed(50000)
    np.random.shuffle(indices)
    
    X_valid_all=np.array(X_valid_all[indices])
    Y_valid_all=np.array(Y_valid_all[indices])
    X_valid_src=np.array(X_valid_src[indices])
    snr_mod_pairs_valid_all=np.array(snr_mod_pairs_valid_all[indices])

    print("\n\nTrain datasets have shapes such:")
    print("Train SNR Modulation pairs all: ", snr_mod_pairs_train_all.shape)
    print("Train X_train allw datasets: ", X_train_all.shape)

    print("Valid SNR Modualtion pairs all: ",snr_mod_pairs_valid_all.shape)
    print("Valid all datasets: ", X_valid_all.shape)

    print("\n"*2,"*"*10,"Train/test dataset split - Done","*"*10,"\n"*2)


def data_stats():
    global X_train, X_valid, snr_mod_pairs_train, snr_mod_pairs_valid
    global no_augm, mods, snrs,seed_SNR
    global X_train_all, X_train_src, X_valid_all, X_valid_src
    global Y_train_all, Y_valid_all
    global snr_mod_pairs_valid_all,snr_mod_pairs_train_all

    for mod in mods:
        for snr in snrs:
            indices_src=[]
            indices_noise=[]
            indices_stretch=[]
            indices_radial_shift=[]
            indices_rotation=[]
            i=0
            j=0
            for snr_mod in snr_mod_pairs_train_all:
                #print("snr mod is ", snr_mod)
                if (snr_mod[1] == str(snr) and  snr_mod[0]== mod):
                    if int(snr_mod[3])==0:
                        indices_src.append(i)
                    if int(snr_mod[3])==1:
                        indices_noise.append(i)
                    if int(snr_mod[3])==2:
                        indices_rotation.append(i)
                    if int(snr_mod[3])==3:
                        indices_stretch.append(i)
                    if int(snr_mod[3])==4:
                        indices_radial_shift.append(i)
                    
                i=i+1

            print("Total number src data for ", mod, "and snr ", snr, " is ", len(indices_src))
            print("Total number noise data for ", mod, "and snr ", snr, " is ", len(indices_noise))
            print("Total number rotation data for ", mod, "and snr ", snr, " is ", len(indices_rotation))
            print("Total number stretch data for ", mod, "and snr ", snr, " is ", len(indices_stretch))
            print("Total number radial shift data for ", mod, "and snr ", snr, " is ", len(indices_radial_shift))

    

def get_data():
    global X_test, X_train,X_valid,Y_test,Y_valid,Y_train,mods, snrs, nsps, snr_mod_pairs_test, snr_mod_pairs_train, snr_mod_pairs_valid
    global X_train_all, X_train_src, X_valid_all, X_valid_src
    global snr_mod_pairs_valid_all, snr_mod_pairs_train_all, Y_valid_all, Y_train_all
    np.random.seed(33000)
    random.seed(33)
    
    parse = True
    paired= True
    addition = False
    if parse:
        gen_data()
        max_norm()
        encode_labels()
        transform_input()
        np.save("X_train.npy",X_train)
        np.save("Y_train.npy",Y_train)

        np.save("X_valid.npy",X_valid)
        np.save("Y_valid.npy",Y_valid)

        np.save("X_test.npy",X_test)
        np.save("Y_test.npy",Y_test)

        np.save("mods.npy",mods)
        np.save("snrs.npy",snrs)
        np.save("nsps.npy",nsps)
        np.save("snr_mod_pairs_train.npy",snr_mod_pairs_train)
        np.save("snr_mod_pairs_valid.npy",snr_mod_pairs_valid)
        np.save("snr_mod_pairs_test.npy",snr_mod_pairs_test)
        
    else:
        X_train = np.load("X_train.npy")
        Y_train = np.load("Y_train.npy")

        X_test  = np.load("X_test.npy")
        Y_test  = np.load("Y_test.npy")

        X_valid = np.load("X_valid.npy")
        Y_valid = np.load("Y_valid.npy")

        mods = np.load("mods.npy").tolist()
        snrs = np.load("snrs.npy").tolist()
        nsps = np.load("nsps.npy").tolist()
        snr_mod_pairs_test=np.load("snr_mod_pairs_test.npy")
        snr_mod_pairs_train=np.load("snr_mod_pairs_train.npy")
        snr_mod_pairs_valid=np.load("snr_mod_pairs_valid.npy")

    if paired:
        #data_augmentation_rotation_first()
        #data_augmentation_second()
        data_augmentation()
        data_augmentation_rotation_new()
        data_augmentation_stretch()
        data_augmentation_radial_shift()
        

        #data_augmentation_addition_rician()
        #data_augmentation_radial_shift()
        #data_augmentation_stretch()
        
        
        
        #data_augmentation_rot_stretch_together()
        
        np.save("X_train_all.npy",X_train_all)
        np.save("X_train_src.npy",X_train_src)
        np.save("Y_train_all.npy",Y_train_all)

        np.save("X_valid_all.npy",X_valid_all)
        np.save("X_valid_src.npy",X_valid_src)
        np.save("Y_valid_all.npy",Y_valid_all)
    
        np.save("snr_mod_pairs_train_all.npy",snr_mod_pairs_train_all)
        np.save("snr_mod_pairs_valid_all.npy",snr_mod_pairs_valid_all)
    else:
        print('else')
        X_train_all=np.load("X_train_all.npy")
        Y_train_all = np.load("Y_train_all.npy")
        X_train_src=np.load("X_train_src.npy")

        X_valid_all=np.load("X_valid_all.npy")
        X_valid_src=np.load("X_valid_src.npy")
        Y_valid_all=np.load("Y_valid_all.npy")

        snr_mod_pairs_train_all=np.load("snr_mod_pairs_train_all.npy")
        snr_mod_pairs_valid_all=np.load("snr_mod_pairs_valid_all.npy")
        data_stats()

    if addition:
        data_augmentation_radial_shift()
        np.save("X_train_all.npy",X_train_all)
        np.save("X_train_src.npy",X_train_src)
        np.save("Y_train_all.npy",Y_train_all)

        np.save("X_valid_all.npy",X_valid_all)
        np.save("X_valid_src.npy",X_valid_src)
        np.save("Y_valid_all.npy",Y_valid_all)
    
        np.save("snr_mod_pairs_train_all.npy",snr_mod_pairs_train_all)
        np.save("snr_mod_pairs_valid_all.npy",snr_mod_pairs_valid_all)
    

get_data()
