import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, Reshape,Conv2D, MaxPooling2D, AveragePooling2D, Add, Multiply,Concatenate

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import *
from tensorflow.keras.utils import *
from tensorflow.keras.regularizers import l1,l2,l1_l2

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import CSVLogger

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


import time
import random
import itertools
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Greys):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,fontsize=12)
    plt.yticks(tick_marks, classes,fontsize=12)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label',fontsize=20)
    plt.xlabel('Predicted label',fontsize=20)
    plt.tight_layout()

def cosine_distance(vects):
    x, y = vects
    x = tf.nn.l2_normalize(x, axis=-1)
    y = tf.nn.l2_normalize(y, axis=-1)

    return -tf.math.reduce_mean(x * y, axis=-1, keepdims=True)

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)


def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

def loss(margin=1):
	def contrastive_loss(y_true, y_pred):
		square_pred = tf.math.square(y_pred)
		margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
		return tf.math.reduce_mean((1 - y_true) * square_pred + (y_true) * margin_square)
	return contrastive_loss

class Evaluate():
	def __init__(self, population=None,X_train_all=None,Y_train_all=None, X_valid_all=None, Y_valid_all=None, X_test=None, Y_test=None,snr_mod_pairs_test=None, num_classes=11, N_samples=256,epochs=10, batch_size=128,max_complexity=100000):
		self.population=population
		self.epochs=epochs
		self.batch_size=batch_size
		self.num_classes=num_classes
		self.N_samples=N_samples
		self.max_complexity=max_complexity
		self.snr_mod_pairs_test=snr_mod_pairs_test

		self.X_train_all=X_train_all
		self.Y_train_all=Y_train_all
		self.X_valid_all=X_valid_all
		self.Y_valid_all=Y_valid_all
		self.X_test=X_test
		self.Y_test=Y_test

		#print("evaluate pop ",self.population)
		#print("n samples ",self.N_samples)

	def evaluate_population(self, gen_no):
		for i in range(self.population.get_pop_size()):
			indi = self.population.get_individual_at(i)
			indi.complexity=100000
			indi.accuracy=0.0
			accuracy=100.0
			try:
				complexity, accuracy = self.evaluate_individual(indi,gen_no)
				indi.complexity = complexity
				indi.accuracy = accuracy
			except:
				indi.complexity=100000
				indi.accuracy=0.0
			gama=0.3
			indi.c_prob=gama*indi.c_prob + (1-gama)*accuracy
			#indi.c_prob=1.0

	def get_activation_type(self,activation):
		#print("activation is ",activation)
		activation_type='relu'

		if activation>=0.25 and activation<0.5:
			activation_type='selu'

		if activation>=0.5 and activation<0.75:
			activation_type='tanh'

		if activation>=0.75:
			activation_type='linear'

		return activation_type

	def get_kernel_regularizer(self,kernel_reg):
		
		kernel_regularizer_type=None

		if kernel_reg>=0.25 and kernel_reg<0.5:
			kernel_regularizer_type=keras.regularizers.l1

		if kernel_reg>=0.5 and kernel_reg<0.75:
			kernel_regularizer_type=keras.regularizers.l2

		if kernel_reg>=0.75:
			kernel_regularizer_type=keras.regularizers.l1_l2

		return kernel_regularizer_type

	def build_block(self,block_layer,block_in, training=True):

		#print("building a block width ", block_layer.width)

		id_x=block_in
		identity_branch=block_layer.identity_branch

		id_x=keras.layers.Conv2D(filters=identity_branch.conv_layer.filters,
				kernel_size=(identity_branch.conv_layer.kernel_size,1),
				padding='same',trainable=training,
				activation=self.get_activation_type(identity_branch.conv_layer.activation))(id_x)


		if identity_branch.before_merge == True:
			block_in=id_x
			
		
		#print("id shape ",id_x.get_shape())
		conv_xs=""

		for i in range(block_layer.width):
			conv_x=block_in
			conv_branch=block_layer.conv_branch
			#print("width ", i)
			#j=0
			for conv in conv_branch:
				#print("depth ", j)
				#j=j+1

				conv_x=keras.layers.Conv2D(filters=conv.filters,
				kernel_size=(conv.kernel_size,1),padding='same',trainable=training,
				activation=self.get_activation_type(conv.activation))(conv_x)
			
			if i==0:
				conv_xs=conv_x
			else:
				conv_xs=keras.layers.Concatenate(axis=3)([conv_xs,conv_x])
				#print("conv xs shape ",conv_xs.get_shape())
			
			#print("conv x shape ",conv_x.get_shape())

		#print(" final conv xs shape ",conv_xs.get_shape())
		#print("id shape ",id_x.get_shape())

		zero_padding=int(id_x.get_shape()[1])-int(conv_xs.get_shape()[1])
		#print("zero padd ", zero_padding)

		if zero_padding>0:
			conv_xs=keras.layers.ZeroPadding2D(padding=((zero_padding,0),(0,0)))(conv_xs)

		if zero_padding<0:
			id_x=keras.layers.ZeroPadding2D(padding=((np.abs(zero_padding),0),(0,0)))(id_x)


		#print(" after padding final conv xs shape ",conv_xs.get_shape())
		#print("id shape ",id_x.get_shape())

		merge_function=block_layer.merge_layer.merge_function
		#0-0.33 Concat, 0.3-0.66 Multiply, 0.66-1 Add
		#block_out=conv_xs+id_x
		merge_function=0.8
		

		if merge_function<0.33:
			#print("concat")
			block_out=keras.layers.Concatenate(axis=3)([conv_xs,id_x])

		if merge_function>=0.33 and merge_function<0.66:
			#print("mult")
			block_out=keras.layers.Multiply()([conv_xs,id_x])

		if merge_function>=0.66:
			block_out=keras.layers.Add()([conv_xs,id_x])
			#print("add")

		if block_layer.pooling_layer!=None:
			if block_layer.pooling_layer.kernel_type<0.5:
				block_out=keras.layers.BatchNormalization(trainable=training)(block_out)
				block_out=keras.layers.MaxPooling2D(pool_size=(block_layer.pooling_layer.kernel_size,1),trainable=training)(block_out)
				block_out=keras.layers.Dropout(0.3,trainable=training)(block_out)
			else:
				block_out=keras.layers.BatchNormalization(trainable=training)(block_out)
				block_out=keras.layers.AveragePooling2D(pool_size=(block_layer.pooling_layer.kernel_size,1),trainable=training)(block_out)
				block_out=keras.layers.Dropout(0.3,trainable=training)(block_out)


		#print("out shape ",block_out.get_shape())


		return block_out

	def build_block_decoder(self,block_layer,block_in, training=True):

		#print("building a block width ", block_layer.width)

		id_x=block_in
		identity_branch=block_layer.identity_branch

		id_x=keras.layers.Conv2DTranspose(filters=identity_branch.conv_layer.filters,
				kernel_size=(identity_branch.conv_layer.kernel_size,1),
				padding='same',trainable=training,
				activation=self.get_activation_type(identity_branch.conv_layer.activation))(id_x)


		if identity_branch.before_merge == True:
			block_in=id_x
			
		
		#print("id shape ",id_x.get_shape())
		conv_xs=""

		for i in range(block_layer.width):
			conv_x=block_in
			conv_branch=block_layer.conv_branch
			#print("width ", i)
			#j=0
			for conv in conv_branch:
				#print("depth ", j)
				#j=j+1

				conv_x=keras.layers.Conv2DTranspose(filters=conv.filters,
				kernel_size=(conv.kernel_size,1),padding='same',trainable=training,
				activation=self.get_activation_type(conv.activation))(conv_x)
			
			if i==0:
				conv_xs=conv_x
			else:
				conv_xs=keras.layers.Concatenate(axis=3)([conv_xs,conv_x])
				#print("conv xs shape ",conv_xs.get_shape())
			
			#print("conv x shape ",conv_x.get_shape())

		#print(" final conv xs shape ",conv_xs.get_shape())
		#print("id shape ",id_x.get_shape())

		zero_padding=int(id_x.get_shape()[1])-int(conv_xs.get_shape()[1])
		#print("zero padd ", zero_padding)

		if zero_padding>0:
			conv_xs=keras.layers.ZeroPadding2D(padding=((zero_padding,0),(0,0)))(conv_xs)

		if zero_padding<0:
			id_x=keras.layers.ZeroPadding2D(padding=((np.abs(zero_padding),0),(0,0)))(id_x)


		#print(" after padding final conv xs shape ",conv_xs.get_shape())
		#print("id shape ",id_x.get_shape())

		merge_function=block_layer.merge_layer.merge_function
		#0-0.33 Concat, 0.3-0.66 Multiply, 0.66-1 Add
		#block_out=conv_xs+id_x
		merge_function=0.8
		

		if merge_function<0.33:
			#print("concat")
			block_out=keras.layers.Concatenate(axis=3)([conv_xs,id_x])

		if merge_function>=0.33 and merge_function<0.66:
			#print("mult")
			block_out=keras.layers.Multiply()([conv_xs,id_x])

		if merge_function>=0.66:
			block_out=keras.layers.Add()([conv_xs,id_x])
			#print("add")

		if block_layer.pooling_layer!=None:
			block_out=keras.layers.BatchNormalization(trainable=training)(block_out)
			block_out=keras.layers.UpSampling2D(size=(2,1))(block_out)
			block_out=keras.layers.Dropout(0.3,trainable=training)(block_out)


		#print("out shape ",block_out.get_shape())


		return block_out

	def get_feature_extractor(self,indi,training=True):

		inp_seq = keras.layers.Input(shape=(self.N_samples,2,1))

		x=inp_seq

		for i in range(indi.get_layer_size()):
			layer=indi.get_layer_at(i)

			if layer.type==0:
				x=self.build_block(layer,x,training)

			if layer.type==1:
				x=keras.layers.Conv2D(filters=layer.filters,
					kernel_size=(layer.kernel_size,1),padding='valid',trainable=training,
					activation=self.get_activation_type(layer.activation))(x)

			if layer.type==2:
				if layer.kernel_type<0.5:
					x=keras.layers.BatchNormalization(trainable=training)(x)
					x=keras.layers.MaxPooling2D(pool_size=(layer.kernel_size,1),trainable=training)(x)
					x=keras.layers.Dropout(0.3,trainable=training)(x)
				else:
					x=keras.layers.BatchNormalization(trainable=training)(x)
					x=keras.layers.AveragePooling2D(pool_size=(layer.kernel_size,1),trainable=training)(x)
					x=keras.layers.Dropout(0.3,trainable=training)(x)

			#if layer.type==3:
			#	activation_type=self.get_activation_type(layer.activation)
		
		#x=keras.layers.Dense(units=128,activation='relu', kernel_initializer='he_normal')(x)

		
		embedding_network = keras.Model(inp_seq, x,name='encoder')
		print("feature encoder")

		embedding_network.summary()
		
		return embedding_network



	
	def get_projection_head(self,indi,shape_t1):
		inp_seq = keras.layers.Input(shape=(shape_t1[1],shape_t1[2],shape_t1[3]))

		x=inp_seq

		for i in range(indi.get_layer_size()):
			layer=indi.get_layer_at(i)
			if layer.type==6:
				x=keras.layers.Flatten()(x)

			if layer.type==7:
				x=keras.layers.GlobalMaxPooling2D()(x)
			if layer.type==8:
				x=keras.layers.GlobalAveragePooling2D()(x)

			if layer.type==3:
				activation_type=self.get_activation_type(layer.activation)
				x=keras.layers.Dense(units=layer.units,activation=activation_type)(x)
				x = keras.layers.Dropout(0.3)(x)

			if layer.type==10:
				break
		projection_head= keras.Model(inp_seq, x,name='projection_head')
		
		return projection_head


	def build_model(self,indi):
		#print(indi)


		embedding_network = self.get_feature_extractor(indi,training=True)

		input_1 =  keras.layers.Input(shape=(self.N_samples,2,1), name='x_1')
		
		
		tower_1 = embedding_network(input_1)
		
		shape_t1=tower_1.get_shape()
		

		#classifier 
		
		x=tower_1
		class_part=False
		for i in range(indi.get_layer_size()):
			layer=indi.get_layer_at(i)
			if layer.type==6:
				x=keras.layers.Flatten()(x)

			if layer.type==7:
				x=keras.layers.GlobalMaxPooling2D()(x)
			if layer.type==8:
				x=keras.layers.GlobalAveragePooling2D()(x)

			if layer.type!=10 and class_part==False:
				continue

			else:
				class_part=True

			if layer.type==3:
				activation_type=self.get_activation_type(layer.activation)
				x=keras.layers.Dense(units=layer.units,activation=activation_type,kernel_initializer='he_normal')(x)
				x = keras.layers.Dropout(0.3)(x)

		amc_out=keras.layers.Dense(self.num_classes,activation='softmax', name='amc_out')(x)

		classifier= keras.Model(input_1, amc_out)

		


		#model = keras.Model(inputs=[input_1, input_2,left_hoc,right_hoc], outputs=output_layer)


		classifier.summary()



		return classifier,classifier

			

	def evaluate_best(self,indi,X_train_all, Y_train_all, X_test, Y_test,X_valid_all,Y_valid_all, snr_mod_pairs_test, snrs, mods,gen_no=100):
		print("Evaluating the best individual....")
		print("snr mod pairs", snr_mod_pairs_test.shape)

		random.seed(33)
		os.environ['PYTHONHASHSEED'] = str(33)
		#print(tf.__version__)
		session_conf = tf.compat.v1.ConfigProto(
			intra_op_parallelism_threads=1, 
			inter_op_parallelism_threads=1)

		sess = tf.compat.v1.Session(
			graph=tf.compat.v1.get_default_graph(), 
			config=session_conf)
		tf.compat.v1.keras.backend.set_session(sess)

		es = EarlyStopping(patience=5, verbose=1, min_delta=0.001, monitor='val_loss', mode='auto')

		model,classifier=self.build_model(indi)


		if indi.learning_rate<0.5:
			opt=keras.optimizers.Adam(learning_rate=0.001)
		elif indi.learning_rate>=0.5 and indi.learning_rate<0.75:
			opt=keras.optimizers.Adam(learning_rate=0.01)
		else:
			opt=keras.optimizers.Adam(learning_rate=0.0001)

		sum_weights=indi.alpha+indi.beta+indi.gama
		indi.alpha=1
		indi.beta=0.5
		indi.gama=0.5
		reduce_lr=keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2,
                              patience=5, min_lr=0.0001)

		model.compile(optimizer='adam',
              loss='categorical_crossentropy',metrics=['accuracy'])

		#model.compile(loss=loss(margin=indi.margin),optimizer=opt,metrics=['accuracy'])
		
		model.summary()
		trainable_count = np.sum([K.count_params(w) for w in classifier.trainable_weights])
		non_trainable_count = np.sum([K.count_params(w) for w in classifier.non_trainable_weights])
		print('Total params: {:,}'.format(trainable_count + non_trainable_count))
		print('Trainable params: {:,}'.format(trainable_count))
		print('Non-trainable params: {:,}'.format(non_trainable_count))
		complexity=trainable_count+non_trainable_count
		

		tb_log_dir = './l_log' 
		
		save_dir = './models/'
		if not os.path.exists(save_dir):
			os.mkdir(save_dir)

		ckpt = ModelCheckpoint(save_dir+'/pure_class_model_{epoch:02d}.h5', monitor='val_accuracy',
		verbose=0, period=1, save_best_only=True, mode='min', save_weights_only=True)#period=args.save_every

		csv_logger = CSVLogger(save_dir+'/pure_cnn_dn_log.csv', append=True, separator=',')

		filepath = save_dir + 'pure_auto_amc_loss_50epoch.wts.h5'
		filepath_cl = save_dir + 'pure_cl_joint_auto_amc_loss_50epoch.wts.h5'
		train=True
		sco_ev=True


		#dot_img_file = '/home/eperenda/multiple_signals/infocom2023/clsr_testing/model_1.png'
		#tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

		if train:
			model.fit({'x_1': self.X_train_all},
			{'amc_out': self.Y_train_all}, validation_data=({'x_1':self.X_valid_all}, {'amc_out':self.Y_valid_all}),
			callbacks = [reduce_lr, keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min',save_weights_only=True), csv_logger, keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=0,mode='auto')],
			epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1)#50

		model.load_weights(filepath)


		times_s={}
		acc={}
		
		for snr in snrs:
			print ("Predicting for SNR of ",snr," \n"*2)
			indices=[]
			i=0
			j=0
			for snr_mod in snr_mod_pairs_test:
				if (snr_mod[1] == str(snr)):
					indices.append(i)
				i=i+1

			print("Total number test data is ", len(indices))

			if len(indices) == 0:
				print("continue")
				continue

			X_test_1=X_test[indices]
			Y_test_1=Y_test[indices]

			start = time.time()
			y_pred=model.predict(X_test_1)
			end = time.time()
			period=end-start
			times_s[snr]=float(period)
			y_el=[]
			y_pred_el=[]
			for i in range(1,len(y_pred)):
				y_pred_el.append(y_pred[i-1].argmax())

			for i in range(1,len(y_pred)):
				y_el.append(Y_test_1[i-1].argmax())

			cnf_matrix=confusion_matrix(y_el, y_pred_el)
			cor=np.trace(cnf_matrix)
			cor_new=np.sum(np.diag(cnf_matrix))
			sum_all=np.sum(cnf_matrix)
			acc[snr]=float(cor)/float(sum_all)

			
			# Plot normalized confusion matrix
			#plt.figure(figsize = (12,10))
			#plot_confusion_matrix(cnf_matrix,classes=mods, normalize=True)
			#plt.savefig("./images/augm_bigds_awng_snr_"+str(snr)+".png")

			

		print("\noverall accuracy is ",acc, " \n\n")

		f = open("best_ind_acc.txt", "a")
		f.write("Best individual ")
		f.write("Acc is  {} ".format(acc))
		f.close()

		X_test=np.load("../X_test_mix.npy")
		Y_test=np.load("../Y_test_mix.npy")
		snr_mod_pairs_test=np.load("../snr_mod_pairs_test_mix.npy")


		times_s={}
		acc={}
		for snr in snrs:
			#print ("Predicting for SNR of ",snr," \n"*2)
			indices=[]
			i=0
			j=0
			for snr_mod in snr_mod_pairs_test:
				if (snr_mod[1] == str(snr)):
					indices.append(i)
				i=i+1

			#print("Total number test data is ", len(indices))

			if len(indices) == 0:
				print("continue")
				continue

			X_test_1=X_test[indices]
			Y_test_1=Y_test[indices]

			start = time.time()
			y_pred=model.predict(X_test_1)
			end = time.time()
			period=end-start
			times_s[snr]=float(period)
			y_el=[]
			y_pred_el=[]
			for i in range(1,len(y_pred)):
				y_pred_el.append(y_pred[i-1].argmax())

			for i in range(1,len(y_pred)):
				y_el.append(Y_test_1[i-1].argmax())

			cnf_matrix=confusion_matrix(y_el, y_pred_el)
			cor=np.trace(cnf_matrix)
			cor_new=np.sum(np.diag(cnf_matrix))
			sum_all=np.sum(cnf_matrix)
			acc[snr]=float(cor)/float(sum_all)


		print("\nMix overall accuracy is ",acc, " \n\n")

		exit()



		X_test=np.load("../ray/X_test.npy")
		Y_test=np.load("../ray/Y_test.npy")
		snr_mod_pairs_test=np.load("../ray/snr_mod_pairs_test.npy")


		times_s={}
		acc={}
		snrs=[10]
		for snr in snrs:
			#print ("Predicting for SNR of ",snr," \n"*2)
			indices=[]
			i=0
			j=0
			for snr_mod in snr_mod_pairs_test:
				if (snr_mod[1] == str(snr)):
					indices.append(i)
				i=i+1

			#print("Total number test data is ", len(indices))

			if len(indices) == 0:
				print("continue")
				continue

			X_test_1=X_test[indices]
			Y_test_1=Y_test[indices]

			start = time.time()
			y_pred=model.predict(X_test_1)
			end = time.time()
			period=end-start
			times_s[snr]=float(period)
			y_el=[]
			y_pred_el=[]
			for i in range(1,len(y_pred)):
				y_pred_el.append(y_pred[i-1].argmax())

			for i in range(1,len(y_pred)):
				y_el.append(Y_test_1[i-1].argmax())

			cnf_matrix=confusion_matrix(y_el, y_pred_el)
			cor=np.trace(cnf_matrix)
			cor_new=np.sum(np.diag(cnf_matrix))
			sum_all=np.sum(cnf_matrix)
			acc[snr]=float(cor)/float(sum_all)

			# Plot normalized confusion matrix
			plt.figure(figsize = (12,10))
			plot_confusion_matrix(cnf_matrix,classes=mods, normalize=True)
			plt.savefig("./images/augm_bigds_ray_snr_"+str(snr)+".png")

		print("\nRayleigh overall accuracy is ",acc, " \n\n")

		X_test=np.load("../rici/X_test.npy")
		Y_test=np.load("../rici/Y_test.npy")
		snr_mod_pairs_test=np.load("../rici/snr_mod_pairs_test.npy")


		times_s={}
		acc={}
		snrs=[10]
		for snr in snrs:
			#print ("Predicting for SNR of ",snr," \n"*2)
			indices=[]
			i=0
			j=0
			for snr_mod in snr_mod_pairs_test:
				if (snr_mod[1] == str(snr)):
					indices.append(i)
				i=i+1

			#print("Total number test data is ", len(indices))

			if len(indices) == 0:
				print("continue")
				continue

			X_test_1=X_test[indices]
			Y_test_1=Y_test[indices]

			start = time.time()
			y_pred=model.predict(X_test_1)
			end = time.time()
			period=end-start
			times_s[snr]=float(period)
			y_el=[]
			y_pred_el=[]
			for i in range(1,len(y_pred)):
				y_pred_el.append(y_pred[i-1].argmax())

			for i in range(1,len(y_pred)):
				y_el.append(Y_test_1[i-1].argmax())

			cnf_matrix=confusion_matrix(y_el, y_pred_el)
			cor=np.trace(cnf_matrix)
			cor_new=np.sum(np.diag(cnf_matrix))
			sum_all=np.sum(cnf_matrix)
			acc[snr]=float(cor)/float(sum_all)

			# Plot normalized confusion matrix
			plt.figure(figsize = (12,10))
			plot_confusion_matrix(cnf_matrix,classes=mods, normalize=True)
			plt.savefig("./images/augm_bigds_rici_snr_"+str(snr)+".png")

		print("\nRician overall accuracy is ",acc, " \n\n")
		
		
		if sco_ev==False:
			return

		scos=np.load("../scos/scos.npy")
		cfos=np.load("../cfos/cfos.npy")
		a_imbs=np.load("../iqimbalance/a_imbs.npy")
		ph_imbs=np.load("../iqimbalance/ph_imbs.npy")

		X_test=np.load("../scos/X_test.npy")
		Y_test=np.load("../scos/Y_test.npy")
		snr_mod_pairs_test=np.load("../scos/snr_mod_pairs_test.npy")
		snrs=[0,10,18]
		snrs=[18]
		
		for snr in snrs:
			times_s={}
			acc={}
			for sco in scos:
				#print ("Predicting for SNR of ",snr," and sco of ",sco," \n"*2)
				indices=[]
				i=0
				j=0
				for snr_mod in snr_mod_pairs_test:
					if (snr_mod[1] == str(snr) and  snr_mod[4]== str(sco)):
						indices.append(i)
					i=i+1
				#print("Total number test data is ", len(indices))
				if len(indices) == 0:
					print("continue")
					continue
				X_test_1=X_test[indices]
				Y_test_1=Y_test[indices]
				start = time.time()
				y_pred=model.predict(X_test_1)
				end = time.time()
				period=end-start
				times_s[sco]=float(period)
				y_el=[]
				y_pred_el=[]
				for i in range(1,len(y_pred)):
					y_pred_el.append(y_pred[i-1].argmax())

				for i in range(1,len(y_pred)):
					y_el.append(Y_test_1[i-1].argmax())

				cnf_matrix=confusion_matrix(y_el, y_pred_el)
				cor=np.trace(cnf_matrix)
				cor_new=np.sum(np.diag(cnf_matrix))
				sum_all=np.sum(cnf_matrix)
				acc[sco]=float(cor)/float(sum_all)

				# Plot normalized confusion matrix
				plt.figure(figsize = (12,10))
				plot_confusion_matrix(cnf_matrix,classes=mods, normalize=True)
				plt.savefig("./images/augm_bigds_sco_snr_"+str(snr)+"_sco_"+str(sco)+".png")


			print("\nSCO overall accuracy for ", snr, " is ",acc, " \n\n")

		X_test=np.load("../cfos/X_test.npy")
		Y_test=np.load("../cfos/Y_test.npy")
		snr_mod_pairs_test=np.load("../cfos/snr_mod_pairs_test.npy")
		#print(snr_mod_pairs_test.shape)
		print("cfos are ",cfos)
		snrs=[18]
		
		for snr in snrs:
			times_s={}
			acc={}
			for cfo in cfos:
				#print ("Predicting for SNR of ",snr," and cfo of ",cfo," \n"*2)
				indices=[]
				i=0
				j=0
				for snr_mod in snr_mod_pairs_test:

					if (snr_mod[1] == str(snr) and  str(snr_mod[4])==str(cfo)):
						indices.append(i)
					i=i+1
				#print("Total number test data is ", len(indices))
				if len(indices) == 0:
					print("continue")
					continue
				X_test_1=X_test[indices]
				Y_test_1=Y_test[indices]
				start = time.time()
				y_pred=model.predict(X_test_1)
				end = time.time()
				period=end-start
				times_s[cfo]=float(period)
				y_el=[]
				y_pred_el=[]
				for i in range(1,len(y_pred)):
					y_pred_el.append(y_pred[i-1].argmax())

				for i in range(1,len(y_pred)):
					y_el.append(Y_test_1[i-1].argmax())

				cnf_matrix=confusion_matrix(y_el, y_pred_el)
				cor=np.trace(cnf_matrix)
				cor_new=np.sum(np.diag(cnf_matrix))
				sum_all=np.sum(cnf_matrix)
				acc[cfo]=float(cor)/float(sum_all)

				# Plot normalized confusion matrix
				plt.figure(figsize = (12,10))
				plot_confusion_matrix(cnf_matrix,classes=mods, normalize=True)
				plt.savefig("./images/augm_bigds_cfo_snr_"+str(snr)+"_cfo_"+str(cfo)+".png")

			print("\nCFOs overall accuracy for ", snr, " is ",acc, " \n\n")

		X_test=np.load("../iqimbalance/X_test.npy")
		Y_test=np.load("../iqimbalance/Y_test.npy")
		snr_mod_pairs_test=np.load("../iqimbalance/snr_mod_pairs_test.npy")
		
		snrs=[18]
		for snr in snrs:
			times_s={}
			acc={}
			for a_imb in a_imbs:
				#print ("Predicting for SNR of ",snr," and a_imb of ",a_imb," \n"*2)
				indices=[]
				i=0
				j=0
				for snr_mod in snr_mod_pairs_test:
					if (snr_mod[1] == str(snr) and  str(snr_mod[4])== str(a_imb)):
						indices.append(i)
					i=i+1
				#print("Total number test data is ", len(indices))
				if len(indices) == 0:
					print("continue")
					continue
				X_test_1=X_test[indices]
				Y_test_1=Y_test[indices]
				start = time.time()
				y_pred=model.predict(X_test_1)
				end = time.time()
				period=end-start
				times_s[a_imb]=float(period)
				y_el=[]
				y_pred_el=[]
				for i in range(1,len(y_pred)):
					y_pred_el.append(y_pred[i-1].argmax())

				for i in range(1,len(y_pred)):
					y_el.append(Y_test_1[i-1].argmax())

				cnf_matrix=confusion_matrix(y_el, y_pred_el)
				cor=np.trace(cnf_matrix)
				cor_new=np.sum(np.diag(cnf_matrix))
				sum_all=np.sum(cnf_matrix)
				acc[a_imb]=float(cor)/float(sum_all)

				# Plot normalized confusion matrix
				plt.figure(figsize = (12,10))
				plot_confusion_matrix(cnf_matrix,classes=mods, normalize=True)
				plt.savefig("./images/augm_bigds_iqimb_snr_"+str(snr)+"_aimb_"+str(a_imb)+".png")

			print("\niq imbalance overall accuracy for ", snr, " is ",acc, " \n\n")



		


	def evaluate_individual(self,indi,gen_no=1,init=False):
		complexity=-1
		accuracy=0

		#complexity=np.random.randint(1000, 100000)
		#accuracy=np.random.rand()

		#return complexity, accuracy

		random.seed(33)

		os.environ['PYTHONHASHSEED'] = str(33)
		#print(tf.__version__)
		session_conf = tf.compat.v1.ConfigProto(
			intra_op_parallelism_threads=1, 
			inter_op_parallelism_threads=1)

		sess = tf.compat.v1.Session(
			graph=tf.compat.v1.get_default_graph(), 
			config=session_conf)
		tf.compat.v1.keras.backend.set_session(sess)

		es = EarlyStopping(patience=5, verbose=1, min_delta=0.001, monitor='val_loss', mode='auto')

		model,classifier=self.build_model(indi)
		#print("indi learning rate is ", indi.learning_rate)

		if indi.learning_rate<0.5:
			opt=keras.optimizers.Adam(learning_rate=0.001)
		elif indi.learning_rate>=0.5 and indi.learning_rate<0.75:
			opt=keras.optimizers.Adam(learning_rate=0.01)
		else:
			opt=keras.optimizers.Adam(learning_rate=0.0001)

		sum_weights=indi.alpha+indi.beta+indi.gama
		indi.beta=0.5
		indi.gama=0.5

		model.compile(optimizer='adam',
              loss={'amc_out': 'categorical_crossentropy',
                    'decoder_out':'mse'},
              loss_weights={'amc_out': indi.beta,
                            'decoder_out': indi.gama})
		#model.compile(loss=loss(margin=indi.margin),optimizer=opt,metrics=['accuracy'])
		
		#model.summary()
		trainable_count = np.sum([K.count_params(w) for w in classifier.trainable_weights])
		non_trainable_count = np.sum([K.count_params(w) for w in classifier.non_trainable_weights])
		print('Total params: {:,}'.format(trainable_count + non_trainable_count))
		print('Trainable params: {:,}'.format(trainable_count))
		print('Non-trainable params: {:,}'.format(non_trainable_count))
		complexity=trainable_count+non_trainable_count

		
		

		if complexity>self.max_complexity or init==True: #don't waste time on super big networks
			accuracy=0
			return complexity,accuracy



		tb_log_dir = './l_log' 
		
		save_dir = './models/'
		if not os.path.exists(save_dir):
			os.mkdir(save_dir)

		ckpt = ModelCheckpoint(save_dir+'/model_{epoch:02d}.h5', monitor='val_loss',
                    verbose=0, period=1, save_best_only=True, mode='min', save_weights_only=True)#period=args.save_every

		csv_logger = CSVLogger(save_dir+'/cnn_dn_log.csv', append=True, separator=',')

		filepath = save_dir + 'joint_auto_amc_loss_50epoch.wts.h5'

		model.fit({'x_1': self.X_train_all},
			{'amc_out': self.Y_train_all,
			'decoder_out':self.X_train_src},
			callbacks = [keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min',save_weights_only=True), csv_logger, keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=0,mode='auto')],
			epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1)#50

		

		#model.fit([self.X_train_paired_1,self.X_train_paired_2], self.Y_train_paired, validation_data=([self.X_valid_paired_1,self.X_valid_paired_2],self.Y_valid_paired), batch_size=self.batch_size,epochs=self.epochs,verbose=1, 
		#	callbacks=[checkpoint, TensorBoard(log_dir=tb_log_dir)])

		#evaluate on the best saved model
		model.load_weights(filepath)
		#print("we are done with siamese training")
		score = model.evaluate([self.X_valid_all], [self.Y_valid_all,self.X_valid_src], verbose=0)
		print('score is ', score)
		#exit()
		y_pred=classifier.predict(self.X_valid_all)
		y_el=[]
		y_pred_el=[]
		for i in range(1,len(y_pred)):
			y_pred_el.append(y_pred[i-1].argmax())
		for i in range(1,len(y_pred)):
			y_el.append(self.Y_valid_paired_1[i-1].argmax())

		cnf_matrix=confusion_matrix(y_el, y_pred_el)
		cor=np.trace(cnf_matrix)
		cor_new=np.sum(np.diag(cnf_matrix))
		sum_all=np.sum(cnf_matrix)
		accuracy_1=float(cor)/float(sum_all)
		print('score class is ', accuracy_1)


		return complexity,accuracy_1


   