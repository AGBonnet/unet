import os
import tensorflow as tf
import numpy as np
from keras.models import load_model
from deepiction_unet import Unet_multiclass
from deepiction_data import Dataset_image_label
from deepiction_prediction import Prediction
from deepiction_tools import Colors, report_resources

report_resources()

n_channels_initial = 32
epochs = 100
batchnorm = False
shuffle = True
dropout = 0
activation = "relu"
batchsize = 8 #16
learning_rate = 0.001  # default 0.001
split_ratio_val = 0.25
n_im_train = 100
n_im_test  = 15
n_pools = 4
skip_connections = True

#dataname = 'ctc-hela-40im' 
#datapath = os.path.abspath(os.path.join('/Users/sage/Desktop/datasets/ctc-hela/', dataname))
#dataname = 'radius25.0_sigmas10.0_1.0_diff0.3'
#dataname = 'data-256 32-25-2.0'
#datapath = os.path.abspath(os.path.join('/Users/demko/Desktop/EPFL/MA1/ML/Projects/ml-project-2-stateoftheart/data_synthesis/generated_datasets/no_overlap', dataname))

#dataname = 'clothes_rgb'
#datapath = os.path.abspath(os.path.join('/Users/sage/Desktop/datasets/', dataname))

dataname = 'mitochondria-SEM-Claire-Boulogne'
datapath = os.path.abspath(os.path.join('/Users/demko/Desktop/EPFL/MA1/ML/Projects/ml-project-2-stateoftheart/datasets', dataname))


# Load data
data_train = Dataset_image_label(os.path.join(datapath, 'train'))              
data_train.load(n_im_train, show_table=False, norm_images='DIV255')
nk = data_train.getNumberOfClasses() 

# Training
name = dataname + '_' + str(nk) + 'K_' + str(data_train.images.shape[0]) + 'I_' + str(epochs) + 'E_'
report_path = os.path.abspath(os.path.join('/Users/demko/Desktop/EPFL/MA1/ML/Projects/ml-project-2-stateoftheart/u_net/report_' + name))
unet = Unet_multiclass(name, report_path, nk, skip_connections, activation, batchnorm, dropout)
unet.build_model(data_train.images.shape, n_channels_initial, n_pools, learning_rate)
unet.model.summary()

unet.train(data_train.images, data_train.labels, epochs, batchsize, split_ratio_val, shuffle, save_epoch=True)
unet.report()

# Prediction
data_test  = Dataset_image_label(os.path.join(datapath, 'test'))   
data_test.load(n_im_test, show_table=False, norm_images='DIV255')
best_model = load_model(os.path.join(unet.report_path, 'model_best.hdf5'))

prediction = Prediction(best_model)
prediction.predict(data_test)
prediction.report(nk)
prediction.plot(unet.report_path, 0, nk)

print('Number of parameters:', best_model.count_params())
print('n_channels_first:', n_channels_initial, ' n_pools:', n_pools)
print('epochs:', epochs, ' learning_rate:', learning_rate)
print('batchnorm:', batchnorm, ' shuffle:', shuffle, ' dropout:', dropout, ' skip_connections', skip_connections)
print('split_ratio_val:', split_ratio_val)
print('Number of images train:', n_im_train, ' Number of image test:', n_im_test)
print('Runtime:', unet.runtime, ' Peak mem:', unet.peakmemory)

print('IoU/classes ', np.mean(prediction.IOU, axis=0))
#print('IoU/Images ', np.mean(prediction.IOU, axis=1))
print('IoU/mean ', np.mean(prediction.IOU))

