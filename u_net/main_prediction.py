import os
import tensorflow as tf
import numpy as np
from keras.models import load_model
from deepiction_unet import Unet_multiclass
from deepiction_data import Dataset_image_label
from deepiction_prediction import Prediction
from deepiction_tools import Colors, report_resources

report_resources()

dataname = '3-Quality_control_data_set_Claire_Boulogne'
dataname = 'Mito_SEM_Claire_Boulogne'
datapath = os.path.abspath(os.path.join('/Users/sage/Desktop/datasets/', dataname))
#dataname = 'clothes_rgb'
#datapath = os.path.abspath(os.path.join('/Users/sage/Desktop/datasets/', dataname))

n_im_test = 6
data_test  = Dataset_image_label(os.path.join(datapath, 'test'))   
data_test.load(n_im_test, show_table=True)
data_test.report()

name = 'report_clothes_rgb_59K_50I_3E_'
name = 'report_3-Quality_control_data_set_Claire_Boulogne_2K_11I_30E_'

name = 'report_Mito_SEM_Claire_Boulogne_2K_75I_50E_'
path = '/Users/sage/Desktop/reports/'
path = os.path.abspath(os.path.join( path, name))

if not os.path.exists(os.path.join(path, 'model_best.hdf5')):
    print('Model not found')
    exit()

model = load_model(os.path.join(path, 'model_best.hdf5'))

prediction = Prediction(model, os.path.join(path, 'prediction_best'))
prediction.predict(data_test)
prediction.save()
prediction.report()
for i in range(0, data_test.images.shape[0]):
    prediction.plot(i)
