import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Dropout
from keras.layers import Activation, MaxPool2D, Concatenate, AveragePooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
import time
from matplotlib import pyplot as plt
from csv import writer
import os
import tracemalloc

class Unet_multiclass:
  '''
  This is a class that implements a customized Unet 
  for a multiple class classification
  '''
  name = 'untitled'
  n_classes = 1
  history = []
  runtime = 0
  peakmemory = 0
  skip_connections = 1
  pool_stride = 2
  pool_type = 'max'
  activation = "relu"
  batchnorm = False
  dropout = 0
  n_images = 0
  n_images_train = 0
  emissions = []
  report_path = ''
  model = []
  runtime = 0 # Computation time of the training

  def __init__(self, name, report_path, n_classes, pool_stride, pool_type, skip_connections, activation, batchnorm, dropout):
    self.name = name
    self.n_classes = n_classes
    self.skip_connections = skip_connections # Modified: now a list of booleans of length num_pools
    self.pool_stride=pool_stride
    self.pool_type=pool_type
    self.activation = activation
    self.batchnorm = batchnorm
    self.dropout = dropout
    if not(os.path.exists(report_path)): 
      os.makedirs(report_path)
    self.report_path = report_path

  def conv_block(self, input, n_filters):
    x = Conv2D(n_filters, 3, padding="same")(input)
    x = BatchNormalization()(x) if self.batchnorm == True else x 
    x = Dropout(self.dropout)(x) if self.dropout > 0 else x 
    x = Activation(self.activation)(x)
    x = Conv2D(n_filters, 3, padding="same")(x)
    x = BatchNormalization()(x) if self.batchnorm == True else x
    x = Dropout(self.dropout)(x) if self.dropout > 0 else x
    x = Activation(self.activation)(x)
    return x

  # Encoder block: Conv block followed by pooling
  def encoder_block(self, input, n_filters):
    x = self.conv_block(input, n_filters)
    if self.pool_type == 'max': # Pooling type and stride added here
      p = MaxPool2D((self.pool_stride, self.pool_stride))(x)
    else:
      p = AveragePooling2D((self.pool_stride, self.pool_stride))(x)
    n_filters = n_filters * 2
    return x, p, n_filters   

  # Decoder block, skip features gets input from encoder for concatenation
  def decoder_block(self, input, skip_features, n_filters, layer):
    skip = self.skip_connections[layer]
    x = Conv2DTranspose(n_filters, (2, 2), strides=(self.pool_stride, self.pool_stride), padding="same")(input) # Added stride here
    x = Concatenate()([x, skip_features]) if skip else x # Skip connection added here. 
    x = self.conv_block(x, n_filters)
    return x

  def split(self, images, labels, split_ratio_val):
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = split_ratio_val, random_state = 1234)
    train_masks_cat = to_categorical(y_train, num_classes=self.n_classes)
    y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], self.n_classes))
    test_masks_cat = to_categorical(y_test, num_classes=self.n_classes)
    y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], self.n_classes))
    return x_train, y_train_cat, x_test, y_test_cat

  #Build Unet using the blocks
  def build_model(self, input_shape, n_channels_initial, n_pools, learning_rate):
    input_shape = (input_shape[1], input_shape[2], input_shape[3])
    
    # Contractive path
    p0 = Input(input_shape) 
    nc0 = n_channels_initial
    (x1, p1, nc1) = self.encoder_block(p0, nc0) if n_pools > 0 else (p0, p0, nc0)
    (x2, p2, nc2) = self.encoder_block(p1, nc1) if n_pools > 1 else (x1, p1, nc1)
    (x3, p3, nc3) = self.encoder_block(p2, nc2) if n_pools > 2 else (x2, p2, nc2)   
    (x4, p4, nc4) = self.encoder_block(p3, nc3) if n_pools > 3 else (x3, p3, nc3)
    (x5, p5, nc5) = self.encoder_block(p4, nc4) if n_pools > 4 else (x4, p4, nc4)
    (x6, p6, nc6) = self.encoder_block(p5, nc5) if n_pools > 5 else (x5, p5, nc5)
    (x7, p7, nc7) = self.encoder_block(p6, nc6) if n_pools > 6 else (x6, p6, nc6)
    (x8, p8, nc8) = self.encoder_block(p7, nc7) if n_pools > 7 else (x7, p7, nc7)
    (x9, p9, nc9) = self.encoder_block(p8, nc8) if n_pools > 8 else (x8, p8, nc8)

    # Expansive path
    u9 = self.conv_block(p9, nc9)
    u8 = self.decoder_block(u9, x9, nc8, 8) if n_pools > 8 else u9 
    u7 = self.decoder_block(u8, x8, nc7, 7) if n_pools > 7 else u8
    u6 = self.decoder_block(u7, x7, nc6, 6) if n_pools > 6 else u7
    u5 = self.decoder_block(u6, x6, nc5, 5) if n_pools > 5 else u6
    u4 = self.decoder_block(u5, x5, nc4, 4) if n_pools > 4 else u5    
    u3 = self.decoder_block(u4, x4, nc3, 3) if n_pools > 3 else u4 
    u2 = self.decoder_block(u3, x3, nc2, 2) if n_pools > 2 else u3
    u1 = self.decoder_block(u2, x2, nc1, 1) if n_pools > 1 else u2
    u0 = self.decoder_block(u1, x1, nc0, 0) if n_pools > 0 else u1
    
    # Compute final layer activation
    activation_final = 'sigmoid' if self.n_classes == 1 else 'softmax' 
    outputs = Conv2D(self.n_classes, 1, padding="same", activation=activation_final)(u0)

    # Compile model
    adam = keras.optimizers.Adam(learning_rate=learning_rate)
    self.model = Model(p0, outputs, name="U-Net") 
    self.model.compile(optimizer=adam, loss='categorical_crossentropy')
    return self.model

  def train(self, images, labels, epochs, batchsize, split_ratio_val, shuffle, save_epoch=False, verbose=True):
    tracemalloc.start()
    start = time.time()
    self.epochs = epochs
    self.split_ratio_val = split_ratio_val
    self.n_images = images.shape[0]
    x_train, y_train, x_val, y_val = self.split(images, labels, split_ratio_val)
    self.n_images_train = x_train.shape[0]

    csv_logger = keras.callbacks.CSVLogger(os.path.join(self.report_path, 'training.log'))
    model_checkpoint = keras.callbacks.ModelCheckpoint(
      os.path.join(self.report_path, 'model_best.hdf5'),
      monitor='val_loss', mode='min',
      save_best_only=True, verbose=verbose)
    #reduce_lr = keras.callbacks.ReduceLROnPlateau(verbose=verbose)
    #earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=verbose)
    #callbacks = [model_checkpoint, csv_logger, earlystop, reduce_lr] if save_epoch == True else []  
    callbacks = [model_checkpoint, csv_logger] if save_epoch == True else []  

    self.history = self.model.fit(x_train, y_train, batch_size = batchsize, verbose=verbose, epochs=epochs, callbacks=callbacks, 
      validation_data=(x_val, y_val),  shuffle=shuffle) # Shuffle False
    
    self.runtime = time.time() - start
    self.peakmemory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    self.model.save(os.path.join(self.report_path, 'model_last.hdf5'))
    #self.emissions: float = tracker.stop()
    return self.history, self.runtime, self.peakmemory

  
  def learning_curve(self):
    loss = self.history.history['loss']
    val_loss = self.history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    fig, axs = plt.subplots(2, figsize=(16,8))
    axs[0].plot(epochs, loss, 'b', label='Training loss')
    axs[0].plot(epochs, val_loss, 'r', label='Validation loss')
    axs[0].legend()
    axs[1].plot(epochs, loss, 'b', label='Training loss')
    axs[1].plot(epochs, val_loss, 'r', label='Validation loss')
    axs[1].set_yscale('log') 
    return plt
          
  def show_learning_curve(self):
    plt = self.learning_curve()
    plt.show(block=False)
  
  def report(self):
    pd.DataFrame(self.history.history).to_csv(os.path.join(self.report_path, 'learning_curve_.csv'))
    plt = self.learning_curve()
    plt.savefig(os.path.join(self.report_path, 'learning_curve.png'), bbox_inches='tight')
    
  def info(self, iou, filename):
    f_tloss = self.history.history['loss'][-1]
    f_vloss = self.history.history['val_loss'][-1]
    f_tacc = self.history.history['loss'][-1]
    f_vacc = self.history.history['val_loss'][-1]
    s = [self.n_images, self.n_images_train,
        self.epochs, 
        self.n_channels_first, 
        self.batchnorm, 
        self.n_classes, 
        self.runtime, iou, f_tloss, f_vloss, f_tacc, f_vacc]
    
    with open(filename, 'a+') as handle:
        writer(handle).writerow(s)
    handle.close()
    return s
