import cv2
import os
import keras
import numpy as np
import pandas as pd
from deepiction_data import Dataset_image_label
from IPython.display import display
from scipy.ndimage import sobel
from matplotlib import pyplot as plt
from deepiction_tools import Colors

class Prediction:
    model = []
    predictions = []
    pred_argmax = []
    data = []
    IOU = []
    predic_stack = []
    labels_stack = []
    path = ''
    nk = 0
    n_images = 0
    
    def __init__(self, model, path=''):
        self.model = model
        self.path = path
        if not path == '':
            if not os.path.exists(path):
                os.mkdir(path)
    
    def predict(self, data, verbose=True):
        self.data = data
        self.predictions = self.model.predict(self.data.images, verbose=verbose)
        self.pred_argmax = np.argmax(self.predictions, axis=3)
        
        self.n_images = self.predictions.shape[0]
        self.nk = self.data.getNumberOfClasses() 
        labels_list = list()
        prediction_list = list()
        for i in range(0,self.nk):
            labels_list.append(np.where(self.data.labels==i, 1, 0))
            prediction_list.append(np.where(self.pred_argmax == i, 1, 0))
        
        self.labels_stack = np.stack(labels_list, axis=3)
        self.predic_stack = np.stack(prediction_list, axis=3)
        self.nk = min(self.nk, self.predic_stack.shape[3])
        if verbose: 
            print('Predic_stack shape:', self.predic_stack.shape)
            print('Labels_stack shape:', self.labels_stack.shape)
            print('Prediction shape:', self.predictions.shape)
            print('Image shape:', self.data.images.shape)
            print()

    def save(self):
        for c in range(0, self.nk):
            filename = self.data.table_info.to_numpy()[1,c]
            print('save as ', filename)
            
    def report(self, verbose=True):
        if verbose: 
            print(Colors.BOLD, '---- report ----', Colors.END)
        self.IOU = np.zeros((self.n_images, self.nk))
        TP  = np.zeros((self.n_images, self.nk))
        table_results = pd.DataFrame({'Image' : [],'Class' : [], 'IoU' : [], 'TP/Area ' : [], 'Accuracy' : [], 'Recall' : [], 'Precision' : []})
        for i in range(0, self.n_images):
            for c in range(0,self.nk):
                bin1 = self.predic_stack[i,:,:,c]
                bin2 = self.labels_stack[i,:,:,c]
                m = keras.metrics.IoU(num_classes=2, target_class_ids=[1])
                m.update_state(bin1, bin2)
                self.IOU[i, c] = m.result().numpy()
                m = keras.metrics.TruePositives()
                m.update_state(bin1, bin2)
                TP[i, c] = m.result().numpy()
                m = keras.metrics.Accuracy()
                m.update_state(bin1, bin2)
                acc = m.result().numpy()
                m = keras.metrics.Recall()
                m.update_state(bin1, bin2)
                recall = m.result().numpy()
                m = keras.metrics.Recall()
                m.update_state(bin1, bin2)
                precision = m.result().numpy()
                table_results.loc[len(table_results)] = [i, c, self.IOU[i, c], TP[i, c], acc, recall, precision]
                if verbose: 
                    print("Class ", c, " Image ", i, " IOU =", self.IOU[i, c], " TP =", TP[i, c])
        if verbose: 
            print('Table results')
            display(table_results)
    
    def subplot(self, axs, x, y, image, title, colorbar=False):
        im = axs[x, y].imshow(image, interpolation='nearest')
        if colorbar == True:
            axs[x, y].figure.colorbar(im)
        axs[x, y].set_title(title)
        axs[x, y].axis('off')

    def plot(self, num_image):
        fig, axs = plt.subplots(3, self.nk, figsize=(5*(self.nk+1),5))
        filename = self.data.table_info.to_numpy()[num_image, 0]
        for c in range(0, self.nk):
            self.subplot(axs, 0, c, self.predictions[num_image,:,:,c], 'proba ' + str(c))
            self.subplot(axs, 1, c, self.labels_stack[num_image,:,:,c], 'Label class:' + str(c))
            self.subplot(axs, 2, c, self.predic_stack[num_image,:,:,c], 'IoU:' + str(round(self.IOU[0, c],3)))
        plt.tight_layout()
        plt.show(block=False)

        if not self.path == '':
            filename1 = os.path.join(self.path, 'figure_per_class_' + filename + '.png')
            print('Saving ', filename1)
            plt.savefig(filename1, bbox_inches='tight')
        
        fig, axs = plt.subplots(2, 2, figsize=(16, 16))
        self.subplot(axs, 0, 0, self.data.images[num_image], 'Image', colorbar=True)
        self.subplot(axs, 1, 0, self.data.labels[num_image], 'Label', colorbar=True)
        self.subplot(axs, 0, 1, np.uint8(self.pred_argmax[num_image]), 'Argmax', colorbar=True)
        plt.tight_layout()
        plt.show(block=False)
        if not self.path == '':
            filename2 = os.path.join(self.path, 'figure_summary_' + filename + '.png')
            print('Saving ', filename2)
            plt.savefig(filename2, bbox_inches='tight')
