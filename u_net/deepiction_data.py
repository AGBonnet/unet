from sys import exit
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd
from IPython.display import display
from deepiction_tools import Colors

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

class Dataset_image_label:
  '''
  This is a class allows to read images and labels
  '''
  table_info = []
  images = []
  labels = []
  folder = ''
  
  def __init__(self, folder):
    self.folder = folder
  
  def compare_shape(self, a, b):
    if (a.ndim != b.ndim):
        return False
    flag = True
    for i in range(0, a.ndim):
        if a.shape[i] != b.shape[i]:
            flag = False
    return flag

  def encode_mask(self, mask_dataset):
    #Encode labels to 0, 1, 2, 3, ... but multi dim array so need to flatten, encode and reshape
    labelencoder = LabelEncoder()
    n, h, w = mask_dataset.shape  
    mask_dataset_reshaped = mask_dataset.reshape(-1,1)
    mask_dataset_reshaped_encoded = labelencoder.fit_transform(mask_dataset_reshaped)
    mask_dataset_encoded = mask_dataset_reshaped_encoded.reshape(n, h, w)
    mask_dataset_encoded = np.expand_dims(mask_dataset_encoded, axis = 3)
    return mask_dataset_encoded

  # Read nimages images in a folder_name
  # Check if no hidden files, various format, all images should have the same size
  def load(self, n_images, show_table=False, norm_images='DIV255'):
    #print()
    #print("Loading ", n_images )
    folder_images = os.path.join(self.folder, 'images')
    folder_labels = os.path.join(self.folder, 'labels')
    if os.path.exists(folder_images) == False:
      print("Path not found " + folder_images)
      exit()
    flag = self.read_dataset(folder_images, folder_labels, n_images) 
    
    if norm_images == 'DIV255':
      self.images = self.images / 255.

    if show_table:
      self.table()
      self.report()

    return flag

  def read_dataset(self, folder_images, folder_labels, n_images):
    mode_imread = -1 #cv2.IMREAD_UNCHANGED

    # check folder images
    if not os.path.exists(folder_images):
      print('Error: not found:', folder_images)
      return False

    # check folder labels
    if not os.path.exists(folder_labels):
      print('Error: not found:', folder_labels)
      return False

    list_images = self.list_files(folder_images)
    list_labels = self.list_files(folder_labels)
    list_inter = set(list_images).intersection(list_labels)
    list_sorted = sorted(list_inter)

    self.images = []
    self.labels = []
    self.table_info = pd.DataFrame({'Filename' : [],'Shape Image' : [], 'Type Image' : [], 'Min/Max Image ' : [], 'Shape Label' : [], 'Type Label' : [], 'Min/Max Label' : []})

    for f in list_sorted:
      if len(self.images) < n_images:
        if f.endswith('.tif') or f.endswith('.tiff') or f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg'):  
          current_image = cv2.imread(os.path.join(folder_images, f), mode_imread)
          current_label = cv2.imread(os.path.join(folder_labels, f), mode_imread)
          if len(self.images) == 0:
            first_image = current_image
          if len(self.labels) == 0:
            first_label = current_label
          if self.compare_shape(first_image, current_image) and self.compare_shape(first_label, current_label):
            self.images.append(current_image)
            self.labels.append(current_label)
            self.table_info.loc[len(self.table_info)] = [f, 
              current_image.shape, current_image.dtype, str(current_image.min()) + '...' + str(current_image.max()),
              current_label.shape, current_label.dtype, str(current_label.min()) + '...' + str(current_label.max()),]
            
    self.images = np.array(self.images)
    self.labels = np.array(self.labels)
    
    if (self.images.ndim == 3):
      self.images = np.expand_dims(self.images, axis=3)
    if (self.labels.ndim == 3):
      self.labels = np.expand_dims(self.labels, axis=3)
    return True

  def list_files(self, folder_local):
    list_files = os.listdir(folder_local)
    list_files.sort()
    list_clean = []
    for filename in (list_files):
      if not filename.startswith('.') and not filename.startswith('~') and not filename.startswith('#'): 
        list_clean.append(filename)
    return list_clean
  
  def report(self):
    print()
    print(Colors.RED + Colors.BOLD + Colors.UNDERLINE + self.folder + Colors.END)
    print("Images: ", self.images.shape, ' min:', self.images.min(), ' max:', self.images.max())   
    print("Labels: ", self.labels.shape, ' min:', self.labels.min(), ' max:', self.labels.max()) 
    print("Classes:", self.getNumberOfClasses(), np.unique(self.labels), )
    
  def getNumberOfClasses(self):
    return self.labels.max()+1
  
  def table(self):
    display(self.table_info)
  
    

