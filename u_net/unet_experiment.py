''' U-Net training and evaluation. '''


# ------------------ Import libraries ------------------#

import sys
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import cv2
from IPython.display import display

from keras.models import load_model
from deepiction_unet import Unet_multiclass
from deepiction_data import Dataset_image_label
from deepiction_prediction import Prediction
from deepiction_tools import Colors, report_resources

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ------------------ Define paths ------------------#

BASE_DIR = os.path.dirname(os.getcwd())
DATASETS_PATH = os.path.join(BASE_DIR, "data_synthesis", "generated_datasets")
RESULTS_PATH = os.path.join(BASE_DIR, "results")
LOG_PATH = os.path.join(RESULTS_PATH, "log.csv")
PLOTS_PATH = os.path.join(BASE_DIR, "plots")

# If these directories do not exist, create them
if not os.path.exists(DATASETS_PATH):
    os.makedirs(DATASETS_PATH)
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)
if not os.path.exists(PLOTS_PATH):
    os.makedirs(PLOTS_PATH)

# ------------------ Fixed architecture parameters ------------------#

BATCHNORM = False
SHUFFLE = True
DROPOUT = 0
ACTIVATION = "relu"
BATCHSIZE = 8 
LEARNING_RATE = 0.001  # default value = 0.001
SPLIT_RATIO_VAL = 0.25
N_IM_TRAIN = 100
N_IM_TEST  = 100
POOL_TYPE = "max"
POOL_STRIDE = 2

# ------------------ Load training and test data ------------------#

def get_similarity(sigma_cell, sigma_back): 
    return sigma_cell/sigma_back 

def load_dataset(avg_radius, sigma_back, verbose=True): 
    """ Given data parameters {avg_radius, sigma_back}, load the corresponding training and test sets. """

    dataset_name = f'dataset_rad{avg_radius:.1f}_sig{sigma_back:.1f}'
    
    # Check that dataset path exists
    dataset_path = os.path.join(DATASETS_PATH, dataset_name)
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path {dataset_path} does not exist!")
        
    # If it exists, load dataset parameters
    dataset_params = pd.read_csv(os.path.join(dataset_path, "parameters.csv"))
    dataset_params['similarity'] = get_similarity(dataset_params['sigma_cell'], dataset_params['sigma_back'])
    if verbose: 
        print('Dataset parameters:')
        display(dataset_params)
        print(' Loading training set...')

    # Load training set
    data_train = Dataset_image_label(os.path.join(dataset_path, 'train'))
    data_train.load(N_IM_TRAIN, show_table=verbose, norm_images='DIV255')

    # Load test set
    if verbose:
        print(' Loading test set...')
    data_test = Dataset_image_label(os.path.join(dataset_path, 'test'))  
    data_test.load(N_IM_TEST, show_table=verbose, norm_images='DIV255')
    
    # Get number of classes in train set
    nk = data_train.getNumberOfClasses()

    return data_train, data_test, dataset_params, nk


# --------------- Training U-Net ----------------#


def train_unet(
    data_train, epochs, avg_radius, sigma_back, num_pools, num_channels, pool_type, pool_stride, skip_connections=1, nk=2, verbose=True
    ):
    """ Train Unet model on training set, predict on test set and save results. """

    # Check that skip connections are valid
    if skip_connections == 1:
        skip_connections = [1] * num_pools
    elif skip_connections == 0:
        skip_connections = [0] * num_pools
    elif type(skip_connections) == list and len(skip_connections) != num_pools:
        raise ValueError(f"Number of skip connections ({len(skip_connections)}) does not match number of pooling layers ({num_pools})!")

    # If report already exists, skip
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)
    skip_str = stringify_skips(skip_connections, num_pools)
    report_name = 'rad{}_sig{}_ch{}_pools{}_{}_stride{}_skip{}_{}E'.format(
            avg_radius, sigma_back, num_channels, num_pools, pool_type, pool_stride, skip_str, epochs)
    report_path = os.path.join(RESULTS_PATH, report_name)

    if os.path.exists(report_path):
        if verbose:
            print(f"Model '{report_name}' already trained, overwriting previous training...")
    
    # Setup U-Net with specified architecture 
    unet = Unet_multiclass(
        name='U-Net', 
        report_path=report_path, 
        n_classes=int(nk), 
        pool_stride=pool_stride, 
        pool_type=pool_type,
        skip_connections=skip_connections, 
        activation=ACTIVATION, 
        batchnorm=BATCHNORM, 
        dropout=DROPOUT)
        
    unet.build_model(
        input_shape=data_train.images.shape, 
        n_channels_initial=num_channels, 
        n_pools=num_pools, 
        learning_rate=LEARNING_RATE)
    
    # Train the model on the training set
    unet.train(data_train.images, data_train.labels, epochs, BATCHSIZE, SPLIT_RATIO_VAL, SHUFFLE, save_epoch=True, verbose=verbose)
    
    # If specified, report performance on training set
    if verbose: 
        unet.report()   
    return report_path


# --------------- Prediction with U-net ----------------#


def test_unet(data_test, report_path, verbose=True, display_plots=0):
    ''' Test the trained network on the test set. 
    Inputs: 
        data_test: Dataset_image_label object with test set
        report_path: path where report was saved
        verbose: Print accuracy
        display_plots: number of prediction images to display (0 to disable)
    '''
    # Load model with lowest validation loss
    best_model = load_model(os.path.join(report_path, 'model_best.hdf5'))
    
    # Predict with best model on test set
    prediction = Prediction(best_model)
    prediction.predict(data_test, verbose=verbose)

    # Report IoU of best model on test set, plot examples of predictions
    prediction.report(verbose)
    
    # Print examples of prediction images
    for i in range(int(display_plots)):
        prediction.plot(i)

    if verbose: 
        print('IoU/classes ', np.mean(prediction.IOU, axis=0))
        print('IoU/mean ', np.mean(prediction.IOU))
        #print('IoU/Images ', np.mean(prediction.IOU, axis=1))

    return prediction


# --------------- Logging results ----------------#


def stringify_skips(skip_connections, num_pools): 
    ''' Convert skip_connections to a string.'''
    if skip_connections == 0: 
        skip_str = '-'.join(['0'] * num_pools)
    elif skip_connections == 1:
        skip_str = '-'.join(['1'] * num_pools)
    else: 
        skip_str = '-'.join(str(i) for i in skip_connections)
    return skip_str


def find_log(epochs, num_pools, num_channels, pool_type, pool_stride, skip_connections, avg_radius, sigma_back):
    ''' Check whether we have already logged this configuration. '''
    if not os.path.exists(LOG_PATH):
        return None
    log_df = pd.read_csv(LOG_PATH)
    skip_str = stringify_skips(skip_connections, num_pools)
    match = log_df[
        (log_df['num_pools'] == num_pools) &
        (log_df['num_channels'] == num_channels) &
        (log_df['pool_type'] == pool_type) &
        (log_df['pool_stride'] == pool_stride) &
        (log_df['skip_connections'] == skip_str) &
        (log_df['avg_radius'] == avg_radius) &
        (log_df['sigma_back'] == sigma_back)
        #(log_df['epochs'] == epochs)
        ]
    # If there is a match, return its index in the log file
    match_idx = match.index[0] if len(match) > 0 else None
    return match_idx


def log_run(num_pools, num_channels, pool_type, pool_stride, skip_connections, avg_radius, sigma_back, epochs, IoU, verbose=True):
    ''' Add a run to the log file, including results path and weighted IoU score.'''
    
    # If log file exists, load it, otherwise create it
    if os.path.exists(LOG_PATH):
        log_df = pd.read_csv(LOG_PATH)
    else:
        columns = [
            'num_pools', 'num_channels', 'pool_type', 'pool_stride', 
            'skip_connections', 'avg_radius', 'sigma_back', 'epochs', 'IoU', 'path']
        log_df = pd.DataFrame(columns=columns)

    # Check whether this run already exists in the log file
    skip_str = stringify_skips(skip_connections, num_pools)

    if len(log_df) > 0:
        match_idx = find_log(epochs, num_pools, num_channels, pool_type, pool_stride, skip_connections, avg_radius, sigma_back)

        # If run already exists, update the IoU score and update the log file
        if match_idx is not None: 
            if verbose: 
                old_IoU = log_df.iloc[match_idx]['IoU']
                print(f'Match found with IOU = {old_IoU}, updating IoU score to {IoU}.')

            # Update IoU score in row by creating a copy
            log_df.loc[match_idx, 'IoU'] = IoU
            log_df.to_csv(LOG_PATH, index=False)
            return 
        
    # If run does not exist, add a new row to the log file
    log_path = 'results/rad{}_sig{}_ch{}_pools{}_{}_stride{}_skip{}_{}E'.format(
        avg_radius, sigma_back, num_channels, num_pools, pool_type, pool_stride, skip_str, epochs)
        
    # Create new row
    new_row = pd.DataFrame({
        'num_pools': num_pools,
        'num_channels': num_channels,
        'pool_type': pool_type,
        'pool_stride': pool_stride,
        'skip_connections': skip_str,
        'avg_radius': avg_radius,
        'sigma_back': sigma_back,
        'epochs': epochs,
        'IoU': IoU,
        'path': log_path
    }, index=[0])

    # Update the log file by appending new row
    log_df.loc[len(log_df)] = new_row.iloc[0]
    log_df.to_csv(LOG_PATH, index=False)
    


# --------------- Running experiments ----------------#

# Weight IoU of each class by the number of pixels of that class in the image
def weighted_IOU(prediction, density):
    ''' Weight IoU of each class by the proportion of pixels of that class in the image. '''
    IoU_back, IoU_cell = np.mean(prediction.IOU, axis=0)
    return IoU_back*(1-density) + IoU_cell*density

def run_unet(num_pools, num_channels, skip_connections, avg_radius, sigma_back, epochs=50, verbose=True):
    ''' Run the U-Net algorithm with the given parameters and store IoU in log file. '''
    

    # Check if configuration was already run; if so, return its IoU score
    match_idx = find_log(epochs, num_pools, num_channels, POOL_TYPE, POOL_STRIDE, skip_connections, avg_radius, sigma_back)
    if match_idx is not None: 
        log_df = pd.read_csv(LOG_PATH)
        IoU = log_df.loc[match_idx, 'IoU']
        if verbose: 
            print('\t\tFound pre-existing log for this configuration with IoU = {}.'.format(IoU))
        return IoU
    
    # Load datasets
    data_train, data_test, dataset_params, nk = load_dataset(avg_radius, sigma_back, verbose=verbose)
    density = float(dataset_params['density'])
    
    # Train U-net on training set
    report_path = train_unet(
        data_train, epochs, avg_radius, sigma_back, num_pools, num_channels, 
        POOL_TYPE, POOL_STRIDE, skip_connections, nk, verbose=verbose)

    # Predict with best model on test set
    results_path = os.path.join(BASE_DIR, report_path)
    prediction = test_unet(data_test, results_path, verbose=verbose, display_plots=0)
    
    # Compute IoU
    IoU = weighted_IOU(prediction, density)
    
    # Update log file with IoU score
    log_run(num_pools, num_channels, POOL_TYPE, POOL_STRIDE, skip_connections, avg_radius, sigma_back, epochs, IoU, verbose=verbose)
    print('\t\tNew run with IoU = {}'.format(IoU))
    return IoU


# --------------- Experiment A: Architecture and data parameters ----------------#


def experimentA(
    avg_radius_vals, similarity_vals, num_channels_vals, num_pools_vals, 
    epochs=50, skip_connections=1, verbose=False
    ):
    ''' Run experiment A: vary the number of pooling layers and number of channels. '''
    
    # Initialize a grid of IoU values
    IoU_gridA = np.zeros((
        len(avg_radius_vals), len(similarity_vals), 
        len(num_channels_vals), len(num_pools_vals))
        )

    for i, avg_radius in enumerate(avg_radius_vals):

        for j, similarity in enumerate(similarity_vals):

            sigma_back = round(1. / similarity, 1)
            if verbose: 
                print('Dataset: Average radius:', avg_radius,'\tSimilarity:', similarity)

            for k, num_channels in enumerate(num_channels_vals):

                for l, num_pools in enumerate(num_pools_vals):
                    if verbose: 
                        print('\tArchitecture:\tNumber of initial channels:', num_channels,'\tNumber of pooling layers:', num_pools)

                    IoU = run_unet(
                        num_pools, num_channels, skip_connections, 
                        avg_radius, sigma_back, epochs=epochs, verbose=verbose
                        )

                    IoU_gridA[i, j, k, l] = IoU
    # Save IoU_gridA 
    np.save(os.path.join(RESULTS_PATH, 'IoU_gridA.npy'), IoU_gridA)
    return IoU_gridA
    

def plot_experimentA(IoU_gridA, avg_radius_vals, similarity_vals, num_pools_vals, num_channels_vals, save=False):
    ''' Visualize effect of architecture and data parameters on IoU. '''
    val_min = np.floor(IoU_gridA[IoU_gridA>0.9].min()*50)/50
    val_max = np.ceil(IoU_gridA.max()*100)/100
    val_step = (val_max-val_min)/10

    avg_radius_vals = avg_radius_vals[::-1]
    similarity_vals = similarity_vals[::-1]

    fig, axs = plt.subplots(len(avg_radius_vals), len(similarity_vals), figsize=(18, 15))

    for i, _ in enumerate(avg_radius_vals):

        for j, _ in enumerate(similarity_vals):
            
            sns.heatmap(
                IoU_gridA[len(avg_radius_vals)-1-i, len(similarity_vals)-1-j, ::-1, :], ax=axs[i, j], 
                annot=True, cbar=i==0 and j==0, vmin=val_min, vmax=val_max, fmt='.3f',
                cbar_kws={'label': 'IoU'}, xticklabels=num_pools_vals, yticklabels=num_channels_vals[::-1])

    # Add common axes labels
    for i, ax in enumerate(axs[:, 0]):
        ax.set_ylabel('Average radius: ' + str(avg_radius_vals[i]), fontsize=12, rotation=90, labelpad=15)
    for i, ax in enumerate(axs[-1, :]):
        ax.set_xlabel('Similarity: ' + str(round(similarity_vals[i], 3)), fontsize=12, labelpad=15)

    # Format colorbar
    axs[0, 0].collections[0].colorbar.remove()
    fig.colorbar(
        axs[0, 0].collections[0], ax=axs, label='IoU', shrink=0.5, 
        ticks=np.arange(val_min, val_max+0.01, val_step), orientation='vertical', pad=0.05)
    axs[0, 0].collections[0].colorbar.ax.yaxis.set_label_position('left')
    axs[0, 0].collections[0].colorbar.ax.yaxis.label.set_size(16)
        
    # Add common title and labels
    fig.suptitle('Experiment A: U-Net test IoU values for different architectures and dataset parameters', fontsize=18, y=0.92, x=0.43)
    fig.text(0.43, 0.04, 'Number of pooling layers', ha='center', fontsize=18)
    fig.text(0.04, 0.5, 'Number of initial channels', va='center', rotation='vertical', fontsize=18)
    
    plt.show()

    if save: 
        plot_name = os.path.join(PLOTS_PATH, 'experimentA.png')
        fig.savefig(plot_name)



def plot_experimentA_inverted(IoU_gridA, avg_radius_vals, similarity_vals, num_pools_vals, num_channels_vals, save=False): 
    ''' Visualize effect of architecture and data parameters on IoU. '''
    val_min = np.floor(IoU_gridA[IoU_gridA>0.9].min()*50)/50
    val_max = np.ceil(IoU_gridA.max()*100)/100
    val_step = (val_max-val_min)/10

    num_channels_vals = num_channels_vals[::-1]
    num_pools_vals = num_pools_vals[::-1]

    fig, axs = plt.subplots(len(num_channels_vals), len(num_pools_vals), figsize=(18, 15))
    for i in range(len(num_channels_vals)):

        for j in range(len(num_pools_vals)):

            sns.heatmap(
                IoU_gridA[::-1, ::-1, len(num_channels_vals)-1-i, j], 
                ax=axs[i, j], annot=True, 
                cbar=i==0 and j==0, vmin=val_min, vmax=val_max, fmt='.3f',
                cbar_kws={'label': 'IoU'}, 
                yticklabels=avg_radius_vals[::-1], 
                xticklabels=[round(s, 2) for s in similarity_vals[::-1]])

    # Add common axes labels
    for i, ax in enumerate(axs[-1, :]):
        num_pool = num_pools_vals[len(num_pools_vals)-1-i]
        if num_pool == 1: 
            ax.set_xlabel('1 pooling layer', fontsize=12, labelpad=15)
        else: 
            ax.set_xlabel(str(num_pool) + ' pooling layers', fontsize=12, labelpad=15)
    for i, ax in enumerate(axs[:, 0]):
        ax.set_ylabel(
            str(round(num_channels_vals[i], 3)) + ' initial channels', 
            fontsize=12, rotation=90, labelpad=15)

    # Format colorbar
    axs[0, 0].collections[0].colorbar.remove()
    fig.colorbar(
        axs[0, 0].collections[0], ax=axs, label='IoU', shrink=0.5, 
        ticks=np.arange(val_min, val_max + 0.01, val_step), 
        orientation='vertical', pad=0.05)
    axs[0, 0].collections[0].colorbar.ax.yaxis.set_label_position('left')
    axs[0, 0].collections[0].colorbar.ax.yaxis.label.set_size(16)
        
    # Add common title 
    fig.suptitle('Experiment A: U-Net test IoU values for different architectures and dataset parameters', fontsize=18, y=0.92, x=0.43)

    # Add outer labels for num_pools and num_channels
    fig.text(0.43, 0.04, 'Texture similarity', ha='center', fontsize=18)
    fig.text(0.04, 0.5, 'Average cell radius', va='center', rotation='vertical', fontsize=18)
    
    plt.show()

    if save: 
        plot_name = os.path.join(PLOTS_PATH, 'experimentA_inverted.png')
        fig.savefig(plot_name)


# --------------- Experiment B: Receptive field ----------------#

def receptive_field(num_pools, kernel_size=3, pool_stride=2):
    ''' Compute the receptive field of a U-Net architecture.'''
    rf = 1
    for _ in range(num_pools):
        rf = (rf + kernel_size -1) * pool_stride
    for _ in range(num_pools):
        rf = rf * pool_stride + kernel_size -1

    return rf

def experimentB(avg_radius_vals, num_pools_vals, num_channels, similarity, epochs=100, skip_connections=1, verbose=False):
    
    sigma_back = round(1/similarity, 1)
    IoU_gridB = np.zeros((len(avg_radius_vals), len(num_pools_vals)))
    for i, avg_radius in enumerate(avg_radius_vals):
        if verbose: 
            print('Dataset: \tSimilarity (fixed):', similarity,'\tAverage radius:', avg_radius)
        for j, num_pools in enumerate(num_pools_vals):
            if verbose: 
                print('\tArchitecture:\tNumber of initial channels (fixed):', num_channels,'\tNumber of pooling layers:', num_pools)
            IoU = run_unet(num_pools, num_channels, skip_connections, avg_radius, sigma_back, epochs=epochs, verbose=verbose)
            if verbose: 
                print('\t\tIoU = ', IoU)
            IoU_gridB[i,j] = IoU

    return IoU_gridB

def plot_experimentB(IoU_gridB, num_pools_vals, avg_radius_vals, save=False): 
    ''' Plot a IoU heatmap with `avg_radius` on the y-axis and `num_pools` on the x-axis using seaborn heatmap. '''

    val_min = np.floor(IoU_gridB[IoU_gridB>0.9].min()*50)/50
    val_max = np.ceil(IoU_gridB.max()*100)/100
    val_step = (val_max-val_min)/10

    # Create a heatmap with minimum 0 and maximum 1 with viridis
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        IoU_gridB[::-1, :], ax=ax, annot=True, cbar=True, vmin=val_min, vmax=val_max, fmt='.3f', 
        cbar_kws={'label': 'IoU'}, xticklabels=num_pools_vals, yticklabels=avg_radius_vals[::-1])
    ax.set_xlabel('Number of pooling layers')
    ax.set_ylabel('Average cell radius')
    ax.set_title('Experiment B: U-Net test IoU values for various number of pooling layers and cell radii', fontsize=14)

    # For each num_pools, compute receptive field size, and add an x-axis below num_pools with receptive field values, ticks at the center of each num_pools
    rf_vals = [receptive_field(num_pools) for num_pools in num_pools_vals]
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(np.arange(len(num_pools_vals))+0.5)
    ax2.set_xticklabels(rf_vals)
    ax2.set_xlabel('Receptive field size')

    ax.collections[0].colorbar.remove()
    fig.colorbar(
        ax.collections[0], ax=ax, label='IoU', shrink=0.5, 
        ticks=np.arange(val_min, val_max + 0.01, val_step), 
        orientation='vertical', pad=0.05)
    ax.collections[0].colorbar.ax.yaxis.set_label_position('left')
    ax.collections[0].colorbar.ax.yaxis.label.set_size(14)

    if save:
        plt.savefig(os.path.join(PLOTS_PATH, 'experimentB.png'))
    plt.show()


# --------------- Experiment C:  ----------------#

def experimentC(
    num_pools_vals, avg_radius_vals, 
    num_channels=16, similarity = 0.2, epochs=50, verbose=False
    ): 
    """ For each combination (avg_radius, num_pools, skip_connections), train U-Net and compute IoU. """

    # Initialize grid 
    width_skips = 2**max(num_pools_vals)
    IoU_gridC = np.zeros((len(num_pools_vals), width_skips, len(avg_radius_vals)))
    sigma_back = round(1/similarity, 1)

    for i, num_pools in enumerate(num_pools_vals):

        # Make a list of skip_connections
        skip_values = [[int(s) for s in bin(layer)[2:].zfill(num_pools)] for layer in range(2**num_pools)]
        if verbose: 
            print('Architecture: num_channels (fixed) = {}, num_pools = {}'.format(num_channels, num_pools))

        for j, skip_connections in enumerate(skip_values):
            if verbose: 
                print('\tSkip connections = {}'.format(skip_connections))
            
            for k, avg_radius in enumerate(avg_radius_vals):

                if verbose: 
                    print('Dataset:\tSimilarity (fixed):', similarity, '\tAverage radius:', avg_radius) 
                
                # For each skip connection combination, train the network and compute IoU
                IoU = run_unet(
                    num_pools, num_channels, skip_connections, 
                    avg_radius, sigma_back, epochs=epochs, verbose=verbose)
                if verbose: 
                    print('\t\tIoU = ', IoU)
                IoU_gridC[i,j,k] = IoU

    np.save(os.path.join(RESULTS_PATH, 'IoU_gridC.npy'), IoU_gridC)
    return IoU_gridC

def plot_experimentC(IoU_gridC, num_pools_vals, avg_radius_vals, save=False): 
    ''' Plot the relationship between IoU, average radius and skip connections for each number of pooling layers. '''

    val_min = np.floor(IoU_gridC[IoU_gridC>0.9].min()*50)/50
    val_max = np.ceil(IoU_gridC.max()*100)/100
    val_step = (val_max-val_min)/10
    skips_lists = []
    for num_pools in (num_pools_vals):
        skips_lists.append(
            [stringify_skips([int(s) for s in bin(layer)[2:].zfill(num_pools)], num_pools) for layer in range(2**num_pools)])

    fig, axes = plt.subplots(2,2, figsize=(12, 10))
    for i, num_pools in enumerate(num_pools_vals):
        row = i // 2
        col = i % 2
        grid = IoU_gridC[i,:,:]
        grid = grid[~np.all(grid == 0, axis=1)]

        sns.heatmap(
            grid, annot=True, fmt='.3f', ax=axes[row,col], cbar_kws={'label': 'IoU'},
            xticklabels=avg_radius_vals, yticklabels=skips_lists[i], 
            cbar=i==0, vmax=val_max, vmin=val_min)
        axes[row,col].set_xticklabels(axes[row,col].get_xticklabels(), rotation=0)
        axes[row,col].set_yticklabels(axes[row,col].get_yticklabels(), rotation=0)

        if num_pools == 1: 
            axes[row,col].set_title('1 pooling layer')
        else: 
            axes[row,col].set_title(str(num_pools) + ' pooling layers')

    # Format colorbar
    axes[0, 0].collections[0].colorbar.remove()
    fig.colorbar(
        axes[0, 0].collections[0], ax=axes, label='IoU', shrink=0.5, 
        ticks=np.arange(val_min, val_max + 0.01, val_step), 
        orientation='vertical', pad=0.05)
    axes[0, 0].collections[0].colorbar.ax.yaxis.set_label_position('left')
    axes[0, 0].collections[0].colorbar.ax.yaxis.label.set_size(14)

    fig.text(0.05, 0.5, 'Skip connections', ha='center', va='center', rotation='vertical', fontsize=14)
    fig.text(0.43, 0.05, 'Average cell radius', ha='center', va='center', fontsize=14)
    fig.suptitle('Experiment C: U-Net test IoU values for various combinations of skip connections and average cell radii', fontsize=14, y=0.95, x=0.43)

    if save:
        plt.savefig(os.path.join(PLOTS_PATH, 'experimentC.png'))
    plt.show()



# ----------------- Visualization tools ---------------- # 

def visualize_IOU(label, pred): 
    ''' Given a label image and its prediction, visualize the IOU by 
    coloring the label image with the prediction colors. '''
    label = label.astype(np.uint8)
    pred = pred.astype(np.uint8)

    TP = np.logical_and(label, pred)
    FP = np.logical_and(np.logical_not(label), pred)
    FN = np.logical_and(label, np.logical_not(pred))
    TN = np.logical_and(np.logical_not(label), np.logical_not(pred))

    # Color TN in black, TP in white, FP in red, FN in blue
    comparison = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    comparison[TP] = [0, 0, 0]
    comparison[FP] = [255, 0, 0]
    comparison[FN] = [0, 0, 255]
    comparison[TN] = [255, 255, 255]

    # Add legend: black for TP, red for FP, blue for FN, white for TN
    legend = np.zeros((50, 200, 3), dtype=np.uint8)
    legend[10:40, 10:40] = [0, 0, 0]
    legend[10:40, 50:80] = [255, 0, 0]
    legend[10:40, 90:120] = [0, 0, 255]
    legend[10:40, 130:160] = [255, 255, 255]
    cv2.putText(legend, 'TP', (45, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(legend, 'FP', (85, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(legend, 'FN', (125, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(legend, 'TN', (165, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # remove axis
    plt.axis('off')
    
    # Plot comparison
    plt.figure(figsize=(10, 10))
    plt.imshow(comparison)
    plt.show()
    return comparison


# ----------------- Main ---------------- #

if __name__ == '__main__':

    # Parse script argument
    parser = argparse.ArgumentParser()
    parser.add_argument('part', type=str, help='Experiment to run')
    args = parser.parse_args()

    if args.part == 'A': # Architecture experiment

        avg_radius_vals = [20, 30, 40, 50]
        similarity_vals = [1/2.5, 1/5.0, 1/7.5, 1/10]
        num_channels_vals = [2, 4, 8, 16, 32]
        num_pools_vals = [1, 2, 3, 4, 5]

        IoU_grid = experimentA(
            avg_radius_vals, similarity_vals, num_channels_vals, num_pools_vals, 
            epochs=50, skip_connections=1, verbose=False)

        plot_experimentA(IoU_grid, avg_radius_vals, similarity_vals, num_pools_vals, num_channels_vals, save=True)

        plot_experimentA_inverted(IoU_grid, avg_radius_vals, similarity_vals, num_pools_vals, num_channels_vals, save=True)


    elif args.part == 'B': # Receptive field experiment

        num_pools_vals = [1, 2, 3, 4, 5]
        avg_radius_vals = [20, 30, 40, 50]

        IoU_gridB = experimentB(num_pools_vals, avg_radius_vals, num_channels=16, similarity=0.2, skip_connections=1, epochs=100, verbose=False)

        plot_experimentB(IoU_gridB, num_pools_vals, avg_radius_vals, save=True)

    elif args.part == 'C': # Skip-connections experiment

        num_pools_vals = [1, 2, 3, 4]
        avg_radius_vals = [20, 30, 40, 50]

        IoU_gridC = experimentC(num_pools_vals, avg_radius_vals, num_channels=16, similarity = 0.2, epochs=50, verbose=False)


    else:
        print('Invalid argument. Please choose between experiments A, B, C and D.\nUsage: python3 experiments.py [A|B|C|D]')

