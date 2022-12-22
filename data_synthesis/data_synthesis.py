""" Generating synthetic data for biological cellular images. """

import os
import sys
import shutil
import matplotlib.image
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
import Augmentor
from PIL import Image
import imageio.v2 as imageio  
import pandas as pd
import cv2

DATASETS_PATH = "generated_datasets"


def generate_mask(image_size, angle, position, radii):
    """Generate a 0-1 Numpy array mask for cell position
    Inputs:
        image_size: (width, height) tuple of integers
        angle: float, angle of the cell
        position: (x_pos, y_pos) tuple of integers, center position of the cell
        radii: (x_rad, y_rad) tuple of integers, radii of the cell in x and y direction

    Outputs:
        mask: Numpy array, 0 = background, 1 = cell
    """
    # Extract parameters
    x_pos, y_pos = position
    width, height = image_size
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    x_rad, y_rad = radii

    # Create a binary mask for cell position
    mask = np.zeros(image_size)
    y, x = np.ogrid[0:height, 0:width]
    xh = x - x_pos
    yh = y - y_pos
    cell = (xh * cos_a + yh * sin_a) ** 2 / x_rad**2 + (
        xh * sin_a - yh * cos_a
    ) ** 2 / y_rad**2 <= 1
    mask[cell] = 1
    return mask


def generate_label(image_size, avg_radius, no_overlap=True, density=0.4):
    """
    Generate label mask of cells until we reach the desired density.
    Input:
        image_size: (width, height) tuple of integers
        avg_radius: integer, average radius of each cell
        density: float, desired density of cells in the image
    Output:
        label: Numpy array corresponding to a synthetic image
        num_cells: integer, number of cells in the image
        current_density: float, actual density of cells in the image
    """
    width, height = image_size
    label = np.zeros(image_size)
    num_cells = 0
    current_density = 0
    searching_center = True

    outer_tries = 0

    while current_density < density:

        searching_center = True
        tries = 0

        while searching_center and tries < 100:

            # Generate radii in x and y direction from normal distribution with minimum 5
            x_rad = max(5, np.random.normal(avg_radius, 1))
            y_rad = max(5, np.random.normal(avg_radius, 1))
            max_rad = int(np.ceil(max(x_rad, y_rad)))
            radii = (x_rad, y_rad)

            # Generate center position of cell within a margin of the border
            margin = int(max_rad)
            x_pos = round(np.random.uniform(margin, width - margin))
            y_pos = round(np.random.uniform(margin, height - margin))
            position = (x_pos, y_pos)

            # Generate random angle of the cell
            angle = round(np.random.uniform(0, 360))

            new_cell = generate_mask(image_size, angle, position, radii)

            # The intersection of the new cell and the existing cells overlaps, try again
            if no_overlap and np.any(label * new_cell):
                tries += 1
                continue

            else:
                searching_center = False
                outer_tries -= 1

            # Generate mask and add it to the image
            label = np.logical_or(label, new_cell)
            current_density = np.sum(label) / label.size
            num_cells += 1

        outer_tries += 1
        if outer_tries > 100:
            print("Could not generate cells with desired density.")
            break

    return label, num_cells, current_density


def smooth_binary(image, sigma_blur):
    new_image = gaussian_filter(image.astype(float), sigma_blur)
    threshold = np.mean(new_image[new_image > 0])
    new_image[new_image >= threshold] = 1
    new_image[new_image < threshold] = 0
    return new_image


def apply_deformation(label, magnitude, grid_size=3, sigma_blur=0.5): 
    ''' Given a Numpy 0-1 array, apply a random elastic deformation to it. '''
    # Create temporary directory
    temp_dir = "temp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.mkdir(temp_dir)

    # Save image to temporary directory 
    label = np.where(label, 255, 0)
    temp_image_path = os.path.join(temp_dir, 'temp.png')
    matplotlib.image.imsave(temp_image_path, label, cmap='gray')

    # These two give black images for some reason:
    #imageio.imwrite(temp_image_path, label.astype(np.uint8))  
    #cv2.imwrite(temp_image_path, label.astype(np.uint8))

    # Create augmentation pipeline in temporary directory, save image
    p = Augmentor.Pipeline(temp_dir)
    p.random_distortion(probability=1, grid_width=grid_size, grid_height=grid_size, magnitude=magnitude)
    p.gaussian_distortion(probability=1, grid_width=grid_size, grid_height=grid_size, magnitude=magnitude, corner='bell', method='in') 
    p.sample(1)

    # Read the deformed image and delete temporary directory
    output_filename = os.listdir(os.path.join(temp_dir, 'output'))[0]
    output_path = os.path.join(temp_dir, 'output', output_filename)
    new_label = np.array(Image.open(output_path).convert('L'))
    new_label = np.where(new_label > 200, 1, 0)
    shutil.rmtree(temp_dir)

    # Smooth out edges if specified
    # if sigma_blur > 0: 
    #     new_label = smooth_binary(new_label, sigma_blur)

    return new_label


def difference_of_gaussians(input, sigma, sigma_ratio=np.sqrt(2)):
    gaussian1 = gaussian_filter(input, sigma, mode="reflect")
    gaussian2 = gaussian_filter(input, sigma * sigma_ratio, mode="reflect")
    return gaussian2 - gaussian1


def generate_image(
    image_size=512,
    density=0.4,
    avg_radius=10,
    diff_mean=0.1,
    sigma_cell=1,
    sigma_back=10,
    magnitude=50, 
    grid_size=3,
    no_overlap=True,
):
    """Generates a noisy image with cells and background of different variance with fixed density
    Inputs:
        density = float, desired density of cells in the image
        avg_radius = float, average radius of each cell
        diff_mean: float, difference in noise mean between background and cells
        sigma_cell: float, std of cell noise
        sigma_back: float, std of background noise
        magnitude: float, magnitude of elastic deformation
    Outputs:
        image: Numpy array, image of cells and background
        labels: Numpy array, 0 = background, 1 = cell
        num_cells: integer, number of cells in the image
    """
    # Generate random label
    label, num_cells, _ = generate_label(
        image_size, avg_radius, no_overlap=no_overlap, density=density
    )

    # Apply elastic deformation
    if magnitude > 0: 
        label = apply_deformation(label, magnitude, grid_size)

    # Create noisy cells
    noise_label = np.random.normal(0, 10, image_size)
    dog_label = difference_of_gaussians(noise_label, sigma_cell)

    # Create noisy background
    noise_back = np.random.normal(0, 10, image_size)
    dog_back = difference_of_gaussians(noise_back, sigma_back)

    # Combine masks into a single image
    image = dog_label * label
    image += (dog_back - diff_mean) * (1 - label)
    return image, label, num_cells


def generate_dataset(
    image_size,
    density,
    avg_radius,
    diff_mean,
    sigma_cell,
    sigma_back,
    magnitude,
    grid_size,
    num_images,
    no_overlap=True,
):
    """Generates train and test sets of size num_images with the given hyperparameters."""

    d_path = DATASETS_PATH
    if not os.path.exists(d_path):
        os.mkdir(d_path)

    # Create dataset directory (overwrite if it already exists)
    dataset_path = os.path.join(
        d_path,
        f"dataset_rad{avg_radius:.1f}_sig{sigma_back:.1f}",
    )
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    os.mkdir(dataset_path)

    # Store parameters in csv file for reference    
    params = {
        "density": density,
        "avg_radius": avg_radius,
        "diff_mean": diff_mean,
        "sigma_cell": sigma_cell,
        "sigma_back": sigma_back,
        "magnitude": magnitude,
        "no_overlap": no_overlap,
        "num_images": num_images
    }
    params_path = os.path.join(dataset_path, "parameters.csv")
    df = pd.DataFrame(params, index=[0])

    # Create train and test directories
    train_path = os.path.join(dataset_path, "train")
    test_path = os.path.join(dataset_path, "test")
    os.mkdir(train_path)
    os.mkdir(test_path)

    # Create num_images in both train and test sets
    for path in [train_path, test_path]:

        # Store number of cells in each image in a csv file
        cell_path = os.path.join(path, "num_cells.csv")
        df_cells = pd.DataFrame(columns=["num_cells"])
        df_cells.index.name = "image number"

        images_path = os.path.join(path, "images")
        labels_path = os.path.join(path, "labels")
        os.mkdir(images_path)
        os.mkdir(labels_path)

        for i in range(1, num_images + 1):
            # Generate images and labels
            image, label, num_cells = generate_image(
                image_size, density, avg_radius, diff_mean, sigma_cell, sigma_back, magnitude, grid_size, no_overlap
            )
            image_path = os.path.join(images_path, f"{i:03d}.tif")
            label_path = os.path.join(labels_path, f"{i:03d}.tif")

            # Normalize image to 0-255 and save as uint8 TIFF
            norm_image = cv2.normalize(
                image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F
                ).astype(np.uint8)
            cv2.imwrite(image_path, norm_image)

            # Save label as 0-1 binary image TIFF
            cv2.imwrite(label_path, label.astype(np.uint8))

            # Save number of cells in each image
            df_cells.loc[i] = num_cells
    
        df_cells.to_csv(cell_path)

    df.assign(num_cells=num_cells).to_csv(params_path, index=False)



if __name__ == "__main__":

    arguments = sys.argv[1:]

    if len(arguments) != 10:
        print(
            "Usage: python3 generate_dataset.py <image_size> <density> <avg_radius> <diff_mean> <sigma_cell> <sigma_back> <magnitude> <grid_size> <num_images> <no_overlap>"
        )
        sys.exit(1)

    arg_names = [
        "image_size",
        "density",
        "avg_radius",
        "diff_mean",
        "sigma_cell",
        "sigma_back",
        "magnitude",
        "grid_size",
        "num_images",
        "no_overlap",
    ]
    arg_check = [int, float, float, float, float, float, float, int, int, int]
    for idx in range(len(arguments)):
        try:
            arg_check[idx](arguments[idx])
        except ValueError:
            print(arg_names[idx] + " should be " + arg_check[idx].__name__)
            print("Here is: " + arguments[idx])
            sys.exit(1)
        arguments[idx] = arg_check[idx](arguments[idx])

    arguments[0] = (arguments[0], arguments[0])
    arguments[-1] = bool(arguments[-1])
    seed = 1
    np.random.seed(seed)

    generate_dataset(*arguments)
