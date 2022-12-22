#!/bin/sh

# To run this script: 
#   cd data_synthesis
#   ./generate_datasets.sh

# This script runs multple instances of the data synthesis tool to generate different datasets. 

# "Usage: 
# python3 data_synthesis.py     <image_size> <density> <avg_radius> <diff_mean> <sigma_cell> <sigma_back> <magnitude> <grid_size> <num_images> <no_overlap>"

# We create 4x4 datasets of 100 (512 x 512) images by varying <avg_radius> and <sigma_back>. 
# We keep the other parameters fixed: {density = 0.4, diff_mean = 0, sigma_cell = 1, magnitude = 40, grid_size = 5}. 

# We define the similarity measure as: {similarity = sigma_cell / sigma_back = 1 / sigma_back}. 
# e.g.  sigma_back = 1 -> similarity = 1
#       sigma_back = 2 -> similarity = 0.5
#       sigma_back = 5 -> similarity = 0.2

# Values over which we iterate: 
#   - Radius = [20, 30, 40, 50]
#   - sigma_back = [2.5, 5, 7.5, 10] -> similarity = [0.4, 0.2, 0.133, 0.1]


# Radius 20

python3 data_synthesis.py 512 0.4 20 0 1 2.5 20 4 100 1

python3 data_synthesis.py 512 0.4 20 0 1 5 20 4 100 1

python3 data_synthesis.py 512 0.4 20 0 1 7.5 20 4 100 1

python3 data_synthesis.py 512 0.4 20 0 1 10 20 4 100 1

# Radius 30

python3 data_synthesis.py 512 0.4 30 0 1 2.5 20 4 100 1

python3 data_synthesis.py 512 0.4 30 0 1 5 20 4 100 1

python3 data_synthesis.py 512 0.4 30 0 1 7.5 20 4 100 1

python3 data_synthesis.py 512 0.4 30 0 1 10 20 4 100 1

# Radius 40

python3 data_synthesis.py 512 0.4 40 0 1 2.5 20 4 100 1

python3 data_synthesis.py 512 0.4 40 0 1 5 20 4 100 1

python3 data_synthesis.py 512 0.4 40 0 1 7.5 20 4 100 1

python3 data_synthesis.py 512 0.4 40 0 1 10 20 4 100 1

# Radius 50

python3 data_synthesis.py 512 0.4 50 0 1 2.5 20 4 100 1

python3 data_synthesis.py 512 0.4 50 0 1 5 20 4 100 1

python3 data_synthesis.py 512 0.4 50 0 1 7.5 20 4 100 1

python3 data_synthesis.py 512 0.4 50 0 1 10 20 4 100 1
