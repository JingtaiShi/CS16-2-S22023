import os
import numpy as np
vipshome = r'F:\research\vips-dev-8.14\bin'
current_directory = os.getcwd()
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']

importpyvips
import glob


# New folder path
input_folder = r'F:\research2'

# Get the paths of all .svs files in the folder
svs_files = glob.glob(os.path.join(input_folder, '*.svs'))

for TCGA_path in svs_files:
img = pyvips.Image.new_from_file(TCGA_path, access='sequential')

    # Get the file name, excluding the path
    filename = os.path.basename(TCGA_path)

    # Use dots to separate file names and get the first part as the output folder name
    output_folder_name = filename.split('.')[0]

    #Create the path to the output folder
    output_folder = os.path.join(current_directory, 'output', output_folder_name)

    # Delete all files in the output folder
existing_files = glob.glob(os.path.join(output_folder, '*'))
    for file in existing_files:
        os.remove(file)

    # Save image slices to the output folder and define the size of the output patch (8192*8192)
    img.dzsave(output_folder, tile_height=8192, tile_width=8192, depth='one')