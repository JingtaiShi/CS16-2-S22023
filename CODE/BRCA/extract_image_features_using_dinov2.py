import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
import os 
import pandas as pd

patch_h = 1
patch_w = 1
feat_dim = 1536  # vitg14

transform = T.Compose([
    T.GaussianBlur(9, sigma=(0.1, 2.0)),
    T.Resize((patch_h * 14,
patch_w * 14)),
    T.CenterCrop((patch_h * 14, patch_w * 14)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

dinov2_vitg14 = torch.hub.load('', 'dinov2_vitg14', source='local').cuda()

print(dinov2_vitg14)

#Set image folder path
image_folder_path = 'images' # Modify the image folder path

#Set the folder path to save vectors
output_vector_folder = 'imagesDino2_vitg14_2' # Modify the folder path to save vectors

# Get the image file list
image_files = [os.path.join(image_folder_path, filename) for filename in os.listdir(image_folder_path) if filename.endswith(".jpeg")] # Get all file names ending with .jpeg

# Create an empty numpy array object to store the feature vectors of all images
all_features = np.empty((0, 6144), dtype=float)

# Extract the vector representation of image data and save it
os.
makedirs(output_vector_folder, exist_ok=True) # Create a folder to save the vector, ignore it if it already exists

#Create an empty numpy array object to store the attrib_name of all images
all_attrib_names = []

for image_file in image_files: # Traverse each image file
    # Get the first part of the file name as attrib_name
    attrib_name = os.path.basename(image_file).split('-')[0]
    #Load images and preprocess them
    features = torch.zeros(4, patch_h * patch_w, feat_dim)
imgs_tensor = torch.zeros(4, 3, patch_h * 14, patch_w * 14).cuda()

    img = Image.open(image_file).convert('RGB')
    imgs_tensor[0] = transform(img)[:3]
    with torch.no_grad():
        features_dict = dinov2_vitg14.forward_features(imgs_tensor)
        features = features_dict['x_norm_patchtokens']

    features = features.reshape(1, -1).
cpu().numpy()
    #Append the feature vector corresponding to each image file to the numpy array object as a row of data
    all_features = np.append(all_features, features, axis=0)

    all_attrib_names.append(attrib_name)

    # Print a line of log showing the name of the currently processed image file and the shape of the feature vector
    print(f"Processed {image_file} and added vector with shape {features.shape}")

# After traversing all image files, save them as csv files
output_csv_path = os.path.join(output_vector_folder
, "all_images.csv")

# Convert all_features into a data frame with column names 'f1', 'f2', ..., 'f6144'
features_df = pd.DataFrame(all_features, columns=['f'+str(i) for i in range(1, 6145)])

# Use the iloc function to select the first 1536 columns of the data frame
features_df = features_df.iloc[:, :1536]

# Convert all_attrib_names into a data frame with the column name 'attrib_name'
attrib_names_df = pd.DataFrame(all_attrib_names, columns=['attrib_name'])
# Merge the two data frames in the horizontal direction, that is, add attrib_name as the first column to features_df
merged_df = pd.concat([attrib_names_df, features_df], axis=1)

# Save the merged data frame as a csv file, do not retain the index, and use commas to separate
merged_df.to_csv(output_csv_path, index=False, sep=',')

# Print a line of log to show that the save was successful.
print(f"All image vectors and attrib_name have been saved to {output_csv_path}")