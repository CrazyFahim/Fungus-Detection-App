import os
import cv2

# Define input and output folders
dataset_folder = '/home/f4h1m/Documents/CSE3200/Binary_classification/project-2/dataset'
H5_folder = '/home/f4h1m/Documents/CSE3200/Binary_classification/project-2/dataset/H5'
H6_folder = '/home/f4h1m/Documents/CSE3200/Binary_classification/project-2/dataset/H6'
output_folder = '/home/f4h1m/Documents/CSE3200/Binary_classification/project-2/preprocessed_dataset'
# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to resize images to 179x178 pixels
def resize_image(image):
    return cv2.resize(image, (179, 179))

# Function to process images in a folder
def process_folder(input_folder, output_subfolder):
    # Create output subfolder
    output_subfolder_path = os.path.join(output_folder, output_subfolder)
    os.makedirs(output_subfolder_path, exist_ok=True)
    
    # Loop through images in the input folder
    for img_file in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_file)
        # Check if it's a file
        if os.path.isfile(img_path):
            # Read the image
            image = cv2.imread(img_path)
            if image is not None:  # Check if image is loaded successfully
                # Resize the image
                resized_image = resize_image(image)
                # Save the resized image
                output_path = os.path.join(output_subfolder_path, img_file)
                cv2.imwrite(output_path, resized_image)
                print(f"Processed: {img_file}")
            else:
                print(f"Failed to load: {img_file}")
        else:
            print(f"{img_path} is not a file.")

# Process H5 folder
process_folder(H5_folder, 'H5')

# Process H6 folder
process_folder(H6_folder, 'H6')