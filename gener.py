import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

def load_and_save_mat_file(mat_file_path, save_dir):
    # Load the .mat file using h5py
    with h5py.File(mat_file_path, 'r') as mat_data:
        # Access the key in the dictionary where the prediction is stored
        prediction = mat_data['predict_Gan_0_dB_Indoor2p4_64ant_32users_8pilot'][:]
    
    # Select the first slice along the third dimension (index 0)
    prediction_image = np.squeeze(prediction[0, :, :, 0])  # Assuming you want to visualize the first slice

    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the predicted image as a PNG file
    file_name = os.path.basename(mat_file_path).replace('.mat', '.png')
    save_path = os.path.join(save_dir, file_name)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(prediction_image)  # Use the default colormap for a colorful image
    plt.title('Predicted Channel Image')
    plt.axis('off')  # Hide axis for cleaner visualization
    plt.savefig(save_path, bbox_inches='tight')  # Save the image
    plt.close()  # Close the plot to free up memory

    print(f"Saved plot as: {save_path}")

# Example: Load and save from a .mat file generated at epoch 10
mat_file_path = 'Eest_cGAN_1_0db_Indoor2p4_64ant_32users_8pilot.mat'
save_dir = 'saved_plots'
load_and_save_mat_file(mat_file_path, save_dir)
