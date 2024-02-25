import pandas as pd
import shutil
import os

# Read the CSV file into a DataFrame
df = pd.read_csv('/home/pengshijie/dybranch/evaluation/metrics/evaluation_record/evaluation_record_exp_EP3_final.csv')

# Specify the destination directory
destination_base_directory = 'metrics/wula_RR/exp_EP3_final/'

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    # Extract the CSV file path from the 'csv' column
    csv_file_path = row['csv']
    
    # Get the GPU value from the 'gpu' column
    gpu_value = row['gpu']
    
    # Extract the path after 'metrics/wula_RR/' to maintain the subsequent structure
    parts = csv_file_path.split('/')
    
    # Take the last three parts to form the relative path
    relative_path = os.path.join(parts[-3], parts[-2], parts[-1])
    
    # Construct the destination directory path
    destination_directory = os.path.join(destination_base_directory, gpu_value, relative_path)
    
    # Create the destination directory if it doesn't exist
    os.makedirs(destination_directory, exist_ok=True)
    
    # Construct the full destination file path
    destination_file_path = os.path.join(destination_directory, os.path.basename(csv_file_path))
    shutil.copytree(csv_file_path, destination_directory)
    # Copy the CSV file to the destination directory
    # shutil.copyfile(csv_file_path, destination_file_path)

print("Files copied successfully.")
