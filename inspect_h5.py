import h5py
import os

def inspect_h5_file(file_path):
    try:
        with h5py.File(file_path, 'r') as file:
            print(f"Inspecting {file_path}")
            print("Available datasets/tags:")
            file.visititems(lambda name, obj: print(f" - {name}"))
            print("\n")
    except Exception as e:
        print(f"Error inspecting {file_path}: {e}")

def inspect_h5_files(file_list_path):
    try:
        with open(file_list_path, 'r') as f:
            file_paths = [line.strip() for line in f]
        
        for file_path in file_paths:
            inspect_h5_file(file_path)
    except Exception as e:
        print(f"Error reading file list: {e}")

if __name__ == "__main__":
    train_file_list = '/home/abhat/wirecell_sbnd/Pytorch-UNet/bnb_target_short.file'
    val_file_list = '/home/abhat/wirecell_sbnd/Pytorch-UNet/bnb_image_short.file'
    
    print("Inspecting training files:")
    inspect_h5_files(train_file_list)
    
    print("Inspecting validation files:")
    inspect_h5_files(val_file_list)
