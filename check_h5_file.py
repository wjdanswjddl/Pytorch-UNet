import h5py

def count_hdf5_groups(file_path):
    group_count = 0

    def count_groups(name, obj):
        nonlocal group_count
        if isinstance(obj, h5py.Group):
            group_count += 1
    
    with h5py.File(file_path, 'r') as file:
        file.visititems(count_groups)
    
    return group_count

# file_img = '/scratch/7DayLifetime/munjung/DNN_ROI/train/smeared/tpc1_bothplanes-rec.h5'
# file_mask = '/scratch/7DayLifetime/munjung/DNN_ROI/train/smeared/tpc1_bothplanes-tru.h5'

# img_group_count = count_hdf5_groups(file_img)
# mask_group_count = count_hdf5_groups(file_mask)

# print(f"Image File '{file_img}' has {img_group_count} groups.")
# print(f"Mask File '{file_mask}' has {mask_group_count} groups.")

# file_img = '/scratch/7DayLifetime/munjung/DNN_ROI/train/smeared/tpc0_plane0_1000-rec.h5'
# file_mask = '/scratch/7DayLifetime/munjung/DNN_ROI/train/smeared/tpc0_plane0_1000-tru.h5'

# img_group_count = count_hdf5_groups(file_img)
# mask_group_count = count_hdf5_groups(file_mask)

# print(f"Image File '{file_img}' has {img_group_count} groups.")
# print(f"Mask File '{file_mask}' has {mask_group_count} groups.")

# file_img = '/scratch/7DayLifetime/munjung/DNN_ROI/train/smeared/tpc0_plane1_1000-rec.h5'
# file_mask = '/scratch/7DayLifetime/munjung/DNN_ROI/train/smeared/tpc0_plane1_1000-rec.h5'

# img_group_count = count_hdf5_groups(file_img)
# mask_group_count = count_hdf5_groups(file_mask)

# print(f"Image File '{file_img}' has {img_group_count} groups.")
# print(f"Mask File '{file_mask}' has {mask_group_count} groups.")

# file_img = '/scratch/7DayLifetime/munjung/DNN_ROI/train/smeared/tpc1_bothplanes_with_prolongedtrks-rec.h5'
# file_mask = '/scratch/7DayLifetime/munjung/DNN_ROI/train/smeared/tpc1_bothplanes_with_prolongedtrks-tru.h5'

# img_group_count = count_hdf5_groups(file_img)
# mask_group_count = count_hdf5_groups(file_mask)

# print(f"Image File '{file_img}' has {img_group_count} groups.")
# print(f"Mask File '{file_mask}' has {mask_group_count} groups.")

# file_img = '/scratch/7DayLifetime/abhat/wirecell/dnn_roi/tpc1_plane0_80-82-rec.h5'
# file_mask = '/scratch/7DayLifetime/abhat/wirecell/dnn_roi/tpc1_plane0_80-82-tru.h5'

# img_group_count = count_hdf5_groups(file_img)
# mask_group_count = count_hdf5_groups(file_mask)

# print(f"Image File '{file_img}' has {img_group_count} groups.")
# print(f"Mask File '{file_mask}' has {mask_group_count} groups.")

# file_img = '/scratch/7DayLifetime/abhat/wirecell/dnn_roi/tpc1_plane1_80-82-rec.h5'
# file_mask = '/scratch/7DayLifetime/abhat/wirecell/dnn_roi/tpc1_plane1_80-82-tru.h5'

# img_group_count = count_hdf5_groups(file_img)
# mask_group_count = count_hdf5_groups(file_mask)

# print(f"Image File '{file_img}' has {img_group_count} groups.")
# print(f"Mask File '{file_mask}' has {mask_group_count} groups.")

file_img = '/scratch/7DayLifetime/abhat/wirecell/dnn_roi/tpc1_plane0_75-80-rec.h5'
file_mask = '/scratch/7DayLifetime/abhat/wirecell/dnn_roi/tpc1_plane0_75-80-tru.h5'

img_group_count = count_hdf5_groups(file_img)
mask_group_count = count_hdf5_groups(file_mask)

print(f"Image File '{file_img}' has {img_group_count} groups.")
print(f"Mask File '{file_mask}' has {mask_group_count} groups.")
