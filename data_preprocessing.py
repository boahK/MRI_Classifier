
import os
import nibabel as nib
import nibabel.processing as nib_processing

###################################################################
#### Please set the following directories proper to yours
directory_niftis = './data/DLDS_dataset'
preprocessed_path = './data/preprocessed'
###################################################################

print('##### Data preprocessing: resampling')
sequence_folders = ['T1w_pre', 'T1w_art', 'T1w_ven', 'T1w_del', 'T2w', 'T2fs', 'DWI', 'ADC'] 
for folder in sequence_folders:
    print('------- Processing %s' % folder)
    os.makedirs(os.path.join(preprocessed_path, folder), exist_ok=True)
    for file in os.listdir(directory_niftis + '/' + folder):

            file_full = file 
            suffix = file.index('.')
            file = file[:suffix]
            
            nii = nib.load(directory_niftis + '/' + folder + '/' + file_full)
            #save spacing info in x, y, and z dimensions
            if len(nii.header.get_zooms()) == 4: 
                time_dim_len = nii.header.get_data_shape()[3]
            else:
                time_dim_len = 1
            
            for x in range(time_dim_len):
                
                if time_dim_len != 1:
                    new_nii = nii.slicer[:,:,:,x]
                    new_file = file + "_" + str(x)
                else:
                    new_nii = nii
                    new_file = file
                
                #save sample sizes
                sx, sy, sz = new_nii.header.get_zooms()
                
                #save original voxels
                voxels_x = new_nii.header.get_data_shape()[0]
                voxels_y = new_nii.header.get_data_shape()[1]
                voxels_z = new_nii.header.get_data_shape()[2]

                #define resample sizes
                new_sx, new_sy, new_sz = 1.5, 1.5, 7.8

                #find the no of voxels in each dimension associated with each new resample size
                target_voxels_x = int((sx * new_nii.header.get_data_shape()[0]) / new_sx)
                target_voxels_y = int((sy * new_nii.header.get_data_shape()[1]) / new_sy)
                target_voxels_z = int((sz * new_nii.header.get_data_shape()[2]) / new_sz)

                #resample image
                new_nii = nib_processing.conform(new_nii, out_shape=(target_voxels_x, target_voxels_y, target_voxels_z), voxel_size=(new_sx, new_sy, new_sz), order=3, cval=0.0, orientation='RAS', out_class=None)
                        
                #save image to directory
                nib.save(new_nii, os.path.join(preprocessed_path, folder, '%s.nii.gz'%new_file)) 
                         
print("##### Complete")