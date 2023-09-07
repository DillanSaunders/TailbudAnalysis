import numpy as np
import glob
import tifffile as tiff
import pandas as pd
from skimage.filters import gaussian, difference_of_gaussians, median
from skimage import exposure
from skimage.restoration import rolling_ball
import skimage.morphology as morph
import scipy.signal as signal
import matplotlib.cm as cm
import matplotlib.colors as col
import scipy.spatial as spatial


def read_images(path):
        
    full_paths = np.sort(glob.glob(path))
    
    filenames = []
    images = []
    voxel_dims = []
    for a_path in full_paths:
        filenames.append(a_path.split("/")[-1])
        images.append(tiff.imread(a_path))
        if a_path[-4:] == '.lsm':
            voxel_size= np.array([tiff.TiffFile(a_path).pages[0].tags['CZ_LSMINFO'].value['VoxelSizeZ'], tiff.TiffFile(full_paths[0]).pages[0].tags['CZ_LSMINFO'].value['VoxelSizeX'], tiff.TiffFile(full_paths[0]).pages[0].tags['CZ_LSMINFO'].value['VoxelSizeY']])
            voxel_size = voxel_size * 1000000
            voxel_dims.append(voxel_size)
        else:
            voxel_dims.append((1,1,1))
    
    voxel_dims = np.array(voxel_dims)
        
    return filenames, images, voxel_dims

#doesnt work for csvs use only for numpy files
def load_files(path, pandas = False):
    files = np.sort(glob.glob(path))
    all_data = []
    for file in files:
        if pandas == False:
            if file[-3:] == 'npy':
                data = np.load(file,allow_pickle='TRUE')
                all_data.append(data)
            elif file[-3:] == 'csv':
                data = np.loadtxt(file,delimiter = ',',skiprows = 1)
                all_data.append(data)
            else:
                print('incorrect format')
        elif pandas == True:
            if file[-3:] == 'csv':
                data = pd.read_csv(file)
            else:
                print('incorrect format for pandas loading')
        else:
            print('incorrect format')
    
    return all_data


def split_channels(images, channel_axis = 1):
    '''splits channels into lists and then assigns them to values in a dictionary. Can split any number of channels. '''
    
    channel_split = {}
    for i in range(len(images)):
        channels = np.split(images[i], images[i].shape[channel_axis],axis = channel_axis)
        if i == 0:
            for j in range(len(channels)):
                print(f'image {i} has {j} channels')
                channel_split[f'ch{j}'] = [channels[j]]
        else:
            for j in range(len(channels)):
                print(f'image {i} has {j} channels')
                channel_split[f'ch{j}'].append(channels[j])
                
    return channel_split

def merge_channels(channel0, channel1, channel2):
    
    images = []
    
    for i in range(len(channel0)):
        image = np.concatenate((channel0[i],channel1[i],channel2[i]), axis = 1)
        images.append(image)
    
    return images

def clahe(images, kernel_clahe = (50,50,10), clahe_limit = 0.05):
    
    assert type(images) == list, 'Input images must be in a list'
    
    images_clahe = []
    
    for image in images:
        if len(image.shape) == 3:
            image = image.transpose()
            image_clahe = exposure.equalize_adapthist(image,kernel_size = kernel_clahe, clip_limit = clahe_limit)
            image_clahe = image_clahe.transpose()
            images_clahe.append(image_clahe)
        else: 
            channel_number = image.shape[1] #assuming that channel number is the second dimension of the np array
            channels_array = np.empty(image.shape)
            for channel in range(channel_number):
                single_channel = image[:,channel,...].transpose()
                single_channel_clahe = exposure.equalize_adapthist(single_channel,kernel_size = kernel_clahe, clip_limit = clahe_limit)
                single_channel_clahe = single_channel_clahe.transpose()
                channels_array[:,channel,...] = single_channel_clahe
            images_clahe.append(channels_array)
    
    return images_clahe

def batch_gaussian(images, sigma = 1, diff_gauss = False, high_sigma = 3, channel_axis = 1, preserve_range = True):
    
    
    images_gauss = []
    
    for i in range(len(images)):
        img = images[i]
    
        if diff_gauss == False:
            img_gauss = gaussian(img, sigma, channel_axis = channel_axis, preserve_range = preserve_range)
        elif diff_gauss == True:
            img_gauss = difference_of_gaussians(img, low_sigma = sigma, high_sigma = high_sigma, channel_axis = channel_axis)
    
        images_gauss.append(img_gauss.reshape((img_gauss.shape[0],)+img_gauss.shape[2:]))
        
    
    return images_gauss

def supress_background(image):
    # create structuring element that is the size of a true HCR nuclear spot (5,5,3 pixels)
    disk = morph.disk(2)
    disk = np.stack((disk,disk,disk),axis=0)
    # use element to remove salt and pepper noise below this size/shape, 2x median filter enhances difference between background and signal
    img_med = median(image,footprint= disk)
    img_med = median(img_med,footprint= disk)
    # remove further background using rolling ball subtraction with same element
    bk_roll = rolling_ball(img_med,kernel=disk)
    img_roll = img_med - bk_roll
    
    return img_roll

def qqplot(array1, array2):
    
    q1 = np.percentile(array1,range(100))
    q2 = np.percentile(array2,range(100))
    
    plt.plot(q1,q1)
    plt.scatter(q1,q2)
    plt.show()
    
    return plt

def plotting_grid_index(cols,rows):
    total_plots = cols+ rows
    row_index = []
    col_index = []
    for i in range(len(total_plots)):
        row_index.append(i//cols)
        col_index.append(i%cols)
        
    return row_index,col_index

def optimal_nn_radius(data,voxel_dims):
    
    #kd tree with a radius of 35 gives most common nn num to be 14 which is the number of nearest neighbours in a cube
    kd_tree = spatial.KDTree(data.loc[:,['z','y','x']][~np.isnan(data.loc[:,['z','y','x']])]*voxel_dims)
    
    mean_nn = 0
    radius = 0
    while mean_nn < 14:
        radius += 1
        nn_index = kd_tree.query_ball_tree(kd_tree,r=radius)
        mean_nn = np.mean([len(pt_nn) for pt_nn in nn_index])
    
    return radius

def data_nn(data,voxel_dims,radius):
    #kd tree with a radius of 35 gives most common nn num to be 14 which is the number of nearest neighbours in a cube
    kd_tree = spatial.KDTree(data.loc[:,['z','y','x']][~np.isnan(data.loc[:,['z','y','x']])]*voxel_dims)
    nn_index = kd_tree.query_ball_tree(kd_tree,r=radius)
    
    nn_num = [len(pt_nn) for pt_nn in nn_index]
    
    nn_data = pd.concat([np.mean(data.iloc[i[1:]],axis=0) for i in nn_index],axis=1).transpose()
    nn_data.index= data.index
    nn_data['num_nn'] = nn_num  

    nn_data_std = pd.concat([np.std(data.iloc[i[1:]],axis=0) for i in nn_index],axis=1).transpose()
    nn_data_std.index= data.index
    nn_data_std['num_nn'] = nn_num
    
    return nn_data,nn_data_std


def color_map(int_array, cmap):
    min_intensity = np.nanmin(int_array)
    max_intensity = np.nanmax(int_array)
    norm = col.Normalize(vmin=min_intensity,vmax = max_intensity)
    col_map = cm.get_cmap(cmap)
    lut=cm.ScalarMappable(norm=norm,cmap=col_map)
    rgb_array = np.empty((len(int_array),4))
    
    for i in range(len(int_array)):
        rgb_array[i,:] = lut.to_rgba(int_array[i])
        
    return rgb_array

def density_map(xaxis_array,yaxis_array,window_size=20,normalize = False):
    
    assert(type(xaxis_array) == type(yaxis_array)), 'Both arrays must be of the same type'
    
    if type(xaxis_array) == np.ndarray or type(xaxis_array) == pd.core.series.Series:
        assert(xaxis_array.shape[0] == yaxis_array.shape[0]),'Both arrays must have the same length'

        if normalize == True:
            xaxis_array = nucseg.decimal_normalization(xaxis_array)
            yaxis_array = nucseg.decimal_normalization(yaxis_array)
        
        xaxis_max = np.round(np.nanmax(xaxis_array))
        yaxis_max = np.round(np.nanmax(yaxis_array))
        xaxis_min = np.round(np.nanmin(xaxis_array))
        yaxis_min = np.round(np.nanmin(yaxis_array))

        xaxis_intervals = np.linspace(xaxis_min,xaxis_max,window_size)
        yaxis_intervals = np.linspace(yaxis_min,yaxis_max,window_size)

        density = np.zeros((window_size,window_size),dtype = np.int16)
        for i in range(len(xaxis_intervals)-1):
            for j in range(len(yaxis_intervals)-1):
                density[-(j+1),i] = xaxis_array[(xaxis_array >= xaxis_intervals[i])&(xaxis_array < xaxis_intervals[i+1]) & (yaxis_array >= yaxis_intervals[j]) & (yaxis_array <yaxis_intervals[j+1])].shape[0]
    
    elif type(xaxis_array) == list:
        density = np.zeros((window_size,window_size),dtype = np.int16)
        xaxis_max = np.round(np.nanmax(np.concatenate(xaxis_array)))
        yaxis_max = np.round(np.nanmax(np.concatenate(yaxis_array)))
        xaxis_min = np.round(np.nanmin(np.concatenate(xaxis_array)))
        yaxis_min = np.round(np.nanmin(np.concatenate(yaxis_array)))

        xaxis_intervals = np.linspace(xaxis_min,xaxis_max,window_size)
        yaxis_intervals = np.linspace(yaxis_min,yaxis_max,window_size)
        for array_index in range(len(xaxis_array)):
            assert(xaxis_array[array_index].shape[0] == yaxis_array[array_index].shape[0]),'Both arrays must have the same length'

            if normalize == True:
                xaxis_array[array_index] = nucseg.decimal_normalization(xaxis_array[array_index])
                yaxis_array[array_index] = nucseg.decimal_normalization(yaxis_array[array_index])
            
            

            for i in range(len(xaxis_intervals)-1):
                for j in range(len(yaxis_intervals)-1):
                    if xaxis_array[array_index][(xaxis_array[array_index] >= xaxis_intervals[i])&(xaxis_array[array_index] < xaxis_intervals[i+1]) & (yaxis_array[array_index] >= yaxis_intervals[j]) & (yaxis_array[array_index] <yaxis_intervals[j+1])].shape[0] > 0:
                        density[-(j+1),i] += 1
    return density