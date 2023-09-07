import numpy as np
import glob
import tifffile as tiff
from stardist.models import StarDist2D
from csbdeep.utils import Path, normalize
import pandas as pd
from skimage.color import label2rgb
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import copy
import scipy.interpolate as interpolate
import scipy.spatial as spatial
import napari
from skimage.measure import regionprops
import math


viewer = napari.Viewer()

@viewer.bind_key('g')
def get_viewer(viewer):
    return viewer

@viewer.bind_key('c',overwrite=True)
def crop_tailbud(viewer):
    img_layer = np.where(['RAW' in layer.name for layer in viewer.layers ]==True)[0][0]
    mask = viewer.layers.selection.active.to_masks(mask_shape = img_layer.shape[1:])
    mask_z = mask
    for i in range(img_layer.shape[0]-1):
        mask_z = np.concatenate((mask_z,mask), axis = 0)
    np.save('masks/MSK'+img_layer.name[3:-3]+'npy', mask_z)
    print(f'saving mask for image {img}')

    return mask_z

def stardist_2d(images,filenames,pretrained = True, model = None):
    
    if pretrained == True:
        model = StarDist2D.from_pretrained('2D_versatile_fluo')
    else:
        model = model
        assert (model != None), 'specify model in function'
    
    images_labels = []
    images_details = []
    for j in range(len(images)):
        print(f'Starting segmentation of image {j}')
        labels_all = np.empty(images[j].shape, dtype = "int32")
        details_all = []
        
        for i in range(len(images[j])):
            img = normalize(images[j][i,...], 1,99.8, axis=(0,1))
            labels, details = model.predict_instances(img)
            labels_all[i] = labels
            details_all.append(details)
        lbl_name = 'LBL' + filenames[j][3:-3] + 'npy'
        detail_name = 'DTL' + filenames[j][3:-3] + 'npy'
        np.save(lbl_name, labels_all) 
        np.save(detail_name, details_all)
        images_labels.append(labels_all)
        images_details.append(details_all)
        print(f'Finishing segmentation of image {j}')
        
    return images_labels, images_details

def label_links_mapping(labels_test, name, iou_threshold = 0.5,trim_labels = False,additional_props=None):    
    # find total number of labels across all z slices 
    num_labels = 0
    for i in range(len(labels_test)):
        num_labels += labels_test[i].max()

    # create global and local mapping matrices     
    global2local = np.zeros((num_labels+1,len(labels_test)),dtype = np.uint)
    local2global = np.zeros((labels_test.max()+1, len(labels_test)), dtype = np.uint)

    num_labels = 0
    for z in range(len(labels_test)-1):
        slice1_masks = labels_test[z].ravel()
        slice2_masks = labels_test[z+1].ravel()

        # calculate overlap and generate IOU matrix for the current slice and the next slice - code adapted from cellpose _label_overlap function. Flatten consecutive slices and create matrix of the number of occurances of each mask. Use this to calculate IoU 

        overlap_matrix = np.zeros((slice1_masks.max()+1,slice2_masks.max()+1), dtype=np.uint)
        for j in range(len(slice1_masks)):
            overlap_matrix[slice1_masks[j],slice2_masks[j]] += 1
        n_pixels_slice1 = np.sum(overlap_matrix, axis=0, keepdims=True)
        n_pixels_slice2 = np.sum(overlap_matrix, axis=1, keepdims=True)
        iou_matrix = overlap_matrix/(n_pixels_slice1 + n_pixels_slice2 - overlap_matrix)
        
        if trim_labels == True:
            props1 = regionprops(labels_test[z])
            props2 = regionprops(labels_test[z+1])
            perim_slice1 = np.concatenate(([1],np.array([i.perimeter for i in props1])))
            perim_slice2 = np.concatenate(([1],np.array([i.perimeter for i in props2])))
            area_slice1 = np.concatenate(([1],np.array([i.area for i in props1])))
            area_slice2 = np.concatenate(([1],np.array([i.area for i in props2])))
            #additional shape properties 
            '''if additional_props != None:
                assert(type(additional_props) == dict), 'additional props must be a dictionary of additional shape properties from skimage regionprops'

                orient_matrix = np.empty(overlap_matrix.shape)
                orient_slice1 = np.concatenate(([0],np.array([i.orientation for i in props1])))
                orient_slice2 = np.concatenate(([0],np.array([i.orientation for i in props2])))

                for i in range(len(orient_slice1)):
                    orient_matrix[i] = orient_slice1[i]/orient_slice2

                shape_matrix = np.empty(overlap_matrix.shape)
                for i in range(len((perim_slice1/np.sqrt(area_slice1)))):
                    shape_matrix[i] = (perim_slice1/np.sqrt(area_slice1))[i]/(perim_slice2/np.sqrt(area_slice2))
                    '''
            size_mask1 =  (area_slice1 > 400)&(area_slice1 < 3000)#((perim_slice1/area_slice1) > 0.03)&((perim_slice1/area_slice1) <0.13)
            size_mask2 =   (area_slice2 > 400)&(area_slice2 < 3000)#((perim_slice2/area_slice2) > 0.03)&((perim_slice2/area_slice2) <0.13)
            #setting the overlaps of all small and large labels to zero
            iou_matrix = iou_matrix * size_mask1.reshape(size_mask1.shape+(1,)) * size_mask2
            
        # find unique labels in current and subsequent slices, convert to global identity using num_labels count. Increment num_labels
        slice1_local = np.unique(slice1_masks)
        slice1_global = slice1_local + num_labels 
        slice2_local = np.unique(slice2_masks)
        num_labels += slice1_local.max()
        slice2_global = slice2_local + num_labels

        # for each local label in the slice get the local and global number, determine overlap with any labels in next slice
        # if overlap the value of the matched label is set both locally and globally
        for i in range(len(slice1_local)-1):
            slice1_label = slice1_local[i+1]
            slice1_label_global = slice1_global[slice1_label]
            slice2_index = np.where(iou_matrix[slice1_label] > iou_threshold)[0]
            #&(orient_matrix[slice1_label] >0)&(orient_matrix[slice1_label] <2)&(shape_matrix[slice1_label] >0.9)&(shape_matrix[slice1_label] <1.1)
            if slice2_index.size == 1:
                slice2_local_match = slice2_index[0]
                slice2_global_match = slice2_global[slice2_local_match]
            else:
                slice2_local_match = 0
                slice2_global_match = 0

            # if the current label is not linked to a previous label it is added into the matrices sequentially
            # if the current label is linked to previous then it is added in with the same index as the first label of the linking
            linkto_previous = np.where(global2local[:,z] == slice1_label)[0]
            if linkto_previous.size == 0:
                local2global[slice1_label,z] = slice1_label_global
                global2local[slice1_label_global, z] = slice1_label
                global2local[slice1_label_global,z+1] = slice2_local_match
            elif linkto_previous.size > 0:
                global2local[linkto_previous[0],z+1] = slice2_local_match
                local2global[slice1_label,z] = linkto_previous[0]
                
    np.save('g2l'+name[3:-3]+'npy', global2local, allow_pickle = True) 
    np.save('l2g'+name[3:-3]+'npy', local2global, allow_pickle = True) 
                
    return global2local, local2global

# create global color map and shuffle to randomise colors
def global_colormap(matplotlib_cmap, cmap_length):

    cmap = LinearSegmentedColormap.from_list('cmap', matplotlib_cmap(range(256)), N= cmap_length)
    global_cmap = cmap(range(cmap_length))[:,:3]
    np.random.shuffle(global_cmap)
    
    return global_cmap


# for each slice finds labels, maps to global label number and that is used to index the global_cmap
# linked set of labels have the same global label ID and therefore the same color
# this is then converted to RGB 
# the slices are then concatenated back together
def label_rgb(labels_test, global_cmap, local2global):
    labels_rgb = []
    for z in range(len(labels_test)):
        local_index = np.unique(labels_test[z])[1:]
        global_index = local2global[local_index, z]
        local_cmap = global_cmap[global_index]
        z_rgb = label2rgb(labels_test[z],colors = local_cmap)
        z_rgb = z_rgb.reshape((1,1024,1024,3))
        labels_rgb.append(z_rgb)

    labels_rgb = np.concatenate(labels_rgb, axis = 0)
    
    return labels_rgb

def subset_cmap(bool_array, labels, g2l, l2g):
    cmap = global_colormap(cm.jet, len(g2l))
    
    for i in range(len(bool_array)):
        if bool_array[i] == 0: #if the number of local labels greater than threshold is zero then they arent part of a bigger label and can be removed
            cmap[i] = 0 #the global index can be used to mask the value of cmap to zero
        
    labels_rgb = label_rgb(labels,cmap,l2g) #then run label_rgb to generate the subset colored image
    
    return labels_rgb

def relabeller(label_image,g2l):
    lbl_image = copy.deepcopy(label_image)
    lbl_image[lbl_image == 0] = len(g2l)
    for i in range(len(g2l)):
        if np.sum(g2l[i] > 0) >= 1:
            local_ids = g2l[i][g2l[i] > 0]
            slice_loc = np.where(g2l[i] > 0)[0]
            
            for j in range(len(local_ids)):
                lbl_image[slice_loc[j]][lbl_image[slice_loc[j]] == local_ids[j]]= i + len(g2l) 
        
        else:
            
            local_ids = g2l[i][g2l[i] > 0]
            slice_loc = np.where(g2l[i] > 0)[0]
            for j in range(len(local_ids)):
                lbl_image[slice_loc[j]][lbl_image[slice_loc[j]] == local_ids[j]]= 0 + len(g2l)
                
    lbl_image[-1] = len(g2l)
    if lbl_image.min() < len(g2l):
        all_labs = np.unique(lbl_image)
        rogue_labs = all_labs[all_labs < len(g2l)]
        for i in range(len(rogue_labs)):
            lbl_image[lbl_image == rogue_labs[i]] = 0 + len(g2l)
    assert(lbl_image.min() >= len(g2l)),'Rogue labels not being captured'
    
    #print(np.unique(lbl_image))
    lbl_image = lbl_image - len(g2l)#int(g2l.max())
    #lbl_image[lbl_image == -int(g2l.max())] = 0 
    #print(np.unique(lbl_image))
    #print(np.unique(lbl_image2))
    assert(lbl_image.min() == 0), 'Minimum label is not zero'
    
    print('image relabelled')
    return lbl_image

def nzero_mean(regionmask, intensity_image):
    
    if np.sum(intensity_image[regionmask]>0) > 0:
        return np.mean(intensity_image[regionmask][intensity_image[regionmask]>0])
    else:
        return 0

def nzero_std(regionmask, intensity_image):
    
    if np.sum(intensity_image[regionmask]>0) > 0:
        return np.std(intensity_image[regionmask][intensity_image[regionmask]>0])
    else:
        return 0

def nzero_min(regionmask, intensity_image):
    
    if np.sum(intensity_image[regionmask]>0) > 0:
        return np.std(intensity_image[regionmask][intensity_image[regionmask]>0])
    else:
        return 0
    
def nzero_median(regionmask, intensity_image):
    
    if np.sum(intensity_image[regionmask]>0) > 0:
        return np.nanmedian(intensity_image[regionmask][intensity_image[regionmask]>0])
    else:
        return 0
    
def nzero_mad(regionmask, intensity_image):
    
    if np.sum(intensity_image[regionmask]>0) > 0:
        return np.nanmedian(abs(intensity_image[regionmask][intensity_image[regionmask]>0] - np.nanmedian(intensity_image[regionmask][intensity_image[regionmask]>0])))
    else:
        return 0
    
def nzero_sum(regionmask, intensity_image):
    
    if np.sum(intensity_image[regionmask]>0)>0:
        return np.nansum(intensity_image[regionmask][intensity_image[regionmask]>0])

    else:
        return 0
    
def nzero_area(regionmask, intensity_image):
    
    if np.sum(intensity_image[regionmask]>0)>0:
        return np.nansum(intensity_image[regionmask]>0)

    else:
        return 0
#Greg Lee stackexchange   
def ellipsoid_axis_lengths(central_moments):
    """Compute ellipsoid major, intermediate and minor axis length.

    Parameters
    ----------
    central_moments : ndarray
        Array of central moments as given by ``moments_central`` with order 2.

    Returns
    -------
    axis_lengths: tuple of float
        The ellipsoid axis lengths in descending order.
    """
    m0 = central_moments[0, 0, 0]
    sxx = central_moments[2, 0, 0] / m0
    syy = central_moments[0, 2, 0] / m0
    szz = central_moments[0, 0, 2] / m0
    sxy = central_moments[1, 1, 0] / m0
    sxz = central_moments[1, 0, 1] / m0
    syz = central_moments[0, 1, 1] / m0
    S = np.asarray([[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]])
    # determine eigenvalues in descending order
    eigvals = np.sort(np.linalg.eigvalsh(S))[::-1]
    return tuple([math.sqrt(20.0 * e) for e in eigvals])

#reduced matrix by removing spaces where the single labels where
def data_matrix3d(label_image, int_ch, global2local, local2global, filename = None, extra_properties = None):
    channel_num = int_ch.shape[-1]
    unique_labels = np.unique(label_image)
    #number of properties for which there is more than 1 value depending on num of intensity channels = 8
    data = np.full((unique_labels.shape[0],(8 * channel_num)+8),np.nan, dtype = np.float32)
    
    props = regionprops(label_image,int_ch, extra_properties = extra_properties)
    
    
    for label in range(len(props)):
        try:
            data[label+1,0] = props[label].label
            data[label+1,1] = props[label].area
            data[label+1,2:5] = props[label].centroid
            data[label+1,5:(1*channel_num)+5] = props[label].nzero_mean
            data[label+1,(1*channel_num)+5:(2*channel_num)+5] = props[label].intensity_max
            data[label+1,(2*channel_num)+5:(3*channel_num)+5] = props[label].nzero_min
            data[label+1,(3*channel_num)+5:(4*channel_num)+5] = props[label].nzero_std
            data[label+1,(4*channel_num)+5:(5*channel_num)+5] = props[label].nzero_median
            data[label+1,(5*channel_num)+5:(6*channel_num)+5] = props[label].nzero_mad
            data[label+1,(6*channel_num)+5:(7*channel_num)+5] = props[label].nzero_sum
            data[label+1,(7*channel_num)+5:(8*channel_num)+5] = props[label].nzero_area
            data[label+1,(8*channel_num)+5:(8*channel_num)+8] = ellipsoid_axis_lengths(props[label].moments_central)
            
            #data[label+1,(8*channel_num)+4:(8*channel_num)+5] = props[label].minor_axis_length
            #data[label+1,(8*channel_num)+5:] = props[label].major_axis_length
        except ValueError:
            print(label,unique_labels)
    col_names = ['label','volume','z','y','x'] + [f'mean_int_ch{ch}' for ch in range(channel_num)] + [f'max_int_ch{ch}' for ch in range(channel_num)] + [f'min_int_ch{ch}' for ch in range(channel_num)] + [f'std_int_ch{ch}' for ch in range(channel_num)] + [f'median_int_ch{ch}' for ch in range(channel_num)] + [f'mad_int_ch{ch}' for ch in range(channel_num)] + [f'sum_int_ch{ch}' for ch in range(channel_num)] + [f'nzerovol_int_ch{ch}' for ch in range(channel_num)] + ['major_axis','intermediate_axis','minor_axis']    
    data = pd.DataFrame(data, columns = col_names)
    data.loc[0,'label'] = 0
    data.loc[:,'label'] = data.loc[:,'label'].astype(np.uint64)
    data.set_index('label',drop=False,inplace=True)
    #because region props does not have corrected z depth the minor axis is always the z axis (less pixels than both xy)
    #this is handy as it orients the elliposids, to make minor axis on same scale as other two multiply by six which is the scale (6,1,1)
    data.loc[:,'minor_axis'] = data.loc[:,'minor_axis'] * 6
    data['ellipticity'] = data.major_axis/data.intermediate_axis
    if type(filename) == str:           
        data.to_csv('label_data/'+'STS_global'+filename[3:-4]+'_v04.csv')
        print('saved data')
        
    return data

@viewer.bind_key('l',overwrite=True)
def get_label_props(viewer):
    
    print(f'volume >> {data[1].loc[viewer.layers.selection.active.selected_label,"volume"]}')
    print(f'major axis >> {data[1].loc[viewer.layers.selection.active.selected_label,"major_axis"]}')
    print(f'intermediate axis >> {data[1].loc[viewer.layers.selection.active.selected_label,"intermediate_axis"]}')
    print(f'minor axis >> {data[1].loc[viewer.layers.selection.active.selected_label,"minor_axis"]}')
    
    return

@viewer.bind_key('m',overwrite=True)
def mean_intensity(viewer):
    print(np.median([viewer.layers.selection.active.properties['mean_int_ch0'][pt] for pt in viewer.layers.selection.active.selected_data]))
    
    return
    
@viewer.bind_key('n',overwrite=True)
def mean_intensity(viewer):
    print(np.median([viewer.layers.selection.active.properties['mean_int_ch1'][pt] for pt in viewer.layers.selection.active.selected_data]))
    
    return

def interpolate_thresholds(data,z_points,tbxta_background,sox2_background):
    #tbxta 
    t, c, k = interpolate.splrep([int(data.z.min())]+z_points+[int(data.z.max())], tbxta_background,k=1)
    xx = np.linspace(int(data.z.min()), int(data.z.max()), int(data.z.max())+1)
    spline = interpolate.BSpline(t, c, k, extrapolate=False)
    thresholds = spline(xx)

    mask = []
    for j in range(len(data)):
        mask.append(data.iloc[j].mean_int_ch1 > thresholds[int(data.iloc[j].z)])

    data[i]['tbxta_thresh'] = mask
    
    #sox2
    t, c, k = interpolate.splrep([int(data.z.min())]+z_points+[int(data.z.max())],sox2_background,k=1)
    xx = np.linspace(int(data.z.min()), int(data.z.max()), int(data.z.max())+1)
    spline = interpolate.BSpline(t, c, k, extrapolate=False)
    thresholds = spline(xx)

    mask = []
    for j in range(len(data)):
        mask.append(data.iloc[j].mean_int_ch0 > thresholds[int(data.iloc[j].z)])
    data['sox2_thresh'] = mask
    
    return data

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
    
    