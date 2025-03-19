import numpy as np
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import pandas as pd
import copy
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects
from skimage.transform import resize
import nucseg_utils_v15 as utils
from skimage.filters import threshold_otsu
from glob import glob
#note: would be good to switch to using pathlib Paths instead of strings so as to make code compatible across different OS
#main python class for storing and manipulating the segmentation  information
class Segobj():
    ''' Segobj is the main python class for storing and manipulating images, segmentation masks, and related data.'''

    def __init__(self,path,channel_axis=1,nuclei_channel = 0,channel_names = None):
        '''Initialization of the segmentation object with given path (str). Returns instance of class.
        Keyword args:
        channel_axis (int) -- which axis of input images is channel default 1 for tifffile loading
        channel_names (list of str or int) -- assign names to channels to make keeping track of individual channels easier, default None
        nuclei_channel (int or str) -- which channel contains the nuclei for segmentation. Can use integer identity or str of name if specified in channel_names. default zero.
        
        Note: for loading in multiple images path can have unix style pathname pattern recognition used by the python package glob.'''

        #basic file and image information
        self.filepath = path
        self.filenames = None
        self.voxel_dims = None
        self.voxel_anisotropy = None
        
        # raw images channels and axis information
        self.image_channels = None
        self.channel_names = channel_names
        self.channel_axis = channel_axis
        self.nuclei_channel = str(nuclei_channel)

        # segmentation outputs
        self.labels2d = None
        self.labels2d_metrics = None
        self.g2l_maps = []
        self.l2g_maps = []
        self.labels3d = []
        self.label_data = []

        # intensity thresholding outputs
        self.intensity_masks = {}
        self.intensity_thresholds = []

        # change this when updating nucseg version
        self.version = 'v15'

    def load_rawdata(self):
        '''Read in raw images located at the filepath that was instansiated with the segobj object. 
        Raw Images are stored in the attributes of the segobj. Returns None.'''
        
        # load in raw images found at self.filepath then splits channels into a dictionary to allow easy computational access to each channel
        # note: utils.read_images is currently only set up to read the metadata and get the voxel dimensions for .lsm files
        print("Loading raw data")
        filenames, images, voxel_dims = utils.read_images(self.filepath)
        self.filenames = filenames
        self.voxel_dims = voxel_dims
        print("Raw data loaded")
        assert(len(images)>0), "Filepath is returning no images, change filepath pattern to correctly idenitfy images."
        #images should be as close to 1024 x 1024 pixels as possible for StarDist 2D pretrained model
        
        for i in range(len(images)):
            if images[i].shape[-1] <1000 or images[i].shape[-1] > 1100:
                print(f"Image {i} is not the right size, resizing...")
                ratio = 1024/images[i].shape[-1]
                images[i] = resize(images[i], images[i].shape[:2] +(1024,1024,))
                self.voxel_dims[i] = [self.voxel_dims[i][0],self.voxel_dims[i][1]/ratio,self.voxel_dims[i][2]/ratio] #need to adjust voxel dims for change in image xy size
        
        print("all resizing complete")
        self.voxel_anisotropy = [[vd[0]/vd[1],1,1] for vd in self.voxel_dims] #assuming that the Z/anisotropic axis is 0

        print("splitting images into component channels")
        # split the channels of each image into a dictionary where the keys are each channel and the values are a list of images for each channel
        # this enables independant manipulation of each channel
        if self.channel_names is None:
            self.channel_names = list(range(images.shape[self.channel_axis]))

        self.image_channels = utils.split_channels(images,channel_axis = self.channel_axis,channel_names = self.channel_names)
        print(f"Finished. All raw image data can be found in the image_channels attribute under the names {self.channel_names}")

    def load_intermediates(self):
        '''Read the intermidate steps of the segmentation into the segobj attributes located at self.filepath. 
        Use if you have only made it part way through the pipeline. '''

        # this function loads in intermediate steps in the segmentation pipeline to allow you to pick up where you left off
        try:
            self.labels2d = utils.load_files("/".join(self.filepath.split("/")[:-1])+"/L2D*.npy")
            print("2D label images loaded in...")
        except FileNotFoundError:
            print("No 2D label segmentation found")

        try:
            self.g2l_maps = utils.load_files("/".join(self.filepath.split("/")[:-1])+"/G2L*.npy")
            print("Global2local mapping loaded in...")
        except FileNotFoundError:
            print("No Global to Local mapping found")

        try:
            self.l2g_maps = utils.load_files("/".join(self.filepath.split("/")[:-1])+"/L2G*.npy")
            print("local2global mapping loaded in...")
        except FileNotFoundError:
            print("No Local to Global mapping found")

        try:
            self.labels3d = utils.load_files("/".join(self.filepath.split("/")[:-1])+"/L3D*.npy")
            print("3D label images loaded in...")
        except FileNotFoundError:
            print("No 3D label segmentation found")

        try:
            fnames = np.sort(glob("/".join(self.filepath.split("/")[:-1])+"/STS*.csv"))
            self.label_data = [pd.read_csv(name) for name in fnames]
            
            if len(self.label_data) == 0:
                print("No label statistics found")
            else:
                print("label statistics loaded in...")
        except FileNotFoundError:
            print("No label statistics found")


    def preprocess_nuclei(self,clahe_kernel = (35,35,18),low_sigma = 1, high_sigma = 3, testing_mode = False):

        '''Preprocess the nuclear channel specified at self.nuclei_channel. 
        Takes images and runs adapative histogram equalization and bandpass filtering to remove background.
        Keyword arguments:
        clahe_kernel (tuple) -- dimensions of the kernel for adapative histogram equalization
        low_sigma (int) -- sigma of low pass gaussian kernel, default 1 
        high_sigma (int) -- sigma of high pass gaussian kernel, default 3
        testing_mode (bool) -- when false runs function and stores final output in class attributes. when true runs function and returns intermdiate steps, use this for trialling parameters.'''

        #preprocessing steps, adapative histogram equalization and difference of gaussian bandpass filtering
        print(f"Beginning adaptive histogram equalization on {self.nuclei_channel} using a kernel of {clahe_kernel}")
        clahe = utils.clahe(self.image_channels[self.nuclei_channel], kernel_clahe=clahe_kernel)
        print(f"Finished\n Begnning difference of gaussian bandpass filtering between {low_sigma} and {high_sigma}")
        gauss = utils.batch_gaussian(clahe,diff_gauss=True, low_sigma=low_sigma, high_sigma=high_sigma)
        print("Preprocessing finished. Nuclear channel has been updated")
        #testing mode returns the intermediate steps of this function as an output, this allows rapid interation of parameters when finding optimal params for new image types, not needed for ultimate processing
        if testing_mode ==True:
            return clahe, gauss

        else:
            self.image_channels.update({self.nuclei_channel:gauss})

    def segmentation2d(self,pretrained = True, model = None,trim_labels = True,trim_thresh=200):

        '''Run stardist default 2D fluorescent model for ellipsoid segmentation across each slice in an image stack.
        Output saved in class attributes and into local folder.
        Output: image details (list of dictionaries) -- metrics for the quality of each segmented object
        Output: image labels 2D (list of arrays) -- segmentation masks 
        keyword args:
        pretrained (bool) -- use pretrained stardist model, default True
        model (stardist model class) -- if not pretrained then provided trained model to this argument, default None
        trim_labels (bool) -- remove segmented objects that are small (significantly improves quality, stardist often creates many small segmentations in between real nuclei)
        trim_thresh (float or int) -- size of object below which trim_labels removes, default 200'''
        
        #load model
        if pretrained == True:
            model = StarDist2D.from_pretrained('2D_versatile_fluo')
        else:
            model = model
            assert (model != None), 'specify model in function'
        
        # iterate over all the nuclei channels from each image
        images_labels = []
        images_details = []
        for j in range(len(self.image_channels[self.nuclei_channel])):
            #create empty label image
            print(f'Starting segmentation of image {j}')
            labels_all = np.empty(self.image_channels[self.nuclei_channel][j].shape, dtype = "int32")
            details_all = []
            
            #iterate over each image z slice
            for i in range(len(self.image_channels[self.nuclei_channel][j])):
                #normalize image and then feed to stardist model to generate 2d labels
                img = normalize(self.image_channels[self.nuclei_channel][j][i,...], 1,99.8, axis=(0,1))
                labels, details = model.predict_instances(img)
                labels_all[i] = labels
                details_all.append(details)
            
            #stardist often generates many small labels in between nuclei, remove these using a size threshold
            if trim_labels == True:
                labels_trimmed = []
                for z in labels_all:
                    labels_trimmed.append(label(remove_small_objects(z,trim_thresh)))
                labels_all = np.array(labels_trimmed)    

            # stardist produces two outputs, labels (the segmentation masks) and details (the segmentation quality), could also use details to remove bad labels  
            lbl_name = '/L2D_' + self.filenames[j][4:-4] + f'_{self.version}.npy'
            detail_name = '/DTL_' + self.filenames[j][4:-4] + f'_{self.version}.npy'
            np.save("/".join(self.filepath.split("/")[:-1])+lbl_name, labels_all) 
            np.save("/".join(self.filepath.split("/")[:-1])+detail_name, details_all)
            images_labels.append(labels_all)
            images_details.append(details_all)
            print(f'Finishing segmentation of image {j}')

        self.labels2d = images_labels
        self.labels2d_metrics = images_details
        print("Finished 2D segmentation")

        return 

    def label_3dlinking_maps(self,labels2d, name, iou_threshold = 0.6,additional_props=None):
        '''Takes 2d label image from output of stardist segmentation and links it to labels below it. These are placed into two relational tables to enable rapid interfacing between the local label id, its new global id, and the Z slice of the image.
        Returns:
        global2local: 2D array with columns global label ID and rows Zslice, containing the local label IDs
        local2global: 2D array with columns local label ID and rows Zslice, containing the global label IDs
        
        Keyword args:
        iou_threshold (float) -- the intersection over union value used to determine whether a 2D label is part of a 3D label
        additional_props (dict) -- default none, never finished, this was to allow threhsolding of labels by other properties as well as IOU
        
        Note: this function processes one 2d label stack at a time, for the batch processing function that integrates with the class object see label_3dlinking() '''
    
        # find total number of labels across all z slices 
        num_labels = 0
        for i in range(len(labels2d)):
            num_labels += labels2d[i].max()

        # create global and local mapping matrices     
        global2local = np.zeros((num_labels+1,len(labels2d)),dtype = np.uint)
        local2global = np.zeros((labels2d.max()+1, len(labels2d)), dtype = np.uint)

        num_labels = 0
        for z in range(len(labels2d)-1):
            slice1_masks = labels2d[z].ravel()
            slice2_masks = labels2d[z+1].ravel()
            
            
            # calculate overlap and generate IOU matrix for the current slice and the next slice - code adapted from cellpose _label_overlap function. Flatten consecutive slices and create matrix of the number of occurances of each mask. Use this to calculate IoU 

            overlap_matrix = np.zeros((slice1_masks.max()+1,slice2_masks.max()+1), dtype=np.uint)
            for j in range(len(slice1_masks)):
                overlap_matrix[slice1_masks[j],slice2_masks[j]] += 1
            n_pixels_slice1 = np.sum(overlap_matrix, axis=0, keepdims=True)
            n_pixels_slice2 = np.sum(overlap_matrix, axis=1, keepdims=True)
            iou_matrix = overlap_matrix/(n_pixels_slice1 + n_pixels_slice2 - overlap_matrix)
            
                
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
                    
        np.save("/".join(self.filepath.split("/")[:-1])+'/G2L_'+name[4:-4]+f'_{self.version}.npy', global2local, allow_pickle = True) 
        np.save("/".join(self.filepath.split("/")[:-1])+'/L2G_'+name[4:-4]+f'_{self.version}.npy', local2global, allow_pickle = True) 
                    
        return global2local, local2global

    def label_3dlinking(self,iou_threshold=0.6):
        ''' Batch process 2d label stacks using the label_3dlinking_maps() function, output result into segobj attribute labels3d
        kwargs:
        iou_threshold (float) -- the intersection over union value used to determine whether a 2D label is part of a 3D label '''
        #USE THIS FUNCTION WHEN RUNNING 3D LABEL LINKING, runs label_3dlinking_maps across all images and appends the result into the class attributes
        for lbl,name in zip(self.labels2d,self.filenames):
            print(f"Starting linking 2d labels for image {name}")
            g2l, l2g = self.label_3dlinking_maps(lbl,name,iou_threshold=iou_threshold)
            self.g2l_maps.append(g2l)
            self.l2g_maps.append(l2g)
            print(f"3D linking complete, linking maps saved as attributes")
        print("Finished")

    def relabeller(self):
        '''Uses global2local segmented object IDs to relabel the objects in the 2D label array. 
        Saves output as new 3D label image into class attributes.'''

        for lbl, g2l,name in zip(self.labels2d,self.g2l_maps,self.filenames):
            print(f"Starting relabelling for image {name}")
            # label must be deepcopied before changing the label IDs otherwise it can overwrite the 2d labels
            lbl_image = copy.deepcopy(lbl)
            
            
            lbl_image[lbl_image == 0] = len(g2l) # so as not to confuse global and local labels during this process the new 3D label ids will count from the maxium number of global IDs found in g2l
            #iterate over every row in g2l matrix, each row is a global label
            for i in range(len(g2l)):
                # if the global label id is associated with at least one 2D label 
                if np.sum(g2l[i] > 0) >= 1: 
                    local_ids = g2l[i][g2l[i] > 0] #get the local ids of the label
                    slice_loc = np.where(g2l[i] > 0)[0] # get the locations of each local label in Z
                    
                    for j in range(len(local_ids)):
                        lbl_image[slice_loc[j]][lbl_image[slice_loc[j]] == local_ids[j]]= i + len(g2l) # in the 2d label image at the correct slice where the local 2d label is change the value to the new global label id
                
                # if the global id is not associated with any 2d labels then set it as background
                else:
                    
                    local_ids = g2l[i][g2l[i] > 0]
                    slice_loc = np.where(g2l[i] > 0)[0]
                    for j in range(len(local_ids)):
                        lbl_image[slice_loc[j]][lbl_image[slice_loc[j]] == local_ids[j]]= 0 + len(g2l)
            # if there are any local 2D labels that for some reason have been missed (they are not part of a new global label) then set them to the new background value len(g2l)          
            lbl_image[-1] = len(g2l)
            if lbl_image.min() < len(g2l):
                all_labs = np.unique(lbl_image)
                rogue_labs = all_labs[all_labs < len(g2l)]
                for i in range(len(rogue_labs)):
                    lbl_image[lbl_image == rogue_labs[i]] = 0 + len(g2l)
            assert(lbl_image.min() >= len(g2l)),'Rogue labels not being captured'
            
            #finally subtract the new background value from the 3D label image so that the background value is zero once more
            lbl_image = lbl_image - len(g2l)
            
            assert(lbl_image.min() == 0), 'Minimum label is not zero'

            print('image relabelled')
            self.labels3d.append(lbl_image)
            np.save("/".join(self.filepath.split("/")[:-1])+'/L3D_'+name[4:-4]+f'_{self.version}.npy', lbl_image, allow_pickle = True)
            print(f"Relabelling complete, 3D label image saved.")
        print("Finished All Relabelling")

    def preprocess_intensity(self,process_channels = [1,2,3],kernelsize = (5,5,3),testing_mode=False):
        '''Method for preprocessing intensity channels prior to threshold and/or quantification. 
        keyword args:
        process_channels (list of int or str) -- list of channels to process either using their int designation or their names as specified in self.channel_names
        kernelsize (3 tuple) -- shape (x,y,z) of the kernel used to process the image to smooth background.
        testing_mode (bool) -- default False, use this to return the processed channels rather than overwriting the image_channels attribute, this can be used for testing parameters.
        '''
        pp_chs = {}
        for ch in process_channels:
            print(f"Processing all images of channel {process_channels}")
            pp_imgs = []
            for img in self.image_channels[str(ch)]:
                pp_imgs.append(utils.supress_background(img,kernel_dims=kernelsize)) # runs the suppress background function from utils, several round of median and rolling ball filtering using a kernel size optimized for HCR signal

            if testing_mode == True:
                pp_chs.update({str(ch):pp_imgs})
            else:
                self.image_channels.update({str(ch):pp_imgs})
            print("Done")

        if testing_mode == True:
            return pp_chs

        print("All processing complete.")
    

    def threshold_intensity(self,process_channels=[1,2,3],autothresh = True, thresh_val = None,testing_mode=False):
        '''Method to generate a threshold cutoff for intensity channel images and generate binary masks that can be used for thresholding/background removal.
        Threshold values and binary masks are saved as segobj class attributes.
        Keyword args:
        process_channels (list of int or str) -- list of channels to process either using their int designation or their names as specified in self.channel_names
        autothresh (bool) -- whether to use an automatic threshold method (True) or to use values provided in thresh_val (False).
        thresh_val (float) -- manual threshold value to apply to all images. this could be improved by taking the threshold value from a pre-exisintg list given to the intensity_thresholds attribute.
        testing_mode (bool) -- default False, use this to return the binary masks rather than saving them into the attributes, this can be used for testing parameters.  '''
        
        thresh_ch = {}
        for ch_id in process_channels:
            ch = self.image_channels[str(ch_id)]
            thresh_img = []
            for img in ch:
                if autothresh == True:
                    thresh = np.mean(img)+(3* np.std(img))#threshold_otsu(img)
                else:
                    thresh = thresh_val
                thresh_img.append(img > thresh)
                self.intensity_thresholds.append((str(ch_id),thresh))
            if testing_mode == True:
                thresh_ch.update({str(ch_id):thresh_img})
            else:
                self.intensity_masks.update({str(ch_id):thresh_img})
        
        if testing_mode == True:
            return thresh_ch

    def apply_intensity_mask(self,process_channels=[1,2,3],testing_mode=False):
        '''Apply the binary intensity masks generated using the threshold intensity method to the image channels.
        Output overwrites intensity image channels unless testing_mode =True
        Keyword args:
        process_channels (list of int or str) -- list of channels to process either using their int designation or their names as specified in self.channel_names
        testing_mode (bool) -- default False, use this to return the threhsolded channels rather than overwriting the image_channels attribute, this can be used for testing parameters.
        '''
        mask_ch = {}
        for ch_id in process_channels:
            ch = self.image_channels[str(ch_id)]
            masks = self.intensity_masks[str(ch_id)]
            mask_img = []
            for img, mask in zip(ch,masks):
                mask_img.append(img * mask)
    
            if testing_mode == True:
                mask_ch.update({str(ch_id):mask_img})
            else:
                self.image_channels.update({str(ch_id):mask_img})
        
        if testing_mode == True:
            return mask_ch

    def data_matrix3d(self,label_image, int_ch, filename = None, extra_properties = None,intensity_chan_ids = None):
        '''For each 3D label image, and associated intensity channels calculates shape and intensity properties for each segmented nucleus.
        Returns data as a pandas table and saves it to self.filepath as .csv file.
        Postional Args:
        label_image (np.Array, integer labels) -- 3D label image 
        int_ch (np.Array, float) -- image containing the intensity channels of the original image, can be processed or unprocessed. Channel axis should be last for regionprops.
        Keyword args:
        filename (str) -- name to save data under
        extra_properties (dict) -- custom properties to give to the regionprops function to calculate for each labe
        intensity_chan_ids (list of int or str) -- this should be a list of the names or integer ids of the intensity channels. used for labelling the resulting data.'''

        channel_num = int_ch.shape[-1]
        unique_labels = np.unique(label_image)
        # create empty data array of NaNs in which to place the label properties, this is of variable length depending on the number of intensity channels used
        #number of properties for which there is more than 1 value depending on num of intensity channels = 8
        data = np.full((unique_labels.shape[0],(8 * channel_num)+9),np.nan, dtype = np.float32)
        
        #use regionprops to calculate props objects for each label
        props = regionprops(label_image,int_ch, extra_properties = extra_properties,spacing = self.voxel_anisotropy[0])
        
        #for each properties object retrieve the respective properties and add them into the data table
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
                data[label+1,(8*channel_num)+5:(8*channel_num)+8] = utils.ellipsoid_axis_lengths(props[label].moments_central)
                data[label+1,(8*channel_num)+8] = props[label].solidity
                

                #data[label+1,(8*channel_num)+9] = props[label].solidity
                
                #data[label+1,(8*channel_num)+4:(8*channel_num)+5] = props[label].minor_axis_length
                #data[label+1,(8*channel_num)+5:] = props[label].major_axis_length
            except ValueError:
                print(label,unique_labels)
        col_names = ['label','volume','z','y','x'] + [f'mean_int_ch{ch}' for ch in intensity_chan_ids] + [f'max_int_ch{ch}' for ch in intensity_chan_ids] + [f'min_int_ch{ch}' for ch in intensity_chan_ids] + [f'std_int_ch{ch}' for ch in intensity_chan_ids] + [f'median_int_ch{ch}' for ch in intensity_chan_ids] + [f'mad_int_ch{ch}' for ch in intensity_chan_ids] + [f'sum_int_ch{ch}' for ch in intensity_chan_ids] + [f'nzerovol_int_ch{ch}' for ch in intensity_chan_ids] + ['major_axis','intermediate_axis','minor_axis','solidity']    
        data = pd.DataFrame(data, columns = col_names)
        #data.loc[0,'label'] = 0
        data = data.iloc[1:] # this should remove the zero label which does not exist in the label image (it is the background id)
        data.loc[:,'label'] = data.loc[:,'label'].astype(np.uint64)
        data.set_index('label',drop=False,inplace=True)
        
        data['ellipticity13'] = data.major_axis/data.minor_axis
        data['ellipticity12'] = data.major_axis/data.intermediate_axis
        data['ellipticity23'] = data.intermediate_axis/data.minor_axis
        if type(filename) == str:           
            data.to_csv("/".join(self.filepath.split("/")[:-1])+'/STS_'+filename[4:-4]+f'_{self.version}.csv')
            print('saved data')
            
        return data

    def nuclei_properties(self,intensity_channels = [1,2,3]):
        '''Use this method to batch process 3D label images to get the resulting label properties from the data_matrix3d method. 
        Output stored as attribute of segobj class
        Keyword args:
        intensity_channels (list of int or str) -- this is the name or integer id of the intensity channels that will be used to calulate the 3d label properties'''
        #USE THIS FUNCTION WHEN RUNNING 3D LABEL LINKING, runs data_matrix3d across all images and appends the result into the class attributes
        for i in range(len(self.labels3d)):
            chans = [self.image_channels[str(ch)][i] for ch in intensity_channels]
            int_ch = np.concatenate(chans,axis=self.channel_axis)
            int_ch = np.moveaxis(int_ch,int(self.channel_axis),-1)
            #print(int_ch.shape,self.labels3d[i].shape)
            print(f"Starting label properties calculations for {self.filenames[i]}")
            self.label_data.append(self.data_matrix3d(self.labels3d[i],int_ch,filename=self.filenames[i],extra_properties=utils.extra_properties,intensity_chan_ids=intensity_channels))
        print("Finished all.")
    
        



    

        


    



   
    