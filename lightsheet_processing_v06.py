
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import spatial
#from sklearn.preprocessing import StandardScaler
#import umap
import seaborn as sns
import math
from scipy.optimize import curve_fit
from scipy.spatial.transform import Rotation
import matplotlib as mpl



#each spot has 1 parent spot (track merging is biologically impossible)

def assign_parent(data,link_data):
    
    if np.isin('manipulation',data.columns) == True:
        data['parent_spot'] = -1
        for i in range(len(link_data)):
            try:
                link_data.target[i] = 'none finished this bit'
            except:
                pass
            
        
    else:
        data['parent_spot'] = -1
        for i in range(len(link_data)):
            try:
                data.loc[link_data.target[i],'parent_spot'] = link_data.source[i]
            except:
                pass
    return data


def assign_id(data):
    #creates new track id for any dividing cells and assures that track id links through parent spots
    #starting spots (no parent spot) has a parent spot id of -1
    '''data['new_track_id'] = np.nan
    new_id_count = 0
    #napari_graph = {}
    for i in range(len(data)):
        if np.isnan(data.new_track_id[i]) == True:
            if data.parent_spot[i] == -1:
                data.loc[i,'new_track_id'] = new_id_count
                new_id_count += 1
            
            else:
                if data.spot_n_links[data.parent_spot[i]] == 3:
                    data.loc[data.id[data.parent_spot == data.parent_spot[i]].iloc[0],'new_track_id'] = new_id_count
                    data.loc[data.id[data.parent_spot == data.parent_spot[i]].iloc[1],'new_track_id'] = new_id_count + 1
                    #napari_graph.update({new_id_count:int(data.new_track_id[data.parent_spot[i]]),new_id_count+1:int(data.new_track_id[data.parent_spot[i]])})
                    new_id_count += 2
                
                else:
                    data.loc[i, 'new_track_id'] = data.new_track_id[data.parent_spot[i]]
                
    data.loc[:,'new_track_id'] = data.new_track_id.astype(np.int64)
    '''
    data['new_track_id'] = np.nan
    unique_tracks = data[(data.parent_spot != -1)&(data.spot_n_links == 1)].copy()
    print(unique_tracks.shape)
    track_stems = pd.DataFrame(columns = data.columns)
    count = 0
    data['new_id'] = np.nan
    for i in range(len(unique_tracks)):
        data.loc[unique_tracks.index[i],'new_track_id'] = i
        data.loc[unique_tracks.index[i],'new_id'] = count
        count += 1
        idx = unique_tracks.index[i]
       
        while get_parent_index(idx,data) != -1:
            idx=get_parent_index(idx,data)
            
            if np.isnan(data.loc[idx,'new_track_id']):
                data.loc[idx,'new_track_id'] = i
                data.loc[idx,'new_id'] = count
                count += 1
            else:
                dup_row = data.loc[idx].copy()
                dup_row.loc['new_track_id'] = i
                dup_row.loc['new_id'] = count
                count += 1
                    #print(dup_row.columns,data.columns)
                track_stems = pd.concat((track_stems,dup_row.to_frame().T),axis=0)
    #data.drop(index=-1,inplace=True)
    #data.loc[:,'new_track_id'] = data.new_track_id.astype(np.int64)
    data = pd.concat((data,track_stems),axis=0)
    
    data.set_index('new_track_id',inplace=True, drop=False)
    data['new_parent_spot'] = np.nan
    data.sort_values(['spot_frame'],inplace=True)
    trks = np.unique(data.index)
    for trk in trks:
        data.loc[trk,'new_parent_spot'] = [-1]+list(data.loc[trk].new_id.iloc[:-1])
        
    data.reset_index(drop=True,inplace =True)

    return data

def get_parent_index(idx,data):
    
    return data.loc[idx,'parent_spot']



def adjust_positions(data,mean_displacement):
    #adjust point location relative to ablation center and calculate distance of each point from ablation center
    points = []
    dist = []

    for i in range(len(data)):
        rel_points = np.asarray(data.loc[i,['x','y','z']]) - np.asarray(mean_displacement.iloc[int(data.loc[i,'spot_frame']),0:])
        points.append(rel_points)
        dist.append(np.sqrt(np.sum((rel_points)**2)))

    new_data = pd.DataFrame(np.array(points), columns = ['rel_x','rel_y','rel_z'])
    new_data['ref_distance'] = dist
    data = pd.concat((data,new_data), axis= 1)
    
    return data

def parent_child(data):
    # this function splits the dataset into two in order to match a parent spot to its child
    # note that all spots are both parent and child except for the first in track (parent only), and last in track (child only)
    #this allows rapid calculation of displacement etc between each timepoint
    # one reason this can work is new id is assigned in a parent-child way from end of track upwards
    data.set_index('new_id',inplace=True,drop=False)
    child = data[data.new_parent_spot != -1].copy()
    parent= data.loc[data.new_parent_spot[data.new_parent_spot != -1]].copy()
    child.set_index('new_parent_spot',inplace=True, drop=False)
    
    if np.nansum(parent.index - child.index) != 0:
        print('indexes do no match')
        
    return parent, child

def spot_properties(data):
    
    parent,child = parent_child(data)
    
    ## now we calculate any properties we want that require a spot and its immediate parent
    parent['diff_x'] = (child.rel_x - parent.rel_x)
    parent['diff_y'] = (child.rel_y - parent.rel_y)
    parent['diff_z'] = (child.rel_z - parent.rel_z)
    
    parent['xy_dist'] = np.sqrt(np.sum(parent.loc[:,['diff_x','diff_y']]**2,axis=1))
    parent['xz_dist'] = np.sqrt(np.sum(parent.loc[:,['diff_x','diff_z']]**2,axis=1))
    parent['yz_dist'] = np.sqrt(np.sum(parent.loc[:,['diff_y','diff_z']]**2,axis=1))
    
    parent['total_dist'] = np.sqrt(np.sum(parent.loc[:,['diff_x','diff_y','diff_z']]**2,axis=1))
    
    parent['int_ch1_change'] = child.int_mean_ch1 - parent.int_mean_ch1
    parent['int_ch2_change'] = child.int_mean_ch2 - parent.int_mean_ch2
    
    parent['xy_anglex'] = ((parent.rel_x * parent.diff_x)/(parent.xy_dist * abs(parent.rel_x))) * (180/np.pi)
    parent['xz_anglex'] = ((parent.rel_x * parent.diff_x)/(parent.xz_dist * abs(parent.rel_x))) * (180/np.pi)
    parent['yz_angley'] = ((parent.rel_y * parent.diff_y)/(parent.yz_dist * abs(parent.rel_y))) * (180/np.pi)
    
    parent['xy_angley'] = 270 - parent.xy_anglex.copy() 
    parent['xz_anglez'] = 270 - parent.xz_anglex.copy() 
    parent['yz_anglez'] = 270 - parent.yz_angley.copy() 
    
    data = pd.concat((parent,child[(child.spot_n_links ==1)&(child.parent_spot !=-1)]),axis=0)

    
    return data

def turning_angle(data):
    
    parent,child = parent_child(data)
        
    dp = (parent.diff_x * child.diff_x) + (parent.diff_y * child.diff_y)+(parent.diff_z* child.diff_z)
    child['turning_angle'] = np.arccos((dp/(parent.total_dist * child.total_dist)))* (180/np.pi)
    
    child.set_index('id',inplace=True,drop=False)
    data = pd.concat((parent[parent.new_parent_spot == -1],child),axis=0)
    
    return data

def NN_matrix(data_subset, radius = 20, max_nn= 30):
    nn_matrix = np.empty((data_subset.spot_frame.max()+1,np.max(np.unique(data_subset.new_track_id))+1,max_nn),dtype = np.float32) # generate nn matrix, this holds the nn for each unique track at each timeframe - 20 is set as the max nn per frame as this is much greater than actual nn
    nn_matrix[:] = np.nan
    for i in range(data_subset.spot_frame.max()+1):
        frame_subset = data_subset[data_subset.spot_frame == i].copy() # data is split into frame by frame data
        kdtree_frame = spatial.KDTree(frame_subset.loc[:,['rel_x','rel_y','rel_z']]) #create kd tree for spots in frame
        nn = kdtree_frame.query_ball_tree(kdtree_frame,r=radius) #use kdtree to index itself to find nn in radius

        for j in range(len(nn)): #use the nn index to find the track ids of the track and its nns in the frame
            nn_trks = np.unique(frame_subset.track_id.iloc[nn[j]])
            #:   len(nn[j])
            nn_matrix[i,frame_subset.new_track_id.iloc[j],:len(nn_trks)] = nn_trks
            # in this case we are counting dividing tracks as one for the purposes of new nn
            
    return nn_matrix

def nn_count(data,nn_matrix):
    # this block calculates the number of new nn a track sees per time point, to make the graph understandable this is added to a cumulative total over time of new nn
    # the number of nn at each timepoint is also calculated
    
    #num nn is the total number nn, new_nn is the number of new nn at this timepoint,lt_nn is the number of new nn that have never before been seen, lost_nn is the number of nn lost at timepoint
    inst_num_nn=[]
    inst_new_nn=[]
    inst_lost_nn=[]
    inst_lt_nn = []
    
    cum_num_nn=[]
    cum_new_nn=[]
    cum_lost_nn=[]
    cum_lt_nn = []
    
    time_frame = []
    track_id = []
    
    data.set_index('new_track_id',inplace=True,drop=False)
    data.sort_values(['spot_frame'],inplace=True)
    tracks = np.unique(data.new_track_id)
    for trk in tracks:
        count_new_nn = 0
        count_nn = 0
        count_lost_nn = 0
        count_lt_nn = 0
        for frame in data.loc[trk,'spot_frame']:
            if count_nn == 0:
                count_nn += np.sum(~np.isnan(nn_matrix[frame,trk,:]))#use sum to count the number of instances that are not NaN in the matrix eg the number of nn
                inst_num_nn.append(count_nn)
                inst_new_nn.append(0)
                inst_lost_nn.append(0)
                ist_lt_nn.append(0)
    
                cum_num_nn.append(count_nn)
                cum_new_nn.append(0)
                cum_lost_nn.append(0)
                cum_lt_nn.append(0)
                
                #we are interested in whether a cell leaves its starting neighbourhood so the initial neighbours do not count as new
                    
                time_frame.append(frame)
                track_id.append(trk)
                
            else: #in the middle or end of a track
                
                inst_num_nn.append(np.sum(~np.isnan(nn_matrix[frame,trk,:]))) # count the number of nn in the matrix using sum
                inst_new_nn.append(np.sum(~np.isin(nn_matrix[frame,trk,:],nn_matrix[frame-1,trk,:]) * ~np.isnan(nn_matrix[frame,trk,:])))
                inst_lost_nn.append(np.sum(~np.isin(nn_matrix[frame-1,trk,:],nn_matrix[frame,trk,:]) * ~np.isnan(nn_matrix[frame-1,trk,:])))
                inst_lt_nn.append(np.sum(~np.isin(nn_matrix[frame,trk,:],nn_matrix[:frame,trk,:]) * ~np.isnan(nn_matrix[frame,trk,:])))
                
                count_nn += np.sum(~np.isnan(nn_matrix[frame,trk,:]))
                count_new_nn += np.sum(~np.isin(nn_matrix[frame,trk,:],nn_matrix[frame-1,trk,:]) * ~np.isnan(nn_matrix[frame,trk,:]))
                count_lost_nn += np.sum(~np.isin(nn_matrix[frame-1,trk,:],nn_matrix[frame,trk,:]) * ~np.isnan(nn_matrix[frame-1,trk,:]))
                count_lt_nn += np.sum(~np.isin(nn_matrix[frame,trk,:],nn_matrix[:frame,trk,:]) * ~np.isnan(nn_matrix[frame,trk,:]))
                
                cum_num_nn.append(count_nn)
                cum_new_nn.append(count_new_nn)
                cum_lost_nn.append(count_lost_nn)
                cum_lt_nn.append(count_lt_nn)
                
                time_frame.append(frame)
                track_id.append(trk)
                
                   
               
    nn_data = pd.DataFrame((inst_num_nn,inst_new_nn,inst_lost_nn,inst_lt_nn,cum_num_nn,cum_new_nn,cum_lost_nn,cum_lt_nn,time_frame,track_id)).transpose()
    nn_data.columns = ['inst_num_nn','inst_new_nn','inst_lost_nn','inst_lt_nn','cum_num_nn','cum_new_nn','cum_lost_nn','cum_lt_nn','spot_frame','new_track_id' ]
    
    #nn_data['prop_nn_cum'] = nn_data.cum_new_nn/nn_data.cum_nn
    #nn_data['prop_nn_instant'] = nn_data.new_nn/nn_data.num_nn
    #nn_data['prop_orig_nn'] = nn_data.orig_nn/nn_data.num_nn
    
    nn_data.set_index(['new_track_id','spot_frame'],drop=True,inplace=True)
    nn_data.sort_index(inplace=True)
    
    data.set_index(['new_track_id','spot_frame'],inplace=True,drop=False)
    data.sort_index(inplace=True)
    data = pd.concat((data,nn_data),axis=1)
    data.reset_index(drop=True,inplace=True)

    return data

def ref_adjacent_early(data_subset,radius = 30,repeats = 3):
    data_subset['ref_adjacent_early'] = 0
    
    data_subset.set_index('new_track_id',inplace=True, drop=False)

    for j in range(repeats,0,-1):
        ref_adjacent_tracks= []

        #int((data.ref_distance//30).max())

        for i in range(10):
            frame_subset = data_subset[data_subset.spot_frame == i]
            kdtree_frame = spatial.KDTree(frame_subset.loc[:,['rel_x','rel_y','rel_z']])
            nn = kdtree_frame.query_ball_point([0,0,0],r=radius*j)
            ref_adjacent_tracks.append(frame_subset.new_track_id.iloc[nn])


        #ref_adjacent_tracks= pd.concat(ref_adjacent_tracks)

        for i in range(len(ref_adjacent_tracks)):
            data_subset.loc[ref_adjacent_tracks[i],'ref_adjacent_early'] = j
            
    data_subset.reset_index(inplace=True,drop=True)
    
    return data_subset

def ref_adjacent_late(data_subset,radius = 30,repeats = 3):
    data_subset['ref_adjacent_late'] = 0
    
    data_subset.set_index('new_track_id',inplace=True, drop=False)

    for j in range(repeats,0,-1):
        ref_adjacent_tracks= []

        #int((data.ref_distance//30).max())

        for i in range(10):
            frame_subset = data_subset[data_subset.spot_frame == i+110]
            kdtree_frame = spatial.KDTree(frame_subset.loc[:,['rel_x','rel_y','rel_z']])
            nn = kdtree_frame.query_ball_point([0,0,0],r=radius*j)
            ref_adjacent_tracks.append(frame_subset.new_track_id.iloc[nn])


        #ref_adjacent_tracks= pd.concat(ref_adjacent_tracks)

        for i in range(len(ref_adjacent_tracks)):
            data_subset.loc[ref_adjacent_tracks[i],'ref_adjacent_late'] = j
            
    data_subset.reset_index(inplace=True,drop=True)
    
    return data_subset

def power_law(x,a,b):
    return a*np.power(x, b)

def msd(track):
    subset = track.loc[:,['rel_x','rel_y','rel_z']]
    trk_len = subset.shape[0]
    
    msd = []
    time_lag = []
    for disp in range(trk_len-1):
        time_lag_vals = []
        for j in range(trk_len):
            if subset.iloc[j:j+trk_len-disp].shape[0]%(trk_len-disp) == 0:
                time_lag_vals.append(np.sum((subset.iloc[j:j+trk_len-disp-1])**2)) #subset.iloc[j+trk_len-disp-1] - subset.iloc[j]
        msd.append(np.nanmean(time_lag_vals))
        time_lag.append(trk_len-1-disp)

    pars, cov = curve_fit(f=power_law, xdata=time_lag, ydata=msd, p0=[0, 0], bounds=(-np.inf, np.inf))
    #stdevs = np.sqrt(np.diag(cov))
    
    return pars

def NN_difference(data_subset,properties, radius = 20):
    #nn_matrix = np.empty((data_subset.spot_frame.max()+1,np.max(np.unique(data_subset.new_track_id))+1,max_nn),dtype = np.float32) # generate nn matrix, this holds the nn for each unique track at each timeframe - 20 is set as the max nn per frame as this is much greater than actual nn
    #nn_matrix[:] = np.nan
    properties_array = np.empty((len(data_subset),(len(properties)*2)+2),dtype=np.float64)
    spot_total = 0
    for i in range(data_subset.spot_frame.max()+1):
        frame_subset = data_subset[data_subset.spot_frame == i] # data is split into frame by frame data
        kdtree_frame = spatial.KDTree(frame_subset.loc[:,['rel_x','rel_y','rel_z']]) #create kd tree for spots in frame
        nn = kdtree_frame.query_ball_tree(kdtree_frame,r=radius) #use kdtree to index itself to find nn in radius
        
        for j in range(len(nn)): #use the nn index to find the track ids of the track and its nns in the frame
            #for k in range(len(properties)):
                #frame_subset.loc[:,properties[k]].iloc[np.array(nn[j])[~np.isin(nn[j],j)]])
            neighbs = frame_subset.loc[:,properties].iloc[np.array(nn[j])]#[~np.isin(nn[j],j)]]
            properties_array[spot_total,:len(properties)] = np.array(frame_subset.loc[:,properties].iloc[np.array(nn[j])[np.isin(nn[j],j)]])-np.nanmean(neighbs,axis=0) # calculates the difference from mean of nn
            properties_array[spot_total,len(properties):2*len(properties)] = np.nanstd(neighbs,axis=0) # get std of nn to see if they are uniform or not
            
            properties_array[spot_total,-2] = i
            properties_array[spot_total,-1] = frame_subset.new_track_id.iloc[j]
            spot_total += 1
    properties_array = np.nan_to_num(properties_array,nan=0.0,posinf=np.inf)       
    new_names = []
    for i in range(len(properties)*2):
        if i//len(properties) == 0:
            new_names.append('nn_diff'+'_'+properties[i])
        elif i//len(properties) == 1:
            new_names.append('nn_std'+'_'+properties[i%len(properties)])
    
    properties_data = pd.DataFrame(properties_array, columns = new_names+['spot_frame','new_track_id'])
    properties_data.set_index(['new_track_id','spot_frame'],drop=True,inplace=True)
    properties_data.sort_index(inplace=True)        
    
    data_subset = pd.concat((data_subset.set_index(['new_track_id','spot_frame'],drop=False).sort_index(),properties_data),axis=1).reset_index(drop=True)
    #data_subset = data_subset.set_index(['new_track_id','spot_frame'],drop=False).sort_index()
    #data_subset = pd.concat((data_subset,properties_data),axis=1)
    #data_subset.reset_index(drop=True,inplace=True)
            
    return data_subset

def NN_polarisation(data_subset,properties, radius = 20):
    #nn_matrix = np.empty((data_subset.spot_frame.max()+1,np.max(np.unique(data_subset.new_track_id))+1,max_nn),dtype = np.float32) # generate nn matrix, this holds the nn for each unique track at each timeframe - 20 is set as the max nn per frame as this is much greater than actual nn
    #nn_matrix[:] = np.nan
    properties_array = np.empty((len(data_subset),len(properties)+2),dtype=np.float64)
    spot_total = 0
    for i in range(data_subset.spot_frame.max()+1):
        frame_subset = data_subset[data_subset.spot_frame == i] # data is split into frame by frame data
        kdtree_frame = spatial.KDTree(frame_subset.loc[:,['rel_x','rel_y','rel_z']]) #create kd tree for spots in frame
        nn = kdtree_frame.query_ball_tree(kdtree_frame,r=radius) #use kdtree to index itself to find nn in radius
        
        for j in range(len(nn)): #use the nn index to find the track ids of the track and its nns in the frame
            #for k in range(len(properties)):
                #frame_subset.loc[:,properties[k]].iloc[np.array(nn[j])[~np.isin(nn[j],j)]])
            neighbs = frame_subset.loc[:,properties].iloc[np.array(nn[j])]#[~np.isin(nn[j],j)]]
            properties_array[spot_total,:-2] = abs(np.nanmean(neighbs/abs(neighbs),axis=0))
            
            properties_array[spot_total,-2] = i
            properties_array[spot_total,-1] = frame_subset.new_track_id.iloc[j]
            spot_total += 1
    #properties_array = np.nan_to_num(properties_array,nan=0.0,posinf=np.inf)       
    new_names = []
    for i in range(len(properties)):
        new_names.append('nn_pol'+'_'+properties[i])
    
    properties_data = pd.DataFrame(properties_array, columns = new_names+['spot_frame','new_track_id'])
    properties_data.set_index(['new_track_id','spot_frame'],drop=True,inplace=True)
    properties_data.sort_index(inplace=True)        
    
    data_subset = pd.concat((data_subset.set_index(['new_track_id','spot_frame'],drop=False).sort_index(),properties_data),axis=1).reset_index(drop=True)
    #data_subset = data_subset.set_index(['new_track_id','spot_frame'],drop=False).sort_index()
    #data_subset = pd.concat((data_subset,properties_data),axis=1)
    #data_subset.reset_index(drop=True,inplace=True)
            
    return data_subset