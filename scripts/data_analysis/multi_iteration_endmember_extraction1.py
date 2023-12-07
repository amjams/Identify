# imports

import numpy as np
from sklearn.decomposition import PCA
import umap
from functions_EDX import *
import time
import matplotlib.pyplot as plt
from scipy.stats import zscore
from datetime import datetime
from skimage.feature import peak_local_max
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler
import os



home_path = ''   
# home_path contains subfolder NPZ, which has the NPZ files and empty subfolders for results (see below)
# download NPZ files here: http://www.nanotomy.org/PW/temp03/Duinkerken2023SUB/index.html 

# home_path
# |   NPZ
# |   Spectrum2D
# |   Results



# Get all file names (no extension)
file_names = []
for file in os.listdir(os.path.join(home_path, 'NPZ')):
    if file.endswith('.npz'):
        file_names.append(file[:-4])



exp_tag = '15Endmembers'

num = ""
for c in scriptname:
    if c.isdigit():
        num = num + c
idx = int(num)-1


# get the number in the script's name.  
scriptname = os.path.basename(sys.argv[0])

# load file
file_name = file_names[idx]
time_str = str(datetime.now())[:16]

loaded_file = np.load(os.path.join(home_path,'NPZ',file_name+'.npz'))
haadf = loaded_file['haadf']
spectrum = loaded_file['spectrum'][:,:,96:]
xray_energies = loaded_file['xray_energies'][96:]
subsample_size = spectrum.shape[2]


# Clean up then bin the spectrum and check if any empty channels remain
n_bins = 250
spectrum = rebin_spectrum(spectrum,n_bins)

# Now bin in XY
subsample_size = 1024
spectrum = rebin_spectrumXY(spectrum,subsample_size)  
haadf = rebin_XY(haadf,subsample_size)       

# normalize each pixel over peak (idx = 3)
#spectrum = np.array([j/j[3] for i in spectrum for j in i]).reshape((1024,1024,250)) 

xray_energies = rebin_energies(xray_energies,n_bins)
where_notempty = ~np.all(spectrum==0,axis=(0,1))
spectrum = spectrum[:,:,where_notempty]
spectral_depth = spectrum.shape[2]
spectrum_2D = np.reshape(spectrum,(subsample_size*subsample_size,spectral_depth))
print("%04d channels remain" % spectral_depth)


# Pre-Filter 
spectrum = MeanFilterCube(spectrum.astype('float32'),sigma=2, size=3)
spectrum_2D = np.reshape(spectrum,(subsample_size*subsample_size,spectral_depth))

# Poisson scale that works https://www.sciencedirect.com/science/article/pii/S0968432816303821
g = np.mean(spectrum,axis=2).reshape(subsample_size*subsample_size,1)
#g = np.ones((subsample_size*subsample_size,1))
h = np.mean(np.mean(spectrum,axis=0),axis=0).reshape(spectral_depth,-1)
W = g@np.transpose(h)
W = np.sqrt(W)
spectrum_2D = np.divide(spectrum_2D,W)
spectrum = np.reshape(spectrum_2D,(subsample_size,subsample_size,spectral_depth))


# Use PCA to reconstruct before UMAP and clustering
start = datetime.now()
pca_partial = PCA(n_components = None)
X_partial = pca_partial.fit_transform(spectrum_2D)


# scree plots
lim = 100
PC_values = np.arange(pca_partial.n_components_)[:lim]+1
exp_var = pca_partial.explained_variance_ratio_[:lim]
f, ax = plt.subplots(figsize=(10,10))
ax.plot(np.log(exp_var), 'o-', linewidth=2, color='k')
ax.set_title('Scree Plot')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance ')

plt.savefig(os.path.join(home_path,'Results',"Scree_%s.png" % file_name), format="png", dpi=600)


# do PCA reconstruction manually
n_keep = 10
spectrum_2D = reconstruct_manual(spectrum_2D,pca_partial,[i for i in range(n_keep)])
spectrum_2D = np.multiply(spectrum_2D,W)

# reshape back to 3D
spectrum = np.reshape(spectrum_2D,(subsample_size,subsample_size,spectral_depth))


np.savez_compressed(os.path.join(home_path,'Spectrum2D','%s_spectrum_2D_%s.npz' % (file_name,exp_tag)), spectrum_2D=spectrum_2D)

# Iterative Endmember extraction
grid = {'dimensions':[(0,1),(0,2),(1,2)],'n_neighbors':[10],'metric':['cosine','minkowski','chebyshev','euclidean','correlation','canberra'],'seed':[i for i in range(2)]}


# desired nEM and other things
nEM_d = 15
rstate = 1989
n_components_umap = 3

# zscore
spectrum_2D_zscored = zscore(spectrum_2D)


# histogram parameters
threshold_rel_list = np.arange(0.2,0.7,0.05)
min_distance_list =  np.arange(10,40,2)
#nEM_list = np.zeros((threshold_rel_list.size*min_distance_list.size,3))

cnt = 1
#end_members = []
num_params = len(ParameterGrid(grid))


# make a directory
path = os.path.join(home_path,'Results','end_members_%s_%s' % (file_name,exp_tag))
os.mkdir(path)

for param in ParameterGrid(grid):
    
    # subsample before training
    nTrain_umap = int(0.20*subsample_size**2)
    np.random.seed(param['seed'])
    n_sample_train = np.random.choice(spectrum_2D.shape[0], nTrain_umap,replace=False)

    nEM_list = np.zeros((threshold_rel_list.size*min_distance_list.size,3))
    umap1 = umap.UMAP(n_components=n_components_umap, n_neighbors=param['n_neighbors'], min_dist=0.0, metric=param['metric'],random_state=rstate)
    embeddings = umap1.fit_transform(spectrum_2D_zscored[n_sample_train,:])
    scaler = MinMaxScaler()
    embeddings = scaler.fit_transform(embeddings)

    dim_a = param['dimensions'][0]
    dim_b = param['dimensions'][1]


    Grid = 256
    xmin, xmax = np.amin(embeddings[:,dim_a]), np.amax(embeddings[:,dim_a])
    ymin, ymax = np.amin(embeddings[:,dim_b]), np.amax(embeddings[:,dim_b])
    counts, xedges, yedges = np.histogram2d(embeddings[:,dim_a], embeddings[:,dim_b], bins=Grid)

    cnt2 = 0
    for threshold_rel in threshold_rel_list:
        for min_distance in min_distance_list:
            #find local maxima in 2D histogram
            #threshold_rel = 0.1
            #min_distance = 20
            local_maxi = peak_local_max(counts, threshold_rel = threshold_rel, min_distance = min_distance, exclude_border = False)

            local_max_x = []
            local_max_y = []

            for i in range(0,len(local_maxi)):
                NewValue_X = (((local_maxi[i][0] ) * (xmax - xmin)) / (Grid - 0)) + xmin
                NewValue_Y = (((local_maxi[i][1] - 0) * (ymax - ymin)) / (Grid - 0)) + ymin
                local_max_x.append(NewValue_X)
                local_max_y.append(NewValue_Y)

            x = [x for x in zip(local_max_x,local_max_y)]
            nEM = len(x)
            nEM_list[cnt2,:] = [threshold_rel,min_distance,nEM]
            cnt2+=1
    
    # find the best one
    best_idx = np.argmin(np.abs(nEM_list[:,2]-nEM_d))
    print(nEM_list[best_idx,0])
    print(nEM_list[best_idx,1])
    local_maxi = peak_local_max(counts, threshold_rel = nEM_list[best_idx,0], min_distance = int(nEM_list[best_idx,1]), exclude_border = False)
    local_max_x = []
    local_max_y = []

    for i in range(0,len(local_maxi)):
        NewValue_X = (((local_maxi[i][0] ) * (xmax - xmin)) / (Grid - 0)) + xmin
        NewValue_Y = (((local_maxi[i][1] - 0) * (ymax - ymin)) / (Grid - 0)) + ymin
        local_max_x.append(NewValue_X)
        local_max_y.append(NewValue_Y)

    x = [x for x in zip(local_max_x,local_max_y)]
    nEM = len(x)


    min_max = 1/256       #determines bounding box size
    points = len(x)
    centroid = []       #empty array for selected UMAP centroids
    F = []              #indicies of UMAP points

    for i in range(0,points):    
        #set bounding square 
        Minx = x[i][0] - min_max
        Maxx = x[i][0] + min_max
        Miny = x[i][1] - min_max
        Maxy = x[i][1] + min_max

        #select points
        z3 = [0 if j == 0 else j for j in [j if j <= Maxx else 0 for j in [j if j >= Minx else 0 for j in embeddings[:,dim_a]]]]
        zz3 = [0 if j == 0 else j for j in [j if j <= Maxy else 0 for j in [j if j >= Miny else 0 for j in embeddings[:,dim_b]]]]
        f = [i for i, e in enumerate([x*y for x,y in zip(z3,zz3)]) if e != 0]
        d = embeddings[f]
        F.append(f)
        centroid.append(d)


    end_members = []
    for i in range(0, len(x)):
        jj = np.mean(spectrum_2D[n_sample_train][F[i]], axis = 0)
        end_members.append(jj)
    
    print('%02d out of %02d' % (cnt,num_params))



    # save each step
    end_members_array = np.array(end_members).transpose()
    savePath = os.path.join(path,'%02d.npz' % cnt)
    np.savez_compressed(savePath, end_members=end_members_array,grid=grid,param=param,threshold_rel=nEM_list[best_idx,0],min_distance=nEM_list[best_idx,1],nEM_d=nEM_d,nEM=nEM,random_state=rstate)
    cnt +=1


