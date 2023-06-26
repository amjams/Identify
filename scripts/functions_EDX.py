# Functions for EDX
import numpy as np
import cv2 as cv
import tifffile as tif
import glob
import os



def reconstruct_manual(X,pca,which_components):
    comps = pca.components_
    mu = pca.mean_
    X_transformed = pca.transform(X)
    return np.dot(X_transformed[:,which_components],comps[which_components,:])+mu


def GaussFilter(im,apply=True,sigma = 2, size=3):
    if apply:
        kernel = np.ones((size,size),np.float32)/(size*size)
        im_filtered = cv.GaussianBlur(im,(size,size),sigmaX = sigma, sigmaY= sigma, borderType =cv.BORDER_DEFAULT)
    else:
        im_filtered= im
    return im_filtered

def MeanFilter(im,apply=True,size=3):
    if apply:
        kernel = np.ones((size,size),np.float32)/(size*size)
        im_filtered = cv.filter2D(im,-1,kernel) 
    else:
        im_filtered= im
    return im_filtered

def GaussFilterCube(spectrum, sigma = 2, size=3):
    spectrum_filtered = np.zeros(spectrum.shape)
    for i in range(spectrum.shape[2]): 
        spectrum_filtered[:,:,i] = GaussFilter(spectrum[:,:,i],apply=True,sigma=sigma,size=size)
    return spectrum_filtered

def MeanFilterCube(spectrum, sigma = 2, size=3):
    spectrum_filtered = np.zeros(spectrum.shape)
    for i in range(spectrum.shape[2]): 
        spectrum_filtered[:,:,i] = MeanFilter(spectrum[:,:,i],apply=True,size=size)
    return spectrum_filtered


def discrete_matshow(data,cmap='RdBu'):
    # get discrete colormap
    cmap = plt.get_cmap(cmap, np.max(data) - np.min(data) + 1)
    # set limits .5 outside true range
    mat = plt.matshow(data, cmap=cmap, vmin=np.min(data) - 0.5, 
                      vmax=np.max(data) + 0.5)
    # tell the colorbar to tick at integers
    cax = plt.colorbar(mat, ticks=np.arange(np.min(data), np.max(data) + 1))

def median_blur(im_in,size=0):
    if size==0:
        im_out = im_in
    else: 
        im_out = cv.medianBlur(np.uint8(im_in),size)
    return im_out

def rebin_spectrum(spectrum,bins=1024):
    x,y,z = spectrum.shape
    spectrum = np.reshape(spectrum,(x,y,bins,int(z/bins)))
    return np.sum(spectrum,axis=-1)

def rebin_XY(img,bins=1024):
    x,y = img.shape
    img= np.reshape(img,(bins,int(x/bins),bins,int(y/bins)))
    return img.mean(axis=-1).mean(axis=1)

def rebin_spectrumXY(spectrum,bins=1024):
    x,y,z = spectrum.shape
    spectrum = np.reshape(spectrum,(bins,int(x/bins),bins,int(y/bins),z))
    spectrum = np.mean(spectrum,axis=-2)
    return spectrum.mean(axis=1)

#def rebin_spectrumXY(spectrum,bins=1024):    # Too slow!
#    x,y,z = spectrum.shape
#    new_spectrum = np.zeros((bins, bins, z))
#    for k in range(z):
#        new_spectrum[:,:,k] = rebin_XY(spectrum[:,:,k])
#        print(k)
#    return new_spectrum


def rebin_energies(energies,bins=1024):
    z = energies.shape[0]
    energies = np.linspace(energies[0],energies[-1],bins)
    return energies

def rebin_energies_01(energies,bins=1024):
    z = energies.shape[0]
    energies = np.linspace(energies[0],energies[-1],bins)
    return energies

def clahe(img, clipLimit=2.0, tileGridSize=(8,8)):
    img = img.astype('uint8')
    lab= cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l_channel, a, b = cv.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    return cv.cvtColor(limg, cv.COLOR_LAB2BGR)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def make_ome_rgb(spectrum_rgb,outpath,lvl=4,downsample_factor=2,pixel_size=2.5,tile_size=32,ch_type='PC'):
    with tif.TiffWriter(outpath, bigtiff=True) as tf:
        data = spectrum_rgb
        data = np.swapaxes(data,2,0)
        data = np.swapaxes(data,1,2)
        options = dict(photometric='rgb', tile=(tile_size, tile_size), compression='None',metadata={'axes': 'CYX'})
        tf.write(data, subifds=lvl, **options)
        # save pyramid levels to the two subifds
        # in production use resampling to generate sub-resolutions
        tf.write(data[:, ::2, ::2], subfiletype=1, **options)
        tf.write(data[:, ::4, ::4], subfiletype=1, **options)
        tf.write(data[:, ::8, ::8], subfiletype=1, **options)


def make_ome(haadf, spectrum_reduced,outpath,lvl=4,downsample_factor=2,pixel_size=2.5,tile_size=32,ch_type='PC'):
    compression = None
    subsample_size = haadf.shape[0]
    haadf_reshape = haadf.reshape((subsample_size,subsample_size,1))
    image_temp = np.concatenate((haadf_reshape,spectrum_reduced),axis=2)
    image_temp = np.swapaxes(image_temp,2,0)
    image_temp = np.swapaxes(image_temp,1,2)
    print(image_temp.shape)

    image = np.zeros(image_temp.shape,dtype='uint8')
    for i in range(image.shape[0]):
        image[i,:,:] = normalize8(image_temp[i,:,:])


    ch_names = ['Haadf']
    for j in range(spectrum_reduced.shape[2]):
        ch_names.append('%s_%02d' % (ch_type,(j+1)))
        
    with tif.TiffWriter(outpath, bigtiff=True) as tf:
        options = {'tile': (tile_size, tile_size),
                           'compression': compression,
                           'metadata':{'PhysicalSizeX': pixel_size*1e9, 'PhysicalSizeXUnit': 'nm',
                                       'PhysicalSizeY': pixel_size*1e9, 'PhysicalSizeYUnit': 'nm',
                                       'axes': 'CYX','Description':"Who dis",
                                       'AcquisitionDate':'Now',
                                       'Name':"oh nana",
                                       'Channel': {'Name':ch_names}}}
        tf.write(image, subifds=lvl, **options)
     
        image2 = image
        
        for i in range(lvl):
            idx = downsample_factor**(i+1)
            tf.write(image2[:,::idx, ::idx], subfiletype=1, **options)


def normalize8(I,normalize_by=None):
  if normalize_by is None:  
    mn = I.min()
    mx = I.max()
  else:
    mn = normalize_by.min()
    mx = normalize_by.max()

  mx -= mn
  I = ((I - mn)/mx) * 255
  return I.astype(np.uint8)

# use this one if you want to normalize by another array
def normalize88(I,normalizer=None):
    if normalizer is None:  
        mn = I.min()
        mx = I.max()
        #print(mx)
    else:
        mn = normalizer.min()
        mx = normalizer.max()
        #print(mx)

    mx -= mn
    I = ((I - mn)/mx) * 255
    return I.astype(np.uint8),mn,mx


# NNLS to FCLS
def nnls2fcls(End_maps):
    EM,x,y,files = End_maps.shape
    for i in range(x):
        for j in range(y):
            for f in range(files):
                End_maps[:,i,j,f] = End_maps[:,i,j,f]/np.sum(End_maps[:,i,j,f])
    return End_maps

def nnls_maxcf(End_maps):
    return End_maps/End_maps.max()


# SAM
def show_anns(anns,display=True,randomColors=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    #ax = plt.gca()
    #ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        if randomColors:
            color_mask = np.concatenate([np.random.random(3), [0.35]])
        else:
            color_mask = np.asarray([1,0,0,0.35])   
        img[m] = color_mask
    
    if display:
        ax.imshow(img)
    else:
        return img
    
# SAM EDX 
def show_anns_EDX(anns,abundance_tile,colors,display=True,alpha=0.35,area_thresh=None):
    if area_thresh is None:
        area_thresh = abundance_tile.shape[0]*abundance_tile.shape[1]
    
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    #ax = plt.gca()
    #ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    img_clr_idx = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]))-1
    for ann in sorted_anns:
        m = ann['segmentation']
        if np.sum(m)<area_thresh:
            tmp_img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]))
            tmp_img[m] = 1

            tmp_abundance_masked = tmp_img*abundance_tile
            temp_sum = np.sum(np.sum(tmp_abundance_masked,axis=1),axis=1)
            color_idx = np.argmax(temp_sum)
            color_mask = np.concatenate([colors[color_idx], [alpha]])
            img[m] = color_mask
            img_clr_idx[m] = color_idx
    if display:
        ax.imshow(img)
    else:
        return img,img_clr_idx




# This is from Peregrine. Possibly not useful anymore
def mosaic_edx(main_folder='/data/p276451/EDX/',rows=[0,2],cols=[0,2],base_dims=[1024,1024,256],hash_sample=None,crop_neg=96):
    haadf = np.zeros((base_dims[0]*(rows[1]-rows[0]),base_dims[1]*(cols[1]-cols[0])))

    if hash_sample is None:
        spectrum = np.zeros((base_dims[0]*(rows[1]-rows[0]),base_dims[1]*(cols[1]-cols[0]),base_dims[2]))
    else:
        hash_dim = len(np.arange(1024)[::9])
        spectrum = np.zeros((hash_dim*(rows[1]-rows[0]),hash_dim*(cols[1]-cols[0]),base_dims[2]))

    for i in range(rows[0],rows[1]):
        for j in range(cols[0],cols[1]):
            file_path = os.path.join(main_folder,'npz_files/row%02d_col%02d.npz' % (i,j))
            loaded_file = np.load(file_path)
            print(file_path)
            haadf_temp = loaded_file['haadf']
            spectrum_temp = rebin_spectrum(loaded_file['spectrum'][:,:,crop_neg:],bins=base_dims[2])

            if hash_sample is not None:
                spectrum_temp = spectrum_temp[::hash_sample,::hash_sample,:]
            
            haadf[base_dims[0]*(i):base_dims[0]*(i)+base_dims[0],
                  base_dims[1]*(j):base_dims[0]*(j)+base_dims[1]] = haadf_temp

            if hash_sample is None:
                spectrum[base_dims[0]*(i):base_dims[0]*(i)+base_dims[0],
                      base_dims[1]*(j):base_dims[0]*(j)+base_dims[1],:] = spectrum_temp
            else:
                spectrum[hash_dim*(i):hash_dim*(i)+hash_dim,
                      hash_dim*(j):hash_dim*(j)+hash_dim,:] = spectrum_temp

    return haadf,spectrum
