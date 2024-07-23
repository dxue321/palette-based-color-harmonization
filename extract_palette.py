import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from collections import Counter


def histogram(img_lab, bin, mode=2, mask=None):
    if mask is None:
        mask = np.ones_like(img_lab[:,:,0])

    if img_lab.ndim != 2:
        img_lab = img_lab.reshape(-1, 3)

    mask = mask.flatten()
    img_lab_masked = img_lab[mask==1]

    if mode == 3:
        hist, edges = np.histogramdd(img_lab_masked, bins=bin)
        xpos, ypos, zpos = np.meshgrid(edges[0][:-1], edges[1][:-1], edges[2][:-1], indexing="ij")
        hist_samples = np.concatenate((xpos.reshape((bin*bin*bin,1)), ypos.reshape((bin*bin*bin,1)), zpos.reshape((bin*bin*bin,1))), axis=1)
        hist_counts = hist.reshape(bin*bin*bin)

    elif mode == 2:  
        hist, xedges, yedges = np.histogram2d(img_lab_masked[:,1], img_lab_masked[:,2], bins=bin, range=None)
        xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
        hist_samples = np.concatenate((xpos.reshape((bin*bin,1)), ypos.reshape((bin*bin,1))), axis=1)
        hist_counts = hist.reshape(bin*bin)
    
    # hist_counts = hist_counts/np.sum(hist_counts)

    return hist_samples, hist_counts




def extract_palette(img_lab, hist_samples, hist_counts, mode=2, lightness=70, threshold=0.93, max_cluster=5, mask=None):
    if mask is None:
        mask = np.ones_like(img_lab[:,:,0])
    
    if img_lab.ndim != 2:
        img_lab = img_lab.reshape(-1, 3)
    
    mask = mask.flatten()
    hist_densities = hist_counts /np.sum(hist_counts)

    ## palette extraction

    # inital cluster center
    index = np.argwhere(hist_densities!=0)
    index = np.squeeze(index, axis=(1,))
    num_nonzero = np.size(index)

    ## directly clustering
    # num_clusters_opt = max_cluster
    # kmeans_f = KMeans(n_clusters=num_clusters_opt, init='k-means++', random_state=0).fit(
    #     hist_samples[index, :], y=None, sample_weight=hist_densities[index])
    
    ## clustering method from matlab code
    inits_all = []
    Cold = np.mean(hist_samples[index, :], 0)
    distortion=np.zeros((max_cluster,1))

    dist = pairwise_distances(hist_samples[index, :], np.expand_dims(Cold, axis=0), metric='euclidean')
    distortion[0] = np.sum(hist_densities[index] * np.squeeze(dist**2, axis=1), 0)

    inits_all.append(Cold)
    


    for k in range(1, max_cluster):
        # Initialize the cluster centers
        k = k+1
        cinits = np.zeros((k, mode))
        cw = hist_densities[index]
        for i in range(k):
            id = np.argmax(cw)
            cinits[i,:] = hist_samples[index, :][id,:]
            d2 = cinits[i,:]* np.ones((num_nonzero, 1)) - hist_samples[index, :]
            d2 = np.sum(np.square(d2), axis=1)
            d2 = d2/np.max(d2)
            cw = cw * (d2**2)

        inits_all.append(cinits)
        kmeans = KMeans(n_clusters=k, init=cinits, n_init=1).fit(
                        hist_samples[index, :], y=None, sample_weight=hist_densities[index])
        
        dist_point = pairwise_distances(hist_samples[index, :], kmeans.cluster_centers_, metric='euclidean')
        distortion[k-1] = np.sum(hist_densities[index] * np.min(dist_point, axis=1)**2)

    variance = distortion[:-1] - distortion[1:]
    distortion_percent = np.cumsum(variance)/(distortion[0]-distortion[-1])

    r=np.argwhere(distortion_percent > threshold)
    num_clusters_opt = np.min(r)+2

    kmeans_f = KMeans(n_clusters=num_clusters_opt, init=inits_all[num_clusters_opt-1], n_init=1).fit(
                      hist_samples[index, :], y=None, sample_weight=hist_densities[index])
    cluster_centers = kmeans_f.cluster_centers_


    if mode == 2:
        cluster_centers = np.insert(cluster_centers, 0, values=lightness, axis=1)

    # if mode ==3:
    #     img_labels = kmeans_f.predict(img_lab)
    # elif mode == 2:
    #     img_labels = kmeans_f.predict(img_lab[:, 1:3])

    # hist_labels = kmeans_f.predict(hist_samples)
    

    # img_labels[mask==0] = 255
    # c_densities = np.zeros(num_clusters_opt)
    
    # dict=Counter(img_labels)
    # for key in np.unique(img_labels):
    #     if key == 255:
    #         continue
    #     c_densities[key] = dict.get(key)

    # c_densities = c_densities / np.sum(c_densities)
    
    # return cluster_centers, c_densities, img_labels, hist_labels

    return cluster_centers