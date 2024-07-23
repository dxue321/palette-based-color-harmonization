import numpy as np
import pandas as pd
import heapq
import scipy
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb, rgb2hsv
from color_naming.color_naming import img2color_rgb, id_joost2label, id_ca2label, id_yu2label




df2 = pd.read_csv('./color_naming/w2c_rgb.csv', header=None)
W2CRGB = df2.values
W2CHSV = np.squeeze(rgb2hsv(np.expand_dims(W2CRGB/255., axis=0)), axis=0)
W2CLAB = np.squeeze(rgb2lab(np.expand_dims(W2CRGB/255., axis=0)), axis=0)


def img2color(img_lab, W2C, name_type='joost'):
    img_lab = np.expand_dims(img_lab, axis=0)
    img_rgb = lab2rgb(img_lab)*255
    color_label, color_nam, color_map, prob_map = img2color_rgb(img_rgb, W2C, name_type)

    return color_label, color_nam, color_map, prob_map



def color_clustering(cluster_num, color_percent, W2C, name_type='joost', extra_num=5, cluster_type='saturation', colorspace='lab', mode=2):

    num, class_num = W2C.shape
    w2cM = np.argmax(W2C, 1)

    sub_color_percent = 0.15

    prototype_rgb, prototype_lab, prototype_hsv, num_prototype = [], [], [], []
    for label in range(class_num):
        pure_colors_idx = np.array(np.where(w2cM==label))
        pure_colors_prob = W2C[pure_colors_idx, label]

        # print(label, pure_colors_idx.shape)
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(4,2))                  
        # ax = plt.subplot(111) 
        # nt, bins, patches = plt.hist(np.array(pure_colors_prob).flatten(), density=True, color='red',bins=10, range=(0,1))
        # # plt.title('Probability distribution of color '+ COLOR_NAME[label])
        # print(id_joost2label[label].name, nt)
        # plt.xlim(0, 1)
        # plt.ylim(0, 10)
        # plt.xlabel('Probability', fontsize=18)
        # plt.ylabel('Frequency', fontsize=18)
        # plt.xticks(fontsize=14)
        # plt.yticks(fontsize=14)

        # # plt.show()
        # plt.savefig('./results/'+ id_ca2label[label].name + '_joost.png', bbox_inches='tight')   
        # plt.close()

        pure_colors_num = np.size(pure_colors_prob, 1)
        sort_idx = np.argsort(pure_colors_prob)
        pure_num= np.round(pure_colors_num * color_percent).astype(np.int32)
        selected_idx = np.array(pure_colors_idx[0:, sort_idx[0, -pure_num:]])
        selected_prob = W2C[selected_idx, label]

        rgb = np.squeeze(W2CRGB[selected_idx, :], axis=0)
        hsv = np.squeeze(W2CHSV[selected_idx, :], axis=0)
        lab = np.squeeze(W2CLAB[selected_idx, :], axis=0)

        if colorspace=='lab':
            if mode == 2:
                samples = lab[:,1:]
            else:
                samples = lab
        elif colorspace=='rgb':
            samples = rgb


        if name_type == 'joost':
            if_extra_colors = id_joost2label[label].extra_color
        elif name_type == 'yu':
            if_extra_colors = id_yu2label[label].extra_color
        elif name_type == 'ca':
            if_extra_colors = id_ca2label[label].extra_color

        cluster_num_m = cluster_num
        if if_extra_colors:
            cluster_num_m = cluster_num + extra_num

        kmeans_f_2 = KMeans(n_clusters=cluster_num_m, random_state=0, n_init=cluster_num_m).fit(
                            samples, y=None, sample_weight=selected_prob.flatten()) 
        

        img_labels = kmeans_f_2.predict(samples)
        top_colors_rgb = np.zeros((cluster_num_m, 3))
        top_colors_hsv = np.zeros((cluster_num_m, 3))
        top_colors_lab = np.zeros((cluster_num_m, 3))

        
        if cluster_type == 'saturation':
            for center in range(cluster_num_m):
                center_idx = np.array(np.where(img_labels == center))
                color = hsv[center_idx, :]
                sat_rank = np.argsort(color[:,:,1])[:,-1]
                top_colors_rgb[center, :] = rgb[center_idx[0, sat_rank.flatten()], :]
                top_colors_hsv[center, :] = hsv[center_idx[0, sat_rank.flatten()], :]
                top_colors_lab[center, :] = lab[center_idx[0, sat_rank.flatten()], :]


        elif cluster_type == 'prob':
            for center in range(cluster_num_m):
                center_idx = np.array(np.where(img_labels == center))
                prob = selected_prob[:, center_idx[0]]
                sat_rank = np.argsort(prob)[:,-1]
                top_colors_rgb[center, :] = rgb[center_idx[0, sat_rank.flatten()], :]
                top_colors_hsv[center, :] = hsv[center_idx[0, sat_rank.flatten()], :]
                top_colors_lab[center, :] = lab[center_idx[0, sat_rank.flatten()], :]


        elif cluster_type == 'probysat':
            for center in range(cluster_num_m):
                center_idx = np.array(np.where(img_labels == center))
                prob = selected_prob[:, center_idx[0]]
                prob_rank = np.argsort(prob)
                
                sub_pure_num = np.round(np.size(prob, 1) * sub_color_percent).astype(np.int32)
                sub_selected_idx = np.array(center_idx[0:, prob_rank[0, -sub_pure_num:]])
                # sub_selected_prob = prob[:, prob_rank[0, -sub_pure_num:]]

                color = hsv[sub_selected_idx, :]
                sat_rank = np.argsort(color[:,:,1])[:,-1]

                top_colors_rgb[center, :] = rgb[sub_selected_idx[0, sat_rank.flatten()], :]
                top_colors_hsv[center, :] = hsv[sub_selected_idx[0, sat_rank.flatten()], :]
                top_colors_lab[center, :] = lab[sub_selected_idx[0, sat_rank.flatten()], :]

        prototype_rgb.append(top_colors_rgb)
        prototype_lab.append(top_colors_lab)
        prototype_hsv.append(top_colors_hsv)
        num_prototype.append(cluster_num_m)


    return prototype_rgb, prototype_lab, prototype_hsv, num_prototype




def compare_color_name(img_src, img_tgt, W2C, name_type='joost', threshold=0.98):

    color_label_org, color_nam_org, _, prob_map_org = img2color(img_src, W2C, name_type=name_type)
    color_label_new, color_nam_new, _, prob_map_new = img2color(img_tgt, W2C, name_type=name_type)

    if threshold==0:
        is_same_color = (color_label_org==color_label_new)
    else:
        diff = np.zeros_like(color_label_org).astype(np.float64)
        for jj in range(np.size(color_label_org, 0)):
            for ii in range(np.size(color_label_org, 1)):
                # difference also can be the l1 , l2 distance, kl divergence between two probablity distribution
                # diff[jj, ii] = prob_map_org[jj, ii, color_label_org[jj, ii]] - prob_map_new[jj, ii, color_label_org[jj, ii]]
                diff[jj, ii] = np.linalg.norm(prob_map_org[jj, ii, :]-prob_map_new[jj, ii, :]) 
        is_same_color = (np.abs(diff) < threshold)

    print(is_same_color)
    return is_same_color



def find_similar_color_topkname(img_src, prototype_rgb, prototype_lab, prototype_hsv, 
                                W2C, name_type='joost', 
                                dist_type='l2', top_k=3, prob_thres = 0.15,
                                colorspace='rgb', mode=2, lightness=70
                                ):
    _, _, _, prob_map = img2color(img_src, W2C, name_type=name_type)
    target_color = np.zeros_like(img_src)

    for j in range(np.size(img_src, 0)):
        prob = prob_map[0, j]
        top_index = heapq.nlargest(top_k, range(len(prob)), prob.take)

        prototype_topk_lab = []
        prototype_topk_rgb = []
        prototype_topk_hsv = []
        for k in top_index:
            if prob[k] > prob_thres:
                prototype_topk_lab.extend(prototype_lab[k])
                prototype_topk_rgb.extend(prototype_rgb[k])
                prototype_topk_hsv.extend(prototype_hsv[k])
        proto_color_num = len(prototype_topk_lab)
        
        ## lab
        if colorspace == 'lab':
            color = np.ones((proto_color_num, 1)) * img_src[j,:]

            if dist_type=='l2':
                diff = (color - prototype_topk_lab)**2
            elif dist_type=='l1':
                diff = (color - prototype_topk_lab)
            elif dist_type=='angle':
                diff = angular(img_src[j, 1:], np.array(prototype_topk_lab)[:, 1:], type='ab')

            if mode == 2:
                diff = np.sum(diff[:, 1:], axis=1)
                idx = np.argmin(diff)
                target_color[j, 1:] = prototype_topk_lab[idx][1:]
                target_color[j, 0] = lightness
            elif mode == 3:
                diff = np.sum(diff, axis=1)
                idx = np.argmin(diff)
                target_color[j, :] = prototype_topk_lab[idx]


        ## rgb
        elif colorspace == 'rgb':
            img_src_1 = np.expand_dims(img_src, axis=0)
            img_src_rgb = lab2rgb(img_src_1)*255
            img_src_rgb = np.squeeze(img_src_rgb)
            # print(img_src_rgb, pure_colors_rgb)

            if dist_type=='l2':
                ## l2 distance
                color = np.ones((proto_color_num, 1)) * img_src_rgb[j]
                diff = (color - prototype_topk_rgb)**2
                diff = np.sqrt(np.sum(diff, axis=1))
            elif dist_type=='l1':
                # l1 distance
                color = np.ones((proto_color_num, 1)) * img_src_rgb[j]
                diff = color - np.array(prototype_topk_rgb)
                diff = np.sum(diff, axis=1)
            elif dist_type=='angle':
                diff = angular(img_src_rgb[j], np.array(prototype_topk_rgb))

            idx = np.argmin(diff)
            target_color[j, :] = prototype_topk_lab[idx]
            # target_color[j, 0] = lightness
        
        elif colorspace == 'hsv':
            img_src_1 = np.expand_dims(img_src, axis=0)
            img_src_rgb = lab2rgb(img_src_1)
            img_src_hsv = rgb2hsv(img_src_rgb)
            img_src_hsv = np.squeeze(img_src_hsv)

            prototype_topk_hsv = np.array(prototype_topk_hsv)
            color = np.ones((proto_color_num, 1)) * img_src_hsv[j,:]
            # diff = (color - prototype_topk_lab)**2
            diff = color[:,0] - prototype_topk_hsv[:,0] 

            idx = np.argmin(diff)
            target_color[j, :] = prototype_topk_lab[idx]

    return target_color




def find_similar_color_topkname_prob(img_src, prototype_lab, W2C, name_type='joost', 
                                     top_k=3, lightness=70., prob_thres=0.15, mode=2):
    _, _, _, prob_map = img2color(img_src, W2C, name_type=name_type)
    target_color = np.zeros_like(img_src)

    for j in range(np.size(img_src, 0)):
        prob = prob_map[0, j]
        top_index = heapq.nlargest(top_k, range(len(prob)), prob.take)

        prototype_topk_lab = []
        for k in top_index:
            if prob[k]>prob_thres:
                prototype_topk_lab.extend(prototype_lab[k])

        proto_color_num = len(prototype_topk_lab)
        prototype_topk_lab = np.array(prototype_topk_lab)
        prototype_topk_lab[:,0] = lightness

        _, _, _, prob_map_pro = img2color(prototype_topk_lab, W2C, name_type=name_type)

        diff = np.zeros((proto_color_num,1))
        for ii in range(proto_color_num):
            diff[ii] = cross_entropy(prob, prob_map_pro[0, ii, :]) 

        idx = np.argmin(diff)
        target_color[j, :] = prototype_topk_lab[idx]

    return target_color



def l2_distence(p, q):
    p = np.float_(p)
    q = np.float_(q)
    return np.linalg.norm(p-q)

def cross_entropy(p, q):
    p = np.float_(p)
    q = np.float_(q)
    return -sum([p[i]*np.log2(q[i]) for i in range(len(p))])


def KL_divergence(p,q):
    return scipy.stats.entropy(p, q)


def JS_divergence(p,q):
    M=(p+q)/2
    return 0.5*scipy.stats.entropy(p,M)+0.5*scipy.stats.entropy(q, M)


def angular(p, q_all, type='rgb'):
    ang = np.zeros((np.size(q_all,0),1))

    for i in range(np.size(q_all,0)):
        q = q_all[i,:]

        if type == 'rgb':
            p_gray = p[0]*0.299 + p[1]*0.587 + p[2]*0.114
            q_gray = q[0]*0.299 + q[1]*0.587 + q[2]*0.114
            # p_gray = 1.
            # q_gray = 1.
        else:
            p_gray = 1.
            q_gray = 1.

        p = p/p_gray
        q = q/q_gray

        l_p = np.sqrt(p.dot(p))
        l_q = np.sqrt(q.dot(q))

        ang[i] = np.arccos(p.dot(q)/(l_p*l_q))

    return ang



