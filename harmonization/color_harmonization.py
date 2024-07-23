#! python3

import cv2
import sys
import numpy as np
import harmonization.util as util



HueTemplates = {
    "i"       : [( 0.00, 0.05)],
    "V"       : [( 0.00, 0.26)],
    "L"       : [( 0.00, 0.05), ( 0.25, 0.22)],
    "mirror_L": [( 0.00, 0.05), (-0.25, 0.22)],
    "I"       : [( 0.00, 0.05), ( 0.50, 0.05)],
    "T"       : [( 0.25, 0.50)],
    "Y"       : [( 0.00, 0.26), ( 0.50, 0.05)],
    "X"       : [( 0.00, 0.26), ( 0.50, 0.26)],
}
template_types = list(HueTemplates.keys())
M = len(template_types)
A = 360


deg_distance = util.deg_distance


class HueSector:

    def __init__(self, center, width):
        # In Degree [0,2 pi)
        self.center = center
        self.width  = width
        self.border = [(self.center - self.width/2), (self.center + self.width/2)]

    def is_in_sector(self, H):
        # True/False matrix if hue resides in the sector
        return deg_distance(H, self.center) < self.width/2

    def distance_to_border(self, H):        
        H_1 = deg_distance(H, self.border[0])
        H_2 = deg_distance(H, self.border[1])
        H_dist2bdr = np.minimum(H_1, H_2)
        return H_dist2bdr
        
    def closest_border(self, H):
        H_1 = deg_distance(H, self.border[0])
        H_2 = deg_distance(H, self.border[1])
        H_cls_bdr = np.argmin((H_1, H_2), axis=0)
        H_cls_bdr = 2*(H_cls_bdr - 0.5)
        return H_cls_bdr


    def closest_border_dir(self, H):
        H_1 = deg_distance(H, self.border[0])
        H_2 = deg_distance(H, self.border[1])
        H_cls_bdr = np.argmin((H_1, H_2), axis=0)
        H_dir = np.zeros(H.shape)
        for i in range(2):
            mask = (H_cls_bdr == i)
            H_dir += util.deg_closest_direction(H, self.border[i]) * mask
        return H_dir

    def distance_to_center(self, H):
        H_dist2ctr = deg_distance(H, self.center)
        return H_dist2ctr



class HarmonicScheme:
    def __init__(self, m, alpha):
        self.m = m
        self.alpha = alpha
        self.reset_sectors()

    def reset_sectors(self):
        self.sectors = []
        for t in HueTemplates[self.m]:
            center = t[0] * 360 + self.alpha
            width  = t[1] * 360
            sector = HueSector(center, width)
            self.sectors.append( sector )

    def harmony_score(self, X):
        # Opencv store H as [0, 180) --> [0, 360)
        H = X[:, :, 0].astype(np.int32)* 2
        # Opencv store S as [0, 255] --> [0, 1]
        S = X[:, :, 1].astype(np.float32) / 255.0
        
        H_dis = self.hue_distance(H)
        H_dis = np.deg2rad(H_dis)
        return np.sum( np.multiply(H_dis, S) )
    

    def harmony_score_inside(self, X):
        # Opencv store H as [0, 180) --> [0, 360)
        H = X[:, :, 0].astype(np.int32) * 2
        # Opencv store S as [0, 255] --> [0, 1]
        S = X[:, :, 1].astype(np.float32) / 255.0

        inside = []
        # print(len(self.sectors))
        for i in range(len(self.sectors)):
            label = self.sectors[i].is_in_sector(H.flatten())
            # print(label.min(), label.max())
            inside.append(label)

        if len(self.sectors)>1:
            inside_sum = inside[0]+inside[1]
        else:
            inside_sum = inside

        score = np.sum(inside_sum)/(np.shape(X)[0]*np.shape(X)[1])
        score_sat = np.sum(np.multiply(inside_sum, S.flatten()))/np.sum(inside_sum)

        return score_sat, score
    

    def harmony_score_hist(self, samples, counts):
        # Opencv store H as [0, 180) --> [0, 360)
        H = (samples[:, :, 0]).astype(np.int32)* 2
        # Opencv store S as [0, 255] --> [0, 1]
        S = samples[:, :, 1].astype(np.float32) / 255.0

        # print(np.max(H), np.min(H), np.max(S), np.min(S))
        
        H_dis = self.hue_distance(H)
        H_dis = np.deg2rad(H_dis)*counts
        return np.sum( np.multiply(H_dis, S) )


    def harmony_score_hist_inside(self, samples, counts):
        # Opencv store H as [0, 180) --> [0, 360)
        H = (samples[:, :, 0]).astype(np.int32)* 2
        # Opencv store S as [0, 255] --> [0, 1]
        S = samples[:, :, 1].astype(np.float32) / 255.0

        inside = []
        # print(len(self.sectors))
        for i in range(len(self.sectors)):
            label = self.sectors[i].is_in_sector(H.flatten())
            # print(label.min(), label.max())
            inside.append(label)
        
        inside = inside*counts
        inside = inside.sum(axis=0)

        score = np.sum(inside)/np.sum(counts)
        score_sat = np.sum( np.multiply(inside, S))/np.sum(counts)
        return score_sat, score


    def hue_distance(self, H):
        H_dis = []
        for i in range(len(self.sectors)):
            sector = self.sectors[i]
            H_dis.append(sector.distance_to_border(H))
            H_dis[i][sector.is_in_sector(H)] = 0
        H_dis = np.asarray(H_dis)        
        H_dis = H_dis.min(axis=0)
        return H_dis


    def hue_shifted(self, X):
        Y = X.copy()
        H = X[:, :, 0].astype(np.int32)*2
        S = X[:, :, 1].astype(np.float32) / 255.0
        
        H_d2b = [ sector.distance_to_border(H) for sector in self.sectors ]
        H_d2b = np.asarray(H_d2b)
        
        H_cls = np.argmin(H_d2b, axis=0)
        
        H_ctr = np.zeros((H.shape))
        H_wid = np.zeros((H.shape))
        H_d2c = np.zeros((H.shape))
        H_dir = np.zeros((H.shape))
        
        for i in range(len(self.sectors)):
            sector = self.sectors[i]
            mask = (H_cls == i)
            H_ctr[mask] = sector.center
            H_wid[mask] = sector.width
            H_dir += sector.closest_border(H) * mask
            H_dist2ctr = sector.distance_to_center(H)
            #H_dist2ctr[sector.is_in_sector(H)] = 0
            H_d2c += H_dist2ctr * mask
            
        H_sgm = H_wid / 2
        H_gau = util.normalized_gaussian(H_d2c, 0, H_sgm)
        H_tmp = np.multiply(H_wid / 2, 1 - H_gau)
        H_shf = np.multiply( H_dir, H_tmp )
        H_new = (H_ctr + H_shf).astype(np.int32)

        for i in range(len(self.sectors)):
            sector = self.sectors[i]
            mask = sector.is_in_sector(H)
            np.copyto(H_new, H, where=sector.is_in_sector(H))
        #H_new = tmp.data

        H_new = np.remainder(H_new, 360)
        H_new = (H_new/2).astype(np.uint8)
        Y[:,:,0] = H_new
        return Y

    def energy_E(self, X, V, P):
        w1 = 1.0
        w2 = 1.0
        e = w1 * self.energy_E1(X, V, P)
        return e

    def energy_E1(self, X, V, P):
        # Opencv store H as [0, 180) --> [0, 360)
        H = X[:, :, 0].astype(np.int32)* 2
        # Opencv store S as [0, 255] --> [0, 1]
        S = X[:, :, 1].astype(np.float32) / 255.0
        
        H_P = H[ P[0], P[1] ]
        V_P = V[ P[0], P[1] ]
        S_P = S[ P[0], P[1] ]

        d = util.deg_distance(H_P, V_P)
        s = S_P
        
        e1 = np.multiply(d, s)
        e1 = np.sum(e1)
        return e1


    def update_alpha(self, alpha):
        self.alpha = alpha
        self.sectors = []
        for t in HueTemplates[self.m]:
            center = t[0] * 360 + alpha
            width  = t[1] * 360
            sector = HueSector(center, width)
            self.sectors.append( sector )

    def update_template(self, m):
        self.m = m
        self.sectors = []
        for t in HueTemplates[m]:
            center = t[0] * 360 + self.alpha
            width  = t[1] * 360
            sector = HueSector(center, width)
            self.sectors.append( sector )


def B(X):
    F_matrix = np.zeros((M, A))
    for i in range(M):
        m = template_types[i]
        for j in range(A):
            # print(i,j)
            alpha = 360/A * j
            harmomic_scheme = HarmonicScheme(m, alpha)
            F_matrix[i, j] = harmomic_scheme.harmony_score(X)
    (best_m_idx, best_alpha) = np.unravel_index( np.argmin(F_matrix), F_matrix.shape )
    best_m = template_types[best_m_idx]

    best_alpha = np.argmin(F_matrix[best_m_idx])
    f_score = np.min(F_matrix[best_m_idx])
    best_harmomic_scheme = HarmonicScheme(best_m, best_alpha)
    return best_harmomic_scheme, f_score



def B_hist(samples, counts):
    dict_temple = {}
    F_matrix = np.zeros((M, A))
    for i in range(M):
        m = template_types[i]
        for j in range(A):
            # print(i,j)
            alpha = 360/A * j
            harmomic_scheme = HarmonicScheme(m, alpha)
            F_matrix[i, j] = harmomic_scheme.harmony_score_hist(samples, counts)
        dict_temple[m] = np.argmin(F_matrix[i, :])

    (best_m_idx, best_alpha) = np.unravel_index( np.argmin(F_matrix), F_matrix.shape )
    best_score = np.min(F_matrix[best_m_idx])
    best_harmomic_scheme = HarmonicScheme(template_types[best_m_idx], best_alpha)

    return best_harmomic_scheme, best_score