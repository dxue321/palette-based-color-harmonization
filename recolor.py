import numpy as np
from skimage.color import rgb2lab, lab2rgb


def rgb_transfer(Iin, C_src, C_tgt, mask=None):

    Iin = np.array(Iin/255.).astype(np.float32)

    C_src = lab2rgb(np.expand_dims(C_src, axis=0))
    C_tgt = lab2rgb(np.expand_dims(C_tgt, axis=0))

    C_src = np.squeeze(C_src, axis=0)
    C_tgt = np.squeeze(C_tgt, axis=0)

    if mask is None:
        mask = np.ones_like(Iin[:,:,0])

    m, n, b = Iin.shape

    Iin = np.reshape(Iin, (m*n, b))
    Iout = Iin.copy()
    mask = mask.flatten()
    
    Iout = ab_transfer(Iin, C_src, C_tgt, mask=mask)
    Iout = np.reshape(Iout, (m,n,b))

    return Iout



def lab_transfer(Iin, C_src, C_tgt, mask=None, mode=2):
    # Convert RGB to Lab
    if mask is None:
        mask = np.ones_like(Iin[:,:,0])

    Pout = C_tgt.copy()
    m, n, b = Iin.shape

    # Iin = rgb2lab(Iin)
    Iin = np.reshape(Iin, (m*n, b))
    Iout = Iin.copy()
    mask = mask.flatten()

    # C_src = rgb2lab(np.expand_dims(C_src, axis=0))
    # C_tgt = rgb2lab(np.expand_dims(C_tgt, axis=0))

    # C_src = np.squeeze(C_src, axis=0)
    # C_tgt = np.squeeze(C_tgt, axis=0)
    
    if mode == 2:
        Iout[:, 1:] = ab_transfer(Iin[:, 1:], C_src[:, 1:], C_tgt[:, 1:], mask=mask)
    else:
        Iout[:, 0:] = ab_transfer(Iin[:, 0:], C_src[:, 0:], C_tgt[:, 0:], mask=mask)

    # Convert Lab to RGB
    Iout = np.reshape(Iout, (m,n,b))
    Iout = lab2rgb(Iout)

    return Iout, Pout



def ab_transfer(I_src, C_src, C_tgt, mask=None):
    if mask is None:
        mask = np.ones_like(I_src[:,0])

    I_tgt = np.zeros_like(I_src)
    [m, b] = I_src.shape

    # remove close color
    k = np.size(C_src, 0)
    eps = 0.0001
    W = np.zeros((m, k))
    for i in range(k):
        D = np.zeros(m)
        for j in range(b):
            D = D + (I_src[:, j] - C_src[i,j])**2
        W[:, i] = 1./(D + eps)

    # print(k,b)

    sumW= np.sum(W, 1)
    for j in range(k):
        W[:, j]= W[:, j] / sumW

    for i in range(k):
        for j in range(b):
            I_tgt[:, j] = I_tgt[:, j] + W[:, i] * (I_src[:, j] + C_tgt[i, j] - C_src[i, j])
    
    idx = np.argwhere(mask == 0)
    
    I_tgt[idx, :] = I_src[idx, :]

    return I_tgt




def lab_transfer_cls(Iin, C_src, C_tgt, mask=None, valid_class=None):
    # Convert RGB to Lab
    if mask is None:
        mask = np.ones_like(Iin[:,:,0])
    
    Pout = C_tgt.copy()
    m, n, b = Iin.shape

    Iin = rgb2lab(Iin)
    Iin = np.reshape(Iin, (m*n, b))
    Iout = Iin.copy()
    Iout_cls = Iin.copy()

    mask = mask.flatten()
    mask_bin = np.zeros_like(mask)

    # C_src = rgb2lab(np.expand_dims(C_src, axis=0))
    # C_tgt = rgb2lab(np.expand_dims(C_tgt, axis=0))

    # C_src = np.squeeze(C_src, axis=0)
    # C_tgt = np.squeeze(C_tgt, axis=0)
    for id_cls, cls in enumerate(valid_class):
        if C_src[id_cls] == 0:
            continue
        mask_bin[mask==cls] = 1
        Iout_cls[:, 1:] = ab_transfer(Iin[:, 1:], C_src[id_cls][:, 1:], C_tgt[id_cls][:, 1:], mask=mask_bin)
        Iout[mask_bin, 1:] = Iout_cls[mask_bin, 1:]
    
    # Convert Lab to RGB
    Iout = np.reshape(np.round(Iout), (m,n,b))
    Iout = lab2rgb(Iout)

    return Iout, Pout

