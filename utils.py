import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
from PIL import Image

from color_histogram.core.hist_2d import Hist2D
from color_histogram.core.hist_3d import Hist3D



def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def hex_to_rgb(hex):
    # print(hex)
    rgb = []
    for i in (1, 3, 5):
        decimal = int(hex[i:i+2], 16)
        rgb.append(decimal)
    return tuple(rgb)



def image_resize(img, c_w, c_h):
    # img : PIL Image
    if type(img) is np.ndarray:
        img = Image.fromarray(img)
    h, w = img.size
    h_factor = c_h / h
    w_factor = c_w / w
    factor = np.minimum(h_factor, w_factor)
    img = img.resize((np.round(h*factor).astype(np.int64), 
                     np.round(w*factor).astype(np.int64)), 
                     Image.Resampling.BILINEAR)
    return img



def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette




def visualize_palette(palette_lab, patch_size=20):
    if palette_lab is 0:
        return np.ones((patch_size, patch_size, 3)) * [1.,1.,1.]
    
    palette_lab = np.expand_dims(palette_lab, axis=0)
    palette_rgb = lab2rgb(palette_lab)
    palette_rgb = np.squeeze(palette_rgb, axis=0)


    for id in range(np.size(palette_rgb, 0)):
        rgb = np.expand_dims(palette_rgb[id,:], axis=(0, 1))
        if id==0:
            img_palette = np.ones((patch_size, patch_size, 3)) * rgb
        else:
            img_palette = np.append(img_palette, np.ones((patch_size, patch_size, 3)) * rgb, axis=1)
    
    return img_palette


def visualize_palette_rgb(palette_rgb, patch_size=20):
    # print(palette_lab)
    if palette_rgb is 0:
        return np.ones((patch_size, patch_size, 3)) * [1.,1.,1.]

    for id in range(np.size(palette_rgb, 0)):
        rgb = np.expand_dims(palette_rgb[id,:], axis=(0, 1))
        if id==0:
            img_palette = np.ones((patch_size, patch_size, 3)) * rgb
        else:
            img_palette = np.append(img_palette, np.ones((patch_size, patch_size, 3)) * rgb, axis=1)
    
    return img_palette



def draw_comparison(c_src, c_target, img_in, img_out, save_path):     
    vis_patch_size = 20

    img_p = visualize_palette(c_src, patch_size=vis_patch_size)
    img_p_name = visualize_palette(c_target, patch_size=vis_patch_size)


    fig = plt.figure(figsize=(8,5), dpi=250)
    fig.tight_layout()
    plt.rcParams['figure.constrained_layout.use'] = True

    ax = fig.add_subplot(231)
    plt.imshow(img_in)
    plt.axis('off')
    plt.title("input image")

    hist2D_org = Hist2D(img_in, num_bins=16, color_space='Lab', channels=[1, 2])
    color_samples = hist2D_org.colorCoordinates()
    xmax, xmin = color_samples[:, 0].max(), color_samples[:, 0].min()
    ymax, ymin = color_samples[:, 1].max(), color_samples[:, 1].min()
    
    hist2D_rec = Hist2D(img_out, num_bins=16, color_space='Lab', channels=[1, 2])
    color_samples = hist2D_rec.colorCoordinates()
    xmax = np.maximum(color_samples[:, 0].max(), xmax)
    xmin = np.minimum(color_samples[:, 0].min(), xmin)
    ymax = np.maximum(color_samples[:, 1].max(), ymax)
    ymin = np.minimum(color_samples[:, 1].min(), ymin)
    xmax = np.round(xmax)+0.1*(xmax-xmin)
    xmin = np.round(xmin)-0.1*(xmax-xmin)
    ymax = np.round(ymax)+0.1*(ymax-ymin)
    ymin = np.round(ymin)-0.1*(ymax-ymin)


    ax = fig.add_subplot(232)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    hist2D_org.plot(ax)

    ax = fig.add_subplot(233)
    plt.imshow(img_p)
    plt.axis('off')

    ax = fig.add_subplot(234)
    plt.imshow(img_out)
    plt.axis('off')
    plt.title("recolored image")

    ax = fig.add_subplot(235)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    hist2D_rec.plot(ax)

    ax = fig.add_subplot(236)
    plt.imshow(img_p_name)
    plt.axis('off')

    plt.savefig(save_path)
    plt.close()



# plot palette distribution for comparision
def plot_palette_distribution(prototype, colorspace, save_dir):
    color_values = [[1.0, 0.0, 0.0], 
                    [1.0, 0.6, 0.0], 
                    [0.4, 0.2, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.7, 0.0, 0.7],
                    [0.8, 0.6, 0.7],
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.5],
                    [1.0, 1.0, 1.0]]
    color_names = ['Red', 'Orange', 'Brown', 'Yellow', 'Green', 'Blue', 'Purple', 'Pink', 'Black', 'Grey', 'White']
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    if colorspace == 'rgb':
        ax.set_xlabel('R Label')
        ax.set_ylabel('G Label')
        ax.set_zlabel('B Label')
        ax.set_xlim([0,255])
        ax.set_ylim([0,255])
        ax.set_zlim([0,255])

    elif colorspace == 'lab':
        ax.set_xlabel('L Label')
        ax.set_ylabel('a Label')
        ax.set_zlabel('b Label')
        ax.set_xlim([0,100])
        ax.set_ylim([-100,100])
        ax.set_zlim([-100,100])

    elif colorspace == 'hsv':
        ax.set_xlabel('H Label')
        ax.set_ylabel('S Label')
        ax.set_zlabel('V Label')
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_zlim([0,1])

    for i in range(11):
        palette = prototype[i]
        # print(colorspace, i, np.mean(palette[:,:], axis=0))
        ax.scatter(palette[:,0], palette[:,1], palette[:,2], color=color_values[i], marker='o')
    # plt.show()
    plt.savefig(os.path.join(save_dir, 'palette_{}.jpg'.format(colorspace)))
    plt.close()



def color_difference(img1, img2):
    h, w, c = img1.shape
    img1_lab = rgb2lab(img1)
    img2_lab = rgb2lab(img2)

    diff=img1_lab-img2_lab
    
    dE = np.sqrt(diff[:,:,0]**2 + diff[:,:,0]**2 + diff[:,:,0]**2)
    dE = np.sum(dE)/(h*w)

    return dE

