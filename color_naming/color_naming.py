import numpy as np
import pandas as pd
from collections import namedtuple

##### color naming methods reference
### ca 
# @article{parraga2009psychophysical,
#   title={Psychophysical measurements to model intercolor regions of color-naming space},
#   author={P{\'a}rraga, C Alejandro and Benavente, Robert and Vanrell, Maria and Baldrich, Ramon},
#   journal={Journal of Imaging Science and Technology},
#   volume={53},
#   number={3},
#   pages={031106},
#   year={2009},
#   publisher={Society for Imaging Science and Technology}
# }

### joost
# @article{van2009learning,
#   title={Learning color names for real-world applications},
#   author={Van De Weijer, Joost and Schmid, Cordelia and Verbeek, Jakob and Larlus, Diane},
#   journal={{IEEE} Trans. Image Process.},
#   volume={18},
#   number={7},
#   pages={1512--1523},
#   year={2009},
#   publisher={IEEE}
# }


### yu
# @article{yu2018beyond,
#   title={Beyond eleven color names for image understanding},
#   author={Yu, Lu and Zhang, Lichao and van de Weijer, Joost and Khan, Fahad Shahbaz and Cheng, Yongmei and Parraga, C Alejandro},
#   journal={Machine Vision and Applications},
#   volume={29},
#   pages={361--373},
#   year={2018},
#   publisher={Springer}
# }

### color in the lookup table range from 0 to 255 in each channel, but with stride 8


name_num = 11
Label = namedtuple( 'Label' , [
                    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                                    # We use them to uniquely name a class
                    'id_joost'    , # An integer ID that is associated with this label. (joost's method)
                    'id_yu'    , # An integer ID that is associated with this label. (yu's method)
                    'id_ca'       , # An integer ID that is associated with this label. (CA's method)
                    'color'       , # The color of this label
                    'extra_color' , # The color of this label

                    ] )


# 11 color names: 'Black', 'Blue', 'Brown', 'Grey', 'Green', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow'

labels = [
    #       name        id_joost  id_yu   id_ca      color          extra_color 
    Label(  'Black'     ,  0  ,   0  ,     8   , [0,     0,   0]      , False  ),
    Label(  'Blue'      ,  1  ,   2  ,     5   , [0,     0, 255]      , True   ),
    Label(  'Brown'     ,  2  ,   1  ,     2   , [102,  51,   0]      , False  ),
    Label(  'Grey'      ,  3  ,   3  ,     9   , [128, 128, 128]      , False  ),
    Label(  'Green'     ,  4  ,   4  ,     4   , [0,   255,   0]      , True   ),
    Label(  'Orange'    ,  5  ,   5  ,     1   , [255, 153,   0]      , False  ),    
    Label(  'Pink'      ,  6  ,   6  ,     7   , [204, 153, 179]      , True   ),
    Label(  'Purple'    ,  7  ,   7  ,     6   , [179,   0, 179]      , True   ),
    Label(  'Red'       ,  8  ,   8  ,     0   , [255,   0,   0]      , True   ),
    Label(  'White'     ,  9  ,   9  ,     10  , [255,   0,   0]      , False  ),
    Label(  'Yellow'    ,  10 ,   10 ,     3   , [255, 255,   0]      , False  ),
]



name2label      = { label.name    : label for label in labels }
id_joost2label  = { label.id_joost: label for label in labels }
id_yu2label     = { label.id_yu   : label for label in labels }
id_ca2label     = { label.id_ca   : label for label in labels }
name2color      = { label.name    : label.color for label in labels }




def load_colornamelut(name_type='joost'):
    if name_type == 'joost':
        df1 = pd.read_csv('./color_naming/w2c.csv', header=None)
        W2C = df1.values
    elif name_type == 'ca':
        df1 = pd.read_csv('./color_naming/w2c_TSE.csv', header=None)
        W2C = df1.values
    elif name_type == 'yu':
        df1 = pd.read_csv('./color_naming/w2c_Yu.csv', header=None)
        W2C = df1.values
    return W2C



def img2color_rgb(img_rgb, W2C, name_type='joost'):
    # input: 
    # img_rgb    --- rgb image (hxwx3)
    # W2C        --- rgb2prob lookup table 
    # name_type  --- color naming methd
    # output:
    # color_label--- numeric color label
    # color_nam  --- color name for 11 color classes
    # color_map  --- color map (hxwx3)
    # prob_map   --- probability map (hxwx11)

    RR = img_rgb[:, :, 0]
    GG = img_rgb[:, :, 1]
    BB = img_rgb[:, :, 2]

    index_im = np.floor(RR.flatten()/8)+ 32*np.floor(GG.flatten()/8)+ 32*32*np.floor(BB.flatten()/8)
    index_im = index_im.astype(np.int32)
    color_map = img_rgb.copy()
    w2cM = np.argmax(W2C, 1)

    h,w,_ = img_rgb.shape

    if name_type == 'joost':
        color_lut = id_joost2label
    elif name_type == 'yu':
        color_lut = id_yu2label
    elif name_type == 'ca':
        color_lut = id_ca2label


    color_label = np.reshape(w2cM[index_im.flatten()], (h,w))
    color_label = color_label.astype(np.int32)
    color_nam = np.empty((h, w), dtype=object)
    # print(color_nam)
    prob_map = np.zeros((h,w, name_num))
    
    
    for jj in range(h):
        for ii in range(w):
            color_map[jj, ii, :] = color_lut[color_label[jj, ii]].color
            color_nam[jj, ii] = color_lut[color_label[jj, ii]].name
            # print(color_lut[color_label[jj, ii]].name)

    for color in range(name_num):
        w2cM = W2C[:, color]
        prob_map[:, :, color] = np.reshape(w2cM[index_im.flatten()], (h, w))

    return color_label, color_nam, color_map, prob_map




if __name__ ==  '__main__':
    import cv2
    img_dir = './a0019-jmac_MG_0653.png'
    img = cv2.imread(img_dir)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    w2c = load_colornamelut(name_type='joost')
    color_label, color_nam, color_map, prob_map = img2color_rgb(img_rgb, w2c, name_type='joost')
    #print(color_label.shape, color_nam, color_map, prob_map)

    cv2.imwrite('./colormap.png', cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR))


