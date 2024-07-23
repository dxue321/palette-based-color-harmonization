import os
import logging
import argparse
import cv2
import time

import numpy as np

from skimage.color import rgb2lab, lab2rgb, rgb2hsv, hsv2rgb
from extract_palette import histogram, extract_palette
from recolor import lab_transfer, rgb_transfer
from color_naming.color_name_compare import color_clustering, find_similar_color_topkname, find_similar_color_topkname_prob
from color_naming.color_naming import load_colornamelut
from utils import visualize_palette_rgb, draw_comparison
from eval import evaluation




def palette_color_harmony(args):

    images = os.listdir(args.data_dir)
    images = [name for name in images if name.endswith('.jpg') or name.endswith('.jpeg') or name.endswith('.png')]

    num_img = len(images)

    if num_img == 0:
        logging.info('There are no images at directory %s. Check the data path.' % (args.data_dir))
    else:
        logging.info('There are %d images to be processed.' % (num_img))
    images.sort()


    exp_label = "palette_lab{}_cluster_{}{}_c{}_mapping_{}{}_recolor_{}{}".format(
                args.palette_mode, args.cluster_space, args.cluster_mode, args.cluster_num, 
                args.mapping_space, args.mapping_mode, args.recolor_space, args.recolor_mode)

    save_dir = os.path.join(args.save_dir, exp_label)


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, 'palette'))
        os.makedirs(os.path.join(save_dir, 'image'))
        os.makedirs(os.path.join(save_dir, 'harmony'))


    with open(f'{save_dir}/{exp_label}.csv', 'w') as f:
        f.write('image,time,niqe,brisque,best_temp,best_alpha,Fscore,score_sat,score_perct,width,height\n')

    #################### Step 1: image agnostic: prototype palettes generation via color naming ####################
    W2C = load_colornamelut(name_type=args.name_method)
    T1 = time.time()
    prototype_rgb, prototype_lab, prototype_hsv, num_proto = color_clustering(args.cluster_num, 
                                                                            args.cluster_prob,
                                                                            W2C, 
                                                                            name_type=args.name_method, 
                                                                            extra_num=args.extra_num, 
                                                                            cluster_type=args.cluster_type,
                                                                            colorspace=args.cluster_space, 
                                                                            mode=args.cluster_mode
                                                                            )
    

    T2 = time.time()
    time_single = T2-T1
    print('Time for prototype palettes generation: ', time_single)

    for label, proto in enumerate(prototype_rgb):
        palette = visualize_palette_rgb(proto, patch_size=20)
        palette = np.array(palette).astype(np.uint8)
        out_img_path = os.path.join(save_dir, 'palette_cluster_label{}.jpg'.format(label))
        cv2.imwrite(out_img_path, cv2.cvtColor(palette, cv2.COLOR_RGB2BGR))

    num, niqe_all, brisque_all, Fscore_all,score_sat_all,score_perct_all, time_all= 0, 0, 0, 0, 0, 0, 0


    for img_id, img_name in enumerate(images):

        img_dir = os.path.join(args.data_dir, img_name)
        img = cv2.imread(img_dir)
        print('processing image {}: {}...'.format(img_id, img_dir))

        ## resize adobe5k images to x*1024 or 1024*x
        anchor = 512
        width = img.shape[1]
        height = img.shape[0]			
        if width >= height:
            dim = (np.floor(width/height*anchor).astype(int), anchor)
        else:
            dim = (anchor, np.floor(height/width*anchor).astype(int))
        img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_lab = rgb2lab(img_rgb)    # lab transfer by function
        w,h,c = img_rgb.shape

        # without semantic label
        label_binary = np.ones_like(img_rgb[:,:,0])

        T1 = time.time()

        #################### Step 2: image-wise: palette extraction  ####################

        ## calcultate the histogram
        hist_samples, hist_counts = histogram(img_lab, args.bin_size, mode=args.palette_mode, mask=label_binary)
        
        ## extract color palette
        # c_center, c_density, c_img_label, hist_labels
        c_center = extract_palette(img_lab, hist_samples, hist_counts, 
                                    mode=args.palette_mode,
                                    lightness=args.lightness,
                                    threshold=args.palette_distortion_thres, 
                                    max_cluster=args.palette_num,
                                    mask=label_binary)
        

        #################### Step 3: image-wise: palette matching  ####################
        
        ## color naming harmonization
        if args.color_dist == 'l1' or args.color_dist == 'l2' or args.color_dist == 'angle':
            c_target = find_similar_color_topkname(c_center, prototype_rgb, prototype_lab, prototype_hsv,
                                                   W2C, name_type=args.name_method, 
                                                   dist_type=args.color_dist,
                                                   top_k=args.topk_names,
                                                   prob_thres=args.prob_thres,
                                                   colorspace=args.mapping_space,
                                                   lightness=args.lightness, 
                                                   mode=args.mapping_mode)

        elif args.color_dist == 'prob':
            c_target = find_similar_color_topkname_prob(c_center, prototype_lab,
                                                        W2C, name_type=args.name_method, 
                                                        top_k=args.topk_names,
                                                        lightness=args.lightness, 
                                                        mode=args.mapping_mode)
        
        elif args.color_dist == 'sat':
            # directly increase color saturation
            c_center_rgb = lab2rgb(np.expand_dims(c_center, axis=0))
            c_center_hsv = rgb2hsv(c_center_rgb)
            c_center_hsv[:,:,1] = c_center_hsv[:,:,1]*2
            c_target_hsv = np.minimum(c_center_hsv, 1)
            c_target_rgb = hsv2rgb(c_target_hsv)
            c_target = np.squeeze(rgb2lab(c_target_rgb), axis=0)



        #################### Step 4: image-wise: image recoloring  ####################
        if args.recolor_space == 'lab':
            img_rgb_naming, _ = lab_transfer(img_lab, c_center, c_target, mask=label_binary, mode=args.recolor_mode)
        else:
            img_rgb_naming = rgb_transfer(img_rgb, c_center, c_target, mask=label_binary)
        
        T2 = time.time()
        time_single = T2-T1
        print('Time for image processing: ', time_single)

        #################### evaluation and visualization  ####################

        
        img_rgb_naming = np.array(img_rgb_naming*255).astype(np.uint8)
        out_img_path = os.path.join(save_dir, 'image', img_name.split('/')[-1][:-4]+'.png')
        cv2.imwrite(out_img_path, cv2.cvtColor(img_rgb_naming, cv2.COLOR_RGB2BGR))


        if args.eval:
            niqe, brisque, temp, alpha, Fscore, score_sat, score_perct, canvas, overlay = evaluation(cv2.cvtColor(img_rgb_naming, cv2.COLOR_RGB2BGR))

            with open(f'{save_dir}/{exp_label}.csv', 'a') as f:
                f.write(f'{img_name}, {time_single}, {niqe}, {brisque}, {temp}, {alpha}, {Fscore}, {score_sat}, {score_perct},{w},{h}\n')

            niqe_all = niqe_all + niqe
            brisque_all = brisque_all + brisque
            Fscore_all = Fscore_all + Fscore
            score_sat_all = score_sat_all + score_sat
            score_perct_all = score_perct_all + score_perct
            time_all = time_all + time_single
            num = num + 1


        out_img_path = os.path.join(save_dir, 'harmony', img_name.split('/')[-1][:-4]+'.jpg')
        cv2.addWeighted(overlay, 0.5, canvas, 1 - 0.5, 0, canvas)
        cv2.imwrite(out_img_path, canvas)

        palette_path = os.path.join(save_dir, 'palette', img_name.split('/')[-1][:-4]+'.jpg')

        draw_comparison(c_center, c_target, img_rgb, img_rgb_naming, palette_path)


    niqe_all = niqe_all / num
    brisque_all = brisque_all / num
    Fscore_all = Fscore_all / num
    score_sat_all = score_sat_all / num
    score_perct_all = score_perct_all / num
    time_all = time_all / num



    with open(f'{save_dir}/{exp_label}.csv', 'a') as f:
        f.write(f'average, {time_all}, {niqe_all}, {brisque_all}, None, None, {Fscore_all}, {score_sat_all}, {score_perct_all},{w},{h}\n')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument('--data_dir', type=str, default='./images/',
                        help='path to the folder containing demo images') # srgb image
    parser.add_argument('--save_dir', type=str, default='./results/',
                        help='path to save results')
    parser.add_argument('--name_method', type=str, default='joost',
                        help='color name method (choose from [joost, ca, yu])')
    parser.add_argument('--extra_num', type=int, default=5,
                        help='the number of extra colors for green, blue, purple, red, pink')
    parser.add_argument('--prob_thres', type=float, default=0.15,
                        help='for top k names, with threshold of probability higher than this')
    parser.add_argument('--topk_names', type=int, default=3,
                        help='number top k color names')
    parser.add_argument('--color_dist', type=str, default='l2',
                        help='type of color difference measures')
    parser.add_argument('--cluster_num', type=int, default=10,
                        help='number of cluster centers for generating prototype palette')
    parser.add_argument('--cluster_space', type=str, default='rgb',
                        help='color space of doing color clustering')
    parser.add_argument('--cluster_mode', type=int, default=3,
                        help='cluster channels, 3 for lab, 2 for ab only')
    parser.add_argument('--cluster_prob', type=float, default=1.,
                        help='cluster, choose the probablity level higher than which')
    parser.add_argument('--cluster_type', type=str, default='saturation',
                        help='strategy of selecting prototype colors, choose from: [saturation, prob, probysat]')
    parser.add_argument('--mapping_space', type=str, default='rgb',
                        help='color space of matching palette')
    parser.add_argument('--mapping_mode', type=int, default=3,
                        help='calculating the difference on which channels, 3 for lab, 2 for ab only')
    parser.add_argument('--palette_mode', type=int, default=2,
                        help='clustering mode (choose from:2,3)')
    parser.add_argument('--bin_size', type=int, default=16,
                        help='bin size of histogram')
    parser.add_argument('--palette_num', type=int, default=7,
                        help='number of indiviadual cluster center')
    parser.add_argument('--palette_distortion_thres', type=float, default=0.93,
                        help='number of indiviadual cluster center')
    parser.add_argument('--recolor_space', type=str, default='lab',
                        help='color space of doing color recoloring')
    parser.add_argument('--recolor_mode', type=int, default=3,
                        help='channel of doing color clustering')
    parser.add_argument('--lightness', type=float, default=70.,
                        help='lightness value of Lab colors')
    parser.add_argument('--eval', type=bool, default=True,
                        help='perform evaluation or not')
    args = parser.parse_args()

    palette_color_harmony(args)


