import cv2
import os
import torch
import numpy as np
import argparse
torch.manual_seed(42)

from extract_palette import histogram
import harmonization.color_harmonization as color_harmonization
from harmonization import util

from metrics.niqe import calculate_niqe
from metrics.brisque import BRISQUE
# from metrics.UNIQUE.BaseCNN import BaseCNN
# from metrics.hyperIQA import models
# from collections import OrderedDict
# from torchvision import transforms


def evaluation(img):

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ## niqe
    niqe = calculate_niqe(img_rgb, 0, input_order='HWC', convert_to='y')

    ## brisque
    calculate_brisque = BRISQUE()
    brisque = calculate_brisque.get_score(img_rgb)

    # ## unique
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # unique_model = BaseCNN().to(device)
    # state_dict = torch.load('./metrics/UNIQUE/model.pt', map_location=device)
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]
    #     new_state_dict[name] = v 
    # unique_model.load_state_dict(new_state_dict)
    # unique_model.eval()

    # test_transform = transforms.Compose([transforms.ToTensor(),
    #                                     transforms.Resize(768),
    #                                     transforms.Normalize(mean=(0.485, 0.456, 0.406),
    #                                                         std=(0.229, 0.224, 0.225))
    #                                     ])
    # img_tensor = test_transform(img_rgb)
    # img_tensor = img_tensor.unsqueeze(0).cuda()
    
    # with torch.no_grad():
    #     unique, _ = unique_model(img_tensor)
    #     unique = np.asarray(unique[0].cpu())
    #     # print(unique)

    # ##hyperIQA
    # model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7)
    # model_hyper.to(device)
    # model_hyper.train(False)

    # state_dict = torch.load('./metrics/hyperIQA/pretrained/koniq_pretrained.pkl', map_location=device)
    # model_hyper.load_state_dict(state_dict)

    # test_transforms = transforms.Compose([transforms.ToTensor(),
    #                                       transforms.Resize(384),
    #                                       transforms.RandomCrop((224, 224)),
    #                                       transforms.Normalize(mean=(0.485, 0.456, 0.406),
    #                                                            std=(0.229, 0.224, 0.225))])


    # # random crop 10 patches and calculate mean quality score
    # pred_scores = []
    # for i in range(10):
    #     img_tensor = test_transforms(img_rgb)
    #     # img_tensor = transforms.functional.crop(img_tensor, i, j, h, w)
    #     img_tensor = img_tensor.unsqueeze(0).cuda()
    #     paras = model_hyper(img_tensor)  # 'paras' contains the network weights conveyed to target network
    #     # Building target network
    #     model_target = models.TargetNet(paras)
    #     # for param in model_target.parameters():
    #     #     param.requires_grad = False

    #     # Quality prediction
    #     with torch.no_grad():
    #         hyperIQA_patch = model_target(paras['target_in_vec'])
    #         hyperIQA_patch = np.asarray(hyperIQA_patch.cpu())
    #         # print(hyperIQA)
    #     pred_scores.append(float(hyperIQA_patch.item()))
    # hyperIQA = np.mean(pred_scores)


    ## harmonization metrics

    # quicker way for computing color harmonization metrics through histogram 
    hist_value, hist_counts = histogram(img_hsv, 32, mode=3)
    hist_value = np.expand_dims(hist_value, axis=0)
    best_harmomic_scheme, f_score = color_harmonization.B_hist(hist_value, hist_counts)
    score_sat_org, score_org = best_harmomic_scheme.harmony_score_hist_inside(hist_value, hist_counts)
    best_temp = best_harmomic_scheme.m
    best_alpha = best_harmomic_scheme.alpha

    # classic brute-force search for template and angle
    # best_harmomic_scheme, f_score = color_harmonization.B(img_hsv)
    # score_sat_org, score_org = best_harmomic_scheme.harmony_score_inside(img_hsv)
    # best_temp = best_harmomic_scheme.m
    # best_alpha = best_harmomic_scheme.alpha

    histo = util.count_hue_histogram(img_hsv)
    canvas = util.draw_polar_histogram(histo)
    overlay = util.draw_harmonic_scheme(best_harmomic_scheme, canvas)
    
    
    return niqe, brisque, best_temp, best_alpha, f_score, score_sat_org, score_org, canvas, overlay

