import os
import sys
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import time

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Interpolation for a series of images')
parser.add_argument('--in_folder', dest='in_folder', type=str, help='Name of folder with input images')
parser.add_argument('--add', dest= 'add', default=1, type=int) 
parser.add_argument('--rmaxcycles', default=8, type=int, help='limit max number of bisectional cycles')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
parser.add_argument('--out_folder', dest='out_folder', type=str, help='Name of folder to save output images') 
parser.add_argument('--out_format', dest='out_format', type=str, help='Format of output files')  
args = parser.parse_args()

folder_name = args.out_folder
if not os.path.exists(folder_name):   # If the folder exists, the images are overwritten
    os.mkdir(folder_name)

# Define number of interpolation cycles
if args.add == 1:
  args.exp = 1
elif args.add == 3:
  args.exp = 2
elif args.add == 7:
  args.exp = 3
else:
  print('Error: Number of additional frames not allowed.') 
  sys.exit()



# Select model
try: 
  from train_log.RIFE_HDv3 import Model
  model = Model()
  model.load_model(args.modelDir, -1)
  print("Loaded HD model.")   # v3.x HD
except:
  from model.RIFE import Model
  model = Model()
  model.load_model(args.modelDir, -1)
  print("Loaded m model")


# Load files from folder
input_files = sorted(os.listdir(args.in_folder))

input_files = [os.path.join(args.in_folder, fname) for fname in input_files if not fname.startswith(".")]
filename = 1


for i in range(len(input_files)-1):
    # Select two consecutive frames
    img0 = cv2.imread(input_files[i],cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH) 
    img1 = cv2.imread(input_files[i+1],cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH) 
    img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

    # Extract format information (use it only if format is not specified)
    if not args.out_format:
      args.out_format = input_files[i][-3:]


    # Extract shape information and pad
    n, c, h, w = img0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    img0 = F.pad(img0, padding)
    img1 = F.pad(img1, padding)


    # Image inference
    img_list = [img0, img1]
    for k in range(args.exp):
        tmp = []
        for j in range(len(img_list) - 1):
            mid = model.inference(img_list[j], img_list[j + 1])
            tmp.append(img_list[j])
            tmp.append(mid)
        tmp.append(img1)  
        img_list = tmp

    for f in range(len(img_list)-1): #-1 so it doesn't save the last one (first on the next sequence)
        if os.path.exists(folder_name + '/img{0:05d}.'.format(filename) + args.out_format):
          filename = filename + 1
        cv2.imwrite(folder_name + '/img{0:05d}.'.format(filename) + args.out_format, (img_list[f][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])

# Save the last frame of the folder
filename = filename + 1
cv2.imwrite(folder_name + '/img{0:05d}.'.format(filename) + args.out_format, (img_list[-1][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])

print('Completed.')
