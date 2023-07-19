import numpy as np
from skimage.io import imread
from skimage.color import rgba2rgb
import sys, os, argparse, cv2

import pandas as pd
import skimage.io
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, Dense
from manifold_function import base_model

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]='2'




def load_sample_images(bag_list_dir):
    samples = {}
    fd = open( bag_list_dir) 
    for line in fd:
        line = line.split(',')
        samples[line[0]] = [ fn.strip() for fn in line[1:] if fn.strip() != '' ]
    return samples

def load_ckpt(ckpt_dir):
    print ('**Model Loading**')
    model = base_model(args.img_shape,  manifold_loss=args.manifold_loss, num_class=args.num_class)
    model.load_weights(ckpt_dir)
    model.summary()
    layer = 'block5_pool'
    pool_size = 8
    px = model.get_layer(layer).output
    if int(pool_size) > 0:
        px = MaxPooling2D((pool_size,pool_size),name='maxpool',data_format = 'channels_last')(px)
    model2 = Model( inputs=model.input, outputs=px )
    model2.summary()
    return model2


def main(args):

    sample_images = load_sample_images(args.bag_list_dir)
    sample_list = []
    for sample,imagelist in sample_images.items():
        sample_list.append(sample)

    ## Loading the checkpoint
    model_fn = os.path.split(args.ckpt_dir)[1]
    model_name = model_fn.split('.hdf5')[0]
    model = load_ckpt(args.ckpt_dir)

    ## Genearte folder for storing feature
    folder_fn = f'feature_{model_name}' 
    npy_dir = args.save_dir+ folder_fn+'/'
    tefolder_fn = folder_fn + '_conc_te/'
    trfolder_fn = folder_fn + '_conc_tr/'
    os.makedirs(args.save_dir +  trfolder_fn, exist_ok=True,mode=0o0777  )
    os.makedirs(args.save_dir +  tefolder_fn, exist_ok=True,mode=0o0777 )

    fold = pd.read_csv (args.split_file,index_col=None)

    for sample,imagelist in sample_images.items():
        if sample in sample_list[:] :
            print('Extracting ==> ',sample)
            batch_input = []
            loc = args.save_dir + trfolder_fn +str(sample)+'.npy'
            loc1 = args.save_dir + tefolder_fn +str(sample)+'.npy'
            for img_fn in imagelist:
                folder_fn2 = img_fn[:15]
                p1 = args.src_dir  + folder_fn2 + '/' + img_fn
                feat_fn = npy_dir+img_fn[:img_fn.rfind('.')]+ '.npy' 
                img =cv2.imread(p1)
                x = (img)/255
                batch_input += [x]
            p = model.predict(np.array(batch_input))
            if len(p.shape) > 2:
                feat = [ p[:,r,c,:].squeeze() for r in range(p.shape[1]) for c in range(p.shape[2]) ]
            else:
                feat = [ p.squeeze() ]
            label = np.array(fold[fold['ID'] == folder_fn2]['label'])
            if label == 'train':
                np.save (loc,feat)
            else:
                np.save (loc1, feat)


######################################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your program description")
    parser.add_argument("--bag_list_dir", type=str, help="Directory of the CSV file which stores directory of train set patches.")
    parser.add_argument("--split_file", type=str, help="Directory of the CSV file which stores train and test set information.")
    parser.add_argument("--ckpt_dir", type=str, help="Checkpoint directory.")
    parser.add_argument("--save_dir", type=str, help="Directory to save the extracted features.")
    parser.add_argument("--src_dir", type=str, help="Source directory of patches.")
    parser.add_argument("--num_class", type=int, default=2, help="Number of class.")
    parser.add_argument("--manifold_loss", action='store_true', default=True,help="If True, then use manifold loss. If False, then use CE loss only.")
    parser.add_argument("--img_shape", default=(256,256,3), help="Patch size.")

    args = parser.parse_args()

    main(args)
