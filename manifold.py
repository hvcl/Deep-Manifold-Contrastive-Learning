from sklearn.neighbors import NearestNeighbors
import numpy as np
import glob, os, sys, cv2, random, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, Conv2D, Dropout, BatchNormalization
import pandas as pd
from random import seed,randint
from collections import Counter
from tqdm import tqdm
from tensorflow.keras import backend as K
from scipy.spatial.distance import directed_hausdorff
from tensorflow.keras.callbacks import TensorBoard
from sklearn.manifold import TSNE,Isomap
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from heapq import nsmallest
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from manifold_function import base_model, extract_feat,clustering,load_data_main, loading_global,compute_prototype, DG_rand




def main(args):
    
    ## Loss Function
    def feat_sc(labels):
        label = labels[args.num_class:]
        feat_pair = np.load(f'{args.save_dir}feature_{save_fn}/cluster{str(int(label.numpy()[0]))}_prototype.npy',allow_pickle=True )
        return feat_pair

    def intra_new(labels, features):
        intra_loss =0
        for i in range(len(features)):
            feat_norm = tf.expand_dims(tf.squeeze(features[i]), axis =0)
            feat_pair_pos = tf.expand_dims(feat_sc(labels[i]), axis=0)
            loss1 = tf.matmul(tf.transpose(feat_pair_pos - feat_norm),(feat_pair_pos - feat_norm))
            intra_loss += loss1
        return tf.reduce_mean(intra_loss)

    def min_haus_tf(feat, labels):
        clus_label = (np.array_split(unique_clus, len(uni_label)))
        gt_label = labels[:args.num_class]
        gt_label = tf.argmax(gt_label).numpy()
        neg_ind = uni_label[uni_label != gt_label]
        neg_clus_ind = []
        for ind in neg_ind:
            neg_clus_ind.append(clus_label[int(ind)]) 
        neg_clus_ind = tf.concat (neg_clus_ind, axis=0)
        neg_proto = []
        for ii in range(len(neg_clus_ind)):
            neg = neg_clus_ind[ii].numpy()
            protype_path = f'{args.save_dir}feature_{save_fn}/cluster{str(int(neg))}_prototype.npy'
            neg_prototype = tf.expand_dims(np.load( protype_path, allow_pickle=True),0)
            feat = tf.expand_dims(tf.squeeze(feat), axis=0)
            neg_proto.append(neg_prototype)
        return neg_proto

    def inter_new(labels, features):
        inter_loss = 0
        inter_loss2 =[]
        low_k = args.num_cluster
        for i in range(len(features)):
            feat = tf.expand_dims(tf.squeeze(features[i]), axis =0) 
            neg_proto = min_haus_tf(feat, labels[i] )
            all_dist = []
            for j in range(len(neg_proto)):
                A = feat
                B = neg_proto[j]
                dist1 = tf.convert_to_tensor(directed_hausdorff( A, B)[0])
                dist2 = tf.convert_to_tensor(directed_hausdorff(  B, A)[0])
                dist = tf.math.maximum(dist1, dist2)
                all_dist.append(dist)
            if low_k == 1:
                max_dist = tf.reduce_min(all_dist, axis =0)
                inter_loss += (max_dist)
                inter_loss2.append (max_dist)
            else: 
                sorted_dist = tf.sort(all_dist)
                max_dist = sorted_dist[:len(neg_proto)]
                inter_loss += tf.reduce_mean(max_dist)
                clus_dist =[]
                for k in range (low_k):
                    clus_dist.append (max_dist[k])
                inter_loss2.append (tf.reduce_mean(clus_dist))
                #print ('loss', inter_loss2)
            return tf.reduce_mean(inter_loss2)

    def final_loss(labels, features):
        intra = intra_new (labels, features)
        inter = inter_new(labels, features)
        return intra + (args.margin - tf.cast(inter, tf.float32))


    def softmax_loss(labels, pred):
        bilabel = labels[:,0:args.num_class]
        sloss = tf.keras.losses.categorical_crossentropy(bilabel, pred)
        return sloss

    def ignore_accuracy_of_class(class_to_ignore=0):
        def ignore_acc(y_true, y_pred):
            y_true = y_true[:,0:args.num_class]
            y_true_class = K.argmax(y_true, axis=-1)
            y_pred_class = K.argmax(y_pred, axis=-1)
            ignore_mask = K.cast(K.not_equal(y_pred_class, class_to_ignore), 'int32')
            matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
            accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
            return accuracy
        return ignore_acc


    ## Loading Data
    img_list_fn = args.tr_patch_list
    img_list_fn_val = args.val_patch_list
    label_file = args.label_file
    img_list, img_list_val, label_list_tr,label_list_val, img_list_fn, uni_label = loading_global(img_list_fn, img_list_fn_val, label_file)

    ## Building model
    model = base_model(args.img_shape,  manifold_loss=args.manifold_loss, num_class=args.num_class)

    ## Feature extractor
    emb_layer = 'maxpool'
    fe = keras.Model(inputs=model.inputs,outputs=model.get_layer(name=emb_layer).output,)
    fe.run_eagerly = True

    update_ep = args.clustering_update_epoch

    ## Learning Schedule and Optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.lr,decay_steps=10000,decay_rate=args.dc,staircase=True) 
    opt = tf.keras.optimizers.Adam(lr_schedule)

    if args.manifold_loss == True:
        save_fn = f'tr_{str(args.num_NN)}NN_{str(args.num_cluster)}clus__m{args.margin}_lr{args.lr}_dc{args.dc}'
        save_fn_val = f'val_{str(args.num_NN)}NN_{str(args.num_cluster)}clus_m{args.margin}_lr{args.lr}_dc{args.dc}'

        if args.clustering_update == True:
            save_fn = f'{save_fn}_upEp{update_ep}_updateClus'
            save_fn_val = f'{save_fn_val}_upEp{update_ep}_updateClus'

        clus_dir = clustering (img_list, label_list_tr, fe, args.num_NN,args.num_cluster,args.save_dir,save_fn, uni_label)
        clus_dir_val = clustering (img_list_val, label_list_val, fe, args.num_NN,args.num_cluster,args.save_dir, save_fn_val, uni_label)
        
        ## Making x:image list , y:ground-truth label, z:cluster label
        batch_x, batch_y, batch_z = load_data_main (label_list_tr, cluster_dir= clus_dir)
        batch_x_val, batch_y_val, batch_z_val =load_data_main (label_list_val, cluster_dir=clus_dir_val)

        ## Cluster info
        cluster_file = pd.concat([pd.DataFrame(batch_x),pd.DataFrame(batch_z)], axis = 1)
        cluster_file.columns = ['sample', 'label']
        unique_clus = np.unique(cluster_file.label)
        num_clus = len(unique_clus)

        cluster_file_val =  pd.concat([pd.DataFrame(batch_x_val),pd.DataFrame(batch_z_val)], axis = 1)
        cluster_file_val.columns = ['sample', 'label']
        cluster_file_val['label'] = cluster_file_val['label'] + (args.num_cluster*len(uni_label))
        unique_clus_val = np.unique(cluster_file_val.label)
        num_clus_val = len(unique_clus_val)

        ## Feature extraction and TSNE plotting for initial stage
        extract_feat(fe, cluster_file_val, args.save_dir,save_fn, unique_clus_val, num_clus, args.img_shape, args.num_cluster, uni_label)
        extract_feat(fe, cluster_file, args.save_dir,save_fn, unique_clus ,num_clus,args.img_shape, args.num_cluster, uni_label, plot_tsne = True)

        ## Compute prototype for each class
        compute_prototype(args.save_dir, save_fn,args.num_cluster)


        model.compile(optimizer=opt,run_eagerly=True,loss = {'maxpool':final_loss, 'softmax':softmax_loss},metrics={'softmax':'accuracy' })

        gen =DG_rand(batch_x, batch_y,batch_z, batch_size = args.batch_size, img_shape=args.img_shape, aug=False)
        gen_val =DG_rand(batch_x_val, batch_y_val, batch_z_val, batch_size= args.batch_size, img_shape= args.img_shape, aug=False)

    else:
        save_fn = f'tr_{in_lr}_{dr}'
        save_fn_val = f'val_{in_lr}_{dr}'

        batch_x, batch_y = load_data_main (label_list_tr, cluster_dir= None)
        batch_x_val, batch_y_val =load_data_main (label_list_val, cluster_dir= None)
        
        model.compile(optimizer=opt,run_eagerly=True,loss = {'softmax':softmax_loss},metrics={'softmax': ignore_accuracy_of_class()})

        gen =DG_rand(batch_x, batch_y, batch_size = args.batch_size, img_shape=args.img_shape, aug=False)
        gen_val =DG_rand(batch_x_val, batch_y_val, batch_size=args.batch_size, img_shape= args.img_shape, aug=False)
        

    ckpt_log_dir= f"{args.save_model_dir}/{save_fn}.hdf5"
    ckpt2 = keras.callbacks.ModelCheckpoint(ckpt_log_dir, monitor='val_loss', verbose=0,
                            save_best_only=True, save_weights_only=True, mode='auto',epoch_per_save=1)  


    for ep in range (args.num_ep):
        print ('Epoch ', ep+1)
        history = model.fit(gen, epochs=1, verbose =2,validation_data=gen_val,workers=1, max_queue_size = 1, callbacks=[ ckpt2])
        fe = keras.Model(inputs=model.inputs,outputs=model.get_layer(name=emb_layer).output,)
        extract_feat(fe, cluster_file_val, args.save_dir,save_fn, unique_clus_val, num_clus, args.img_shape, args.num_cluster, uni_label)
        extract_feat(fe, cluster_file, args.save_dir,save_fn, unique_clus ,num_clus,args.img_shape, args.num_cluster, uni_label, plot_tsne = True, epoch_num=ep+1)
        compute_prototype(args.save_dir, save_fn,args.num_cluster)
        if (ep+1)% update_ep == 0:
            if args.clustering_update == True:
                clus_dir = clustering (img_list, label_list_tr, fe, args.num_NN,args.num_cluster,args.save_dir,save_fn, uni_label)
                clus_dir_val = clustering (img_list_val, label_list_val, fe, args.num_NN,args.num_cluster,args.save_dir, save_fn_val, uni_label)
                batch_x, batch_y, batch_z = load_data_main (label_list_tr, cluster_dir= clus_dir)
                batch_x_val, batch_y_val, batch_z_val =load_data_main (label_list_val, cluster_dir=clus_dir_val)
                gen =DG_rand(batch_x, batch_y,batch_z, batch_size = args.batch_size, img_shape= args.img_shape, aug=False)
                gen_val =DG_rand(batch_x_val, batch_y_val, batch_z_val, batch_size=args.batch_size, img_shape= args.img_shape, aug=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your program description")

    parser.add_argument("--manifold_loss", action='store_true', default=True,help="If True, then use manifold loss. If False, then use CE loss only.")
    parser.add_argument("--tr_patch_list", type=str, help="Directory of the CSV file which stores directory of train set patches.")
    parser.add_argument("--val_patch_list", type=str, help="Directory of the CSV file which stores directory of validation set patches.")
    parser.add_argument("--label_file", type=str, help="Directory of the CSV file which stores ground truth label.")
    parser.add_argument("--save_dir", type=str, help="Folder directory to store all information.")
    parser.add_argument("--num_class", type=int, default=2, help="Number of class.")
    parser.add_argument("--num_NN", type=int, default=10, help="Number of nearest neighbor.")
    parser.add_argument("--num_cluster", type=int, default=5, help="Number of cluster for each class.")
    parser.add_argument("--clustering_update", action='store_true', default=True, help="If True, then update cluster.")
    parser.add_argument("--clustering_update_epoch", type=int, default=1, help="Update the cluster every k epochs.")
    parser.add_argument("--margin", type=int, default=1, help="Number of cluster for each class.")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of batch.")
    parser.add_argument("--save_model_dir", type=str, help="Folder directory to store training model.")
    parser.add_argument("--img_shape", default=(256,256,3), help="Patch size.")
    parser.add_argument("--lr", type=int, default=1e-2, help="Learning rate")
    parser.add_argument("--dc", type=int, default=1e-4, help="decay rate")
    parser.add_argument("--num_ep", type=int, default=50, help="Number of epoch")

    args = parser.parse_args()

    main(args)
