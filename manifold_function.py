from sklearn.neighbors import NearestNeighbors
import numpy as np
import glob, os, sys 
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from random import seed,randint
from collections import Counter
from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff
import random
from sklearn.manifold import TSNE,Isomap
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from heapq import nsmallest
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import KFold
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, Conv2D, Dropout, BatchNormalization

def get_output(path,label_file):
    file_name_slide = os.path.split(os.path.split(path)[0])[1]   
    with open(label_file, 'r',encoding='utf-8') as inF:
        for line in inF:
            if file_name_slide in line:
                label = line[-2].strip()   
                label = np.array(label)
                return (label)

def loading_global(img_list_fn, img_list_fn_val, label_file):
    img_list = np.concatenate(np.array(pd.read_csv(img_list_fn, header=None, index_col=None)))
    img_list_val = np.concatenate(np.array(pd.read_csv(img_list_fn_val, header=None, index_col=None)))
    label_list = pd.read_csv( label_file)
    label_list.columns = ['sample','label']
    uni_label = np.unique(label_list['label'].to_numpy())
    img_list_all = np.concatenate ([img_list,img_list_val], axis =0)
    gt_label = []
    for fn in img_list_all:
        label = get_output (fn, label_file)
        gt_label.append(label)
    over_sampler = RandomOverSampler(random_state=42)
    img_list, gt_label_tr = over_sampler.fit_resample(np.reshape(img_list, (-1,1)), gt_label[:len(img_list)])
    print(f"Training target statistics: {Counter(gt_label_tr)}")
    img_list2 = np.concatenate ([np.squeeze(img_list),img_list_val], axis =0)
    label_list_tr = pd.concat((pd.DataFrame(img_list),pd.DataFrame(gt_label_tr)), axis =1)
    label_list_val = pd.concat((pd.DataFrame(img_list_val),pd.DataFrame(gt_label[-len(img_list_val):])), axis =1)
    label_list_tr.columns = ['sample','label']
    label_list_val.columns = ['sample','label']
    img_list = np.concatenate(img_list)
    return img_list,img_list_val, label_list_tr, label_list_val, img_list_fn, uni_label



def base_model(img_shape, manifold_loss=True, num_class=2):
    base_model = tf.keras.applications.VGG16(input_shape= img_shape, weights = 'imagenet', include_top = False)
    feats1 = base_model.output
    feats1.trainable = False
    if manifold_loss == True:
        feats_max = MaxPooling2D((8,8),name='maxpool')(feats1)                                                                                                                                                                           
        feats = tf.keras.layers.Flatten()(feats_max)
        output = tf.keras.layers.Dense(num_class,activation='softmax', name = 'softmax')(feats)
        model = Model(inputs=base_model.input, outputs=[feats_max, output])
    elif manifold_loss ==False:
        feats = tf.keras.layers.Flatten()(feats1)
        feats = tf.keras.layers.Dense(512,activation='relu')(feats)
        feats = tf.keras.layers.Dense(128,activation='relu')(feats)
        output = tf.keras.layers.Dense(num_class,activation='softmax', name = 'softmax')(feats)
        model = Model(inputs=base_model.input, outputs=output)
    return model



def clustering (img_list, label_list, fe, num_NN, num_cluster, save_dir, save_filename, uni_label):
    img_nm = [] 
    cluster_label = []
    num_gt_label = len(uni_label)
    for ind in tqdm (range(num_gt_label),desc='Generating sub-clusters'):
        label = uni_label[ind]
        wsi_list = label_list.loc[label_list['label'] == str(label)]
        if len(wsi_list)==0:
            wsi_list = label_list.loc[label_list['label'] == label] 
        group = np.array(wsi_list['sample'])
        img_feat = []
        for img_fn in img_list:
                wsi_nm = img_fn
                if img_fn in group:
                    wsi_nm = img_fn
                    img =cv2.imread(str(img_fn))/255
                    feat =fe.predict(np.expand_dims(img, axis = 0), verbose=0) 
                    img_feat.append (np.squeeze(feat))
                    img_nm.append (wsi_nm)
        img_feat = np.array(img_feat)
        isomap = Isomap(n_neighbors=5, n_components=2, path_method = 'D',n_jobs = -1, p=2 )
        X_digits_isomap = isomap.fit(img_feat)
        dist_mat_A = isomap.dist_matrix_
        clustering = AgglomerativeClustering(n_clusters=num_cluster,affinity='precomputed',linkage='average').fit_predict(dist_mat_A)
        cluster_label.append(clustering)
    final_clus = []
    for c in range (len(cluster_label)):
        cluster = [int(x) + int(uni_label[c])*num_cluster for x in cluster_label[c]]
        final_clus.append(pd.DataFrame(cluster))
    final_clus = pd.concat(final_clus).to_numpy()
    img_nm = pd.DataFrame(img_nm).to_numpy()
    assert len(img_nm)==len(final_clus), f"number of images is diffrent with the number of label: {len(img_nm)}, {len(final_clus)}"
    comb = pd.DataFrame(np.concatenate([img_nm, final_clus], axis=1))
    comb.columns = ['sample', 'label']
    save_path = f'{save_dir}{save_filename}.csv'
    print ('saving ===>', save_path)
    comb.to_csv(save_path, header=None, index=None)
    return save_path


def load_data_main (label_list, cluster_dir=None):
    if cluster_dir != None:
        cluster_file = pd.read_csv(cluster_dir, header=None)
        cluster_file.columns = ['sample', 'label']
        df_m = pd.merge(label_list,cluster_file, on=['sample'])
        df_m.columns=['sample', 'gt_label', 'clus_label']
        batch_x = df_m['sample'].to_numpy()
        batch_y = df_m['gt_label'].to_numpy()
        batch_z = df_m['clus_label'].to_numpy()
        return batch_x, batch_y, batch_z
    elif cluster_dir == None:
        df_m = label_list
        batch_x = df_m['sample'].to_numpy()
        batch_y = df_m['label'].to_numpy()
        return batch_x, batch_y




def extract_feat(fe, cluster_file,save_dir,model_fn, unique_clus, num_clus, img_shape, num_cluster,uni_label, plot_tsne = False, epoch_num=None):
    print ('********** Extracting Feature in progress ***********')
    label = []
    feat_tsne = [] 
    for i in range(num_clus):
        c_label = unique_clus[i]
        folder = save_dir + 'feature_'+ model_fn+'/'
        os.makedirs(folder, exist_ok=True, mode = 0o777)
        save_loc = folder + 'feat_cluster' + str(c_label) + '.npy' 
        res_patch_list = np.array(cluster_file[cluster_file['label'] == int(c_label)]['sample'])
        _feat=[]
        bsize = 64
        index2 = np.arange(len(res_patch_list))
        a = [index2[i:i + bsize] for i in range(0, len(res_patch_list), bsize)]
        for batch_ind in a:
            class_img = []
            for img_ind in batch_ind:
                patch_fn = res_patch_list[img_ind]
                img = cv2.imread (patch_fn)/255
                label.append(c_label)
                if img.shape != img_shape:
                    img = cv2.resize(img, (img_shape[:2]))
                class_img.append(img)
            feat_pair = fe.predict(np.array(class_img).astype('float32'), verbose=2)
            if len(feat_pair) ==1:
                feat_pair = np.expand_dims(np.squeeze(feat_pair), axis=0)
            else:
                feat_pair = np.squeeze(feat_pair)
            _feat.append((feat_pair))
        feat = np.concatenate (_feat)
        np.save(save_loc, feat)
    if plot_tsne == True:
        print ('********** Plotting TSNE in progress ***********') 
        num_cluster_class = num_cluster*len(uni_label)*2
        feature = []
        label = []
        feature_val = []
        label_val = []
        for index in range (int(num_cluster_class)):
            feat_fn = folder+'feat_cluster' + str(index)+'.npy'
            feat = np.load(feat_fn,allow_pickle=True)
            if len(feat.shape)<2:
                feat= np.expand_dims(feat, 0)
            if index > num_cluster_class//2:
                feature_val.append(feat)
            else:
                feature.append(feat)
            clus = os.path.split(feat_fn)[1].split('.npy')[0].split('feat_cluster')[1]
            for kk in range (len(feat)):
                if index > num_cluster_class//2:
                    label_val.append(int(clus))
                else:
                    label.append(int(clus))
        feat_tsne = np.concatenate(feature)
        label = np.array(label)
        feat_tsne_val = np.concatenate(feature_val)
        label_val = np.array(label_val)
        if len(feat_tsne.shape)>2:
            feat_tsne =  np.reshape(feat_tsne,(feat_tsne.shape[0],feat_tsne.shape[1]*feat_tsne.shape[2]*feat_tsne.shape[3]))
        if len(feat_tsne_val.shape)>2:
            feat_tsne_val =  np.reshape(feat_tsne_val,(feat_tsne_val.shape[0],feat_tsne_val.shape[1]*feat_tsne_val.shape[2]*feat_tsne_val.shape[3]))
        n_components = 2
        tsne = TSNE(n_components)
        tsne_result = tsne.fit_transform(feat_tsne)
        tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': label})
        fig, ax = plt.subplots(1)
        sns.scatterplot(x='tsne_1', y='tsne_2', hue='label',  data=tsne_result_df , ax=ax,s=8,palette=sns.color_palette("hls", len(np.unique(label))))
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        save_loc = save_dir+model_fn 
        os.makedirs(save_loc,exist_ok=True)
        if epoch_num == None:
            plt.savefig(save_loc + '/epoch_0.png')
        else:
            plt.savefig(save_loc + '/epoch_' + str(epoch_num)+'.png')
        plt.close('all')
        tsne = TSNE(n_components)
        tsne_result_val = tsne.fit_transform(feat_tsne_val)
        tsne_result_df_val = pd.DataFrame({'tsne_1': tsne_result_val[:,0], 'tsne_2': tsne_result_val[:,1], 'label': label_val})
        fig, ax = plt.subplots(1)
        sns.scatterplot(x='tsne_1', y='tsne_2', hue='label',  data=tsne_result_df_val , ax=ax,s=8,palette=sns.color_palette("hls", len(np.unique(label_val))))
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        if epoch_num == None:
            plt.savefig(save_loc + '/epoch_0v.png')
        else:
            plt.savefig(save_loc + '/epoch_' + str(epoch_num)+'v.png')
        plt.close('all')





class DG_rand(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, cluster_labels = None, batch_size=3, img_shape = (256,256,3),shuffle=False, 
                    aug = False):
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.cluster_labels = cluster_labels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.img_shape = img_shape
        self.aug = aug
        self.num_gtClass = len(np.unique(labels))
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __len__(self):
        leng = int(np.floor(len(self.list_IDs) / self.batch_size))
        return leng
    def __data_augmentation(self, img):
        ''' function for apply some data augmentation '''
        img = tf.image.random_contrast(img, 0.1,0.3)
        img = tf.keras.preprocessing.image.random_zoom(img, [224,224])
        img = tf.image.random_flip_left_right(img)
        img = tf.keras.preprocessing.image.random_rotation(img,0.15)
        img = tf.image.per_image_standardization(img)
        img = tf.image.resize(img, [334, 334])
        img = tf.image.central_crop(img, 1)
        mean=[0.485, 0.456, 0.406]
        variance=[0.229, 0.224, 0.225]
        img =  tf.divide(tf.subtract(img, mean), variance)
        return img
    def __data_generation(self, list_IDs_temp, label_temp, clus_temp=None):
        X = []
        y = []
        z = np.empty((self.batch_size, 1), dtype=int)
        for i, ID in enumerate(list_IDs_temp):
            img = cv2.imread (ID)/255
            if img.shape != self.img_shape:
                img = cv2.resize(img, (self.img_shape[:2]))
            if self.aug == True:
                img = self.__data_augmentation(img)
            label = label_temp[i]
            zero_label = np.zeros (self.num_gtClass)
            zero_label[int(label)] = 1
            X.append(img)
            y.append(zero_label)
            if clus_temp !=None :
                z[i,] = clus_temp[i]
        if clus_temp !=None:
            return X,np.array(y), z
        else:
            return X, y
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        label_temp =  [self.labels[k] for k in indexes]
        clus_temp = [self.cluster_labels[k] for k in indexes]        
        X,y,z= self.__data_generation(list_IDs_temp, label_temp, clus_temp)
        Y = pd.concat((pd.DataFrame(y), pd.DataFrame(z)),axis=1).to_numpy(dtype = np.uint8)
        return (np.array(X, dtype='float32'), np.array(Y, dtype='float32'))


def compute_prototype(save_dir, model_fn, num_cluster):
    feat_fol = f'{save_dir}feature_{model_fn}'
    feat_list = sorted(glob.glob ( feat_fol +'/feat_*.npy'))
    total_clus = len(feat_list)
    for feat_path in feat_list:
        feat = np.load(feat_path,allow_pickle=True)
        cluster_name = os.path.split(feat_path)[1].split('.npy')[0].split('feat_')[1]
        cluster_num = int(os.path.split(feat_path)[1].split('.npy')[0].split('feat_')[1].split('cluster')[1])
        class_num = cluster_num//num_cluster+1
        if len(feat.shape)==1:
            feat = tf.expand_dims(feat, axis=0)
        prototype = np.median (feat, axis =0)
        #print (prototype.shape)
        np.save (feat_fol + '/' + cluster_name +'_prototype.npy',prototype )
