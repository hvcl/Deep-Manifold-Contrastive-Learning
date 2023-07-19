import cv2
import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import sys
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('WARNING')
from tensorflow import keras
from tensorflow.keras.models  import Sequential, Model
from tensorflow.keras.layers import Input,Dense, Flatten, Dropout, Conv2D, MaxPooling2D,BatchNormalization, Reshape, Layer
from tensorflow.keras import regularizers
from PIL import Image
from tensorflow.keras.constraints import max_norm
import numpy as np
from skimage.io import imread
from skimage.color import rgba2rgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold as KFold
import sklearn.metrics
import ast
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from sklearn.utils import class_weight
from tensorflow.keras.regularizers import l1, l2,l1_l2
from sklearn.metrics import confusion_matrix
import argparse



def get_output(path,label_file):
    file_name_slide = (os.path.split(path)[1]).split('.npy')[0][:15]
    with open(label_file, 'r',encoding='utf-8') as inF:
        for line in inF:
            if file_name_slide in line:
                label = line[-2]      
                return (label)


def load_data (files, label_file):
    batch_input = []
    batch_output = []
    file_fn = [] 
    for input_path in files:
        output = get_output(input_path,label_file)
        batch_input += [input_path]
        batch_output += [output]
    batch_x = np.array(batch_input)
    batch_y = np.array(batch_output)                                           
    return batch_x, batch_y

class DG_train(keras.utils.Sequence):
    def __init__(self, list_IDs,label_fn, batch_size=1, shuffle=True):
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.label_fn = label_fn
        self.on_epoch_end()
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    def __data_generation(self, list_IDs_temp):
        X = []
        y = []
        for i, ID in enumerate(list_IDs_temp):
            feat = np.squeeze(np.load(str(ID), allow_pickle = True))/255
            label = get_output(ID,self.label_fn)
            X.append(feat)
            y.append(label)
        return np.array(X),np.array(y)
    def __getitem__(self, index):
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
            list_IDs_temp = [self.list_IDs[k] for k in indexes]
            X, y = self.__data_generation(list_IDs_temp)
            X = np.squeeze(X)
            y1 = keras.utils.to_categorical (y, 2)
            return X,y1 



def create_model2(input_shape, lr, decay, num_class=2):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_class,activation='softmax'))
    model.compile(loss = "binary_crossentropy",
                optimizer = tf.keras.optimizers.SGD(learning_rate=lr,decay=decay),
                metrics=["accuracy"])
    return(model)       

             
def deploy(model, test_data,label_fn, bsize=16):
    test_img_path, test_label = load_data (test_data, label_fn)
    index2 = np.arange(len(test_img_path))
    bsize = bsize
    a = [index2[i:i + bsize] for i in range(0, len(index2), bsize)]
    y = []
    fn_test = []
    y_predicted2 =[]
    for batch_ind in a:
        X = []
        for ii in batch_ind:
            path_fn = test_img_path[ii]
            fn_test.append(os.path.split(path_fn)[1][:15] )
            img= np.load(path_fn)/255
            X.append(img)
            y.append(test_label[ii])
        prediction = model.predict(np.squeeze(X))
        y_predicted2.append(prediction)
    return np.concatenate(y_predicted2), test_label, fn_test


def acc(y_pred,fn_test, test_labels):   
    y_pred2 = np.argmax(y_pred, axis=1)
    sam_la = np.c_[fn_test, test_labels]
    ff = pd.DataFrame(sam_la)
    ff.columns = ['samples', 'label']
    ff = ff.sort_values(by=['samples'])
    ff["samples"] = (ff["samples"].str[:15])
    ff = ff.drop_duplicates()
    y_testt = pd.to_numeric(ff.label)
    t = np.c_[fn_test, y_pred2]
    df = pd.DataFrame(t)
    df.columns = ['samples', 'predicted_labels']
    b = df.groupby(df.samples.str[:15])['predicted_labels'].apply(list)
    b = b.astype(str).str.replace(r'\[|\]|', '', regex=True)
    c = b.str.split(', ', expand=True)
    c= c.reset_index()
    #df =c.astype(int)
    c['majority'] = c.mode(axis=1)[0]
    c2 = c[['samples', 'majority']]
    c3 = c2.majority.astype(str).str.replace(r'\'|\'|', '', regex=True)
    f_pred2 = pd.to_numeric(c3.astype(str).str.replace(r'\'|\'|', '', regex=True))
    final = ff.merge(c2)
    print ('----Testing Result----')
    print (sklearn.metrics.classification_report(y_testt, f_pred2, digits = 4))
    K.clear_session()



def main(args):
    day_time = datetime.now().strftime("%Y%m%d%H%M")
    tr_path = f'{args.feat_dir}_conc_tr/'
    img_dir = glob.glob(os.path.join(tr_path, '*'))
    te_path = f'{args.feat_dir}_conc_te/'
    test_img_dir = glob.glob(os.path.join(te_path, '*')) 

    print ('Number of training bag: ', len(img_dir))
    print ('Number of testing bag: ', len(test_img_dir))

    input_shape = np.squeeze(np.load(img_dir[0])).shape
    fn = os.path.split(args.feat_dir)[1]

    ckpt_log_dir= f"/home/Paris/jingwei/ckpt/mlp/{fn}_{day_time}.hdf5"
    ckpt = keras.callbacks.ModelCheckpoint(ckpt_log_dir, monitor='val_acc', verbose=1,save_best_only=False, save_weights_only=False, mode='max')  
    params = {'batch_size': args.batch_size,'shuffle': True}

    ## Training
    print ('************ Training *************')
    model = create_model2(input_shape, args.lr, args.dc, args.num_class)

    indices = np.arange(np.array(img_dir).shape[0])
    np.random.shuffle(indices)
    shuffled_img_dir = np.array(img_dir)[indices]
    validation_split = 0.2
    num_validation_samples = int(validation_split * np.array(img_dir).shape[0])

    X_train = shuffled_img_dir[:-num_validation_samples]
    X_val = shuffled_img_dir[-num_validation_samples:]

    train_gen = DG_train(X_train, args.label_file, **params)
    val_gen = DG_train(X_val, args.label_file, **params)

    history = model.fit(train_gen, epochs =args.num_ep, validation_data = val_gen,shuffle = True, verbose = 1,callbacks=[ckpt])

    ##Testing
    print ('************ Testing *************')
    pred_label, test_labels,test_fn  = deploy(model,test_img_dir, args.label_file)
    acc(pred_label, test_fn, test_labels)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your program description")
    parser.add_argument("--feat_dir", type=str, help="Directory of the CSV file which stores directory of train set patches.")
    parser.add_argument("--label_file", type=str, help="Directory of the CSV file which stores ground truth label.")
    parser.add_argument("--save_dir", type=str, help="Folder directory to store all information.")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of batch.")
    parser.add_argument("--save_model_dir", type=str, help="Folder directory to store training model.")
    parser.add_argument("--num_class", type=int, default=2, help="Number of class.")
    parser.add_argument("--lr", type=int, default=1e-2, help="Learning rate")
    parser.add_argument("--dc", type=int, default=1e-4, help="decay rate")
    parser.add_argument("--num_ep", type=int, default=50, help="Number of epoch")

    args = parser.parse_args()

    main(args)

