import os
import glob
import pandas as pd
import numpy as np
import csv
from itertools import groupby
from operator import itemgetter
from collections import defaultdict
from tqdm import tqdm
import argparse


def main(args):

    tr_list = pd.read_csv (args.tr_patch_list, header=None)
    val_list = pd.read_csv (args.val_patch_list, header=None)
    te_list = pd.read_csv (args.te_patch_list, header=None)
    all_list= pd.concat ([tr_list, val_list, te_list]).to_numpy().flatten().tolist()

    maxOfBag = args.num_bag
    patchNum = args.num_patchPerbag
    t=0
    res = [list(i) for j, i in groupby(all_list, lambda a: os.path.split(os.path.split(a)[0])[1])]

    data, data2 , k1, k2= [], [], [], []
    for p in tqdm(range(maxOfBag), desc = 'Buiding bag ...'):
        for i in range (len(res)):
            group = res[i]
            k = os.path.split(os.path.split(group[0])[0])[1]
            b = len(group)
            batch_paths = np.random.choice(a = group, size = patchNum, replace=True)
            gr = [os.path.split(x)[1].split('\n')[0].split(',')[0]  for x in batch_paths]
            gr1 = pd.DataFrame (batch_paths)
            t = t+1
            ww = gr1.iloc[:, 0].tolist()
            data2.append (gr)
            data.append (gr1)
            new_k = str(k)+'_'+ str(p+1).zfill(2) 
            k2.append (new_k)
            k1.append (k)


    df1 = pd.DataFrame(data2)
    df2 = pd.DataFrame(k2)
    df_con = pd.concat([df2, df1], axis=1)
    print (df_con)
    export_csv = df_con.to_csv (args.save_dir +'bag_list.csv', index=None, header = False) 



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your program description")
    parser.add_argument("--tr_patch_list", type=str, help="Directory of the CSV file which stores directory of train set patches.")
    parser.add_argument("--val_patch_list", type=str, help="Directory of the CSV file which stores directory of validation set patches.")
    parser.add_argument("--te_patch_list", type=str, help="Directory of the CSV file which stores directory of testing set patches.")
    parser.add_argument("--save_dir", type=str, help="Folder directory to store all information.")
    parser.add_argument("--num_bag", type=int, default=50, help="Number of bag per WSI.")
    parser.add_argument("--num_patchPerbag", type=int, default=100, help="Number of patch per bag.")

    args = parser.parse_args()

    main(args)
