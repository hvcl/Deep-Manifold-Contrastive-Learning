# Histopathology Image Classification using Deep Manifold Contrastive Learning (MICCAI 2023)

[[Paper](https://arxiv.org/abs/2306.14459)]

<p align="center">
  <img src="Fig.2_1.jpg"  >
</p>

# Framework 
Tensorflow 2

# Dataset 
The hepatocellular carcinomas (HCCs) dataset can be downloaded from [Pathology AI Platform](http://www.wisepaip.org/paip).

# Preprocessing
1. Download the raw WSI data.
2. Prepare the patches.
3. Store all the patches directory in a .csv file (refer CSV File example).


# Deep Manifold Embedding Learning
Manifold encoder training. 
```python
python manifold.py --tr_patch_list [path] --lr 1e-2 --weight_decay 1e-4 
```


# Bag formation
1. Generate bag list in bag_list.csv (refer CSV File example).
   
```python
python bag_list_generation.py --all_patch_list [.csv] --num_bag 50 --num_patchPerbag 100 
```

2. Feature extration based on bag_list.csv.
   Refer the [split_file.csv](Deep-Manifold-Contrastive-Learning/csv_example/) File in CSV File example.
```python
python feature_extraction.py --bag_list_dir [path] --saved_encoder_dir [path] --split_file [.csv]
```
   
# MIL training 
```python
python mil.py --bag_dir [path] --num_class 2 --lr 1e-2 --weight_decay 1e-4 
```


# Citation
Please cite us if you use our work. 
