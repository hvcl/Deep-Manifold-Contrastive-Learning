## [MICCAI 2023] Histopathology Image Classification using Deep Manifold Contrastive Learning 
By [Jing Wei Tan](https://scholar.google.com/citations?user=_PMI46gAAAAJ&hl=en) and  [Won-Ki Jeong](https://scholar.google.com/citations?user=bnyKqkwAAAAJ&hl=en&oi=sra)

[[Paper](https://arxiv.org/abs/2306.14459)]


This is the official implementation code.


<p align="center">
  <img src="Fig.2_1.jpg"  >
</p>

### Abstract
Contrastive learning has gained popularity due to its robustness with good feature representation performance. However, cosine distance, the commonly used similarity metric in contrastive learning, is not well suited to represent the distance between two data points, especially on a nonlinear feature manifold. Inspired by manifold learning, we propose a novel extension of contrastive learning that leverages geodesic distance between features as a similarity metric for histopathology whole slide image classification. To reduce the computational overhead in manifold learning, we propose geodesic-distance-based feature clustering for efficient contrastive loss evaluation using prototypes without time-consuming pairwise feature similarity comparison. The efficacy of the proposed method is evaluated on two real-world histopathology image datasets. Results demonstrate that our method outperforms state-of-the-art cosine-distance-based contrastive learning methods.
## Framework 
Tensorflow 2

## Dataset 
The hepatocellular carcinomas (HCCs) dataset can be downloaded from [Pathology AI Platform](http://www.wisepaip.org/paip).

## Preprocessing
1. Download the raw WSI data.
2. Prepare the patches.
3. Store all the patches directory in a .csv file (refer [patch_list.csv](csv_example/patch_list_example.csv)).


## Deep Manifold Embedding Learning
Manifold encoder training. 
```python
python manifold.py --tr_patch_list [CSV path] --val_patch_list [CSV path] --label_file [CSV path] --save_dir [folder path] --num_class 2 --num_NN 5 --num_cluster 10 --save_model_dir [folder path]
```


## Bag formation
1. Generate bag list in [bag_list.csv](csv_example/bag_list_example.csv).
   
```python
python bag_list_generation.py --tr_patch_list [CSV path] --val_patch_list [CSV path] --te_patch_list [CSV path] --save_dir [folder path] --num_bag 50 --num_patchPerbag 100 
```

2. Feature extration based on [bag_list.csv](csv_example/bag_list_example.csv).
   Example of split_file.csv is in [split_file.csv](csv_example/split_file_example.csv).
```python
python feature_extraction.py --bag_list_dir  [CSV path]  --split_file [CSV path] --ckpt_dir [checkpoint path] --save_dir [folder path] --src_dir [folder_path]
```
   
## MIL training 
Train simple MIL for classification.
```python
 python mil.py --feat_dir  [folder path]  --label_file [CSV path] --num_class 2 --save_model_dir [CSV path]

```


## Citation
### Please cite us if you use our work. 
Tan, Jing Wei, and Won-Ki Jeong. "Histopathology image classification using deep manifold contrastive learning." In International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 683-692. Cham: Springer Nature Switzerland, 2023.
