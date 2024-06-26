# Diff-DGMN
Diff-DGMN: A Diffusion-based Dual Graph Multi-attention Network for POI Recommendation

-  ***Dual-graph-driven Representation***: Direction-aware Sequence Graph Multi-scale Representation Module (SeqGraphRep) and Global-based Distance Graph Geographical Representation Module (DisGraphRep).
-  ***Novel Diffusion-based User Preference Sampling (DiffGenerator)***: leverage the Variance-Preserving Stochastic Differential Equation (VP-SDE) to sample user future preferences by reverse-time generation.
-  ***Pure (noise-free) Location Archetype Vector***: capable of depicting the diffusion path from a source distribution to the target distribution and allowing for the exploration of evolving user interests.
![model](https://github.com/JKZuo/Diff-DGMN/assets/73583962/e7a5de80-2a4b-4cc2-bd49-9f6ca1bcdaed)

## Requirements
The code has been tested running under Python 3.8.

The required packages are as follows: 
- Python == 3.8.13
- torch == 1.12.1
- torchsde == 0.2.6
- torch_geometric == 2.3.1
- pandas == 2.0.3
- numpy == 1.23.3

## Data
Due to the large datasets (the data file uploaded by GitHub cannot be larger than 25MB), you can download them through this Baidu Cloud link:

link: https://pan.baidu.com/s/1Tbjzi8qh7C0dHULoV5axjQ?pwd=diff 

password: diff

This folder (data/processed) contains 5 datasets, including

(1) **IST** (Istanbul in Turkey); 

(2) **JK** (Jakarta in Indonesia); 

(3) **SP** (Sao Paulo in Brazil); 

(4) **NYC** (New York City in USA); 

(5) **LA** (Los Angeles in USA).

We also provided the raw files at (data/raw).

All datasets are sourced from https://sites.google.com/site/yangdingqi/home/foursquare-dataset

where 5. Global-scale Check-in Dataset with User Social Networks. 

This dataset includes long-term (about 22 months from Apr. 2012 to Jan. 2014) global-scale check-in data collected from Foursquare.
The check-in dataset contains 22,809,624 check-ins by 114,324 users on 3,820,891 venues.

## Running
**Attention: Please modify the datasets in your path: DATA_PATH = '../DiffDGMN/data/processed' in the **[gol.py]** file

Then, you can use the small-scale LA dataset as an example to run it as：

```shell
nohup python main.py --dataset LA --gpu 0 --dp 0.4 > LA.log 2>&1 &
```



