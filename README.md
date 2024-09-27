# Diff-DGMN
Diff-DGMN: A Diffusion-based Dual Graph Multi-attention Network for POI Recommendation

-  ***Dual-graph-driven Representation***: Direction-aware Sequence Graph Multi-scale Representation Module (SeqGraphRep) and Global-based Distance Graph Geographical Representation Module (DisGraphRep).
-  ***Novel Diffusion-based User Preference Sampling (DiffGenerator)***: leverage the Variance-Preserving Stochastic Differential Equation (VP-SDE) to sample user future preferences by reverse-time generation.
-  ***Pure (noise-free) Location Archetype Vector***: capable of depicting the diffusion path from a source distribution to the target distribution and allowing for the exploration of evolving user interests.

The overall framework of our proposed Diff-DGMN model is illustrated in **Fig_1**.
<p align="center">
<img align="middle" src="https://github.com/JKZuo/Diff-DGMN/blob/main/Figures/Fig_1.png"/>
</p>
<p align = "center">
<b>Figure 1. The overall framework of the proposed Diffusion-based Dual Graph Multi-attention Network (Diff-DGMN) model. </b> 
</p>

## Methodology
In order to capture individual user visit preferences and behavior patterns, and reflect the local regularity of user transitions between different POIs, we propose a direction-aware sequence graph multi-scale representation module to gain the POI sequence encoding on the user-oriented POI transition graph. The detailed process is depicted in **Fig_2**.
<p align="center">
<img align="middle" src="https://github.com/JKZuo/Diff-DGMN/blob/main/Figures/Fig_2.png"/>
</p>
<p align = "center">
<b>Figure 2. The direction-aware sequence graph multi-scale representation module. </b> 
</p>

Inspired by the achievements in diffusion models, we propose a Diffusion-based User Preference Sampling module to generate a pure (noise-free) location archetype vector from noise, leveraging the variance-preserving stochastic differential equation (VP-SDE). Specifically, this module is divided into two steps: 1) Forward VP-SDE Diffusion Process; and 2) Reverse-time VP-SDE Generation Process. The forward diffusion process aims to convert the pure ground truth (target POI) into noise by continuous time sampling. 
Then build a model to learn this relative reverse-time process, step by step eliminate noise, and reconstruct the original pure ground truth. It can be described as shown in **Fig_3**.
<p align="center">
<img align="middle" src="https://github.com/JKZuo/Diff-DGMN/blob/main/Figures/Fig_3.png" width="750"/>
</p>
<p align = "center">
<b>Figure 3. Illustration of forward diffusion and reverse generation. </b> 
</p>

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

We evaluate our proposed Diff-DGMN model on five cities: Istanbul (IST) in Turkey, Jakarta (JK) in Indonesia, Sao Paulo (SP) in Brazil, New York City (NYC), and Los Angeles (LA) in the USA. 
They are all collected from the most popular location-based social networks (LBSNs) service providers--Foursquare. 
The time range of all datasets is about 22 months from Apr. 2012 to Jan. 2014. The detailed statistics are summarized in **Fig_4**.
<p align="center">
<img align="middle" src="https://github.com/JKZuo/Diff-DGMN/blob/main/Figures/Fig_4.png" width="550"/>
</p>
<p align = "center">
<b>Figure 4. Basic data statistics of five cities. </b> 
</p>

## Running
**Attention: Please modify the datasets in your path: DATA_PATH = '../DiffDGMN/data/processed' in the **[gol.py]** file

Then, you can use the small-scale LA dataset as an example to run it as：

```shell
nohup python main.py --dataset LA --gpu 0 --dp 0.4 > LA.log 2>&1 &
```
## Cite
If you feel that this work has been helpful for your research, please cite it as: 

- J. Zuo and Y. Zhang, "Diff-DGMN: A Diffusion-Based Dual Graph Multi-Attention Network for POI Recommendation," in IEEE Internet of Things Journal, Early Access, https://doi.org/10.1109/JIOT.2024.3446048.

or

```tex
@ARTICLE{Diff-DGMN,
  author={Zuo, Jiankai and Zhang, Yaying},
  journal={IEEE Internet of Things Journal}, 
  title={Diff-DGMN: A Diffusion-Based Dual Graph Multi-Attention Network for POI Recommendation}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/JIOT.2024.3446048}
  publisher={IEEE}
}

```

keywords---Diffusion models; POI recommendation; Graph neural network; Self-attention; Location-based social networks; Long short term memory; Social networking (online); Semantics; Recurrent neural networks; Internet of Things.
