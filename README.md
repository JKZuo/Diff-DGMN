# Diff-DGMN
Diff-DGMN: A Diffusion-based Dual Graph Multi-attention Network for POI Recommendation

## Requirements
The code has been tested running under Python 3.8.13.

The required packages are as follows: 
- Python == 3.8.13
  
- torch == 1.12.1
  
- torchsde == 0.2.6
  
- torch_geometric == 2.3.1
  
- pandas == 2.0.3
  
- numpy == 1.23.3

## Running
```shell
nohup python main.py --dataset 1_saopaulo --gpu 1 --layer 2 --dropoutrate 0.4 > 1SP_1.log 2>&1 & 
```
You will see on the screen the result: 

## Data
Due to the large dataset (the data file uploaded by GitHub cannot be larger than 25MB), you can download it through this link:

https://pan.baidu.com/s/19NG8Vn3u4fhsUK1P_kEr0Q?pwd=poi1

