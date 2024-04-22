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
04/22 10:49  Batch 0 / 219, Loss: 9.73333 = CE Loss: 9.42493 + Fisher Loss: 1.54201
04/22 10:56  Batch 109 / 219, Loss: 8.49752 = CE Loss: 8.27136 + Fisher Loss: 1.13082
04/22 11:03  Batch 218 / 219, Loss: 7.75034 = CE Loss: 7.53010 + Fisher Loss: 1.10121
04/22 11:05  Epoch 0 / 100, Loss: 8.68333 = CE Loss: 8.44584 + Fisher Loss: 1.18745
04/22 11:05  Valid NDCG@5: 0.11307, Recall@2: 0.10275, Recall@5: 0.15164
04/22 11:08  New test result:
 {'MRR': 0.10684251092186374,
 'NDCG': array([0.06797906, 0.08968369, 0.11144677, 0.12435918, 0.13579692]),
 'Recall': array([0.06797906, 0.10238009, 0.1508378 , 0.19096685, 0.2361326 ])}
04/22 11:08  Best valid Recall@5 at epoch 0
04/22 11:08  Test NDCG@5: 0.11145, Recall@2: 0.10238, Recall@5: 0.15084

## Data
Due to the large dataset (the data file uploaded by GitHub cannot be larger than 25MB), you can download it through this link:

https://pan.baidu.com/s/19NG8Vn3u4fhsUK1P_kEr0Q?pwd=poi1

