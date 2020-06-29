root@jian-csn:/home/dev/src# python train.py --model selfatt --max-num-epochs=10
==== Epoch 9 ====
Epoch 9 (train) took 6396.94s [processed 294 samples/second]
Training Loss: 0.298605
Epoch 9 (valid) took 124.54s [processed 714 samples/second]
Validation: Loss: 2.002781 | MRR: 0.668763
2020-06-17 08:24:32.718338: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-06-17 08:24:32.718392: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-06-17 08:24:32.718407: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-06-17 08:24:32.718420: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-06-17 08:24:32.718511: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10761 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 0001:00:00.0, compute capability: 3.7)
"\_runtime": 68618.1518611908,
"train-loss": 0.2986052342033333,
"\_step": 980,
"train-mrr": 0.930153799409919,
"\_timestamp": 1592383364.814416,
"epoch": 9,
"val-time-sec": 124.54219222068787,
"val-mrr": 0.6687632002241156,
"train-time-sec": 6396.942043066025,
"val-loss": 2.002780837959118,
"best_val_mrr_loss": 1.9531075330262773,
"best_val_mrr": 0.669361386074109,
"best_epoch": 8,
Test-All MRR (bs=1,000): 0.677
FuncNameTest-All MRR (bs=1,000): 0.697
Validation-All MRR (bs=1,000): 0.684
Test-java MRR (bs=1,000): 0.554
FuncNameTest-java MRR (bs=1,000): 0.770
Validation-java MRR (bs=1,000): 0.528
Test-python MRR (bs=1,000): 0.653
FuncNameTest-python MRR (bs=1,000): 0.629
Validation-python MRR (bs=1,000): 0.604
Test-php MRR (bs=1,000): 0.574
FuncNameTest-php MRR (bs=1,000): 0.842
Validation-php MRR (bs=1,000): 0.581
Test-ruby MRR (bs=1,000): 0.293
FuncNameTest-ruby MRR (bs=1,000): 0.471
Validation-ruby MRR (bs=1,000): 0.311
Test-go MRR (bs=1,000): 0.663
FuncNameTest-go MRR (bs=1,000): 0.193
Validation-go MRR (bs=1,000): 0.753
Test-javascript MRR (bs=1,000): 0.406
FuncNameTest-javascript MRR (bs=1,000): 0.132
Validation-javascript MRR (bs=1,000): 0.398
wandb: val-time-sec 124.54219222068787
wandb: val-mrr 0.6687632002241156
wandb: train-time-sec 6396.942043066025
wandb: val-loss 2.002780837959118
wandb: best_val_mrr_loss 1.9531075330262773
wandb: best_val_mrr 0.669361386074109
wandb: best_epoch 8
wandb: Test-All MRR (bs=1,000) 0.6767331882676832
wandb: val-time-sec 124.54219222068787
wandb: val-mrr 0.6687632002241156
wandb: train-time-sec 6396.942043066025
wandb: val-loss 2.002780837959118
wandb: best_val_mrr_loss 1.9531075330262773
wandb: best_val_mrr 0.669361386074109
wandb: best_epoch 8
wandb: Test-All MRR (bs=1,000) 0.6767331882676832
wandb: FuncNameTest-All MRR (bs=1,000) 0.6965546223804764
wandb: Validation-All MRR (bs=1,000) 0.6835175149669632
wandb: Test-java MRR (bs=1,000) 0.5542163909809896
wandb: FuncNameTest-java MRR (bs=1,000) 0.7704387757981261
wandb: Validation-java MRR (bs=1,000) 0.5280393938645641
wandb: Test-python MRR (bs=1,000) 0.6528557715249695
wandb: FuncNameTest-python MRR (bs=1,000) 0.6290712005731268
wandb: FuncNameTest-All MRR (bs=1,000) 0.6965546223804764
wandb: Validation-All MRR (bs=1,000) 0.6835175149669632
wandb: Test-java MRR (bs=1,000) 0.5542163909809896
wandb: FuncNameTest-java MRR (bs=1,000) 0.7704387757981261
wandb: Validation-java MRR (bs=1,000) 0.5280393938645641
wandb: Test-python MRR (bs=1,000) 0.6528557715249695
wandb: FuncNameTest-python MRR (bs=1,000) 0.6290712005731268
wandb: Validation-python MRR (bs=1,000) 0.6042772245820232
wandb: Test-php MRR (bs=1,000) 0.5735524312970738
wandb: FuncNameTest-php MRR (bs=1,000) 0.8416365177160439
wandb: Validation-php MRR (bs=1,000) 0.5810513489092148
wandb: FuncNameTest-All MRR (bs=1,000) 0.6965546223804764
wandb: Validation-All MRR (bs=1,000) 0.6835175149669632
wandb: Test-java MRR (bs=1,000) 0.5542163909809896
wandb: FuncNameTest-java MRR (bs=1,000) 0.7704387757981261
wandb: Validation-java MRR (bs=1,000) 0.5280393938645641
wandb: Test-python MRR (bs=1,000) 0.6528557715249695
wandb: FuncNameTest-python MRR (bs=1,000) 0.6290712005731268
wandb: Validation-python MRR (bs=1,000) 0.6042772245820232
wandb: Test-php MRR (bs=1,000) 0.5735524312970738
wandb: FuncNameTest-php MRR (bs=1,000) 0.8416365177160439
wandb: Validation-php MRR (bs=1,000) 0.5810513489092148
wandb: Test-All MRR (bs=1,000) 0.6767331882676832
wandb: FuncNameTest-All MRR (bs=1,000) 0.6965546223804764
wandb: Validation-All MRR (bs=1,000) 0.6835175149669632
wandb: Test-java MRR (bs=1,000) 0.5542163909809896
wandb: FuncNameTest-java MRR (bs=1,000) 0.7704387757981261
wandb: Validation-java MRR (bs=1,000) 0.5280393938645641
wandb: Test-python MRR (bs=1,000) 0.6528557715249695
wandb: FuncNameTest-python MRR (bs=1,000) 0.6290712005731268
wandb: Validation-python MRR (bs=1,000) 0.6042772245820232
wandb: Test-php MRR (bs=1,000) 0.5735524312970738
wandb: FuncNameTest-php MRR (bs=1,000) 0.8416365177160439
wandb: Validation-php MRR (bs=1,000) 0.5810513489092148
wandb: Test-ruby MRR (bs=1,000) 0.29283033058732627
wandb: FuncNameTest-ruby MRR (bs=1,000) 0.47110121773269664
wandb: Validation-ruby MRR (bs=1,000) 0.3112554132086458
wandb: Test-go MRR (bs=1,000) 0.6633660939157314
wandb: FuncNameTest-go MRR (bs=1,000) 0.1929624695962833
wandb: Validation-go MRR (bs=1,000) 0.7530708666659935
wandb: Test-javascript MRR (bs=1,000) 0.40603536841300647
wandb: FuncNameTest-javascript MRR (bs=1,000) 0.13164631892288506
wandb: Validation-javascript MRR (bs=1,000) 0.39778517139453407
wandb: Syncing files in wandb/run-20200616_133908-v6mrdtzt:
wandb: selfatt-2020-06-16-13-39-08-graph.pbtxt
wandb: selfatt-2020-06-16-13-39-08.train_log
wandb: selfatt-2020-06-16-13-39-08_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced selfatt-2020-06-16-13-39-08: https://app.wandb.ai/jianguda/CodeSearchNet/runs/v6mrdtzt

# computed by submitter (so that we get all NDCG scores without updating leaderboard)

root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/v6mrdtzt
Evaluating language: go
https://app.wandb.ai/jianguda/CodeSearchNet/runs/fx0hvqj4
NDCG Average: 0.063727197
Evaluating language: java
https://app.wandb.ai/jianguda/CodeSearchNet/runs/15n9ct49
NDCG Average: 0.087003303
Evaluating language: javascript
https://app.wandb.ai/jianguda/CodeSearchNet/runs/1od1awr3
NDCG Average: 0.041069050
Evaluating language: php
https://app.wandb.ai/jianguda/CodeSearchNet/runs/3w1cxtn1
NDCG Average: 0.100180917
Evaluating language: python
https://app.wandb.ai/jianguda/CodeSearchNet/runs/1oq7om9c
NDCG Average: 0.215069473
Evaluating language: ruby
https://app.wandb.ai/jianguda/CodeSearchNet/runs/20ytz53w
NDCG Average: 0.158151091
