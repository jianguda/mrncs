# vanilla

# KNN

Epoch 32 (valid) took 6.19s [processed 14368 samples/second]
Validation: Loss: 1.030382 | MRR: 0.516010
2020-05-18 23:57:01.466809: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-18 23:57:01.466875: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-18 23:57:01.466891: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-18 23:57:01.466902: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-18 23:57:01.467000: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10761 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 3476:00:00.0, compute capability: 3.7)Test-All MRR (bs=1,000): 0.620
FuncNameTest-All MRR (bs=1,000): 0.572
Validation-All MRR (bs=1,000): 0.638
Test-javascript MRR (bs=1,000): 0.460
FuncNameTest-javascript MRR (bs=1,000): 0.226
Validation-javascript MRR (bs=1,000): 0.457
Test-java MRR (bs=1,000): 0.522
FuncNameTest-java MRR (bs=1,000): 0.625
Validation-java MRR (bs=1,000): 0.505
Test-go MRR (bs=1,000): 0.647
FuncNameTest-go MRR (bs=1,000): 0.317
Validation-go MRR (bs=1,000): 0.778
Test-php MRR (bs=1,000): 0.485
FuncNameTest-php MRR (bs=1,000): 0.585
Validation-php MRR (bs=1,000): 0.494
Test-ruby MRR (bs=1,000): 0.420
FuncNameTest-ruby MRR (bs=1,000): 0.405
Validation-ruby MRR (bs=1,000): 0.482
Test-python MRR (bs=1,000): 0.580
FuncNameTest-python MRR (bs=1,000): 0.475
Validation-python MRR (bs=1,000): 0.552

wandb: Waiting for W&B process to finish, PID 322
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_runtime 9908.826846837997
wandb: train-loss 0.8401613493549063
wandb: \_step 680
wandb: train-mrr 0.7928175668594685
wandb: \_timestamp 1589846761.3812
wandb: val-mrr 0.5160098729508646
wandb: val-time-sec 6.194053411483765
wandb: train-time-sec 204.95770263671875
wandb: epoch 32
wandb: val-loss 1.0303819253203574
wandb: best_val_mrr_loss 1.030174804537484
wandb: best_val_mrr 0.5160657098320093
wandb: best_epoch 27
wandb: Test-All MRR (bs=1,000) 0.6198406089608407
wandb: FuncNameTest-All MRR (bs=1,000) 0.5716961599599323
wandb: Validation-All MRR (bs=1,000) 0.6384438958660721
wandb: Test-javascript MRR (bs=1,000) 0.4603978056147358
wandb: FuncNameTest-javascript MRR (bs=1,000) 0.22646553388750035
wandb: Validation-javascript MRR (bs=1,000) 0.45733729257897776
wandb: Test-java MRR (bs=1,000) 0.522344666811285
wandb: FuncNameTest-java MRR (bs=1,000) 0.625122819117425
wandb: Validation-java MRR (bs=1,000) 0.5052997134055793
wandb: Test-go MRR (bs=1,000) 0.6467017259401601
wandb: FuncNameTest-go MRR (bs=1,000) 0.31663808964386375
wandb: Validation-go MRR (bs=1,000) 0.7784259990971278
wandb: Test-php MRR (bs=1,000) 0.4849080294878009
wandb: FuncNameTest-php MRR (bs=1,000) 0.5852448419041058
wandb: Validation-php MRR (bs=1,000) 0.49383688113945906
wandb: Test-ruby MRR (bs=1,000) 0.41970822508592487
wandb: FuncNameTest-ruby MRR (bs=1,000) 0.4053380352079744
wandb: Validation-ruby MRR (bs=1,000) 0.4820010592665092
wandb: Test-python MRR (bs=1,000) 0.5800741691942761
wandb: FuncNameTest-python MRR (bs=1,000) 0.47528084923742997
wandb: Validation-python MRR (bs=1,000) 0.5518863281554993
wandb: Syncing files in wandb/run-20200518_212054-ratdb76s:
wandb: treeraw-2020-05-18-21-20-54-graph.pbtxt
wandb: treeraw-2020-05-18-21-20-54.train_log
wandb: treeraw-2020-05-18-21-20-54_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-18-21-20-54: https://app.wandb.ai/jianguda/CodeSearchNet/runs/ratdb76s
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/ratdb76s
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-18-21-20-54_model_best.pkl.gz
2020-05-19 00:12:09.332533: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-19 00:12:14.547166: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 3476:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-19 00:12:14.547212: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-19 00:12:14.829536: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-19 00:12:14.829595: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-19 00:12:14.829611: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-19 00:12:14.829724: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10761 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 3476:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-18-21-20-54_model_best.pkl.gz
2020-05-19 00:12:09.332533: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-19 00:12:14.547166: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 3476:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-19 00:12:14.547212: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-19 00:12:14.829536: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-19 00:12:14.829595: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-19 00:12:14.829611: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-19 00:12:14.829724: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10761 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 3476:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
2020-05-19 00:12:14.547212: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-19 00:12:14.829536: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-19 00:12:14.829595: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-19 00:12:14.829611: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-19 00:12:14.829724: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10761 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 3476:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 195164.08it/s]Evaluating language: go
100%|████████████████████████████████████████████████████████████████████████████████████████| 726768/726768 [00:00<00:00, 747289.72it/s]Evaluating language: javascript
100%|██████████████████████████████████████████████████████████████████████████████████████| 1857835/1857835 [00:13<00:00, 137486.59it/s]Evaluating language: java
100%|███████████████████████████████████████████████████████████████████████████████████████| 1569889/1569889 [00:16<00:00, 96167.13it/s]Evaluating language: php
100%|████████████████████████████████████████████████████████████████████████████████████████| 977821/977821 [00:01<00:00, 610031.70it/s]Evaluating language: ruby
100%|████████████████████████████████████████████████████████████████████████████████████████| 164048/164048 [00:00<00:00, 333271.42it/s]Uploading predictions to W&B
NDCG Average: 0.370379094
