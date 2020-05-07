# raw

Test-All MRR (bs=1,000): 0.642
FuncNameTest-All MRR (bs=1,000): 0.520
Validation-All MRR (bs=1,000): 0.607
Test-python MRR (bs=1,000): 0.642
FuncNameTest-python MRR (bs=1,000): 0.520
Validation-python MRR (bs=1,000): 0.607

wandb: Waiting for W&B process to finish, PID 54
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_step 89
wandb: train-loss 0.8792753895220248
wandb: \_runtime 1447.47190451622
wandb: \_timestamp 1589017409.2978396
wandb: train-mrr 0.843405099665077
wandb: val-mrr 0.47118912671959917
wandb: val-loss 1.0636994216753088
wandb: train-time-sec 39.094627141952515
wandb: val-time-sec 1.4557223320007324
wandb: epoch 13
wandb: best_val_mrr_loss 1.0634935368662295
wandb: best_val_mrr 0.47282196708347485
wandb: best_epoch 8
wandb: Test-All MRR (bs=1,000) 0.6423246369422994
wandb: FuncNameTest-All MRR (bs=1,000) 0.5198656606375424
wandb: Validation-All MRR (bs=1,000) 0.6068565747372077
wandb: Test-python MRR (bs=1,000) 0.6423246369422994
wandb: FuncNameTest-python MRR (bs=1,000) 0.5198656606375424
wandb: Validation-python MRR (bs=1,000) 0.6068565747372077
wandb: Syncing files in wandb/run-20200509_091923-54f9fe7d:
wandb: treeraw-2020-05-09-09-19-23-graph.pbtxt
wandb: treeraw-2020-05-09-09-19-23.train_log
wandb: treeraw-2020-05-09-09-19-23_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-09-09-19-23: https://app.wandb.ai/jianguda/CodeSearchNet/runs/54f9fe7d
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/54f9fe7d
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-09-09-19-23_model_best.pkl.gz
2020-05-09 09:43:53.837342: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-09 09:43:58.732852: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/54f9fe7d
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-09-09-19-23_model_best.pkl.gz
2020-05-09 09:43:53.837342: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-09 09:43:58.732852: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
wandb: treeraw-2020-05-09-09-19-23_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-09-09-19-23: https://app.wandb.ai/jianguda/CodeSearchNet/runs/54f9fe7d
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/54f9fe7d  
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-09-09-19-23_model_best.pkl.gz
2020-05-09 09:43:53.837342: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-09 09:43:58.732852: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: e684:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-09 09:43:58.732899: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-09 09:43:59.008692: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-09 09:43:59.008758: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-09 09:43:59.008775: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-09 09:43:59.008886: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10761 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: e684:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 195408.99it/s]1156085it [00:19, 57838.10it/s]
Uploading predictions to W&B
NDCG Average: 0.263834884

# raw-preprocessing

wandb: Waiting for W&B process to finish, PID 218
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-mrr 0.8478999406577897
wandb: \_runtime 1322.3198087215424
wandb: train-loss 0.874376712779349
wandb: \_step 89
wandb: \_timestamp 1589025993.9466066
wandb: epoch 13
wandb: val-loss 1.0585332892157815
wandb: val-mrr 0.4882650742964311
wandb: val-time-sec 1.4677467346191406
wandb: train-time-sec 40.40000796318054
wandb: best_val_mrr_loss 1.0574984550476074
wandb: best_val_mrr 0.4910310377641158
wandb: best_epoch 8
wandb: Test-All MRR (bs=1,000) 0.6478484334281543
wandb: FuncNameTest-All MRR (bs=1,000) 0.5208790949325063
wandb: Validation-All MRR (bs=1,000) 0.6180422828128215
wandb: Test-python MRR (bs=1,000) 0.6478484334281543
wandb: FuncNameTest-python MRR (bs=1,000) 0.5208790949325063
wandb: Validation-python MRR (bs=1,000) 0.6180422828128215
wandb: Syncing files in wandb/run-20200509_114433-zny4wn7d:
wandb: treeraw-2020-05-09-11-44-33-graph.pbtxt
wandb: treeraw-2020-05-09-11-44-33.train_log
wandb: treeraw-2020-05-09-11-44-33_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-09-11-44-33: https://app.wandb.ai/jianguda/CodeSearchNet/runs/zny4wn7d
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/zny4wn7d
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-09-11-44-33_model_best.pkl.gz
2020-05-09 12:07:37.139734: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-09 12:07:42.048151: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: e684:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-09 12:07:42.048199: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-09 12:07:42.335545: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-09 12:07:42.335604: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-09 12:07:42.335617: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-09 12:07:42.335732: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10761 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: e684:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 193587.40it/s]1156085it [00:19, 57844.98it/s]
Uploading predictions to W&B
NDCG Average: 0.286767027

# leaf

Epoch 17 (valid) took 1.48s [processed 15493 samples/second]
Validation: Loss: 1.070042 | MRR: 0.455643
2020-05-09 12:50:34.413709: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-09 12:50:34.413766: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-09 12:50:34.413781: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-09 12:50:34.413788: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-09 12:50:34.413885: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10761 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: e684:00:00.0, compute capability: 3.7)Test-All MRR (bs=1,000): 0.620
FuncNameTest-All MRR (bs=1,000): 0.580
Validation-All MRR (bs=1,000): 0.593
Test-python MRR (bs=1,000): 0.620
FuncNameTest-python MRR (bs=1,000): 0.580
Validation-python MRR (bs=1,000): 0.593

wandb: Waiting for W&B process to finish, PID 427
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_timestamp 1589029139.6861541
wandb: train-loss 0.8771363460612529
wandb: \_runtime 2158.160849094391
wandb: train-mrr 0.8456747618740045
wandb: \_step 113
wandb: train-time-sec 41.35736536979675
wandb: epoch 17
wandb: val-mrr 0.45564281894849695
wandb: val-loss 1.0700422266255254
wandb: val-time-sec 1.4845168590545654
wandb: best_val_mrr_loss 1.070032264875329
wandb: best_val_mrr 0.4586684762705927
wandb: best_epoch 12
wandb: Test-All MRR (bs=1,000) 0.6199923840728658
wandb: FuncNameTest-All MRR (bs=1,000) 0.5799845026952148
wandb: Validation-All MRR (bs=1,000) 0.5930079225754412
wandb: Test-python MRR (bs=1,000) 0.6199923840728658
wandb: FuncNameTest-python MRR (bs=1,000) 0.5799845026952148
wandb: Validation-python MRR (bs=1,000) 0.5930079225754412
wandb: Syncing files in wandb/run-20200509_122303-67isyv5p:
wandb: treeleaf-2020-05-09-12-23-03-graph.pbtxt
wandb: treeleaf-2020-05-09-12-23-03.train_log
wandb: treeleaf-2020-05-09-12-23-03_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeleaf-2020-05-09-12-23-03: https://app.wandb.ai/jianguda/CodeSearchNet/runs/67isyv5p
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/67isyv5p
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeleaf-2020-05-09-12-23-03_model_best.pkl.gz
2020-05-09 12:59:25.070023: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-09 12:59:30.507281: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: e684:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-09 12:59:30.507327: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-09 12:59:30.799049: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-09 12:59:30.799099: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-09 12:59:30.799114: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-09 12:59:30.799229: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10761 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: e684:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:06<00:00, 189097.57it/s]1156085it [00:20, 56512.10it/s]
Uploading predictions to W&B
NDCG Average: 0.267117210

# leaf-preprocessing

# path

# path-attention

# path-preprocessing

# tree

# tree-attention

# tree-preprocessing
