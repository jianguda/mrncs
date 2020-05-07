# convert-normal

Epoch 17 (valid) took 1.53s [processed 15026 samples/second]
Validation: Loss: 1.064817 | MRR: 0.471930
2020-05-10 12:59:12.880207: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-10 12:59:12.880267: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-10 12:59:12.880282: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-10 12:59:12.880294: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-10 12:59:12.880395: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 5117:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.645
FuncNameTest-All MRR (bs=1,000): 0.530
Validation-All MRR (bs=1,000): 0.613
Test-python MRR (bs=1,000): 0.645
FuncNameTest-python MRR (bs=1,000): 0.530
Validation-python MRR (bs=1,000): 0.613

wandb: Waiting for W&B process to finish, PID 261
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_runtime 1553.401748418808
wandb: train-loss 0.8731856139250171
wandb: train-mrr 0.8503593675928208
wandb: \_timestamp 1589115677.687397
wandb: \_step 113
wandb: train-time-sec 40.92247414588928
wandb: epoch 17
wandb: val-time-sec 1.5306415557861328
wandb: val-loss 1.0648169362026711
wandb: val-mrr 0.47193014393682065
wandb: best_val_mrr_loss 1.0631857540296472
wandb: best_val_mrr 0.47431985208262567
wandb: best_epoch 12
wandb: Test-All MRR (bs=1,000) 0.6448155613973404
wandb: FuncNameTest-All MRR (bs=1,000) 0.5302835691613995
wandb: Validation-All MRR (bs=1,000) 0.6131650694531959
wandb: Test-python MRR (bs=1,000) 0.6448155613973404
wandb: FuncNameTest-python MRR (bs=1,000) 0.5302835691613995
wandb: Validation-python MRR (bs=1,000) 0.6131650694531959
wandb: Syncing files in wandb/run-20200510_123525-wogikt4b:
wandb: treeraw-2020-05-10-12-35-25-graph.pbtxt
wandb: treeraw-2020-05-10-12-35-25.train_log
wandb: treeraw-2020-05-10-12-35-25_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-10-12-35-25: https://app.wandb.ai/jianguda/CodeSearchNet/runs/wogikt4b
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/wogikt4b
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-10-12-35-25_model_best.pkl.gz
2020-05-10 13:01:42.915462: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-10 13:01:47.749975: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 5117:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-10 13:01:47.750024: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-10 13:01:48.035091: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-10 13:01:48.035151: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-10 13:01:48.035167: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-10 13:01:48.035289: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 5117:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 195485.93it/s]1156085it [00:19, 58152.57it/s]
Uploading predictions to W&B
NDCG Average: 0.281564237

# discard-normal

Epoch 12 (valid) took 1.45s [processed 15890 samples/second]
Validation: Loss: 1.064884 | MRR: 0.474182
2020-05-10 13:38:21.694584: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-10 13:38:21.694645: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-10 13:38:21.694659: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-10 13:38:21.694669: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-10 13:38:21.694765: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 5117:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.646
FuncNameTest-All MRR (bs=1,000): 0.530
Validation-All MRR (bs=1,000): 0.614
Test-python MRR (bs=1,000): 0.646
FuncNameTest-python MRR (bs=1,000): 0.530
Validation-python MRR (bs=1,000): 0.614

wandb: Waiting for W&B process to finish, PID 470
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_runtime 1304.978660583496
wandb: \_timestamp 1589118021.375357
wandb: train-mrr 0.8450127600512458
wandb: \_step 83
wandb: train-loss 0.8769904172536239
wandb: val-loss 1.0648840095685876
wandb: epoch 12
wandb: train-time-sec 39.80772662162781
wandb: val-time-sec 1.447425365447998
wandb: val-mrr 0.47418150196904724
wandb: best_val_mrr_loss 1.0645566660424937
wandb: best_val_mrr 0.47601575503141985
wandb: best_epoch 7
wandb: Test-All MRR (bs=1,000) 0.6460568612144302
wandb: FuncNameTest-All MRR (bs=1,000) 0.5296884056890842
wandb: Validation-All MRR (bs=1,000) 0.613652863124518
wandb: Test-python MRR (bs=1,000) 0.6460568612144302
wandb: FuncNameTest-python MRR (bs=1,000) 0.5296884056890842
wandb: Validation-python MRR (bs=1,000) 0.613652863124518
wandb: Syncing files in wandb/run-20200510_131837-cv4mpssv:
wandb: treeraw-2020-05-10-13-18-37-graph.pbtxt
wandb: treeraw-2020-05-10-13-18-37.train_log
wandb: treeraw-2020-05-10-13-18-37_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-10-13-18-37: https://app.wandb.ai/jianguda/CodeSearchNet/runs/cv4mpssv
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/cv4mpssv
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-10-13-18-37_model_best.pkl.gz
2020-05-10 13:40:45.465182: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-10 13:40:50.432077: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 5117:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-10 13:40:50.432125: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-10 13:40:50.706525: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-10 13:40:50.706587: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-10 13:40:50.706604: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-10 13:40:50.706716: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 5117:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 193504.41it/s]1156085it [00:20, 56624.05it/s]
Uploading predictions to W&B
NDCG Average: 0.255463245

# normal-option1

Epoch 13 (valid) took 1.40s [processed 16470 samples/second]
Validation: Loss: 1.064329 | MRR: 0.470671
2020-05-10 14:17:16.836288: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-10 14:17:16.836347: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-10 14:17:16.836362: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-10 14:17:16.836373: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-10 14:17:16.836455: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 5117:00:00.0, compute capability: 3.7)Test-All MRR (bs=1,000): 0.636
FuncNameTest-All MRR (bs=1,000): 0.507
Validation-All MRR (bs=1,000): 0.609
Test-python MRR (bs=1,000): 0.636
FuncNameTest-python MRR (bs=1,000): 0.507
Validation-python MRR (bs=1,000): 0.609

wandb: Waiting for W&B process to finish, PID 678
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-loss 0.8787775330462502
wandb: \_step 89
wandb: train-mrr 0.8435899418210521
wandb: \_timestamp 1589120347.2733161
wandb: \_runtime 1319.5408642292023
wandb: val-mrr 0.47067105765964673
wandb: val-loss 1.0643287067827971
wandb: train-time-sec 38.0255560874939
wandb: val-time-sec 1.396451711654663
wandb: epoch 13
wandb: best_val_mrr_loss 1.0645783880482549
wandb: best_val_mrr 0.4716947339928668
wandb: best_epoch 8
wandb: Test-All MRR (bs=1,000) 0.6357590686134024
wandb: FuncNameTest-All MRR (bs=1,000) 0.507364539885093
wandb: Validation-All MRR (bs=1,000) 0.6085602674681447
wandb: Test-python MRR (bs=1,000) 0.6357590686134024
wandb: FuncNameTest-python MRR (bs=1,000) 0.507364539885093
wandb: Validation-python MRR (bs=1,000) 0.6085602674681447
wandb: Syncing files in wandb/run-20200510_135709-aya2x4n6:
wandb: treeraw-2020-05-10-13-57-09-graph.pbtxt
wandb: treeraw-2020-05-10-13-57-09.train_log
wandb: treeraw-2020-05-10-13-57-09_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-10-13-57-09: https://app.wandb.ai/jianguda/CodeSearchNet/runs/aya2x4n6
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/aya2x4n6
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-10-13-57-09_model_best.pkl.gz
2020-05-10 14:19:32.174229: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-10 14:19:37.192221: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 5117:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-10 14:19:37.192268: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-10 14:19:37.472116: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-10 14:19:37.472181: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-10 14:19:37.472199: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-10 14:19:37.472315: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 5117:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 194519.77it/s]1156085it [00:20, 57732.47it/s]
Uploading predictions to W&B
NDCG Average: 0.314850832

# normal-option2

# normal-option3

# normal-option4

Epoch 11 (valid) took 1.50s [processed 15301 samples/second]
Validation: Loss: 1.071973 | MRR: 0.447556
2020-05-10 14:54:22.751156: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-10 14:54:22.751219: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-10 14:54:22.751234: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-10 14:54:22.751244: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-10 14:54:22.751348: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 5117:00:00.0, compute capability: 3.7)Test-All MRR (bs=1,000): 0.610
FuncNameTest-All MRR (bs=1,000): 0.497
Validation-All MRR (bs=1,000): 0.579
Test-python MRR (bs=1,000): 0.610
FuncNameTest-python MRR (bs=1,000): 0.497
Validation-python MRR (bs=1,000): 0.579

wandb: Waiting for W&B process to finish, PID 887
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_step 77
wandb: \_timestamp 1589122574.370859
wandb: train-loss 0.8909476452950135
wandb: train-mrr 0.8270129204907464
wandb: \_runtime 1268.768427848816
wandb: epoch 11
wandb: train-time-sec 40.08960294723511
wandb: val-time-sec 1.5031523704528809
wandb: val-loss 1.0719726344813472
wandb: val-mrr 0.4475564880371094
wandb: best_val_mrr_loss 1.0721352204032566
wandb: best_val_mrr 0.4509873743471892
wandb: best_epoch 6
wandb: Test-All MRR (bs=1,000) 0.6097355210128913
wandb: FuncNameTest-All MRR (bs=1,000) 0.4967660609488884
wandb: Validation-All MRR (bs=1,000) 0.57948711795114
wandb: Test-python MRR (bs=1,000) 0.6097355210128913
wandb: FuncNameTest-python MRR (bs=1,000) 0.4967660609488884
wandb: Validation-python MRR (bs=1,000) 0.57948711795114
wandb: Syncing files in wandb/run-20200510_143507-0tkqujzg:
wandb: treeraw-2020-05-10-14-35-07-graph.pbtxt
wandb: treeraw-2020-05-10-14-35-07.train_log
wandb: treeraw-2020-05-10-14-35-07_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-10-14-35-07: https://app.wandb.ai/jianguda/CodeSearchNet/runs/0tkqujzg
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/0tkqujzg
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-10-14-35-07_model_best.pkl.gz
2020-05-10 14:56:34.991162: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-10 14:56:39.985666: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 5117:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-10 14:56:39.985711: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-10 14:56:40.266549: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-10 14:56:40.266613: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-10 14:56:40.266629: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-10 14:56:40.266739: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 5117:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 197148.54it/s]1156085it [00:19, 57866.21it/s]
Uploading predictions to W&B
NDCG Average: 0.257384607

# normal-option5

Epoch 17 (valid) took 1.44s [processed 15935 samples/second]
Validation: Loss: 1.063813 | MRR: 0.470008
2020-05-10 15:36:34.629577: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-10 15:36:34.629636: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-10 15:36:34.629651: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-10 15:36:34.629662: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-10 15:36:34.629763: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 5117:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.641
FuncNameTest-All MRR (bs=1,000): 0.521
Validation-All MRR (bs=1,000): 0.609
Test-python MRR (bs=1,000): 0.641
FuncNameTest-python MRR (bs=1,000): 0.521
Validation-python MRR (bs=1,000): 0.609

wandb: Waiting for W&B process to finish, PID 1096
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_step 113
wandb: \_timestamp 1589125130.0660582
wandb: train-loss 0.8751411181919783
wandb: train-mrr 0.850477572839237
wandb: \_runtime 1576.1443212032318
wandb: epoch 17
wandb: val-loss 1.0638129296510115
wandb: val-mrr 0.4700077700407609
wandb: val-time-sec 1.443343162536621
wandb: train-time-sec 39.273043632507324
wandb: best_val_mrr_loss 1.0640695872514143
wandb: best_val_mrr 0.4706083307680876
wandb: best_epoch 12
wandb: Test-All MRR (bs=1,000) 0.6409359521115373
wandb: FuncNameTest-All MRR (bs=1,000) 0.5210399821632006
wandb: Validation-All MRR (bs=1,000) 0.6085602194504132
wandb: Test-python MRR (bs=1,000) 0.6409359521115373
wandb: FuncNameTest-python MRR (bs=1,000) 0.5210399821632006
wandb: Validation-python MRR (bs=1,000) 0.6085602194504132
wandb: Syncing files in wandb/run-20200510_151235-y0qxa7a3:
wandb: treeraw-2020-05-10-15-12-35-graph.pbtxt
wandb: treeraw-2020-05-10-15-12-35.train_log
wandb: treeraw-2020-05-10-15-12-35_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-10-15-12-35: https://app.wandb.ai/jianguda/CodeSearchNet/runs/y0qxa7a3
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/y0qxa7a3
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-10-15-12-35_model_best.pkl.gz
2020-05-10 15:39:30.283241: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-10 15:39:35.251710: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 5117:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-10 15:39:35.251758: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-10 15:39:35.524989: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-10 15:39:35.525052: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-10 15:39:35.525063: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-10 15:39:35.525180: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 5117:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 194440.74it/s]1156085it [00:20, 57653.39it/s]
Uploading predictions to W&B
NDCG Average: 0.303023241

# normal-option6

Epoch 21 (valid) took 1.40s [processed 16404 samples/second]
Validation: Loss: 1.066220 | MRR: 0.466056
2020-05-10 17:09:14.823155: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-10 17:09:14.823221: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-10 17:09:14.823236: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-10 17:09:14.823249: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-10 17:09:14.823348: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 5117:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.638
FuncNameTest-All MRR (bs=1,000): 0.509
Validation-All MRR (bs=1,000): 0.607
Test-python MRR (bs=1,000): 0.638
FuncNameTest-python MRR (bs=1,000): 0.509
Validation-python MRR (bs=1,000): 0.607

wandb: Waiting for W&B process to finish, PID 1512
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_timestamp 1589130684.2700675
wandb: \_runtime 1706.1777889728546
wandb: train-loss 0.878668606425952
wandb: train-mrr 0.8433767033549189
wandb: \_step 137
wandb: val-time-sec 1.4020195007324219
wandb: val-loss 1.066219635631727
wandb: val-mrr 0.46605552739682404
wandb: epoch 21
wandb: train-time-sec 38.68764662742615
wandb: best_val_mrr_loss 1.0658983095832493
wandb: best_val_mrr 0.46843522445015284
wandb: best_epoch 16
wandb: Test-All MRR (bs=1,000) 0.6378593480816698
wandb: FuncNameTest-All MRR (bs=1,000) 0.5090063345041885
wandb: Validation-All MRR (bs=1,000) 0.6072488718337352
wandb: Test-python MRR (bs=1,000) 0.6378593480816698
wandb: FuncNameTest-python MRR (bs=1,000) 0.5090063345041885
wandb: Validation-python MRR (bs=1,000) 0.6072488718337352
wandb: Syncing files in wandb/run-20200510_164259-m0oliuq1:
wandb: treeraw-2020-05-10-16-42-59-graph.pbtxt
wandb: treeraw-2020-05-10-16-42-59.train_log
wandb: treeraw-2020-05-10-16-42-59_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-10-16-42-59: https://app.wandb.ai/jianguda/CodeSearchNet/runs/m0oliuq1
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/m0oliuq1
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-10-16-42-59_model_best.pkl.gz
2020-05-10 17:11:52.365082: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-10 17:11:57.336419: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 5117:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-10 17:11:57.336466: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-10 17:11:57.616289: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-10 17:11:57.616350: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-10 17:11:57.616366: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-10 17:11:57.616472: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 5117:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:06<00:00, 192333.57it/s]1156085it [00:20, 57381.84it/s]
Uploading predictions to W&B
NDCG Average: 0.256317155

# normal-option7

Epoch 21 (valid) took 1.44s [processed 15943 samples/second]
Validation: Loss: 1.064444 | MRR: 0.468987
2020-05-10 17:52:28.164504: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-10 17:52:28.164568: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-10 17:52:28.164582: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-10 17:52:28.164593: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-10 17:52:28.164695: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 5117:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.638
FuncNameTest-All MRR (bs=1,000): 0.526
Validation-All MRR (bs=1,000): 0.606
Test-python MRR (bs=1,000): 0.638
FuncNameTest-python MRR (bs=1,000): 0.526
Validation-python MRR (bs=1,000): 0.606

wandb: Waiting for W&B process to finish, PID 1720
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-loss 0.8740121755495812
wandb: \_runtime 1649.869518995285
wandb: \_step 137
wandb: \_timestamp 1589133258.283026
wandb: train-mrr 0.8523500318804991
wandb: val-loss 1.0644444641859636
wandb: epoch 21
wandb: val-time-sec 1.4426319599151611
wandb: val-mrr 0.4689871116306471
wandb: train-time-sec 38.254721879959106
wandb: best_val_mrr_loss 1.0638397621071858
wandb: best_val_mrr 0.4704143709929093
wandb: best_epoch 16
wandb: Test-All MRR (bs=1,000) 0.6384280538589352
wandb: FuncNameTest-All MRR (bs=1,000) 0.5264271440406211
wandb: Validation-All MRR (bs=1,000) 0.6063642955578752
wandb: Test-python MRR (bs=1,000) 0.6384280538589352
wandb: FuncNameTest-python MRR (bs=1,000) 0.5264271440406211
wandb: Validation-python MRR (bs=1,000) 0.6063642955578752
wandb: Syncing files in wandb/run-20200510_172649-osmg7e3y:
wandb: treeraw-2020-05-10-17-26-49-graph.pbtxt
wandb: treeraw-2020-05-10-17-26-49.train_log
wandb: treeraw-2020-05-10-17-26-49_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-10-17-26-49: https://app.wandb.ai/jianguda/CodeSearchNet/runs/osmg7e3y
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/osmg7e3y
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-10-17-26-49_model_best.pkl.gz
2020-05-10 18:20:12.504499: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-10 18:20:17.457159: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 5117:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-10 18:20:17.457208: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-10 18:20:17.741053: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-10 18:20:17.741114: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-10 18:20:17.741131: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-10 18:20:17.741238: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 5117:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 194107.07it/s]1156085it [00:20, 57729.01it/s]
Uploading predictions to W&B
NDCG Average: 0.285133542

# normal-option8

Epoch 12 (valid) took 1.45s [processed 15853 samples/second]
Validation: Loss: 1.065110 | MRR: 0.470705
2020-05-10 22:47:12.723409: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-10 22:47:12.723474: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-10 22:47:12.723489: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-10 22:47:12.723500: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-10 22:47:12.723590: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 5117:00:00.0, compute capability: 3.7)Test-All MRR (bs=1,000): 0.639
FuncNameTest-All MRR (bs=1,000): 0.515
Validation-All MRR (bs=1,000): 0.607
Test-python MRR (bs=1,000): 0.639
FuncNameTest-python MRR (bs=1,000): 0.515
Validation-python MRR (bs=1,000): 0.607

wandb: Waiting for W&B process to finish, PID 2975
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-loss 0.8807120942375035
wandb: \_runtime 1288.9840688705444
wandb: \_step 83
wandb: train-mrr 0.8410794692548733
wandb: \_timestamp 1589150944.7144926
wandb: train-time-sec 39.185790061950684
wandb: val-loss 1.0651098800742107
wandb: val-time-sec 1.450805425643921
wandb: val-mrr 0.4707052466351053
wandb: epoch 12
wandb: best_val_mrr_loss 1.0641338514245076
wandb: best_val_mrr 0.4709259019934613
wandb: best_epoch 7
wandb: Test-All MRR (bs=1,000) 0.6387156766401199
wandb: FuncNameTest-All MRR (bs=1,000) 0.5152049362036988
wandb: Validation-All MRR (bs=1,000) 0.6069307674807819
wandb: Test-python MRR (bs=1,000) 0.6387156766401199
wandb: FuncNameTest-python MRR (bs=1,000) 0.5152049362036988
wandb: Validation-python MRR (bs=1,000) 0.6069307674807819
wandb: Syncing files in wandb/run-20200510_222737-ptzdp6z2:
wandb: treeraw-2020-05-10-22-27-37-graph.pbtxt
wandb: treeraw-2020-05-10-22-27-37.train_log
wandb: treeraw-2020-05-10-22-27-37_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-10-22-27-37: https://app.wandb.ai/jianguda/CodeSearchNet/runs/ptzdp6z2
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/ptzdp6z2
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-10-22-27-37_model_best.pkl.gz
2020-05-10 22:49:21.749393: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-10 22:49:26.667873: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 5117:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-10 22:49:26.667919: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-10 22:49:26.945197: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-10 22:49:26.945255: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-10 22:49:26.945269: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-10 22:49:26.945381: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 5117:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 193248.48it/s]1156085it [00:19, 58195.45it/s]
Uploading predictions to W&B
NDCG Average: 0.264903674

# normal-option9

Epoch 17 (valid) took 1.42s [processed 16173 samples/second]
Validation: Loss: 1.064865 | MRR: 0.469342
2020-05-10 22:06:23.499339: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-10 22:06:23.499401: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-10 22:06:23.499417: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-10 22:06:23.499427: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-10 22:06:23.499511: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 5117:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.643
FuncNameTest-All MRR (bs=1,000): 0.511
Validation-All MRR (bs=1,000): 0.610
Test-python MRR (bs=1,000): 0.643
FuncNameTest-python MRR (bs=1,000): 0.511
Validation-python MRR (bs=1,000): 0.610

wandb: Waiting for W&B process to finish, PID 2767
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-loss 0.8747873871939854
wandb: \_timestamp 1589148493.492285
wandb: \_step 113
wandb: \_runtime 1478.617031097412
wandb: train-mrr 0.8499791748639449
wandb: train-time-sec 38.07616400718689
wandb: epoch 17
wandb: val-time-sec 1.4220964908599854
wandb: val-loss 1.0648652781610903
wandb: val-mrr 0.4693418260657269
wandb: best_val_mrr_loss 1.063511998757072
wandb: best_val_mrr 0.47225580497409986
wandb: best_epoch 12
wandb: Test-All MRR (bs=1,000) 0.6425101674810189
wandb: FuncNameTest-All MRR (bs=1,000) 0.5114578399524105
wandb: Validation-All MRR (bs=1,000) 0.6103572484572787
wandb: Test-python MRR (bs=1,000) 0.6425101674810189
wandb: FuncNameTest-python MRR (bs=1,000) 0.5114578399524105
wandb: Validation-python MRR (bs=1,000) 0.6103572484572787
wandb: Syncing files in wandb/run-20200510_214336-78me5zhm:
wandb: treeraw-2020-05-10-21-43-36-graph.pbtxt
wandb: treeraw-2020-05-10-21-43-36.train_log
wandb: treeraw-2020-05-10-21-43-36_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-10-21-43-36: https://app.wandb.ai/jianguda/CodeSearchNet/runs/78me5zhm
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/78me5zhm
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-10-21-43-36_model_best.pkl.gz
2020-05-10 22:08:35.699034: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-10 22:08:40.764663: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 5117:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-10 22:08:40.764710: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-10 22:08:41.042224: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-10 22:08:41.042288: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-10 22:08:41.042298: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-10 22:08:41.042408: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 5117:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 193561.33it/s]1156085it [00:19, 57886.00it/s]
Uploading predictions to W&B
NDCG Average: 0.280330876

# normal-option89

Epoch 15 (valid) took 1.45s [processed 15913 samples/second]
Validation: Loss: 1.063589 | MRR: 0.471461
2020-05-10 21:26:02.757404: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-10 21:26:02.757485: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-10 21:26:02.757501: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-10 21:26:02.757512: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-10 21:26:02.757609: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 5117:00:00.0, compute capability: 3.7)Test-All MRR (bs=1,000): 0.641
FuncNameTest-All MRR (bs=1,000): 0.512
Validation-All MRR (bs=1,000): 0.611
Test-python MRR (bs=1,000): 0.641
FuncNameTest-python MRR (bs=1,000): 0.512
Validation-python MRR (bs=1,000): 0.611

wandb: Waiting for W&B process to finish, PID 2558
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-loss 0.8764846627284022
wandb: \_timestamp 1589146074.1052907
wandb: train-mrr 0.847620155556688
wandb: \_runtime 1415.1167886257172
wandb: \_step 101
wandb: epoch 15
wandb: val-time-sec 1.4453341960906982
wandb: val-mrr 0.47146095740276833
wandb: train-time-sec 39.28307485580444
wandb: val-loss 1.0635889924090842
wandb: best_val_mrr_loss 1.0630604236022285
wandb: best_val_mrr 0.47407832535453465
wandb: best_epoch 10
wandb: Test-All MRR (bs=1,000) 0.6410737875550834
wandb: FuncNameTest-All MRR (bs=1,000) 0.5124579325083604
wandb: Validation-All MRR (bs=1,000) 0.611024848920198
wandb: Test-python MRR (bs=1,000) 0.6410737875550834
wandb: FuncNameTest-python MRR (bs=1,000) 0.5124579325083604
wandb: Validation-python MRR (bs=1,000) 0.611024848920198
wandb: Syncing files in wandb/run-20200510_210420-horrctzh:
wandb: treeraw-2020-05-10-21-04-20-graph.pbtxt
wandb: treeraw-2020-05-10-21-04-20.train_log
wandb: treeraw-2020-05-10-21-04-20_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-10-21-04-20: https://app.wandb.ai/jianguda/CodeSearchNet/runs/horrctzh
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/horrctzh
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-10-21-04-20_model_best.pkl.gz
2020-05-10 21:28:32.932140: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-10 21:28:37.845463: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 5117:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-10 21:28:37.845510: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-10 21:28:38.122903: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-10 21:28:38.122968: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-10 21:28:38.122984: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-10 21:28:38.123093: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 5117:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 193306.94it/s]1156085it [00:20, 57725.66it/s]
Uploading predictions to W&B
NDCG Average: 0.291761372

# normal-option17

https://app.wandb.ai/jianguda/CodeSearchNet/runs/36kzy789
jianguda/CodeSearchNet/36kzy789
NDCG Average: 0.269775313

# normal-option157

https://app.wandb.ai/jianguda/CodeSearchNet/runs/fuuwk1rg
jianguda/CodeSearchNet/fuuwk1rg
NDCG Average: 0.269775313

# convert-option157

Epoch 22 (valid) took 1.42s [processed 16208 samples/second]
Validation: Loss: 1.064645 | MRR: 0.473349
2020-05-10 20:34:38.184718: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-10 20:34:38.184784: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-10 20:34:38.184800: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-10 20:34:38.184812: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-10 20:34:38.184901: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 5117:00:00.0, compute capability: 3.7)Test-All MRR (bs=1,000): 0.640
FuncNameTest-All MRR (bs=1,000): 0.519
Validation-All MRR (bs=1,000): 0.613
Test-python MRR (bs=1,000): 0.640
FuncNameTest-python MRR (bs=1,000): 0.519
Validation-python MRR (bs=1,000): 0.613

wandb: Waiting for W&B process to finish, PID 2349
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_step 143
wandb: train-loss 0.8691289106619011
wandb: train-mrr 0.8555626866609147
wandb: \_runtime 1722.5554497241974
wandb: \_timestamp 1589143001.589192
wandb: epoch 22
wandb: val-time-sec 1.4190268516540527
wandb: train-time-sec 38.79450559616089
wandb: val-mrr 0.47334898442807405
wandb: val-loss 1.0646452540936677
wandb: best_val_mrr_loss 1.0652561654215273
wandb: best_val_mrr 0.4744010195524796
wandb: best_epoch 17
wandb: Test-All MRR (bs=1,000) 0.6404499223382737
wandb: FuncNameTest-All MRR (bs=1,000) 0.5190728982181346
wandb: Validation-All MRR (bs=1,000) 0.6127828303923523
wandb: Test-python MRR (bs=1,000) 0.6404499223382737
wandb: FuncNameTest-python MRR (bs=1,000) 0.5190728982181346
wandb: Validation-python MRR (bs=1,000) 0.6127828303923523
wandb: Syncing files in wandb/run-20200510_200800-qyle6khj:
wandb: treeraw-2020-05-10-20-08-00-graph.pbtxt
wandb: treeraw-2020-05-10-20-08-00.train_log
wandb: treeraw-2020-05-10-20-08-00_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-10-20-08-00: https://app.wandb.ai/jianguda/CodeSearchNet/runs/qyle6khj
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/qyle6khj
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-10-20-08-00_model_best.pkl.gz
2020-05-10 20:38:43.448384: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-10 20:38:48.379256: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 5117:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-10 20:38:48.379314: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-10 20:38:48.668263: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-10 20:38:48.668328: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-10 20:38:48.668344: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-10 20:38:48.668454: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 5117:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 193127.53it/s]1156085it [00:20, 56610.19it/s]
Uploading predictions to W&B
NDCG Average: 0.264931444
