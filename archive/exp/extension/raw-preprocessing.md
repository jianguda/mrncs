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

# (normal)(non-len1-words)

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

# (normal)(non-len2-words)

Epoch 13 (valid) took 1.41s [processed 16271 samples/second]
Validation: Loss: 1.064445 | MRR: 0.470010
2020-05-11 12:51:22.127729: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-11 12:51:22.127793: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-11 12:51:22.127809: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-11 12:51:22.127820: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-11 12:51:22.127921: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 7ec4:00:00.0, compute capability: 3.7)Test-All MRR (bs=1,000): 0.636
FuncNameTest-All MRR (bs=1,000): 0.502
Validation-All MRR (bs=1,000): 0.608
Test-python MRR (bs=1,000): 0.636
FuncNameTest-python MRR (bs=1,000): 0.502
Validation-python MRR (bs=1,000): 0.608

wandb: Waiting for W&B process to finish, PID 56
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-mrr 0.8428648032771731
wandb: train-loss 0.8788658808735967
wandb: \_timestamp 1589201592.6712606
wandb: \_runtime 1361.6442196369171
wandb: \_step 89
wandb: val-mrr 0.47001015903638754
wandb: train-time-sec 38.64889740943909
wandb: val-loss 1.064444521199102
wandb: epoch 13
wandb: val-time-sec 1.4135358333587646
wandb: best_val_mrr_loss 1.063230794409047
wandb: best_val_mrr 0.47081900024414064
wandb: best_epoch 8
wandb: Test-All MRR (bs=1,000) 0.6356064615300454
wandb: FuncNameTest-All MRR (bs=1,000) 0.5019714401314472
wandb: Validation-All MRR (bs=1,000) 0.6081785651015543
wandb: Test-python MRR (bs=1,000) 0.6356064615300454
wandb: FuncNameTest-python MRR (bs=1,000) 0.5019714401314472
wandb: Validation-python MRR (bs=1,000) 0.6081785651015543
wandb: Syncing files in wandb/run-20200511_123041-xtmvk0z0:
wandb: treeraw-2020-05-11-12-30-41-graph.pbtxt
wandb: treeraw-2020-05-11-12-30-41.train_log
wandb: treeraw-2020-05-11-12-30-41_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-11-12-30-41: https://app.wandb.ai/jianguda/CodeSearchNet/runs/xtmvk0z0
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/xtmvk0z0
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-11-12-30-41_model_best.pkl.gz
2020-05-11 12:53:37.535302: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 12:53:42.454388: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 7ec4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-11 12:53:42.454433: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-11 12:53:42.730374: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-11 12:53:42.730436: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-11 12:53:42.730452: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-11 12:53:42.730562: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 7ec4:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 197786.67it/s]1156085it [00:19, 58281.05it/s]
Uploading predictions to W&B
NDCG Average: 0.250727438

# (normal)(non-len3-words)

Epoch 12 (valid) took 1.44s [processed 15943 samples/second]
Validation: Loss: 1.073981 | MRR: 0.448111
2020-05-11 13:28:10.554304: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-11 13:28:10.554367: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-11 13:28:10.554383: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-11 13:28:10.554394: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-11 13:28:10.554489: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 7ec4:00:00.0, compute capability: 3.7)Test-All MRR (bs=1,000): 0.607
FuncNameTest-All MRR (bs=1,000): 0.454
Validation-All MRR (bs=1,000): 0.574
Test-python MRR (bs=1,000): 0.607
FuncNameTest-python MRR (bs=1,000): 0.454
Validation-python MRR (bs=1,000): 0.574

wandb: Waiting for W&B process to finish, PID 264
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_step 83
wandb: train-loss 0.8888688796353572
wandb: \_timestamp 1589203800.66355
wandb: \_runtime 1287.5365467071533
wandb: train-mrr 0.8253858349253831
wandb: val-loss 1.07398129546124
wandb: val-time-sec 1.4426188468933105
wandb: val-mrr 0.4481107973845109
wandb: train-time-sec 39.17598795890808
wandb: epoch 12
wandb: best_val_mrr_loss 1.0741339299989783
wandb: best_val_mrr 0.4485954749065897
wandb: best_epoch 7
wandb: Test-All MRR (bs=1,000) 0.60723399423807
wandb: FuncNameTest-All MRR (bs=1,000) 0.45423917755631465
wandb: Validation-All MRR (bs=1,000) 0.574450422107332
wandb: Test-python MRR (bs=1,000) 0.60723399423807
wandb: FuncNameTest-python MRR (bs=1,000) 0.45423917755631465
wandb: Validation-python MRR (bs=1,000) 0.574450422107332
wandb: Syncing files in wandb/run-20200511_130834-45f0d33e:
wandb: treeraw-2020-05-11-13-08-34-graph.pbtxt
wandb: treeraw-2020-05-11-13-08-34.train_log
wandb: treeraw-2020-05-11-13-08-34_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-11-13-08-34: https://app.wandb.ai/jianguda/CodeSearchNet/runs/45f0d33e
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/45f0d33e
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-11-13-08-34_model_best.pkl.gz
2020-05-11 13:30:18.995667: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 13:30:23.890550: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 7ec4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-11-13-08-34_model_best.pkl.gz
2020-05-11 13:30:18.995667: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 13:30:23.890550: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 7ec4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-11 13:30:23.890595: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-11 13:30:24.168349: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edtotalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-11 13:30:23.890595: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-11 13:30:24.168349: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-11 13:30:24.168408: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-11 13:30:24.168424: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-11 13:30:24.168532: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 7ec4:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 195205.22it/s]1156085it [00:19, 57889.72it/s]
Uploading predictions to W&B
NDCG Average: 0.226398244

# (normal)(non-digit)

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

# (normal)(non-punctuation)

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

# (normal)(non-digit&non-punctuation)

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

# (normal)(only-alpha)

Epoch 17 (valid) took 1.41s [processed 16275 samples/second]
Validation: Loss: 1.072790 | MRR: 0.450186
2020-05-12 20:45:00.816502: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-12 20:45:00.816555: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-12 20:45:00.816571: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-12 20:45:00.816582: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-12 20:45:00.816679: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 1926:00:00.0, compute capability: 3.7)Test-All MRR (bs=1,000): 0.611
FuncNameTest-All MRR (bs=1,000): 0.503
Validation-All MRR (bs=1,000): 0.582
Test-python MRR (bs=1,000): 0.611
FuncNameTest-python MRR (bs=1,000): 0.503
Validation-python MRR (bs=1,000): 0.582

wandb: Waiting for W&B process to finish, PID 286
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-loss 0.8842674987119379
wandb: \_timestamp 1589316409.7986412
wandb: \_step 113
wandb: \_runtime 1462.2672295570374
wandb: train-mrr 0.8376852563654335
wandb: epoch 17
wandb: val-time-sec 1.4131662845611572
wandb: val-loss 1.072790353194527
wandb: val-mrr 0.4501864657194718
wandb: train-time-sec 38.31377959251404
wandb: best_val_mrr_loss 1.0710925019305686
wandb: best_val_mrr 0.4507065177585768
wandb: best_epoch 12
wandb: Test-All MRR (bs=1,000) 0.6112968097235321
wandb: FuncNameTest-All MRR (bs=1,000) 0.5030362240837644
wandb: Validation-All MRR (bs=1,000) 0.5819153650359862
wandb: Test-python MRR (bs=1,000) 0.6112968097235321
wandb: FuncNameTest-python MRR (bs=1,000) 0.5030362240837644
wandb: Validation-python MRR (bs=1,000) 0.5819153650359862
wandb: Syncing files in wandb/run-20200512_202229-i8kpg6hl:
wandb: treeraw-2020-05-12-20-22-29-graph.pbtxt
wandb: treeraw-2020-05-12-20-22-29.train_log
wandb: treeraw-2020-05-12-20-22-29_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-12-20-22-29: https://app.wandb.ai/jianguda/CodeSearchNet/runs/i8kpg6hl
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/i8kpg6hl
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-12-20-22-29_model_best.pkl.gz
2020-05-12 20:48:27.106129: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 20:48:31.910828: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 1926:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-12 20:48:31.910875: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-12 20:48:32.189181: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-12 20:48:32.189243: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-12 20:48:32.189260: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-12 20:48:32.189369: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 1926:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 193326.16it/s]1156085it [00:19, 57892.76it/s]
Uploading predictions to W&B
NDCG Average: 0.262802144

# (normal)(non-stop-words)

Epoch 17 (valid) took 1.43s [processed 16120 samples/second]
Validation: Loss: 1.065126 | MRR: 0.469098
2020-05-12 20:02:59.487868: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-12 20:02:59.487923: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-12 20:02:59.487938: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-12 20:02:59.487950: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-12 20:02:59.488044: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 1926:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.643
FuncNameTest-All MRR (bs=1,000): 0.519
Validation-All MRR (bs=1,000): 0.608
Test-python MRR (bs=1,000): 0.643
FuncNameTest-python MRR (bs=1,000): 0.519
Validation-python MRR (bs=1,000): 0.608

wandb: Waiting for W&B process to finish, PID 76
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-mrr 0.850284471085928
wandb: \_step 113
wandb: \_runtime 1569.1861732006073
wandb: train-loss 0.8745202047442927
wandb: \_timestamp 1589313916.2774389
wandb: train-time-sec 38.69535946846008
wandb: val-time-sec 1.4267463684082031
wandb: val-loss 1.0651264397994331
wandb: epoch 17
wandb: val-mrr 0.46909837938391646
wandb: best_val_mrr_loss 1.0634327090304831
wandb: best_val_mrr 0.4717544529127038
wandb: best_epoch 12
wandb: Test-All MRR (bs=1,000) 0.6433320031547024
wandb: FuncNameTest-All MRR (bs=1,000) 0.5188892113446477
wandb: Validation-All MRR (bs=1,000) 0.608186934730255
wandb: Test-python MRR (bs=1,000) 0.6433320031547024
wandb: FuncNameTest-python MRR (bs=1,000) 0.5188892113446477
wandb: Validation-python MRR (bs=1,000) 0.608186934730255
wandb: Syncing files in wandb/run-20200512_193908-wjyfvv7u:
wandb: treeraw-2020-05-12-19-39-08-graph.pbtxt
wandb: treeraw-2020-05-12-19-39-08.train_log
wandb: treeraw-2020-05-12-19-39-08_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-12-19-39-08: https://app.wandb.ai/jianguda/CodeSearchNet/runs/wjyfvv7u
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/wjyfvv7u
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-12-19-39-08_model_best.pkl.gz
2020-05-12 20:05:52.836144: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 20:05:57.653073: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 1926:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-12 20:05:57.653073: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 1926:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-12 20:05:57.653121: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-12 20:05:57.930108: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-12 20:05:57.930169: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-12 20:05:57.930185: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-12 20:05:57.930292: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 1926:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 192748.00it/s]1156085it [00:19, 57944.63it/s]
Uploading predictions to W&B
NDCG Average: 0.311699735

# (normal)(stemming)

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

# (normal)(deduplicate)

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

# (normal)(non-len1-words&non-stop-words)

Epoch 25 (valid) took 1.46s [processed 15739 samples/second]
Validation: Loss: 1.063523 | MRR: 0.471258
2020-05-12 22:11:32.195669: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-12 22:11:32.195730: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-12 22:11:32.195745: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-12 22:11:32.195757: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-12 22:11:32.195845: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 1926:00:00.0, compute capability: 3.7)Test-All MRR (bs=1,000): 0.640
FuncNameTest-All MRR (bs=1,000): 0.508
Validation-All MRR (bs=1,000): 0.612
Test-python MRR (bs=1,000): 0.640
FuncNameTest-python MRR (bs=1,000): 0.508
Validation-python MRR (bs=1,000): 0.612

wandb: Waiting for W&B process to finish, PID 699
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_timestamp 1589321627.743747
wandb: train-loss 0.8691218132243573
wandb: train-mrr 0.8571507618728194
wandb: \_step 161
wandb: \_runtime 1894.1802611351013
wandb: val-mrr 0.4712581070609715
wandb: train-time-sec 39.42647409439087
wandb: epoch 25
wandb: val-loss 1.0635230955870256
wandb: val-time-sec 1.4613220691680908
wandb: best_val_mrr_loss 1.0636138138563738
wandb: best_val_mrr 0.4728614024286685
wandb: best_epoch 20
wandb: Test-All MRR (bs=1,000) 0.6395784763036757
wandb: FuncNameTest-All MRR (bs=1,000) 0.5080391111825671
wandb: Validation-All MRR (bs=1,000) 0.6117720761201667
wandb: Test-python MRR (bs=1,000) 0.6395784763036757
wandb: FuncNameTest-python MRR (bs=1,000) 0.5080391111825671
wandb: Validation-python MRR (bs=1,000) 0.6117720761201667
wandb: Syncing files in wandb/run-20200512_214215-cvhd3h4n:
wandb: treeraw-2020-05-12-21-42-15-graph.pbtxt
wandb: treeraw-2020-05-12-21-42-15.train_log
wandb: treeraw-2020-05-12-21-42-15_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-12-21-42-15: https://app.wandb.ai/jianguda/CodeSearchNet/runs/cvhd3h4n
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/cvhd3h4n
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-12-21-42-15_model_best.pkl.gz
2020-05-12 22:14:59.079947: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 22:15:03.785135: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 1926:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-12 22:15:03.785184: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-12 22:15:04.064800: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-12 22:15:04.064854: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-12 22:15:04.064870: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-12 22:15:04.064985: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 1926:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 193935.54it/s]1156085it [00:19, 58072.88it/s]
Uploading predictions to W&B
NDCG Average: 0.290811424

# (normal)(non-punctuation&non-stop-words)

Epoch 10 (valid) took 1.44s [processed 15947 samples/second]
Validation: Loss: 1.063359 | MRR: 0.471165
2020-05-12 22:53:51.504311: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-12 22:53:51.504375: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-12 22:53:51.504390: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-12 22:53:51.504402: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-12 22:53:51.504503: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 1926:00:00.0, compute capability: 3.7)Test-All MRR (bs=1,000): 0.637
FuncNameTest-All MRR (bs=1,000): 0.507
Validation-All MRR (bs=1,000): 0.608
Test-python MRR (bs=1,000): 0.637
FuncNameTest-python MRR (bs=1,000): 0.507
Validation-python MRR (bs=1,000): 0.608

wandb: Waiting for W&B process to finish, PID 904
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_step 71
wandb: train-mrr 0.8375420065574276
wandb: \_timestamp 1589324167.0405529
wandb: train-loss 0.8821777053249692
wandb: \_runtime 1268.248066186905
wandb: val-mrr 0.4711649422023607
wandb: val-loss 1.0633587889049365
wandb: epoch 10
wandb: train-time-sec 38.88594937324524
wandb: val-time-sec 1.442262887954712
wandb: best_val_mrr_loss 1.0631294820619666
wandb: best_val_mrr 0.47219273840862774
wandb: best_epoch 5
wandb: Test-All MRR (bs=1,000) 0.6372260321041735
wandb: FuncNameTest-All MRR (bs=1,000) 0.5071899089943488
wandb: Validation-All MRR (bs=1,000) 0.6084835757450499
wandb: Test-python MRR (bs=1,000) 0.6372260321041735
wandb: FuncNameTest-python MRR (bs=1,000) 0.5071899089943488
wandb: Validation-python MRR (bs=1,000) 0.6084835757450499
wandb: Syncing files in wandb/run-20200512_223500-p02wtizd:
wandb: treeraw-2020-05-12-22-35-00-graph.pbtxt
wandb: treeraw-2020-05-12-22-35-00.train_log
wandb: treeraw-2020-05-12-22-35-00_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-12-22-35-00: https://app.wandb.ai/jianguda/CodeSearchNet/runs/p02wtizd
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/p02wtizd
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-12-22-35-00_model_best.pkl.gz
2020-05-12 22:58:06.214900: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 22:58:11.010564: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 1926:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-12 22:58:11.010607: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-12 22:58:11.287914: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-12 22:58:11.287976: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-12 22:58:11.287992: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-12 22:58:11.288102: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 1926:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:06<00:00, 191334.69it/s]1156085it [00:20, 57655.97it/s]
Uploading predictions to W&B
NDCG Average: 0.265213984

# (normal)(non-digit&non-punctuation&non-stop-words)

Epoch 13 (valid) took 1.47s [processed 15693 samples/second]
Validation: Loss: 1.063679 | MRR: 0.469795
2020-05-12 21:25:13.673964: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-12 21:25:13.674034: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-12 21:25:13.674050: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-12 21:25:13.674062: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-12 21:25:13.674160: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 1926:00:00.0, compute capability: 3.7)Test-All MRR (bs=1,000): 0.638
FuncNameTest-All MRR (bs=1,000): 0.504
Validation-All MRR (bs=1,000): 0.607
Test-python MRR (bs=1,000): 0.638
FuncNameTest-python MRR (bs=1,000): 0.504
Validation-python MRR (bs=1,000): 0.607

wandb: Waiting for W&B process to finish, PID 490
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_timestamp 1589318849.6297758
wandb: train-loss 0.8778404658569873
wandb: train-mrr 0.8437237954741543
wandb: \_runtime 1401.1649057865143
wandb: \_step 89
wandb: epoch 13
wandb: val-mrr 0.4697946578316067
wandb: val-time-sec 1.4655656814575195
wandb: val-loss 1.063678793285204
wandb: train-time-sec 39.29214286804199
wandb: best_val_mrr_loss 1.0628560885139133
wandb: best_val_mrr 0.4723705178965693
wandb: best_epoch 8
wandb: Test-All MRR (bs=1,000) 0.6379793710581608
wandb: FuncNameTest-All MRR (bs=1,000) 0.5039106395916392
wandb: Validation-All MRR (bs=1,000) 0.607173589812417
wandb: Test-python MRR (bs=1,000) 0.6379793710581608
wandb: FuncNameTest-python MRR (bs=1,000) 0.5039106395916392
wandb: Validation-python MRR (bs=1,000) 0.607173589812417
wandb: Syncing files in wandb/run-20200512_210410-4ha33e11:
wandb: treeraw-2020-05-12-21-04-10-graph.pbtxt
wandb: treeraw-2020-05-12-21-04-10.train_log
wandb: treeraw-2020-05-12-21-04-10_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-12-21-04-10: https://app.wandb.ai/jianguda/CodeSearchNet/runs/4ha33e11
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/4ha33e11
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-12-21-04-10_model_best.pkl.gz
2020-05-12 21:27:51.516519: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 21:27:56.293693: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 1926:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-12 21:27:56.293739: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-12 21:27:56.570275: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-12 21:27:56.570337: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-12 21:27:56.570353: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-12 21:27:51.516519: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 21:27:56.293693: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 1926:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-12 21:27:56.293739: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-12 21:27:56.570275: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-12 21:27:56.570337: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-12 21:27:56.570353: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-12 21:27:56.570461: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 1926:00:00.0, compute capability: 3.7)root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/4ha33e11
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-12-21-04-10_model_best.pkl.gz
2020-05-12 21:27:51.516519: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 21:27:56.293693: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 1926:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-12 21:27:56.293739: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-12 21:27:56.570275: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 ed2020-05-12 21:27:56.293693: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 1926:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-12 21:27:56.293739: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-12 21:27:56.570275: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-12 21:27:56.570337: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-12 21:27:56.570353: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-12 21:27:56.570461: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 1926:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:06<00:00, 191945.20it/s]1156085it [00:20, 57208.88it/s]
Uploading predictions to W&B
NDCG Average: 0.254628082
