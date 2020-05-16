# raw softmax

Epoch 12 (valid) took 1.45s [processed 15845 samples/second]
Validation: Loss: 4.375512 | MRR: 0.379371
2020-05-11 14:07:34.199791: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-11 14:07:34.199854: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-11 14:07:34.199868: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-11 14:07:34.199879: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-11 14:07:34.199976: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 7ec4:00:00.0, compute capability: 3.7)Test-All MRR (bs=1,000): 0.582
FuncNameTest-All MRR (bs=1,000): 0.467
Validation-All MRR (bs=1,000): 0.554
Test-python MRR (bs=1,000): 0.582
FuncNameTest-python MRR (bs=1,000): 0.467
Validation-python MRR (bs=1,000): 0.554

wandb: Waiting for W&B process to finish, PID 57
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_timestamp 1589206164.136222
wandb: \_runtime 1295.0329637527466
wandb: train-loss 1.2299168115680656
wandb: train-mrr 0.8051595023442241
wandb: \_step 83
wandb: val-loss 4.375511729198953
wandb: val-time-sec 1.4514923095703125
wandb: train-time-sec 38.726616859436035
wandb: val-mrr 0.37937126955778705
wandb: epoch 12
wandb: best_val_mrr_loss 4.108651472174603
wandb: best_val_mrr 0.3831314975904382
wandb: best_epoch 7
wandb: Test-All MRR (bs=1,000) 0.5816936133747298
wandb: FuncNameTest-All MRR (bs=1,000) 0.4668331772042539
wandb: Validation-All MRR (bs=1,000) 0.5536940765760092
wandb: Test-python MRR (bs=1,000) 0.5816936133747298
wandb: FuncNameTest-python MRR (bs=1,000) 0.4668331772042539
wandb: Validation-python MRR (bs=1,000) 0.5536940765760092
wandb: Syncing files in wandb/run-20200511_134750-2nhe1z7d:
wandb: treeraw-2020-05-11-13-47-50-graph.pbtxt
wandb: treeraw-2020-05-11-13-47-50.train_log
wandb: treeraw-2020-05-11-13-47-50_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-11-13-47-50: https://app.wandb.ai/jianguda/CodeSearchNet/runs/2nhe1z7d
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/2nhe1z7d
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-11-13-47-50_model_best.pkl.gz
2020-05-11 14:09:50.715772: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 14:09:55.652910: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 7ec4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-11 14:09:55.652955: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-11 14:09:55.927757: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-11 14:09:55.927819: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-11 14:09:55.927835: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-11 14:09:55.927947: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 7ec4:00:00.0, compute capability: 3.7)Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 195000.24it/s]1156085it [00:19, 57879.71it/s]
Uploading predictions to W&B
NDCG Average: 0.172546973

# raw cosine

Epoch 13 (valid) took 1.41s [processed 16354 samples/second]
Validation: Loss: 1.065036 | MRR: 0.468127
2020-05-11 18:21:31.941823: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-11 18:21:31.941907: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-11 18:21:31.941923: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-11 18:21:31.941933: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-11 18:21:31.942031: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 7ec4:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.638
FuncNameTest-All MRR (bs=1,000): 0.518
Validation-All MRR (bs=1,000): 0.606
Test-python MRR (bs=1,000): 0.638
FuncNameTest-python MRR (bs=1,000): 0.518
Validation-python MRR (bs=1,000): 0.606

wandb: Waiting for W&B process to finish, PID 881
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_timestamp 1589221401.0904858
wandb: \_step 89
wandb: train-mrr 0.8437633864023153
wandb: train-loss 0.8792643130404277
wandb: \_runtime 1315.5687441825867
wandb: val-time-sec 1.4063692092895508
wandb: val-mrr 0.46812699823794157
wandb: train-time-sec 38.489323139190674
wandb: epoch 13
wandb: val-loss 1.0650362242823062
wandb: best_val_mrr_loss 1.0642786077831103
wandb: best_val_mrr 0.4696956979502802
wandb: best_epoch 8
wandb: Test-All MRR (bs=1,000) 0.6378405845031299
wandb: FuncNameTest-All MRR (bs=1,000) 0.518248812212671
wandb: Validation-All MRR (bs=1,000) 0.6057542802435112
wandb: Test-python MRR (bs=1,000) 0.6378405845031299
wandb: FuncNameTest-python MRR (bs=1,000) 0.518248812212671
wandb: Validation-python MRR (bs=1,000) 0.6057542802435112
wandb: Syncing files in wandb/run-20200511_180127-877djtj5:
wandb: treeraw-2020-05-11-18-01-27-graph.pbtxt
wandb: treeraw-2020-05-11-18-01-27.train_log
wandb: treeraw-2020-05-11-18-01-27_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-11-18-01-27: https://app.wandb.ai/jianguda/CodeSearchNet/runs/877djtj5
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/877djtj5
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-11-18-01-27_model_best.pkl.gz
2020-05-11 18:23:44.817967: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 18:23:49.641964: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 7ec4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-11 18:23:49.642009: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-11 18:23:49.921449: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-11 18:23:49.921512: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-11 18:23:49.921527: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-11 18:23:49.921638: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 7ec4:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 196525.05it/s]1156085it [00:19, 57911.13it/s]
Uploading predictions to W&B
NDCG Average: 0.292252404

# raw max-margin

Epoch 12 (valid) took 1.42s [processed 16146 samples/second]
Validation: Loss: 1.611590 | MRR: 0.390383
2020-05-11 14:46:05.739728: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-11 14:46:05.739790: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-11 14:46:05.739805: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-11 14:46:05.739816: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-11 14:46:05.739905: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 7ec4:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.583
FuncNameTest-All MRR (bs=1,000): 0.476
Validation-All MRR (bs=1,000): 0.551
Test-python MRR (bs=1,000): 0.583
FuncNameTest-python MRR (bs=1,000): 0.476
Validation-python MRR (bs=1,000): 0.551

wandb: Waiting for W&B process to finish, PID 262
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_runtime 1276.7082316875458
wandb: train-mrr 0.7964783285196545
wandb: train-loss 0.6556637420237643
wandb: \_step 83
wandb: \_timestamp 1589208475.4803205
wandb: val-loss 1.611590053724206
wandb: train-time-sec 38.53566861152649
wandb: val-mrr 0.3903830161716627
wandb: epoch 12
wandb: val-time-sec 1.4245004653930664
wandb: best_val_mrr_loss 1.4890402814616328
wandb: best_val_mrr 0.3950152004076087
wandb: best_epoch 7
wandb: Test-All MRR (bs=1,000) 0.5834037742193405
wandb: FuncNameTest-All MRR (bs=1,000) 0.47619932276524574
wandb: Validation-All MRR (bs=1,000) 0.5513148013486656
wandb: Test-python MRR (bs=1,000) 0.5834037742193405
wandb: FuncNameTest-python MRR (bs=1,000) 0.47619932276524574
wandb: Validation-python MRR (bs=1,000) 0.5513148013486656
wandb: Syncing files in wandb/run-20200511_142640-vsy1q4en:
wandb: treeraw-2020-05-11-14-26-40-graph.pbtxt
wandb: treeraw-2020-05-11-14-26-40.train_log
wandb: treeraw-2020-05-11-14-26-40_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-11-14-26-40: https://app.wandb.ai/jianguda/CodeSearchNet/runs/vsy1q4en
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/vsy1q4en
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-11-14-26-40_model_best.pkl.gz
2020-05-11 14:48:21.693460: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 14:48:26.525659: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 7ec4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-11 14:48:26.525706: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-11 14:48:26.801471: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-11 14:48:26.801534: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-11 14:48:26.801551: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-11 14:48:26.801671: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 7ec4:00:00.0, compute capability: 3.7)Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 195234.45it/s]1156085it [00:20, 57181.36it/s]
Uploading predictions to W&B
NDCG Average: 0.175590978

# raw triplet

Epoch 31 (valid) took 2.34s [processed 9833 samples/second]
Validation: Loss: 0.646401 | MRR: 0.420742
2020-05-11 16:36:49.597287: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-11 16:36:49.597349: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-11 16:36:49.597363: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-11 16:36:49.597374: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-11 16:36:49.597463: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 7ec4:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.611
FuncNameTest-All MRR (bs=1,000): 0.493
Validation-All MRR (bs=1,000): 0.575
Test-python MRR (bs=1,000): 0.611
FuncNameTest-python MRR (bs=1,000): 0.493
Validation-python MRR (bs=1,000): 0.575

wandb: Waiting for W&B process to finish, PID 469
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_step 197
wandb: train-loss 0.27098001239513886
wandb: train-mrr 0.8152768621352112
wandb: \_timestamp 1589215119.6832435
wandb: \_runtime 5772.710038661957
wandb: val-time-sec 2.3388476371765137
wandb: train-time-sec 152.754159450531
wandb: val-mrr 0.4207416408372962
wandb: epoch 31
wandb: val-loss 0.646400801513506
wandb: best_val_mrr_loss 0.6401179458784021
wandb: best_val_mrr 0.4219675392482592
wandb: best_epoch 26
wandb: Test-All MRR (bs=1,000) 0.6114543703166584
wandb: FuncNameTest-All MRR (bs=1,000) 0.49342328170029753
wandb: Validation-All MRR (bs=1,000) 0.5745320088508006
wandb: Test-python MRR (bs=1,000) 0.6114543703166584
wandb: FuncNameTest-python MRR (bs=1,000) 0.49342328170029753
wandb: Validation-python MRR (bs=1,000) 0.5745320088508006
wandb: Syncing files in wandb/run-20200511_150228-9dxgw9kk:
wandb: treeraw-2020-05-11-15-02-28-graph.pbtxt
wandb: treeraw-2020-05-11-15-02-28.train_log
wandb: treeraw-2020-05-11-15-02-28_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-11-15-02-28: https://app.wandb.ai/jianguda/CodeSearchNet/runs/9dxgw9kk
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/9dxgw9kk
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-11-15-02-28_model_best.pkl.gz
2020-05-11 16:39:04.658879: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 16:39:09.496323: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 7ec4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-11 16:39:09.496372: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-11 16:39:09.782852: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-11 16:39:09.782918: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-11 16:39:09.782933: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-11 16:39:09.783042: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 7ec4:00:00.0, compute capability: 3.7)Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 193349.66it/s]1156085it [00:20, 57589.98it/s]
Uploading predictions to W&B
NDCG Average: 0.266458079

# row-KNN

Epoch 15 (valid) took 1.48s [processed 15495 samples/second]
Validation: Loss: 1.063652 | MRR: 0.471070
2020-05-18 20:41:13.980681: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-18 20:41:13.980737: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-18 20:41:13.980752: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-18 20:41:13.980763: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-18 20:41:13.980845: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10761 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 3476:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.643
FuncNameTest-All MRR (bs=1,000): 0.521
Validation-All MRR (bs=1,000): 0.611
Test-python MRR (bs=1,000): 0.643
FuncNameTest-python MRR (bs=1,000): 0.521
Validation-python MRR (bs=1,000): 0.611

wandb: Waiting for W&B process to finish, PID 54
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-loss 0.8772346663822248
wandb: train-mrr 0.8472810202293025
wandb: \_runtime 1449.498173713684
wandb: \_step 101
wandb: \_timestamp 1589834585.2846868
wandb: val-mrr 0.4710700471297554
wandb: epoch 15
wandb: train-time-sec 40.00382852554321
wandb: val-loss 1.0636521215024202
wandb: val-time-sec 1.4842910766601562
wandb: best_val_mrr_loss 1.0633603697237761
wandb: best_val_mrr 0.4724775390625
wandb: best_epoch 10
wandb: Test-All MRR (bs=1,000) 0.6428236145615925
wandb: FuncNameTest-All MRR (bs=1,000) 0.520749357434897
wandb: Validation-All MRR (bs=1,000) 0.6111837302175785
wandb: Test-python MRR (bs=1,000) 0.6428236145615925
wandb: FuncNameTest-python MRR (bs=1,000) 0.520749357434897
wandb: Validation-python MRR (bs=1,000) 0.6111837302175785
wandb: Syncing files in wandb/run-20200518_201857-8y9ixqz5:
wandb: treeraw-2020-05-18-20-18-57-graph.pbtxt
wandb: treeraw-2020-05-18-20-18-57.train_log
wandb: treeraw-2020-05-18-20-18-57_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-18-20-18-57: https://app.wandb.ai/jianguda/CodeSearchNet/runs/8y9ixqz5
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/8y9ixqz5
NDCG Average: 0.499120563

# CNN-cosine

Epoch 12 (valid) took 3.93s [processed 5849 samples/second]
Validation: Loss: 1.000094 | MRR: 0.011335
2020-05-11 19:52:44.808160: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-11 19:52:44.808223: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-11 19:52:44.808238: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-11 19:52:44.808249: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-11 19:52:44.808333: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 7ec4:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.008
FuncNameTest-All MRR (bs=1,000): 0.008
Validation-All MRR (bs=1,000): 0.008
Test-python MRR (bs=1,000): 0.008
FuncNameTest-python MRR (bs=1,000): 0.008
Validation-python MRR (bs=1,000): 0.008

wandb: Waiting for W&B process to finish, PID 1193
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-loss 1.0006620063654428
wandb: \_runtime 3688.4801852703094
wandb: \_timestamp 1589226887.9488523
wandb: \_step 83
wandb: train-mrr 0.010353357318535593
wandb: val-time-sec 3.931854248046875
wandb: train-time-sec 218.98445844650269
wandb: epoch 12
wandb: val-mrr 0.011334702989329462
wandb: val-loss 1.000093532645184
wandb: best_val_mrr_loss 1.0001308917999268
wandb: best_val_mrr 0.1707715994793436
wandb: best_epoch 7
wandb: Test-All MRR (bs=1,000) 0.007612385978962976
wandb: FuncNameTest-All MRR (bs=1,000) 0.007551561260681489
wandb: Validation-All MRR (bs=1,000) 0.007602137277146758
wandb: Test-python MRR (bs=1,000) 0.007612385978962976
wandb: FuncNameTest-python MRR (bs=1,000) 0.007551561260681489
wandb: Validation-python MRR (bs=1,000) 0.007602137277146758
wandb: Syncing files in wandb/run-20200511_185321-80nztbz9:
wandb: treeraw-2020-05-11-18-53-21-graph.pbtxt
wandb: treeraw-2020-05-11-18-53-21.train_log
wandb: treeraw-2020-05-11-18-53-21_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-11-18-53-21: https://app.wandb.ai/jianguda/CodeSearchNet/runs/80nztbz9
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/80nztbz9
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-11-18-53-21_model_best.pkl.gz
2020-05-11 19:59:29.467424: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-11 19:59:34.306276: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 7ec4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-11 19:59:34.306324: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-11 19:59:34.585146: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-11 19:59:34.585208: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-11 19:59:34.585225: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-11 19:59:34.585331: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 7ec4:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 195098.42it/s]1156085it [00:19, 58058.36it/s]
Uploading predictions to W&B
NDCG Average: 0.041662995

# RNN-cosine

Epoch 16 (valid) took 10.42s [processed 2207 samples/second]
Validation: Loss: 1.000251 | MRR: 0.200405
2020-05-12 13:01:40.185567: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-12 13:01:40.185629: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-12 13:01:40.185644: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-12 13:01:40.185655: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-12 13:01:40.185745: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 1926:00:00.0, compute capability: 3.7)Test-All MRR (bs=1,000): 0.008
FuncNameTest-All MRR (bs=1,000): 0.008
Validation-All MRR (bs=1,000): 0.008
Test-python MRR (bs=1,000): 0.008
FuncNameTest-python MRR (bs=1,000): 0.008
Validation-python MRR (bs=1,000): 0.008

wandb: Waiting for W&B process to finish, PID 56
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_timestamp 1589288663.0228195
wandb: train-mrr 0.24082739691132482
wandb: train-loss 0.9981757924684043
wandb: \_step 107
wandb: \_runtime 7619.634813785553
wandb: epoch 16
wandb: train-time-sec 386.30909752845764
wandb: val-mrr 0.20040488831893258
wandb: val-time-sec 10.421247720718384
wandb: val-loss 1.0002508889073911
wandb: best_val_mrr_loss 1.0000688666882722
wandb: best_val_mrr 0.24384867527173912
wandb: best_epoch 11
wandb: Test-All MRR (bs=1,000) 0.007805075249383647
wandb: FuncNameTest-All MRR (bs=1,000) 0.007684403977688315
wandb: Validation-All MRR (bs=1,000) 0.0077467231358031826
wandb: Test-python MRR (bs=1,000) 0.007805075249383647
wandb: FuncNameTest-python MRR (bs=1,000) 0.007684403977688315
wandb: Validation-python MRR (bs=1,000) 0.0077467231358031826
wandb: Syncing files in wandb/run-20200512_105735-8zyax1yz:
wandb: treeraw-2020-05-12-10-57-34-graph.pbtxt
wandb: treeraw-2020-05-12-10-57-34.train_log
wandb: treeraw-2020-05-12-10-57-34_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeraw-2020-05-12-10-57-34: https://app.wandb.ai/jianguda/CodeSearchNet/runs/8zyax1yz
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/8zyax1yz
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeraw-2020-05-12-10-57-34_model_best.pkl.gz
2020-05-12 13:15:23.969261: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-12 13:15:28.864865: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 1926:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-12 13:15:28.864915: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-12 13:15:29.143580: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-12 13:15:29.143640: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-12 13:15:29.143657: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-12 13:15:29.143766: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 1926:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 197131.38it/s]1156085it [00:19, 58360.88it/s]
Uploading predictions to W&B
NDCG Average: 0.066400327

# BERT-cosine

...
