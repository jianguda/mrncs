# raw

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

# leaf-preprocessing(non-len1-words)

Epoch 15 (valid) took 1.43s [processed 16045 samples/second]
Validation: Loss: 1.070236 | MRR: 0.456979
2020-05-13 11:34:56.499186: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-13 11:34:56.499242: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-13 11:34:56.499257: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-13 11:34:56.499268: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-13 11:34:56.499351: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 713e:00:00.0, compute capability: 3.7)Test-All MRR (bs=1,000): 0.616
FuncNameTest-All MRR (bs=1,000): 0.575
Validation-All MRR (bs=1,000): 0.594
Test-python MRR (bs=1,000): 0.616
FuncNameTest-python MRR (bs=1,000): 0.575
Validation-python MRR (bs=1,000): 0.594

wandb: Waiting for W&B process to finish, PID 274
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_step 101
wandb: \_runtime 2052.1392505168915
wandb: \_timestamp 1589370202.5854619
wandb: train-mrr 0.8434713368832486
wandb: train-loss 0.8782250222939889
wandb: epoch 15
wandb: val-time-sec 1.4333908557891846
wandb: val-loss 1.070236112760461
wandb: train-time-sec 40.06779479980469
wandb: val-mrr 0.45697907688306727
wandb: best_val_mrr_loss 1.0696853036465852
wandb: best_val_mrr 0.459366931417714
wandb: best_epoch 10
wandb: Test-All MRR (bs=1,000) 0.6161964633783772
wandb: FuncNameTest-All MRR (bs=1,000) 0.5750612911450822
wandb: Validation-All MRR (bs=1,000) 0.594268878819137
wandb: Test-python MRR (bs=1,000) 0.6161964633783772
wandb: FuncNameTest-python MRR (bs=1,000) 0.5750612911450822
wandb: Validation-python MRR (bs=1,000) 0.594268878819137
wandb: Syncing files in wandb/run-20200513_110912-9rkw0jd7:
wandb: treeleaf-2020-05-13-11-09-12-graph.pbtxt
wandb: treeleaf-2020-05-13-11-09-12.train_log
wandb: treeleaf-2020-05-13-11-09-12_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeleaf-2020-05-13-11-09-12: https://app.wandb.ai/jianguda/CodeSearchNet/runs/9rkw0jd7
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/9rkw0jd7
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeleaf-2020-05-13-11-09-12_model_best.pkl.gz
2020-05-13 11:43:47.006493: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 11:43:51.804988: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 713e:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-13 11:43:51.805035: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-13 11:43:52.077447: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-13 11:43:52.077511: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-13 11:43:52.077526: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-13 11:43:52.077633: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 713e:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 196617.59it/s]1156085it [00:20, 56856.60it/s]
Uploading predictions to W&B

# leaf-preprocessing(non-stop-words)

Epoch 10 (valid) took 1.41s [processed 16261 samples/second]
Validation: Loss: 1.070220 | MRR: 0.458410
2020-05-13 13:15:03.025216: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-13 13:15:03.025280: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-13 13:15:03.025294: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-13 13:15:03.025305: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-13 13:15:03.025404: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 713e:00:00.0, compute capability: 3.7)Test-All MRR (bs=1,000): 0.612
FuncNameTest-All MRR (bs=1,000): 0.573
Validation-All MRR (bs=1,000): 0.592
Test-python MRR (bs=1,000): 0.612
FuncNameTest-python MRR (bs=1,000): 0.573
Validation-python MRR (bs=1,000): 0.592

wandb: Waiting for W&B process to finish, PID 478
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-mrr 0.8336081725185357
wandb: train-loss 0.8843383817996794
wandb: \_runtime 1935.0882823467255
wandb: \_step 71
wandb: \_timestamp 1589376247.524795
wandb: val-mrr 0.4584103731901749
wandb: val-time-sec 1.4143702983856201
wandb: epoch 10
wandb: train-time-sec 40.18769574165344
wandb: val-loss 1.0702199417611826
wandb: best_val_mrr_loss 1.0686701069707456
wandb: best_val_mrr 0.4590649201766304
wandb: best_epoch 5
wandb: Test-All MRR (bs=1,000) 0.6116498774181688
wandb: FuncNameTest-All MRR (bs=1,000) 0.5727390513933928
wandb: Validation-All MRR (bs=1,000) 0.5918950030417786
wandb: Test-python MRR (bs=1,000) 0.6116498774181688
wandb: FuncNameTest-python MRR (bs=1,000) 0.5727390513933928
wandb: Validation-python MRR (bs=1,000) 0.5918950030417786
wandb: Syncing files in wandb/run-20200513_125153-zx3x3yez:
wandb: treeleaf-2020-05-13-12-51-53-graph.pbtxt
wandb: treeleaf-2020-05-13-12-51-53.train_log
wandb: treeleaf-2020-05-13-12-51-53_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeleaf-2020-05-13-12-51-53: https://app.wandb.ai/jianguda/CodeSearchNet/runs/zx3x3yez
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/zx3x3yez
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeleaf-2020-05-13-12-51-53_model_best.pkl.gz
2020-05-13 13:36:17.577466: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-13 13:36:22.337051: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 713e:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-13 13:36:22.337100: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-13 13:36:22.611728: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-13 13:36:22.611795: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-13 13:36:22.611811: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-13 13:36:22.611922: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 713e:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 195806.14it/s]1156085it [00:19, 57832.39it/s]
Uploading predictions to W&B
NDCG Average: 0.259327673

# path

# path-preprocessing

# tree

# tree-attention

# tree-preprocessing
