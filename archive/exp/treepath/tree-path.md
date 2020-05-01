# NBOW

Epoch 35 (valid) took 2.80s [processed 8224 samples/second]
Validation: Loss: 4.608176 | MRR: 0.277430
2020-05-03 09:42:42.498174: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-03 09:42:42.498234: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-03 09:42:42.498249: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-03 09:42:42.498260: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-03 09:42:42.498342: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: c810:00:00.0, compute ca2020-05-03 09:42:42.498174: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-03 09:42:42.498234: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-03 09:42:42.498249: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-03 09:42:42.498260: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-03 09:42:42.498342: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: c810:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.394
FuncNameTest-All MRR (bs=1,000): 0.313
Validation-All MRR (bs=1,000): 0.360
Test-python MRR (bs=1,000): 0.394
FuncNameTest-python MRR (bs=1,000): 0.313
Validation-python MRR (bs=1,000): 0.360

wandb: Waiting for W&B process to finish, PID 50
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_timestamp 1588499146.9591432
wandb: train-mrr 0.4320679907937652
wandb: \_runtime 4987.847124814987
wandb: \_step 221
wandb: train-loss 3.4455810854735884
wandb: val-loss 4.608176065527874
wandb: val-time-sec 2.7964582443237305
wandb: epoch 35
wandb: val-mrr 0.2774297120467476
wandb: train-time-sec 88.76752281188965
wandb: best_val_mrr_loss 4.6225603352422295
wandb: best_val_mrr 0.27844631692637567
wandb: best_epoch 30
wandb: Test-All MRR (bs=1,000) 0.3940742339648006
wandb: FuncNameTest-All MRR (bs=1,000) 0.3127811260260263
wandb: Validation-All MRR (bs=1,000) 0.360054112087913
wandb: Test-python MRR (bs=1,000) 0.3940742339648006
wandb: FuncNameTest-python MRR (bs=1,000) 0.3127811260260263
wandb: Validation-python MRR (bs=1,000) 0.360054112087913
wandb: Syncing files in wandb/run-20200503_082249-qbb0l57j:
wandb: tree-2020-05-03-08-22-49-graph.pbtxt
wandb: tree-2020-05-03-08-22-49.train_log
wandb: tree-2020-05-03-08-22-49_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced tree-2020-05-03-08-22-49: https://app.wandb.ai/jianguda/CodeSearchNet/runs/qbb0l57j
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/qbb0l57j
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./tree-2020-05-03-08-22-49_model_best.pkl.gz
2020-05-03 09:49:22.017270: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-03 09:49:26.986957: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: c810:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
wandb: Validation-All MRR (bs=1,000) 0.360054112087913
wandb: Test-python MRR (bs=1,000) 0.3940742339648006
wandb: FuncNameTest-python MRR (bs=1,000) 0.3127811260260263
wandb: Validation-python MRR (bs=1,000) 0.360054112087913
wandb: Syncing files in wandb/run-20200503_082249-qbb0l57j:
wandb: Validation-All MRR (bs=1,000) 0.360054112087913
wandb: Test-python MRR (bs=1,000) 0.3940742339648006
wandb: FuncNameTest-python MRR (bs=1,000) 0.3127811260260263
wandb: Validation-python MRR (bs=1,000) 0.360054112087913
wandb: Syncing files in wandb/run-20200503_082249-qbb0l57j:
wandb: tree-2020-05-03-08-22-49-graph.pbtxt
wandb: tree-2020-05-03-08-22-49.train_log
wandb: tree-2020-05-03-08-22-49_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced tree-2020-05-03-08-22-49: https://app.wandb.ai/jianguda/CodeSearchNet/runs/qbb0l57j
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/qbb0l57j  
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./tree-2020-05-03-08-22-49_model_best.pkl.gz
2020-05-03 09:49:22.017270: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-03 09:49:26.986957: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: c810:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-03 09:49:26.987007: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-03 09:49:27.262940: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-03 09:49:27.263006: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-03 09:49:27.263022: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-03 09:49:27.263131: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: c810:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|███████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 195940.60it/s]1156085it [00:19, 58042.19it/s]
Uploading predictions to W&B
NDCG Average: 0.092186092

# NBOW attention

Epoch 32 (valid) took 3.38s [processed 6795 samples/second]
Validation: Loss: 4.594876 | MRR: 0.282271
2020-05-03 11:51:57.643794: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-03 11:51:57.643882: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-03 11:51:57.643899: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-03 11:51:57.643910: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-03 11:51:57.643998: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: c810:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.416
FuncNameTest-All MRR (bs=1,000): 0.343
Validation-All MRR (bs=1,000): 0.389
Test-python MRR (bs=1,000): 0.416
FuncNameTest-python MRR (bs=1,000): 0.343
Validation-python MRR (bs=1,000): 0.389

wandb: Waiting for W&B process to finish, PID 259
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_timestamp 1588506900.790967
wandb: train-loss 3.355389615285744
wandb: train-mrr 0.44383506737866446
wandb: \_step 203
wandb: \_runtime 5829.753486394882
wandb: train-time-sec 123.7327401638031
wandb: val-time-sec 3.3845534324645996
wandb: val-mrr 0.28227132183572523
wandb: epoch 32
wandb: val-loss 4.594875708870266
wandb: best_val_mrr_loss 4.590098546898884
wandb: best_val_mrr 0.282828253041143
wandb: best_epoch 27
wandb: Test-All MRR (bs=1,000) 0.4157099905847242
wandb: FuncNameTest-All MRR (bs=1,000) 0.34257942044519235
wandb: Validation-All MRR (bs=1,000) 0.3894786345781966
wandb: Test-python MRR (bs=1,000) 0.4157099905847242
wandb: FuncNameTest-python MRR (bs=1,000) 0.34257942044519235
wandb: Validation-python MRR (bs=1,000) 0.3894786345781966
wandb: train-loss 3.355389615285744
wandb: train-mrr 0.44383506737866446
wandb: \_step 203
wandb: \_runtime 5829.753486394882
wandb: train-time-sec 123.7327401638031
wandb: val-time-sec 3.3845534324645996
wandb: val-mrr 0.28227132183572523
wandb: epoch 32
wandb: val-loss 4.594875708870266
wandb: best_val_mrr_loss 4.590098546898884
wandb: best_val_mrr 0.282828253041143
wandb: best_epoch 27
wandb: Test-All MRR (bs=1,000) 0.4157099905847242
wandb: FuncNameTest-All MRR (bs=1,000) 0.34257942044519235
wandb: Validation-All MRR (bs=1,000) 0.3894786345781966
wandb: Test-python MRR (bs=1,000) 0.4157099905847242
wandb: FuncNameTest-python MRR (bs=1,000) 0.34257942044519235
wandb: Validation-python MRR (bs=1,000) 0.3894786345781966
wandb: Syncing files in wandb/run-20200503_101752-n5ri9yc8:
wandb: tree-2020-05-03-10-17-52-graph.pbtxt
wandb: tree-2020-05-03-10-17-52.train_log
wandb: tree-2020-05-03-10-17-52_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced tree-2020-05-03-10-17-52: https://app.wandb.ai/jianguda/CodeSearchNet/runs/n5ri9yc8
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/n5ri9yc8
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./tree-2020-05-03-10-17-52_model_best.pkl.gz
2020-05-03 11:56:21.073536: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-03 11:56:26.048450: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: c810:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-03 11:56:26.048498: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-03 11:56:26.325508: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-03 11:56:26.325563: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-03 11:56:26.325583: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-03 11:56:26.325694: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: c810:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|███████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 194942.95it/s]1156085it [00:20, 57627.80it/s]
Uploading predictions to W&B
NDCG Average: 0.092559123

# RNN

Epoch 31 (valid) took 11.88s [processed 1935 samples/second]
Validation: Loss: 3.111318 | MRR: 0.487859
2020-05-03 21:05:00.770082: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-03 21:05:00.770140: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-03 21:05:00.770156: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-03 21:05:00.770167: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-03 21:05:00.770271: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: c810:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.615
FuncNameTest-All MRR (bs=1,000): 0.609
Validation-All MRR (bs=1,000): 0.570
Test-python MRR (bs=1,000): 0.615
FuncNameTest-python MRR (bs=1,000): 0.609
Validation-python MRR (bs=1,000): 0.570

wandb: Waiting for W&B process to finish, PID 676
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-mrr 0.6499773572903235
wandb: \_runtime 16321.842170000076
wandb: train-loss 2.0844939458138736
wandb: \_step 197
wandb: \_timestamp 1588540133.607574
wandb: epoch 31
wandb: val-time-sec 11.88135051727295
wandb: train-time-sec 445.18568205833435
wandb: val-mrr 0.4878585974651834
wandb: val-loss 3.111318422400433
wandb: best_val_mrr_loss 3.1132699406665303
wandb: best_val_mrr 0.4879697544263757
wandb: best_epoch 26
wandb: Test-All MRR (bs=1,000) 0.6154155893499696
wandb: FuncNameTest-All MRR (bs=1,000) 0.6086325177687664
wandb: Validation-All MRR (bs=1,000) 0.5695364771724583
wandb: Test-python MRR (bs=1,000) 0.6154155893499696
wandb: FuncNameTest-python MRR (bs=1,000) 0.6086325177687664
wandb: Validation-python MRR (bs=1,000) 0.5695364771724583
wandb: Syncing files in wandb/run-20200503_163652-z5dphggj:
wandb: tree-2020-05-03-16-36-52-graph.pbtxt
wandb: tree-2020-05-03-16-36-52.train_log
wandb: tree-2020-05-03-16-36-52_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced tree-2020-05-03-16-36-52: https://app.wandb.ai/jianguda/CodeSearchNet/runs/z5dphggj
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/z5dphggj
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./tree-2020-05-03-16-36-52_model_best.pkl.gz
2020-05-03 21:09:14.579768: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-03 21:09:19.529132: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: c810:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-03 21:09:19.529183: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-03 21:09:19.809380: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./tree-2020-05-03-16-36-52_model_best.pkl.gz
2020-05-03 21:09:14.579768: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-03 21:09:19.529132: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: c810:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-03 21:09:19.529183: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-03 21:09:19.809380: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-03 21:09:19.809449: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-03 21:09:19.809466: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-03 21:09:19.809573: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: c810:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|███████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 195878.08it/s]1156085it [00:19, 57835.53it/s]
Uploading predictions to W&B
NDCG Average: 0.139912708

# RNN attention

Epoch 19 (valid) took 13.23s [processed 1738 samples/second]
Validation: Loss: 3.133402 | MRR: 0.483747
2020-05-03 15:32:36.464524: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-03 15:32:36.464588: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-03 15:32:36.464603: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-03 15:32:36.464614: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-03 15:32:36.464711: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: c810:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.609
FuncNameTest-All MRR (bs=1,000): 0.609
Validation-All MRR (bs=1,000): 0.567
Test-python MRR (bs=1,000): 0.609
FuncNameTest-python MRR (bs=1,000): 0.609
Validation-python MRR (bs=1,000): 0.567

wandb: Waiting for W&B process to finish, PID 467
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_timestamp 1588520195.5083153
wandb: train-loss 2.097659603484626
wandb: train-mrr 0.6478051036353251
wandb: \_step 125
wandb: \_runtime 11720.166335344315
wandb: val-loss 3.1334024201268735
wandb: val-mrr 0.4837473834493886
wandb: train-time-sec 486.26394844055176
wandb: val-time-sec 13.226419925689697
wandb: epoch 19
wandb: best_val_mrr_loss 3.103885443314262
wandb: best_val_mrr 0.4876134444527004
wandb: best_epoch 14
wandb: Test-All MRR (bs=1,000) 0.6090821881125048
wandb: FuncNameTest-All MRR (bs=1,000) 0.6085024735062728
wandb: Validation-All MRR (bs=1,000) 0.5668182598569602
wandb: Test-python MRR (bs=1,000) 0.6090821881125048
wandb: FuncNameTest-python MRR (bs=1,000) 0.6085024735062728
wandb: Validation-python MRR (bs=1,000) 0.5668182598569602
wandb: Syncing files in wandb/run-20200503_122116-yvdtk4mn:
wandb: tree-2020-05-03-12-21-16-graph.pbtxt
wandb: val-loss 3.1334024201268735
wandb: val-mrr 0.4837473834493886
wandb: train-time-sec 486.26394844055176
wandb: val-time-sec 13.226419925689697
wandb: epoch 19
wandb: best_val_mrr_loss 3.103885443314262
wandb: best_val_mrr 0.4876134444527004
wandb: best_epoch 14
wandb: Test-All MRR (bs=1,000) 0.6090821881125048
wandb: FuncNameTest-All MRR (bs=1,000) 0.6085024735062728
wandb: Validation-All MRR (bs=1,000) 0.5668182598569602
wandb: Test-python MRR (bs=1,000) 0.6090821881125048
wandb: FuncNameTest-python MRR (bs=1,000) 0.6085024735062728
wandb: Validation-python MRR (bs=1,000) 0.5668182598569602
wandb: Syncing files in wandb/run-20200503_122116-yvdtk4mn:
wandb: tree-2020-05-03-12-21-16-graph.pbtxt
wandb: tree-2020-05-03-12-21-16.train_log
wandb: tree-2020-05-03-12-21-16_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb: Validation-python MRR (bs=1,000) 0.5668182598569602
wandb: Syncing files in wandb/run-20200503_122116-yvdtk4mn:
wandb: tree-2020-05-03-12-21-16-graph.pbtxt
wandb: tree-2020-05-03-12-21-16.train_log
wandb: tree-2020-05-03-12-21-16_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced tree-2020-05-03-12-21-16: https://app.wandb.ai/jianguda/CodeSearchNet/runs/yvdtk4mn
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/yvdtk4mn  
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./tree-2020-05-03-12-21-16_model_best.pkl.gz
2020-05-03 16:01:01.331846: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-03 16:01:06.373668: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: c810:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-03 16:01:06.373717: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-03 16:01:06.655788: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-03 16:01:06.655876: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-03 16:01:06.655893: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-03 16:01:06.656004: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: c810:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|███████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 193656.79it/s]1156085it [00:20, 57494.18it/s]
Uploading predictions to W&B
NDCG Average: 0.189185019
