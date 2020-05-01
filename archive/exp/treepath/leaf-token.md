# NBOW

Epoch 32 (valid) took 2.75s [processed 8374 samples/second]
Validation: Loss: 4.310367 | MRR: 0.346463
2020-04-25 11:35:52.256624: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-25 11:35:52.256687: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-25 11:35:52.256703: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-25 11:35:52.256714: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-25 11:35:52.256803: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 9780:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.532
FuncNameTest-All MRR (bs=1,000): 0.432
Validation-All MRR (bs=1,000): 0.497
Test-python MRR (bs=1,000): 0.532
FuncNameTest-python MRR (bs=1,000): 0.432
Validation-python MRR (bs=1,000): 0.497

wandb: Waiting for W&B process to finish, PID 48
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_timestamp 1587814702.6169987
wandb: train-mrr 0.6173275139077196
wandb: train-loss 2.2465361026884283
wandb: \_step 203
wandb: \_runtime 3900.0309431552887
wandb: val-loss 4.310366754946501
wandb: epoch 32
wandb: val-time-sec 2.74627423286438
wandb: train-time-sec 88.0218677520752
wandb: val-mrr 0.34646273538340694
wandb: best_val_mrr_loss 4.2975712859112285
wandb: val-time-sec 2.74627423286438
wandb: train-time-sec 88.0218677520752
wandb: val-mrr 0.34646273538340694
wandb: best_val_mrr_loss 4.2975712859112285
wandb: best_val_mrr 0.34772780841329826
wandb: best_epoch 27
wandb: Test-All MRR (bs=1,000) 0.5320587353990976
wandb: FuncNameTest-All MRR (bs=1,000) 0.4319475057124949
wandb: Validation-All MRR (bs=1,000) 0.4974595728869418
wandb: Test-python MRR (bs=1,000) 0.5320587353990976
wandb: FuncNameTest-python MRR (bs=1,000) 0.4319475057124949
wandb: Validation-python MRR (bs=1,000) 0.4974595728869418
wandb: Syncing files in wandb/run-20200425_103323-84zwhdw4:
wandb: tree-2020-04-25-10-33-23-graph.pbtxt
wandb: tree-2020-04-25-10-33-23.train_log
wandb: tree-2020-04-25-10-33-23_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced tree-2020-04-25-10-33-23: https://app.wandb.ai/jianguda/CodeSearchNet/runs/84zwhdw4
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/84zwhdw4  
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./tree-2020-04-25-10-33-23_model_best.pkl.gz
2020-04-25 11:43:44.026641: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-25 11:43:48.908568: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 9780:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-25 11:43:48.908614: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-25 11:43:49.183956: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-25 11:43:49.184012: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
wandb: tree-2020-04-25-10-33-23_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced tree-2020-04-25-10-33-23: https://app.wandb.ai/jianguda/CodeSearchNet/runs/84zwhdw4
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/84zwhdw4  
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./tree-2020-04-25-10-33-23_model_best.pkl.gz
2020-04-25 11:43:44.026641: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-25 11:43:48.908568: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 9780:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-25 11:43:48.908614: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-25 11:43:49.183956: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-25 11:43:49.184012: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-25 11:43:49.184028: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-25 11:43:49.184136: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 9780:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 200957.20it/s]1156085it [00:20, 57231.05it/s]
Uploading predictions to W&B
NDCG Average: 0.132435162

# NBOW attention

Epoch 29 (valid) took 3.35s [processed 6874 samples/second]
Validation: Loss: 4.236795 | MRR: 0.349014
2020-04-25 13:37:09.164928: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-25 13:37:09.164994: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-25 13:37:09.165009: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-25 13:37:09.165020: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-25 13:37:09.165111: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 9780:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.540
FuncNameTest-All MRR (bs=1,000): 0.448
Validation-All MRR (bs=1,000): 0.506
Test-python MRR (bs=1,000): 0.540
FuncNameTest-python MRR (bs=1,000): 0.448
Validation-python MRR (bs=1,000): 0.506

wandb: Waiting for W&B process to finish, PID 48
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-loss 2.217941819464119
wandb: \_runtime 4422.420333623886
wandb: \_timestamp 1587821979.1241062
wandb: \_step 185
wandb: train-mrr 0.6209053741751366
wandb: train-time-sec 120.0040876865387
wandb: epoch 29
wandb: val-loss 4.236794720525327
wandb: val-time-sec 3.3459033966064453
wandb: val-mrr 0.3490137189782184
wandb: best_val_mrr_loss 4.2540039186892304
wandb: best_val_mrr 0.34920158850628397
wandb: best_epoch 24
wandb: Test-All MRR (bs=1,000) 0.5395651137013103
wandb: FuncNameTest-All MRR (bs=1,000) 0.44814100382425665
wandb: Validation-All MRR (bs=1,000) 0.5056592911195057
wandb: Test-python MRR (bs=1,000) 0.5395651137013103
wandb: FuncNameTest-python MRR (bs=1,000) 0.44814100382425665
wandb: Validation-python MRR (bs=1,000) 0.5056592911195057
wandb: Syncing files in wandb/run-20200425_122557-gvcot972:
wandb: tree-2020-04-25-12-25-57-graph.pbtxt
wandb: tree-2020-04-25-12-25-57.train_log
wandb: tree-2020-04-25-12-25-57_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced tree-2020-04-25-12-25-57: https://app.wandb.ai/jianguda/CodeSearchNet/runs/gvcot972
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/gvcot972
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./tree-2020-04-25-12-25-57_model_best.pkl.gz
2020-04-25 13:40:31.410465: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-25 13:40:36.271828: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 9780:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-25 13:40:36.271872: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-25 13:40:36.543540: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-25 13:40:36.543598: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-25 13:40:36.543617: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-25 13:40:36.543725: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 9780:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 195214.73it/s]1156085it [00:19, 57903.95it/s]
Uploading predictions to W&B
NDCG Average: 0.141742677

# CNN

Epoch 35 (valid) took 5.35s [processed 4298 samples/second]
Validation: Loss: 3.958797 | MRR: 0.355874
2020-04-25 22:41:13.706835: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-25 22:41:13.706896: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-25 22:41:13.706911: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-25 22:41:13.706923: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-25 22:41:13.707007: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: e9e1:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.276
FuncNameTest-All MRR (bs=1,000): 0.345
Validation-All MRR (bs=1,000): 0.252
Test-python MRR (bs=1,000): 0.276
FuncNameTest-python MRR (bs=1,000): 0.345
Validation-python MRR (bs=1,000): 0.252

wandb: Waiting for W&B process to finish, PID 50
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_step 221
wandb: train-mrr 0.5083195357091218
wandb: \_timestamp 1587854636.6307259
wandb: \_runtime 10820.222137212753
wandb: train-loss 2.8603884068507592
wandb: val-mrr 0.355873582922894
wandb: val-loss 3.9587967603102974
wandb: val-time-sec 5.351243495941162
wandb: epoch 35
wandb: train-time-sec 270.62750816345215
wandb: best_val_mrr_loss 3.9388750221418296
wandb: best_val_mrr 0.36005307205863624
wandb: best_epoch 30
wandb: Test-All MRR (bs=1,000) 0.27586086262807147
wandb: FuncNameTest-All MRR (bs=1,000) 0.34525544912986067
wandb: Validation-All MRR (bs=1,000) 0.25200654103857145
wandb: Test-python MRR (bs=1,000) 0.27586086262807147
wandb: FuncNameTest-python MRR (bs=1,000) 0.34525544912986067
wandb: Validation-python MRR (bs=1,000) 0.25200654103857145
wandb: Syncing files in wandb/run-20200425_194337-j6yslw7s:
wandb: tree-2020-04-25-19-43-37-graph.pbtxt
wandb: tree-2020-04-25-19-43-37.train_log
wandb: tree-2020-04-25-19-43-37_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced tree-2020-04-25-19-43-37: https://app.wandb.ai/jianguda/CodeSearchNet/runs/j6yslw7s
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/j6yslw7s
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./tree-2020-04-25-19-43-37_model_best.pkl.gz
2020-04-26 01:46:35.263823: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-26 01:46:40.100320: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: e9e1:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-26 01:46:40.100368: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-26 01:46:40.372997: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-26 01:46:40.373061: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-26 01:46:40.373078: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-26 01:46:40.373184: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: e9e1:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 199129.90it/s]1156085it [00:19, 57976.60it/s]
Uploading predictions to W&B
NDCG Average: 0.077165589

# CNN attention

Epoch 31 (valid) took 5.38s [processed 4271 samples/second]
Validation: Loss: 4.252490 | MRR: 0.305305
2020-04-26 04:58:18.101255: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-26 04:58:18.101313: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-26 04:58:18.101328: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-26 04:58:18.101335: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-26 04:58:18.101454: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: e9e1:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.208
FuncNameTest-All MRR (bs=1,000): 0.240
Validation-All MRR (bs=1,000): 0.190
Test-python MRR (bs=1,000): 0.208
FuncNameTest-python MRR (bs=1,000): 0.240
Validation-python MRR (bs=1,000): 0.190

wandb: Waiting for W&B process to finish, PID 263
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_step 197
wandb: \_timestamp 1587877258.1563706
wandb: train-loss 3.2960684826073137
wandb: train-mrr 0.43481431401817544
wandb: \_runtime 9959.231295585632
wandb: val-loss 4.252490043640137
wandb: val-mrr 0.3053050079345703
wandb: epoch 31
wandb: train-time-sec 279.0783152580261
wandb: val-time-sec 5.384861707687378
wandb: best_val_mrr_loss 4.254505934922592
wandb: best_val_mrr 0.31109731458581014
wandb: best_epoch 26
wandb: Test-All MRR (bs=1,000) 0.20805823825116485
wandb: FuncNameTest-All MRR (bs=1,000) 0.24006087573870485
wandb: Validation-All MRR (bs=1,000) 0.19034929220508626
wandb: Test-python MRR (bs=1,000) 0.20805823825116485
wandb: FuncNameTest-python MRR (bs=1,000) 0.24006087573870485
wandb: Validation-python MRR (bs=1,000) 0.19034929220508626
wandb: Syncing files in wandb/run-20200426_021500-mimpe060:
wandb: tree-2020-04-26-02-15-00-graph.pbtxt
wandb: tree-2020-04-26-02-15-00.train_log
wandb: tree-2020-04-26-02-15-00_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced tree-2020-04-26-02-15-00: https://app.wandb.ai/jianguda/CodeSearchNet/runs/mimpe060
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/mimpe060
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./tree-2020-04-26-02-15-00_model_best.pkl.gz
2020-04-26 05:03:07.894201: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-26 05:03:12.654280: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: e9e1:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-26 05:03:12.654325: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-26 05:03:12.925587: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-26 05:03:12.925646: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-26 05:03:12.925662: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-26 05:03:12.925775: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: e9e1:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 198425.00it/s]1156085it [00:19, 58171.25it/s]
Uploading predictions to W&B
NDCG Average: 0.043087838

# RNN

# RNN attention

# BERT

Epoch 44 (valid) took 11.01s [processed 2088 samples/second]
Validation: Loss: 3.606224 | MRR: 0.421470
2020-04-26 11:26:49.301216: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-26 11:26:49.301278: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-26 11:26:49.301293: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-26 11:26:49.301305: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-26 11:26:49.301385: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: e9e1:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.429
FuncNameTest-All MRR (bs=1,000): 0.426
Validation-All MRR (bs=1,000): 0.403
Test-python MRR (bs=1,000): 0.429
FuncNameTest-python MRR (bs=1,000): 0.426
Validation-python MRR (bs=1,000): 0.403

wandb: Waiting for W&B process to finish, PID 471
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-mrr 0.5978235263268924
wandb: train-loss 2.3798899899408656
wandb: \_step 275
wandb: \_runtime 21728.683203458786
wandb: \_timestamp 1587900591.8157635
wandb: val-time-sec 11.013863801956177
wandb: epoch 44
wandb: val-loss 3.6062244954316514
wandb: val-mrr 0.4214704125445822
wandb: train-time-sec 453.8499433994293
wandb: best_val_mrr_loss 3.5426017512445864
wandb: best_val_mrr 0.42654627327297046
wandb: best_epoch 39
wandb: Test-All MRR (bs=1,000) 0.4285978946439927
wandb: FuncNameTest-All MRR (bs=1,000) 0.42596566687720494
wandb: Validation-All MRR (bs=1,000) 0.40346022670825166
wandb: Test-python MRR (bs=1,000) 0.4285978946439927
wandb: FuncNameTest-python MRR (bs=1,000) 0.42596566687720494
wandb: Validation-python MRR (bs=1,000) 0.40346022670825166
wandb: Syncing files in wandb/run-20200426_052744-52zi2b1u:
wandb: epoch 44
wandb: val-loss 3.6062244954316514
wandb: val-mrr 0.4214704125445822
wandb: train-time-sec 453.8499433994293
wandb: best_val_mrr_loss 3.5426017512445864
wandb: best_val_mrr 0.42654627327297046
wandb: best_epoch 39
wandb: Test-All MRR (bs=1,000) 0.4285978946439927
wandb: FuncNameTest-All MRR (bs=1,000) 0.42596566687720494
wandb: Validation-All MRR (bs=1,000) 0.40346022670825166
wandb: Test-python MRR (bs=1,000) 0.4285978946439927
wandb: FuncNameTest-python MRR (bs=1,000) 0.42596566687720494
wandb: Validation-python MRR (bs=1,000) 0.40346022670825166
wandb: Syncing files in wandb/run-20200426_052744-52zi2b1u:
wandb: tree-2020-04-26-05-27-44-graph.pbtxt
wandb: tree-2020-04-26-05-27-44.train_log
wandb: tree-2020-04-26-05-27-44_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb: tree-2020-04-26-05-27-44.train_log
wandb: tree-2020-04-26-05-27-44_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced tree-2020-04-26-05-27-44: https://app.wandb.ai/jianguda/CodeSearchNet/runs/52zi2b1u
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/52zi2b1u  
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./tree-2020-04-26-05-27-44_model_best.pkl.gz
2020-04-26 12:46:41.253808: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-26 12:46:46.191348: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: e9e1:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-26 12:46:46.191392: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-26 12:46:46.470760: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-26 12:46:46.470820: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-26 12:46:46.470836: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-26 12:46:46.470942: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: e9e1:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 198181.04it/s]1156085it [00:19, 58017.82it/s]
Uploading predictions to W&B
NDCG Average: 0.066512942

# BERT attention

Epoch 15 (valid) took 11.71s [processed 1964 samples/second]
Validation: Loss: 5.749025 | MRR: 0.099795
2020-04-26 15:26:04.202407: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-26 15:26:04.202470: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-26 15:26:04.202485: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-26 15:26:04.202496: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-26 15:26:04.202590: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: e9e1:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.099
FuncNameTest-All MRR (bs=1,000): 0.125
Validation-All MRR (bs=1,000): 0.090
Test-python MRR (bs=1,000): 0.099
FuncNameTest-python MRR (bs=1,000): 0.125
Validation-python MRR (bs=1,000): 0.090

wandb: Waiting for W&B process to finish, PID 682
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-mrr 0.11662209720056034
wandb: \_timestamp 1587914947.8332124
wandb: \_step 101
wandb: \_runtime 8332.120104312897
wandb: train-loss 5.4979608382993534
wandb: val-mrr 0.09979548993317977
wandb: train-time-sec 457.65494441986084
wandb: val-time-sec 11.708226442337036
wandb: epoch 15
wandb: val-loss 5.749024805815323
wandb: best_val_mrr_loss 5.58309849448826
wandb: best_val_mrr 0.11975570446511974
wandb: best_epoch 10
wandb: Test-All MRR (bs=1,000) 0.0994736031411533
wandb: FuncNameTest-All MRR (bs=1,000) 0.12522414867352252
wandb: Validation-All MRR (bs=1,000) 0.08963009738741817
wandb: Test-python MRR (bs=1,000) 0.0994736031411533
wandb: FuncNameTest-python MRR (bs=1,000) 0.12522414867352252
wandb: Validation-python MRR (bs=1,000) 0.08963009738741817
wandb: Syncing files in wandb/run-20200426_131016-ntjbalto:
wandb: tree-2020-04-26-13-10-16-graph.pbtxt
wandb: tree-2020-04-26-13-10-16.train_log
wandb: tree-2020-04-26-13-10-16_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced tree-2020-04-26-13-10-16: https://app.wandb.ai/jianguda/CodeSearchNet/runs/ntjbalto
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/ntjbalto
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./tree-2020-04-26-13-10-16_model_best.pkl.gz
2020-04-26 16:23:49.403079: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-26 16:23:54.242527: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: e9e1:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-26 16:23:54.242574: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-26 16:23:54.523938: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-26 16:23:54.524033: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-26 16:23:54.524074: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-26 16:23:54.524185: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: e9e1:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 198551.88it/s]1156085it [00:19, 57989.68it/s]
Uploading predictions to W&B
NDCG Average: 0.004460423

# NBOW+RNN

Epoch 36 (valid) took 2.72s [processed 8468 samples/second]
Validation: Loss: 4.416185 | MRR: 0.342909
2020-05-01 21:52:18.113723: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-01 21:52:18.113785: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-01 21:52:18.113799: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-01 21:52:18.113811: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-01 21:52:18.113897: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 59e9:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.531
FuncNameTest-All MRR (bs=1,000): 0.438
Validation-All MRR (bs=1,000): 0.492
Test-python MRR (bs=1,000): 0.531
FuncNameTest-python MRR (bs=1,000): 0.438
Validation-python MRR (bs=1,000): 0.492
NDCG Average: 0.129065019

# NBOW+RNN attention

Epoch 47 (valid) took 11.95s [processed 1925 samples/second]
Validation: Loss: 4.085803 | MRR: 0.349671
2020-05-03 00:42:51.140382: I tensorflow/core/common*runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-03 00:42:51.140444: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-03 00:42:51.140458: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-03 00:42:51.140469: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-03 00:42:51.140554: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 7bd0:00:00.0, compute capability: 3.7)
*@_@_@_@_@_@_@_@_@_@_@_@_@_@_@\_@
Tensor("code_encoder/python/tree_encoder/MatMul_1:0", shape=(2, ?, 128), dtype=float32)
Tensor("code_encoder/python/tree_encoder/Shape_3:0", shape=(3,), dtype=int32)
Test-All MRR (bs=1,000): 0.008
FuncNameTest-All MRR (bs=1,000): 0.008
Validation-All MRR (bs=1,000): 0.008
Test-python MRR (bs=1,000): 0.008
FuncNameTest-python MRR (bs=1,000): 0.008
Validation-python MRR (bs=1,000): 0.008

wandb: Waiting for W&B process to finish, PID 48
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_step 293
wandb: \_runtime 22996.463274002075
wandb: train-mrr 0.5859921777225235
wandb: train-loss 2.3594511764720805
wandb: \_timestamp 1588466773.2661593
wandb: val-mrr 0.3496708924666695
wandb: train-time-sec 448.20907402038574
wandb: val-time-sec 11.94708776473999
wandb: val-loss 4.085803114849588
wandb: epoch 47
wandb: best*val_mrr_loss 4.163796611454176
wandb: best_val_mrr 0.35001164510975713
wandb: best_epoch 42
wandb: Test-All MRR (bs=1,000) 0.007744362385912137
wandb: FuncNameTest-All MRR (bs=1,000) 0.007686700811841948
wandb: Validation-All MRR (bs=1,000) 0.007688682732382063
wandb: Test-python MRR (bs=1,000) 0.007744362385912137
wandb: FuncNameTest-python MRR (bs=1,000) 0.007686700811841948
wandb: Validation-python MRR (bs=1,000) 0.007688682732382063
wandb: best_val_mrr_loss 4.163796611454176
wandb: best_val_mrr 0.35001164510975713
wandb: best_epoch 42
wandb: Test-All MRR (bs=1,000) 0.007744362385912137
wandb: FuncNameTest-All MRR (bs=1,000) 0.007686700811841948
wandb: Validation-All MRR (bs=1,000) 0.007688682732382063
wandb: Test-python MRR (bs=1,000) 0.007744362385912137
wandb: FuncNameTest-python MRR (bs=1,000) 0.007686700811841948
wandb: Validation-python MRR (bs=1,000) 0.007688682732382063
wandb: Syncing files in wandb/run-20200502_182258-2mhapg3r:
wandb: tree-2020-05-02-18-22-58-graph.pbtxt
wandb: tree-2020-05-02-18-22-58.train_log
wandb: tree-2020-05-02-18-22-58_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced tree-2020-05-02-18-22-58: https://app.wandb.ai/jianguda/CodeSearchNet/runs/2mhapg3r
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/2mhapg3r
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./tree-2020-05-02-18-22-58_model_best.pkl.gz
2020-05-03 06:34:50.742420: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-03 06:34:55.632308: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 7bd0:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-03 06:34:55.632355: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-03 06:34:55.903607: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-03 06:34:55.903671: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-03 06:34:55.903686: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-03 06:34:55.903796: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 7bd0:00:00.0, compute capability: 3.7)
WARNING:tensorflow:From /home/dev/code/encoders/tree_tmp_encoder.py:483: calling softmax (from tensorflow.python.ops.nn_ops) with
dim is deprecated and will be removed in a future version.
Instructions for updating:
dim is deprecated, use axis instead
*@_@_@_@_@_@_@_@_@_@_@_@_@_@_@\_@
Tensor("code_encoder/python/tree_encoder/MatMul_1:0", shape=(2, ?, 128), dtype=float32)
Tensor("code_encoder/python/tree_encoder/Shape_3:0", shape=(3,), dtype=int32)
Evaluating language: python
100%|███████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 200123.75it/s]1156085it [00:19, 58207.13it/s]
Uploading predictions to W&B
NDCG Average: 0.005242633
