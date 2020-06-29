# NBOW

Epoch 46 (valid) took 2.73s [processed 8416 samples/second]
Validation: Loss: 4.401977 | MRR: 0.341080
2020-05-04 21:55:23.153577: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-04 21:55:23.153631: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-04 21:55:23.153646: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-04 21:55:23.153660: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-04 21:55:23.153757: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fec2:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.530
FuncNameTest-All MRR (bs=1,000): 0.439
Validation-All MRR (bs=1,000): 0.493
Test-python MRR (bs=1,000): 0.530
FuncNameTest-python MRR (bs=1,000): 0.439
Validation-python MRR (bs=1,000): 0.493

wandb: Waiting for W&B process to finish, PID 262
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-mrr 0.6175396689998294
wandb: \_step 287
wandb: train-loss 2.241131802207058
wandb: \_runtime 5085.625137805939
wandb: \_timestamp 1588629472.9732616
wandb: val-time-sec 2.7327966690063477
wandb: train-time-sec 87.32311511039734
wandb: val-loss 4.401976709780485
wandb: val-mrr 0.3410798227061396
wandb: epoch 46
wandb: best_val_mrr_loss 4.359511707140052
wandb: best_val_mrr 0.3451665311067001
wandb: best_epoch 41
wandb: Test-All MRR (bs=1,000) 0.5301450014969094
wandb: FuncNameTest-All MRR (bs=1,000) 0.4390447669347795
wandb: Validation-All MRR (bs=1,000) 0.4933176303644359
wandb: Test-python MRR (bs=1,000) 0.5301450014969094
wandb: FuncNameTest-python MRR (bs=1,000) 0.4390447669347795
wandb: Validation-python MRR (bs=1,000) 0.4933176303644359
wandb: Syncing files in wandb/run-20200504_203308-3k89mjyb:
wandb: treeleaf-2020-05-04-20-33-08-graph.pbtxt
wandb: treeleaf-2020-05-04-20-33-08.train_log
wandb: treeleaf-2020-05-04-20-33-08_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeleaf-2020-05-04-20-33-08: https://app.wandb.ai/jianguda/CodeSearchNet/runs/3k89mjyb
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/3k89mjyb
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeleaf-2020-05-04-20-33-08_model_best.pkl.gz
2020-05-04 22:19:46.691978: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-04 22:19:51.511902: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fec2:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-04 22:19:51.511949: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-04 22:19:51.788890: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-04 22:19:51.788970: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-04 22:19:51.788989: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-04 22:19:51.789101: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fec2:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|███████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:10<00:00, 107456.08it/s]1156085it [00:20, 57501.60it/s]
Uploading predictions to W&B
NDCG Average: 0.152013968

# NBOW attention

Epoch 26 (valid) took 2.93s [processed 7852 samples/second]
Validation: Loss: 4.325847 | MRR: 0.344159
2020-05-05 19:45:56.011893: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-05 19:45:56.011952: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-05 19:45:56.011966: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-05 19:45:56.011976: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-05 19:45:56.012077: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fec2:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.539
FuncNameTest-All MRR (bs=1,000): 0.447
Validation-All MRR (bs=1,000): 0.506
Test-python MRR (bs=1,000): 0.539
FuncNameTest-python MRR (bs=1,000): 0.447
Validation-python MRR (bs=1,000): 0.506

wandb: Waiting for W&B process to finish, PID 1187
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-loss 2.224069040955849
wandb: \_timestamp 1588708109.7829528
wandb: train-mrr 0.6195040939479198
wandb: \_runtime 3541.0226306915283
wandb: \_step 167
wandb: train-time-sec 98.28933930397034
wandb: val-loss 4.325847128163213
wandb: val-mrr 0.34415925465459407
wandb: epoch 26
wandb: val-time-sec 2.9290342330932617
wandb: best_val_mrr_loss 4.280250694440759
wandb: best_val_mrr 0.34502858700959577
wandb: best_epoch 21
wandb: Test-All MRR (bs=1,000) 0.5390118703067048
wandb: FuncNameTest-All MRR (bs=1,000) 0.44703435951179243
wandb: Validation-All MRR (bs=1,000) 0.5060024857413213
wandb: Test-python MRR (bs=1,000) 0.5390118703067048
wandb: FuncNameTest-python MRR (bs=1,000) 0.44703435951179243
wandb: Validation-python MRR (bs=1,000) 0.5060024857413213
wandb: Syncing files in wandb/run-20200505_184929-06ax1aum:
wandb: treeleaf-2020-05-05-18-49-29-graph.pbtxt
wandb: treeleaf-2020-05-05-18-49-29.train_log
wandb: treeleaf-2020-05-05-18-49-29_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeleaf-2020-05-05-18-49-29: https://app.wandb.ai/jianguda/CodeSearchNet/runs/06ax1aum
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/06ax1aum
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeleaf-2020-05-05-18-49-29_model_best.pkl.gz
2020-05-05 19:55:08.728579: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-05 19:55:13.550302: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fec2:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-05 19:55:13.550350: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-05 19:55:13.824742: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-05 19:55:13.824799: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-05 19:55:13.824816: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-05 19:55:13.824947: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fec2:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|███████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:10<00:00, 108218.51it/s]1156085it [00:20, 57803.18it/s]
Uploading predictions to W&B
NDCG Average: 0.156450608

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

Epoch 31 (valid) took 5.41s [processed 4253 samples/second]
Validation: Loss: 4.166027 | MRR: 0.320992
2020-05-05 22:58:29.819382: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-05 22:58:29.819443: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-05 22:58:29.819458: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-05 22:58:29.819469: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-05 22:58:29.819569: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fec2:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.257
FuncNameTest-All MRR (bs=1,000): 0.295
Validation-All MRR (bs=1,000): 0.229
Test-python MRR (bs=1,000): 0.257
FuncNameTest-python MRR (bs=1,000): 0.295
Validation-python MRR (bs=1,000): 0.229

wandb: Waiting for W&B process to finish, PID 1393
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_timestamp 1588719673.9058187
wandb: \_runtime 9970.590512275696
wandb: train-loss 3.192250777216791
wandb: train-mrr 0.45334377088824523
wandb: \_step 197
wandb: val-time-sec 5.407455205917358
wandb: train-time-sec 278.7632312774658
wandb: val-mrr 0.3209916985553244
wandb: val-loss 4.166027100189872
wandb: epoch 31
wandb: best_val_mrr_loss 4.145311790963878
wandb: best_val_mrr 0.32368926338527515
wandb: best_epoch 26
wandb: Test-All MRR (bs=1,000) 0.256573866873149
wandb: FuncNameTest-All MRR (bs=1,000) 0.2952721488870248
wandb: Validation-All MRR (bs=1,000) 0.22887311347211903
wandb: Test-python MRR (bs=1,000) 0.256573866873149
wandb: FuncNameTest-python MRR (bs=1,000) 0.2952721488870248
wandb: Validation-python MRR (bs=1,000) 0.22887311347211903
wandb: Syncing files in wandb/run-20200505_201504-dk487j6e:
wandb: treeleaf-2020-05-05-20-15-04-graph.pbtxt
wandb: treeleaf-2020-05-05-20-15-04.train_log
wandb: treeleaf-2020-05-05-20-15-04_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeleaf-2020-05-05-20-15-04: https://app.wandb.ai/jianguda/CodeSearchNet/runs/dk487j6e
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/dk487j6e
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeleaf-2020-05-05-20-15-04_model_best.pkl.gz
2020-05-06 04:59:36.164433: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-06 04:59:41.065915: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fec2:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-06 04:59:41.065961: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-06 04:59:41.342576: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-06 04:59:41.342639: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-06 04:59:41.342654: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-06 04:59:41.342763: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fec2:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|███████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:10<00:00, 107945.18it/s]1156085it [00:20, 57719.48it/s]
Uploading predictions to W&B
NDCG Average: 0.087153617

# RNN

Epoch 23 (valid) took 11.85s [processed 1941 samples/second]
Validation: Loss: 2.878355 | MRR: 0.523161
2020-05-05 10:42:31.977039: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-05 10:42:31.977142: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-05 10:42:31.977160: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-05 10:42:31.977171: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-05 10:42:31.977273: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fec2:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.666
FuncNameTest-All MRR (bs=1,000): 0.654
Validation-All MRR (bs=1,000): 0.614
Test-python MRR (bs=1,000): 0.666
FuncNameTest-python MRR (bs=1,000): 0.654
Validation-python MRR (bs=1,000): 0.614

wandb: Waiting for W&B process to finish, PID 681
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_step 149
wandb: train-mrr 0.724998977216702
wandb: \_timestamp 1588675554.3824427
wandb: train-loss 1.6171477228692435
wandb: \_runtime 11773.593623876572
wandb: epoch 23
wandb: val-mrr 0.5231608422320821
wandb: train-time-sec 442.43261981010437
wandb: val-time-sec 11.846516609191895
wandb: val-loss 2.87835461160411
wandb: best_val_mrr_loss 2.8769357722738516
wandb: best_val_mrr 0.5240779073963995
wandb: best_epoch 18
wandb: Test-All MRR (bs=1,000) 0.665838209518347
wandb: FuncNameTest-All MRR (bs=1,000) 0.6541089960176248
wandb: Validation-All MRR (bs=1,000) 0.6144882206526685
wandb: Test-python MRR (bs=1,000) 0.665838209518347
wandb: FuncNameTest-python MRR (bs=1,000) 0.6541089960176248
wandb: Validation-python MRR (bs=1,000) 0.6144882206526685
wandb: Syncing files in wandb/run-20200505_072941-5cx1o58d:
wandb: treeleaf-2020-05-05-07-29-41-graph.pbtxt
wandb: treeleaf-2020-05-05-07-29-41.train_log
wandb: treeleaf-2020-05-05-07-29-41_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeleaf-2020-05-05-07-29-41: https://app.wandb.ai/jianguda/CodeSearchNet/runs/5cx1o58d
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/5cx1o58d
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeleaf-2020-05-05-07-29-41_model_best.pkl.gz
2020-05-05 12:54:15.497593: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-05 12:54:20.417650: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeleaf-2020-05-05-07-29-41: https://app.wandb.ai/jianguda/CodeSearchNet/runs/5cx1o58d
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/5cx1o58d  
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeleaf-2020-05-05-07-29-41_model_best.pkl.gz
2020-05-05 12:54:15.497593: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-05 12:54:20.417650: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fec2:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-05 12:54:20.417744: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-05 12:54:20.696212: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-05 12:54:20.696276: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-05 12:54:20.696294: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-05 12:54:20.696408: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fec2:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|███████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:10<00:00, 107763.53it/s]1156085it [00:20, 57323.27it/s]
Uploading predictions to W&B
NDCG Average: 0.170924428

# RNN attention

Epoch 22 (valid) took 12.02s [processed 1913 samples/second]
Validation: Loss: 2.948690 | MRR: 0.510720
2020-05-05 16:40:35.042740: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-05 16:40:35.042802: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-05 16:40:35.042816: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-05 16:40:35.042827: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-05 16:40:35.042918: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fec2:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.653
FuncNameTest-All MRR (bs=1,000): 0.632
Validation-All MRR (bs=1,000): 0.602
Test-python MRR (bs=1,000): 0.653
FuncNameTest-python MRR (bs=1,000): 0.632
Validation-python MRR (bs=1,000): 0.602

wandb: Waiting for W&B process to finish, PID 890
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-loss 1.7044454251678245
wandb: \_runtime 11503.67028093338
wandb: \_timestamp 1588697037.115977
wandb: \_step 143
wandb: train-mrr 0.7104337067835539
wandb: val-time-sec 12.017933130264282
wandb: val-mrr 0.5107200516410496
wandb: train-time-sec 451.05411553382874
wandb: val-loss 2.9486896991729736
wandb: epoch 22
wandb: best_val_mrr_loss 2.931273190871529
wandb: best_val_mrr 0.5174847717285156
wandb: best_epoch 17
wandb: Test-All MRR (bs=1,000) 0.6532624647423653
wandb: FuncNameTest-All MRR (bs=1,000) 0.631571495420014
wandb: Validation-All MRR (bs=1,000) 0.601813669055348
wandb: Test-python MRR (bs=1,000) 0.6532624647423653
wandb: FuncNameTest-python MRR (bs=1,000) 0.631571495420014
wandb: Validation-python MRR (bs=1,000) 0.601813669055348
wandb: Syncing files in wandb/run-20200505_133214-lnmcop2b:
wandb: treeleaf-2020-05-05-13-32-14-graph.pbtxt
wandb: treeleaf-2020-05-05-13-32-14.train_log
wandb: treeleaf-2020-05-05-13-32-14_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeleaf-2020-05-05-13-32-14: https://app.wandb.ai/jianguda/CodeSearchNet/runs/lnmcop2b
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/lnmcop2b
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeleaf-2020-05-05-13-32-14_model_best.pkl.gz
2020-05-05 17:00:41.232544: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlwandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeleaf-2020-05-05-13-32-14: https://app.wandb.ai/jianguda/CodeSearchNet/runs/lnmcop2b
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/lnmcop2b
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeleaf-2020-05-05-13-32-14_model_best.pkl.gz
2020-05-05 17:00:41.232544: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-05 17:00:45.947762: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fec2:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-05 17:37:06.757364: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-05 17:37:07.036941: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-05 17:37:07.036999: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-05 17:37:07.037015: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-05 17:37:07.037122: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fec2:00:00.0, compute capability: 3.7)
WARNING:tensorflow:From /home/dev/code/encoders/tree/common.py:55: calling softmax (from tensorflow.python.ops.nn_ops) with dim is deprecated and will be removed in a future version.
Instructions for updating:
dim is deprecated, use axis instead
Evaluating language: python
100%|███████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:10<00:00, 107919.79it/s]1156085it [00:19, 58031.29it/s]
Uploading predictions to W&B
NDCG Average: 0.180373398

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
