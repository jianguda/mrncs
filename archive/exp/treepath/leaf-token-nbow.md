# NBOW

Epoch 13 (valid) took 1.44s [processed 15966 samples/second]
Validation: Loss: 4.341366 | MRR: 0.381824
2020-05-07 13:50:24.930775: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-07 13:50:24.930843: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-07 13:50:24.930859: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-07 13:50:24.930866: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-07 13:50:24.930958: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fec2:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.587
FuncNameTest-All MRR (bs=1,000): 0.473
Validation-All MRR (bs=1,000): 0.554
Test-python MRR (bs=1,000): 0.587
FuncNameTest-python MRR (bs=1,000): 0.473
Validation-python MRR (bs=1,000): 0.554

wandb: Waiting for W&B process to finish, PID 3292
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_step 89
wandb: \_timestamp 1588859565.4959273
wandb: train-loss 1.4357363156323295
wandb: train-mrr 0.7729243932927696
wandb: \_runtime 1322.6756613254547
wandb: val-time-sec 1.4404850006103516
wandb: val-loss 4.341365710548732
wandb: train-time-sec 37.76922917366028
wandb: val-mrr 0.3818241358217986
wandb: epoch 13
wandb: best_val_mrr_loss 4.108479976654053
wandb: best_val_mrr 0.38624475628396737
wandb: best_epoch 8
wandb: Test-All MRR (bs=1,000) 0.5873896315836288
wandb: FuncNameTest-All MRR (bs=1,000) 0.4729197242706705
wandb: Validation-All MRR (bs=1,000) 0.5541633195557877
wandb: Test-python MRR (bs=1,000) 0.5873896315836288
wandb: FuncNameTest-python MRR (bs=1,000) 0.4729197242706705
wandb: Validation-python MRR (bs=1,000) 0.5541633195557877
wandb: Syncing files in wandb/run-20200507_133044-2hjh0hkg:
wandb: treeleaf-2020-05-07-13-30-44-graph.pbtxt
wandb: treeleaf-2020-05-07-13-30-44.train_log
wandb: treeleaf-2020-05-07-13-30-44_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeleaf-2020-05-07-13-30-44: https://app.wandb.ai/jianguda/CodeSearchNet/runs/2hjh0hkg
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/2hjh0hkg
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeleaf-2020-05-07-13-30-44_model_best.pkl.gz
2020-05-07 13:54:23.920906: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-07 13:54:28.785001: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fec2:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-07 13:54:28.785047: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-07 13:54:29.069336: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-07 13:54:29.069391: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-07 13:54:29.069406: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-07 13:54:29.069514: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fec2:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|███████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:10<00:00, 108107.51it/s]1156085it [00:20, 57184.23it/s]
Uploading predictions to W&B
NDCG Average: 0.167121640

# NBOW attention

Epoch 13 (valid) took 1.58s [processed 14529 samples/second]
Validation: Loss: 4.344932 | MRR: 0.377789
2020-05-06 10:16:22.810397: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-06 10:16:22.810460: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-06 10:16:22.810475: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-06 10:16:22.810485: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-06 10:16:22.810578: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fec2:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.579
FuncNameTest-All MRR (bs=1,000): 0.457
Validation-All MRR (bs=1,000): 0.543
Test-python MRR (bs=1,000): 0.579
FuncNameTest-python MRR (bs=1,000): 0.457
Validation-python MRR (bs=1,000): 0.543

wandb: Waiting for W&B process to finish, PID 1811
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-loss 1.4608496181594515
wandb: \_step 89
wandb: \_timestamp 1588760326.2576997
wandb: train-mrr 0.7677069668075414
wandb: \_runtime 1458.2178795337677
wandb: val-loss 4.344932286635689
wandb: train-time-sec 45.989672899246216
wandb: val-time-sec 1.5830388069152832
wandb: val-mrr 0.3777891394573709
wandb: epoch 13
wandb: best_val_mrr_loss 4.1989748063294785
wandb: best_val_mrr 0.3795210339090099
wandb: best_epoch 8
wandb: Test-All MRR (bs=1,000) 0.5788127097954586
wandb: FuncNameTest-All MRR (bs=1,000) 0.4567779377878573
wandb: Validation-All MRR (bs=1,000) 0.5427860508241721
wandb: Test-python MRR (bs=1,000) 0.5788127097954586
wandb: FuncNameTest-python MRR (bs=1,000) 0.4567779377878573
wandb: Validation-python MRR (bs=1,000) 0.5427860508241721
wandb: Syncing files in wandb/run-20200506_095429-ql5whq97:
wandb: treeleaf-2020-05-06-09-54-29-graph.pbtxt
wandb: treeleaf-2020-05-06-09-54-29.train_log
wandb: treeleaf-2020-05-06-09-54-29_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeleaf-2020-05-06-09-54-29: https://app.wandb.ai/jianguda/CodeSearchNet/runs/ql5whq97
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/ql5whq97
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeleaf-2020-05-06-09-54-29_model_best.pkl.gz
2020-05-06 10:23:27.382932: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-06 10:23:32.272203: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fec2:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-06 10:23:32.272250: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-06 10:23:32.560855: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-06 10:23:32.560934: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-06 10:23:32.560953: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-06 10:23:32.561080: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fec2:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|███████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:10<00:00, 108860.99it/s]1156085it [00:20, 57341.88it/s]
Uploading predictions to W&B
NDCG Average: 0.201082369

# CNN

Epoch 35 (valid) took 3.87s [processed 5940 samples/second]
Validation: Loss: 4.116877 | MRR: 0.342616
2020-05-07 13:02:23.388813: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-07 13:02:23.388878: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-07 13:02:23.388893: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-07 13:02:23.388904: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-07 13:02:23.389020: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fec2:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.209
FuncNameTest-All MRR (bs=1,000): 0.190
Validation-All MRR (bs=1,000): 0.186
Test-python MRR (bs=1,000): 0.209
FuncNameTest-python MRR (bs=1,000): 0.190
Validation-python MRR (bs=1,000): 0.186

wandb: Waiting for W&B process to finish, PID 3085
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_runtime 8544.82233595848
wandb: train-loss 2.6509582776467777
wandb: \_timestamp 1588856697.2200558
wandb: train-mrr 0.5564079446144474
wandb: \_step 221
wandb: train-time-sec 210.4310302734375
wandb: val-time-sec 3.8719241619110107
wandb: val-mrr 0.34261602252462636
wandb: epoch 35
wandb: val-loss 4.116877089376035
wandb: best_val_mrr_loss 4.1015716013701065
wandb: best_val_mrr 0.3512777743132218
wandb: best_epoch 30
wandb: Test-All MRR (bs=1,000) 0.2093206200270621
wandb: FuncNameTest-All MRR (bs=1,000) 0.19028876726870003
wandb: Validation-All MRR (bs=1,000) 0.1863551593886448
wandb: Test-python MRR (bs=1,000) 0.2093206200270621
wandb: FuncNameTest-python MRR (bs=1,000) 0.19028876726870003
wandb: Validation-python MRR (bs=1,000) 0.1863551593886448
wandb: Syncing files in wandb/run-20200507_104233-iutec0py:
wandb: treeleaf-2020-05-07-10-42-33-graph.pbtxt
wandb: treeleaf-2020-05-07-10-42-33.train_log
wandb: treeleaf-2020-05-07-10-42-33_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeleaf-2020-05-07-10-42-33: https://app.wandb.ai/jianguda/CodeSearchNet/runs/iutec0py
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/iutec0py
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeleaf-2020-05-07-10-42-33_model_best.pkl.gz
2020-05-07 13:07:18.928851: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-07 13:07:23.818523: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fec2:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-07 13:07:23.818570: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-07 13:07:24.097406: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-07 13:07:24.097472: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-07 13:07:24.097482: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-07 13:07:24.097610: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fec2:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|███████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:10<00:00, 108207.09it/s]1156085it [00:20, 57377.74it/s]
Uploading predictions to W&B
NDCG Average: 0.140916976

# CNN attention

Epoch 37 (valid) took 3.91s [processed 5883 samples/second]
Validation: Loss: 4.303890 | MRR: 0.320821
2020-05-06 16:47:12.289016: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-06 16:47:12.289076: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-06 16:47:12.289090: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-06 16:47:12.289100: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-06 16:47:12.289186: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fec2:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.137
FuncNameTest-All MRR (bs=1,000): 0.127
Validation-All MRR (bs=1,000): 0.116
Test-python MRR (bs=1,000): 0.137
FuncNameTest-python MRR (bs=1,000): 0.127
Validation-python MRR (bs=1,000): 0.116

wandb: Waiting for W&B process to finish, PID 2229
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_timestamp 1588783787.760451
wandb: train-loss 2.8870081131898084
wandb: \_step 233
wandb: train-mrr 0.516682615039418
wandb: \_runtime 9280.383166074753
wandb: train-time-sec 217.6907024383545
wandb: val-mrr 0.32082115637737774
wandb: val-time-sec 3.9089913368225098
wandb: epoch 37
wandb: val-loss 4.303889585577923
wandb: best_val_mrr_loss 4.296566216841988
wandb: best_val_mrr 0.3249516375997792
wandb: best_epoch 32
wandb: Test-All MRR (bs=1,000) 0.1374928128453073
wandb: FuncNameTest-All MRR (bs=1,000) 0.1271211127900929
wandb: Validation-All MRR (bs=1,000) 0.11555150543691076
wandb: Test-python MRR (bs=1,000) 0.1374928128453073
wandb: FuncNameTest-python MRR (bs=1,000) 0.1271211127900929
wandb: Validation-python MRR (bs=1,000) 0.11555150543691076
wandb: Syncing files in wandb/run-20200506_141508-g0zg4qy2:
wandb: treeleaf-2020-05-06-14-15-08-graph.pbtxt
wandb: treeleaf-2020-05-06-14-15-08.train_log
wandb: treeleaf-2020-05-06-14-15-08_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeleaf-2020-05-06-14-15-08: https://app.wandb.ai/jianguda/CodeSearchNet/runs/g0zg4qy2
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/g0zg4qy2
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeleaf-2020-05-06-14-15-08_model_best.pkl.gz
Fetching run files from W&B...
Restoring model from ./treeleaf-2020-05-06-14-15-08_model_best.pkl.gz
Restoring model from ./treeleaf-2020-05-06-14-15-08_model_best.pkl.gz
2020-05-06 16:56:30.717220: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-06 16:56:35.507610: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fec2:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-06 16:56:35.507657: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-06 16:56:35.787475: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-06 16:56:35.787532: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-06 16:56:35.787549: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-06 16:56:35.787659: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fec2:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|███████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:10<00:00, 108369.10it/s]1156085it [00:20, 57784.93it/s]
Uploading predictions to W&B
NDCG Average: 0.101690621

# RNN

Epoch 26 (valid) took 10.44s [processed 2202 samples/second]
Validation: Loss: 3.342869 | MRR: 0.499170
2020-05-07 10:03:44.131753: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-07 10:03:44.131815: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-07 10:03:44.131829: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-07 10:03:44.131840: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-07 10:03:44.131936: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fec2:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.588
FuncNameTest-All MRR (bs=1,000): 0.520
Validation-All MRR (bs=1,000): 0.542
Test-python MRR (bs=1,000): 0.588
FuncNameTest-python MRR (bs=1,000): 0.520
Validation-python MRR (bs=1,000): 0.542

wandb: Waiting for W&B process to finish, PID 2873
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_timestamp 1588846018.9438121
wandb: train-loss 1.2343623768357397
wandb: \_runtime 11605.881103992462
wandb: train-mrr 0.7961667776755916
wandb: \_step 167
wandb: train-time-sec 387.37630820274353
wandb: epoch 26
wandb: val-time-sec 10.44313645362854
wandb: val-loss 3.3428694268931514
wandb: val-mrr 0.4991702283776325
wandb: best_val_mrr_loss 3.25432128491609
wandb: best_val_mrr 0.5028214655337127
wandb: best_epoch 21
wandb: Test-All MRR (bs=1,000) 0.588200447283016
wandb: FuncNameTest-All MRR (bs=1,000) 0.5195320951539375
wandb: Validation-All MRR (bs=1,000) 0.5420324290049641
wandb: Test-python MRR (bs=1,000) 0.588200447283016
wandb: FuncNameTest-python MRR (bs=1,000) 0.5195320951539375
wandb: Validation-python MRR (bs=1,000) 0.5420324290049641
wandb: Syncing files in wandb/run-20200507_065334-wtamf1v0:
wandb: treeleaf-2020-05-07-06-53-34-graph.pbtxt
wandb: treeleaf-2020-05-07-06-53-34.train_log
wandb: treeleaf-2020-05-07-06-53-34_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeleaf-2020-05-07-06-53-34: https://app.wandb.ai/jianguda/CodeSearchNet/runs/wtamf1v0
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/wtamf1v0
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeleaf-2020-05-07-06-53-34_model_best.pkl.gz
2020-05-07 10:12:55.921111: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-07 10:13:00.813652: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fec2:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-07 10:13:00.813698: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-07 10:13:01.096412: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-07 10:13:01.096472: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-07 10:13:01.096490: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-07 10:13:01.096597: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fec2:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|███████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:10<00:00, 107799.43it/s]1156085it [00:20, 57283.78it/s]
Uploading predictions to W&B
NDCG Average: 0.186183029

# RNN attention

Epoch 19 (valid) took 10.52s [processed 2187 samples/second]
Validation: Loss: 3.319820 | MRR: 0.496116
2020-05-06 13:08:20.775834: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-06 13:08:20.775895: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-06 13:08:20.775908: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-06 13:08:20.775916: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-06 13:08:20.776003: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fec2:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.440
FuncNameTest-All MRR (bs=1,000): 0.371
Validation-All MRR (bs=1,000): 0.407
Test-python MRR (bs=1,000): 0.440
FuncNameTest-python MRR (bs=1,000): 0.371
Validation-python MRR (bs=1,000): 0.407

wandb: Waiting for W&B process to finish, PID 2020
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_step 125
wandb: train-mrr 0.7965564290759632
wandb: train-loss 1.24972928554109
wandb: \_runtime 8940.90264749527
wandb: \_timestamp 1588770693.568604
wandb: val-loss 3.3198201241700547
wandb: train-time-sec 395.19605231285095
wandb: val-mrr 0.49611565963081694
wandb: val-time-sec 10.515795946121216
wandb: epoch 19
wandb: best_val_mrr_loss 3.254382371902466
wandb: best_val_mrr 0.4961867277725883
wandb: best_epoch 14
wandb: Test-All MRR (bs=1,000) 0.4402357190749701
wandb: FuncNameTest-All MRR (bs=1,000) 0.37050693443207794
wandb: Validation-All MRR (bs=1,000) 0.4065717273875332
wandb: Test-python MRR (bs=1,000) 0.4402357190749701
wandb: FuncNameTest-python MRR (bs=1,000) 0.37050693443207794
wandb: Validation-python MRR (bs=1,000) 0.4065717273875332
wandb: Syncing files in wandb/run-20200506_104233-s0emhojf:
wandb: treeleaf-2020-05-06-10-42-33-graph.pbtxt
wandb: treeleaf-2020-05-06-10-42-33.train_log
wandb: treeleaf-2020-05-06-10-42-33_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeleaf-2020-05-06-10-42-33: https://app.wandb.ai/jianguda/CodeSearchNet/runs/s0emhojf
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/s0emhojf
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeleaf-2020-05-06-10-42-33_model_best.pkl.gz
2020-05-06 13:16:23.459268: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-06 13:16:28.415728: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fec2:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-06 13:16:28.415777: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-06 13:16:28.698201: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-06 13:16:28.698264: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-06 13:16:28.698280: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-06 13:16:28.698391: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fec2:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|███████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:10<00:00, 107488.33it/s]1156085it [00:20, 57763.87it/s]
Uploading predictions to W&B
NDCG Average: 0.184851180

# BERT

Epoch 37 (valid) took 9.55s [processed 2407 samples/second]
Validation: Loss: 3.447859 | MRR: 0.520949
2020-05-06 23:37:17.715797: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-06 23:37:17.715859: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-06 23:37:17.715874: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-06 23:37:17.715886: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-06 23:37:17.715971: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fec2:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.508
FuncNameTest-All MRR (bs=1,000): 0.436
Validation-All MRR (bs=1,000): 0.472
Test-python MRR (bs=1,000): 0.508
FuncNameTest-python MRR (bs=1,000): 0.436
Validation-python MRR (bs=1,000): 0.472

wandb: Waiting for W&B process to finish, PID 2661
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-mrr 0.8639058806780473
wandb: \_step 233
wandb: train-loss 0.82292313888235
wandb: \_timestamp 1588808412.6897874
wandb: \_runtime 16049.34818816185
wandb: train-time-sec 391.29078936576843
wandb: val-time-sec 9.552696704864502
wandb: val-loss 3.4478590281113335
wandb: val-mrr 0.5209485420558764
wandb: epoch 37
wandb: best_val_mrr_loss 3.440080621968145
wandb: best_val_mrr 0.5221430212933085
wandb: best_epoch 32
wandb: Test-All MRR (bs=1,000) 0.508479269995541
wandb: FuncNameTest-All MRR (bs=1,000) 0.43613458696450014
wandb: Validation-All MRR (bs=1,000) 0.47226922318431386
wandb: Test-python MRR (bs=1,000) 0.508479269995541
wandb: FuncNameTest-python MRR (bs=1,000) 0.43613458696450014
wandb: Validation-python MRR (bs=1,000) 0.47226922318431386
wandb: Syncing files in wandb/run-20200506_191244-8t7lyjco:
wandb: treeleaf-2020-05-06-19-12-44-graph.pbtxt
wandb: treeleaf-2020-05-06-19-12-44.train_log
wandb: treeleaf-2020-05-06-19-12-44_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeleaf-2020-05-06-19-12-44: https://app.wandb.ai/jianguda/CodeSearchNet/runs/8t7lyjco
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/8t7lyjco
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeleaf-2020-05-06-19-12-44_model_best.pkl.gz
2020-05-07 06:14:19.461860: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-07 06:14:24.405900: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fec2:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-07 06:14:24.405947: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-07 06:14:24.684886: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-07 06:14:24.684966: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-07 06:14:24.684983: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-07 06:14:24.685097: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fec2:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|███████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:10<00:00, 107807.14it/s]1156085it [00:20, 57479.89it/s]
Uploading predictions to W&B
NDCG Average: 0.155742830

# BERT attention

Epoch 9 (valid) took 9.68s [processed 2376 samples/second]
Validation: Loss: 3.381434 | MRR: 0.532611
2020-05-06 18:37:37.149726: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-06 18:37:37.149789: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-06 18:37:37.149803: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-06 18:37:37.149814: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-06 18:37:37.149902: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fec2:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.481
FuncNameTest-All MRR (bs=1,000): 0.432
Validation-All MRR (bs=1,000): 0.439
Test-python MRR (bs=1,000): 0.481
FuncNameTest-python MRR (bs=1,000): 0.432
Validation-python MRR (bs=1,000): 0.439

wandb: Waiting for W&B process to finish, PID 2441
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-loss 0.8269011168514641
wandb: \_runtime 4941.911527872086
wandb: \_step 65
wandb: \_timestamp 1588790436.8753588
wandb: train-mrr 0.8689468238608351
wandb: val-loss 3.381434171096138
wandb: val-mrr 0.5326107297150985
wandb: train-time-sec 402.57220363616943
wandb: val-time-sec 9.680066347122192
wandb: epoch 9
wandb: best_val_mrr_loss 3.0735921030459195
wandb: best_val_mrr 0.5424874108355978
wandb: best_epoch 4
wandb: Test-All MRR (bs=1,000) 0.48090670677187697
wandb: FuncNameTest-All MRR (bs=1,000) 0.4324042072361882
wandb: Validation-All MRR (bs=1,000) 0.43887063351638567
wandb: Test-python MRR (bs=1,000) 0.48090670677187697
wandb: FuncNameTest-python MRR (bs=1,000) 0.4324042072361882
wandb: Validation-python MRR (bs=1,000) 0.43887063351638567
wandb: Syncing files in wandb/run-20200506_171816-lrhg6mdy:
wandb: treeleaf-2020-05-06-17-18-16-graph.pbtxt
wandb: treeleaf-2020-05-06-17-18-16.train_log
wandb: treeleaf-2020-05-06-17-18-16_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treeleaf-2020-05-06-17-18-16: https://app.wandb.ai/jianguda/CodeSearchNet/runs/lrhg6mdy
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/lrhg6mdy
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treeleaf-2020-05-06-17-18-16_model_best.pkl.gz
2020-05-06 18:43:21.787620: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-06 18:43:26.686848: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fec2:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-06 18:43:26.686896: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-06 18:43:26.969600: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-06 18:43:26.969664: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-06 18:43:26.969681: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-06 18:43:26.969791: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fec2:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|███████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:10<00:00, 107090.81it/s]1156085it [00:19, 57856.80it/s]
Uploading predictions to W&B
NDCG Average: 0.126119780

# NBOW+RNN attention
