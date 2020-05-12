# path

Epoch 5 (valid) took 1.61s [processed 14246 samples/second]
Validation: Loss: 1.000445 | MRR: 0.027055
2020-05-15 16:52:33.721784: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-15 16:52:33.721868: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-15 16:52:33.721885: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-15 16:52:33.721908: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-15 16:52:33.722007: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: d50b:00:00.0, compute capability: 3.7)Test-All MRR (bs=1,000): 0.030
FuncNameTest-All MRR (bs=1,000): 0.013
Validation-All MRR (bs=1,000): 0.026
Test-python MRR (bs=1,000): 0.030
FuncNameTest-python MRR (bs=1,000): 0.013
Validation-python MRR (bs=1,000): 0.025

wandb: Waiting for W&B process to finish, PID 57
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-loss 0.9977958870744242
wandb: \_step 41
wandb: \_timestamp 1589563524.8975682
wandb: \_runtime 6651.210229873657
wandb: train-mrr 0.03424822818885729
wandb: train-time-sec 40.799808740615845
wandb: val-loss 1.0004450393759685
wandb: val-mrr 0.027055366060008173
wandb: epoch 5
wandb: val-time-sec 1.614405870437622
wandb: best_val_mrr_loss 1.0046641360158506
wandb: best_val_mrr 0.1354088940827743
wandb: best_epoch 0
wandb: Test-All MRR (bs=1,000) 0.030044906043869158
wandb: FuncNameTest-All MRR (bs=1,000) 0.013023173410663648
wandb: Validation-All MRR (bs=1,000) 0.02609787323078061
wandb: Test-python MRR (bs=1,000) 0.030479113623771446
wandb: FuncNameTest-python MRR (bs=1,000) 0.012931617800338734
wandb: Validation-python MRR (bs=1,000) 0.025360441830172522
wandb: Syncing files in wandb/run-20200515_153447-c9p15ygq:
wandb: treepath-2020-05-15-15-34-47-graph.pbtxt
wandb: treepath-2020-05-15-15-34-47.train_log
wandb: treepath-2020-05-15-15-34-47_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced treepath-2020-05-15-15-34-47: https://app.wandb.ai/jianguda/CodeSearchNet/runs/c9p15ygq
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/c9p15ygq
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data] Package stopwords is already up-to-date!
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treepath-2020-05-15-15-34-47_model_best.pkl.gz
2020-05-15 17:26:55.917268: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 17:27:01.504086: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: d50b:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-15 17:27:01.504141: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
Fetching run files from W&B...
Restoring model from ./treepath-2020-05-15-15-34-47_model_best.pkl.gz
2020-05-15 17:26:55.917268: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 17:27:01.504086: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: d50b:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-15 17:27:01.504141: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-15 17:27:01.829496: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./treepath-2020-05-15-15-34-47_model_best.pkl.gz
2020-05-15 17:26:55.917268: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-15 17:27:01.504086: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: d50b:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-05-15 17:27:01.504141: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-05-15 17:27:01.829496: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-05-15 17:27:01.829565: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-05-15 17:27:01.829576: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-05-15 17:27:01.829724: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: d50b:00:00.0, compute capability: 3.7)WARNING:tensorflow:From /home/dev/code/models/model.py:320: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:06<00:00, 172288.32it/s]1156085it [00:19, 58819.66it/s]
Uploading predictions to W&B
NDCG Average: 0.003320833
