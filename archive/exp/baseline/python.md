# NBOW

root@jian-csn:/home/dev/src# python train.py --model neuralbow ../resources/saved_models ../resources/data/python/final/jsonl/train ../resources/data/python/final/jsonl/valid ../resources/data/python/final/jsonl/test
wandb: W&B is a tool that helps track and visualize machine learning experiments
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 2
wandb: You chose 'Use an existing W&B account'
wandb: You can find your API key in your browser here: https://app.wandb.ai/authorize
wandb: Paste an API key from your profile and hit enter
wandb: Appending key for api.wandb.ai to your netrc file: /root/.netrc
wandb: Started W&B process version 0.8.12 with PID 27
wandb: Wandb version 0.8.32 is available! To upgrade, please run:
wandb: \$ pip install wandb --upgrade
wandb: Local directory: wandb/run-20200421_221535-f8b0ko92  
wandb: Syncing run neuralbow-2020-04-21-22-15-35: https://app.wandb.ai/jianguda/CodeSearchNet/runs/f8b0ko92
wandb: Run `wandb off` to turn off syncing.

2020-04-21 22:15:52.825555: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-21 22:15:52.904974: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: f389:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-21 22:15:52.905039: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-21 22:15:53.186887: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-21 22:15:53.186947: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-21 22:15:53.186965: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-21 22:15:53.187078: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: f389:00:00.0, compute capability: 3.7)
Tokenizing and building vocabulary for code snippets and queries. This step may take several hours.
WARNING:tensorflow:From /home/dev/src/models/model.py:299: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Starting training run neuralbow-2020-04-21-22-15-35 of model NeuralBoWModel with following hypers:
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': False, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_nbow_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_nbow_pool_mode': 'weighted_mean', 'batch_size': 1000, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.01, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'cosine', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
Loading training and validation data.
Begin Training.
Training on 412178 python samples.
Validating on 23107 python samples.
==== Epoch 0 ====
Epoch 0 (train) took 39.90s [processed 10325 samples/second]
Training Loss: 1.005567
Epoch 0 (valid) took 1.49s [processed 15393 samples/second]
Validation: Loss: 1.002565 | MRR: 0.295129
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-21-22-15-35_model_best.pkl.gz'.
==== Epoch 1 ====
Epoch 1 (train) took 39.04s [processed 10552 samples/second]
Training Loss: 0.997605
Epoch 1 (valid) took 1.45s [processed 15807 samples/second]
Validation: Loss: 1.038714 | MRR: 0.391607
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-21-22-15-35_model_best.pkl.gz'.
==== Epoch 2 ====
Epoch 2 (train) took 39.40s [processed 10457 samples/second]
Training Loss: 0.963177
Epoch 2 (valid) took 1.45s [processed 15850 samples/second]
Validation: Loss: 1.066582 | MRR: 0.453797
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-21-22-15-35_model_best.pkl.gz'.
==== Epoch 3 ====
Epoch 3 (train) took 39.04s [processed 10552 samples/second]
Training Loss: 0.923722
Epoch 3 (valid) took 1.44s [processed 16015 samples/second]
Validation: Loss: 1.064832 | MRR: 0.464521
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-21-22-15-35_model_best.pkl.gz'.
==== Epoch 4 ====
Epoch 4 (train) took 39.40s [processed 10457 samples/second]
Training Loss: 0.908688
Epoch 4 (valid) took 1.48s [processed 15573 samples/second]
Validation: Loss: 1.065067 | MRR: 0.466978
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-21-22-15-35_model_best.pkl.gz'.
==== Epoch 5 ====
Epoch 5 (train) took 39.06s [processed 10547 samples/second]
Training Loss: 0.900389
Epoch 5 (valid) took 1.43s [processed 16049 samples/second]
Validation: Loss: 1.062576 | MRR: 0.468829
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-21-22-15-35_model_best.pkl.gz'.
==== Epoch 6 ====
Epoch 6 (train) took 39.31s [processed 10479 samples/second]
Training Loss: 0.894874
Epoch 6 (valid) took 1.44s [processed 15955 samples/second]
Validation: Loss: 1.062973 | MRR: 0.469452
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-21-22-15-35_model_best.pkl.gz'.
==== Epoch 7 ====
Epoch 7 (train) took 39.07s [processed 10544 samples/second]
Training Loss: 0.891059
Epoch 7 (valid) took 1.43s [processed 16040 samples/second]
Validation: Loss: 1.062950 | MRR: 0.469788
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-21-22-15-35_model_best.pkl.gz'.
==== Epoch 8 ====
Epoch 8 (train) took 39.09s [processed 10540 samples/second]
Training Loss: 0.887801
Epoch 8 (valid) took 1.46s [processed 15804 samples/second]
Validation: Loss: 1.062888 | MRR: 0.469741
==== Epoch 9 ====
Epoch 9 (train) took 39.08s [processed 10543 samples/second]
Training Loss: 0.885608
Epoch 9 (valid) took 1.45s [processed 15894 samples/second]
Validation: Loss: 1.063307 | MRR: 0.470618
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-21-22-15-35_model_best.pkl.gz'.
==== Epoch 10 ====
Epoch 10 (train) took 39.18s [processed 10516 samples/second]
Training Loss: 0.883883
Epoch 10 (valid) took 1.44s [processed 15971 samples/second]
Validation: Loss: 1.063784 | MRR: 0.472021
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-21-22-15-35_model_best.pkl.gz'.
==== Epoch 11 ====
Epoch 11 (train) took 39.15s [processed 10524 samples/second]
Training Loss: 0.881999
Epoch 11 (valid) took 1.45s [processed 15809 samples/second]
Validation: Loss: 1.064635 | MRR: 0.469373
==== Epoch 12 ====
Epoch 12 (train) took 39.18s [processed 10515 samples/second]
Training Loss: 0.880793
Epoch 12 (valid) took 1.44s [processed 16008 samples/second]
Validation: Loss: 1.063191 | MRR: 0.470703
==== Epoch 13 ====
Epoch 13 (train) took 39.07s [processed 10545 samples/second]
Training Loss: 0.879368
Epoch 13 (valid) took 1.43s [processed 16040 samples/second]
Validation: Loss: 1.064610 | MRR: 0.470415
==== Epoch 14 ====
Epoch 14 (train) took 39.29s [processed 10485 samples/second]
Training Loss: 0.878086
Epoch 14 (valid) took 1.46s [processed 15793 samples/second]
Validation: Loss: 1.064254 | MRR: 0.470189
==== Epoch 15 ====
Epoch 15 (train) took 39.04s [processed 10552 samples/second]
Training Loss: 0.877076
Epoch 15 (valid) took 1.44s [processed 15945 samples/second]
Validation: Loss: 1.065250 | MRR: 0.469467
2020-04-21 22:37:14.517883: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-21 22:37:14.517941: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-21 22:37:14.517956: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-21 22:37:14.517967: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-21 22:37:14.518063: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: f389:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.643
FuncNameTest-All MRR (bs=1,000): 0.525
Validation-All MRR (bs=1,000): 0.609
Test-python MRR (bs=1,000): 0.643
FuncNameTest-python MRR (bs=1,000): 0.525
Validation-python MRR (bs=1,000): 0.609

wandb: Waiting for W&B process to finish, PID 27
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_step 101
wandb: \_runtime 1410.0628073215485
wandb: train-mrr 0.8473652760033469
wandb: train-loss 0.877076058277806
wandb: \_timestamp 1587508744.0080469
wandb: epoch 15
wandb: val-mrr 0.46946718431555706
wandb: train-time-sec 39.04250693321228
wandb: val-time-sec 1.4423763751983643
wandb: val-loss 1.0652501686759617
wandb: best_val_mrr_loss 1.0637843090554941
wandb: best_val_mrr 0.4720205383300781
wandb: best_epoch 10
wandb: Test-All MRR (bs=1,000) 0.6431706756011462
wandb: FuncNameTest-All MRR (bs=1,000) 0.5250481424332979
wandb: Validation-All MRR (bs=1,000) 0.6093811653047474
wandb: Test-python MRR (bs=1,000) 0.6431706756011462
wandb: FuncNameTest-python MRR (bs=1,000) 0.5250481424332979
wandb: Validation-python MRR (bs=1,000) 0.6093811653047474
wandb: Syncing files in wandb/run-20200421_221535-f8b0ko92:
wandb: neuralbow-2020-04-21-22-15-35-graph.pbtxt
wandb: neuralbow-2020-04-21-22-15-35.train_log
wandb: neuralbow-2020-04-21-22-15-35_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced neuralbow-2020-04-21-22-15-35: https://app.wandb.ai/jianguda/CodeSearchNet/runs/f8b0ko92
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/f8b0ko92
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./neuralbow-2020-04-21-22-15-35_model_best.pkl.gz
2020-04-21 22:55:52.407717: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-21 22:55:57.253600: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: f389:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-21 22:55:57.253646: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-21 22:55:57.535840: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-21 22:55:57.535906: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-21 22:55:57.535923: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-21 22:55:57.536033: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: f389:00:00.0, compute capability: 3.7)
WARNING:tensorflow:From /home/dev/src/models/model.py:299: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:10<00:00, 112794.05it/s]1156085it [00:19, 58221.61it/s]
Uploading predictions to W&B
NDCG Average: 0.299385593

# CNN

root@jian-csn:/home/dev/src# python train.py --model 1dcnn ../resources/saved_models ../resources/data/python/final/jsonl/train ../resources/data/python/final/jsonl/valid ../resources/data/python/final/jsonl/test
wandb: Started W&B process version 0.8.12 with PID 232
wandb: Wandb version 0.8.32 is available! To upgrade, please run:
wandb: \$ pip install wandb --upgrade
wandb: Local directory: wandb/run-20200421_172210-0ibonvr0
wandb: Syncing run 1dcnn-2020-04-21-17-22-10: https://app.wandb.ai/jianguda/CodeSearchNet/runs/0ibonvr0
wandb: Run `wandb off` to turn off syncing.

2020-04-21 17:22:15.954719: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-21 17:22:16.032757: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: f389:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-21 17:22:16.032798: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-21 17:22:16.307719: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-21 17:22:16.307783: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-21 17:22:16.307799: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-21 17:22:16.307926: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: f389:00:00.0, compute capability: 3.7)
Tokenizing and building vocabulary for code snippets and queries. This step may take several hours.
Starting training run 1dcnn-2020-04-21-17-22-10 of model ConvolutionalModel with following hypers:
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': False, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_1dcnn_position_encoding': 'learned', 'code_1dcnn_layer_list': [128, 128, 128], 'code_1dcnn_kernel_width': [16, 16, 16], 'code_1dcnn_add_residual_connections': True, 'code_1dcnn_activation': 'tanh', 'code_1dcnn_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_1dcnn_position_encoding': 'learned', 'query_1dcnn_layer_list': [128, 128, 128], 'query_1dcnn_kernel_width': [16, 16, 16], 'query_1dcnn_add_residual_connections': True, 'query_1dcnn_activation': 'tanh', 'query_1dcnn_pool_mode': 'weighted_mean', 'batch_size': 1000, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.0005, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'softmax', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
Loading training and validation data.
Begin Training.
Training on 412178 python samples.
Validating on 23107 python samples.
==== Epoch 0 ====
Epoch 0 (train) took 282.54s [processed 1458 samples/second]
Training Loss: 5.878838
Epoch 0 (valid) took 4.88s [processed 4708 samples/second]
Validation: Loss: 5.499555 | MRR: 0.103939
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-21-17-22-10_model_best.pkl.gz'.
==== Epoch 1 ====
Epoch 1 (train) took 277.66s [processed 1483 samples/second]
Training Loss: 4.567062
Epoch 1 (valid) took 4.81s [processed 4778 samples/second]
Validation: Loss: 4.688248 | MRR: 0.229268
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-21-17-22-10_model_best.pkl.gz'.
==== Epoch 2 ====
Epoch 2 (train) took 278.60s [processed 1478 samples/second]
Training Loss: 3.613206
Epoch 2 (valid) took 4.78s [processed 4813 samples/second]
Validation: Loss: 4.200490 | MRR: 0.308035
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-21-17-22-10_model_best.pkl.gz'.
==== Epoch 3 ====
Epoch 3 (train) took 278.09s [processed 1481 samples/second]
Training Loss: 3.032420
Epoch 3 (valid) took 4.76s [processed 4828 samples/second]
Validation: Loss: 3.914650 | MRR: 0.354160
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-21-17-22-10_model_best.pkl.gz'.
==== Epoch 4 ====
Epoch 4 (train) took 277.79s [processed 1483 samples/second]
Training Loss: 2.632094
Epoch 4 (valid) took 4.78s [processed 4811 samples/second]
Validation: Loss: 3.776596 | MRR: 0.382525
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-21-17-22-10_model_best.pkl.gz'.
==== Epoch 5 ====
Epoch 5 (train) took 277.57s [processed 1484 samples/second]
Training Loss: 2.330840
Epoch 5 (valid) took 4.81s [processed 4781 samples/second]
Validation: Loss: 3.634266 | MRR: 0.404921
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-21-17-22-10_model_best.pkl.gz'.
==== Epoch 6 ====
Epoch 6 (train) took 278.14s [processed 1481 samples/second]
Training Loss: 2.101094
Epoch 6 (valid) took 4.81s [processed 4780 samples/second]
Validation: Loss: 3.572151 | MRR: 0.424287
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-21-17-22-10_model_best.pkl.gz'.
==== Epoch 7 ====
Epoch 7 (train) took 278.08s [processed 1481 samples/second]
Training Loss: 1.934784
Epoch 7 (valid) took 4.79s [processed 4803 samples/second]
Validation: Loss: 3.524823 | MRR: 0.436636
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-21-17-22-10_model_best.pkl.gz'.
==== Epoch 8 ====
Epoch 8 (train) took 277.82s [processed 1482 samples/second]
Training Loss: 1.799696
Epoch 8 (valid) took 4.79s [processed 4798 samples/second]
Validation: Loss: 3.500468 | MRR: 0.441169
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-21-17-22-10_model_best.pkl.gz'.
==== Epoch 9 ====
Epoch 9 (train) took 277.72s [processed 1483 samples/second]
Training Loss: 1.684016
Epoch 9 (valid) took 4.77s [processed 4823 samples/second]
Validation: Loss: 3.476304 | MRR: 0.447281
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-21-17-22-10_model_best.pkl.gz'.
==== Epoch 10 ====
Epoch 10 (train) took 277.27s [processed 1485 samples/second]
Training Loss: 1.587343
Epoch 10 (valid) took 4.81s [processed 4778 samples/second]
Validation: Loss: 3.467007 | MRR: 0.456319
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-21-17-22-10_model_best.pkl.gz'.
==== Epoch 11 ====
Epoch 11 (train) took 277.24s [processed 1486 samples/second]
Training Loss: 1.503658
Epoch 11 (valid) took 4.76s [processed 4829 samples/second]
Validation: Loss: 3.496499 | MRR: 0.459434
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-21-17-22-10_model_best.pkl.gz'.
==== Epoch 12 ====
Epoch 12 (train) took 277.14s [processed 1486 samples/second]
Training Loss: 1.435708
Epoch 12 (valid) took 4.78s [processed 4807 samples/second]
Validation: Loss: 3.484481 | MRR: 0.461386
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-21-17-22-10_model_best.pkl.gz'.
==== Epoch 13 ====
Epoch 13 (train) took 277.32s [processed 1485 samples/second]
Training Loss: 1.374179
Epoch 13 (valid) took 4.76s [processed 4835 samples/second]
Validation: Loss: 3.528935 | MRR: 0.462549
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-21-17-22-10_model_best.pkl.gz'.
==== Epoch 14 ====
Epoch 14 (train) took 277.39s [processed 1485 samples/second]
Training Loss: 1.316981
Epoch 14 (valid) took 4.74s [processed 4852 samples/second]
Validation: Loss: 3.537969 | MRR: 0.467476
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-21-17-22-10_model_best.pkl.gz'.
==== Epoch 15 ====
Epoch 15 (train) took 276.90s [processed 1487 samples/second]
Training Loss: 1.268822
Epoch 15 (valid) took 4.76s [processed 4828 samples/second]
Validation: Loss: 3.529026 | MRR: 0.469727
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-21-17-22-10_model_best.pkl.gz'.
==== Epoch 16 ====
Epoch 16 (train) took 276.80s [processed 1488 samples/second]
Training Loss: 1.220365
Epoch 16 (valid) took 4.74s [processed 4847 samples/second]
Validation: Loss: 3.538559 | MRR: 0.470539
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-21-17-22-10_model_best.pkl.gz'.
==== Epoch 17 ====
Epoch 17 (train) took 276.82s [processed 1488 samples/second]
Training Loss: 1.180363
Epoch 17 (valid) took 4.75s [processed 4844 samples/second]
Validation: Loss: 3.565952 | MRR: 0.469898
==== Epoch 18 ====
Epoch 18 (train) took 276.90s [processed 1487 samples/second]
Training Loss: 1.148456
Epoch 18 (valid) took 4.75s [processed 4843 samples/second]
Validation: Loss: 3.545369 | MRR: 0.474084
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-21-17-22-10_model_best.pkl.gz'.
==== Epoch 19 ====
Epoch 19 (train) took 277.20s [processed 1486 samples/second]
Training Loss: 1.112200
Epoch 19 (valid) took 4.75s [processed 4837 samples/second]
Validation: Loss: 3.582220 | MRR: 0.473571
==== Epoch 20 ====
Epoch 20 (train) took 276.93s [processed 1487 samples/second]
Training Loss: 1.080291
Epoch 20 (valid) took 4.76s [processed 4828 samples/second]
Validation: Loss: 3.594737 | MRR: 0.473743
==== Epoch 21 ====
Epoch 21 (train) took 276.70s [processed 1488 samples/second]
Training Loss: 1.052395
Epoch 21 (valid) took 4.73s [processed 4861 samples/second]
Validation: Loss: 3.586856 | MRR: 0.475795
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-21-17-22-10_model_best.pkl.gz'.
==== Epoch 22 ====
Epoch 22 (train) took 276.94s [processed 1487 samples/second]
Training Loss: 1.025242
Epoch 22 (valid) took 4.76s [processed 4836 samples/second]
Validation: Loss: 3.633659 | MRR: 0.478206
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-21-17-22-10_model_best.pkl.gz'.
==== Epoch 23 ====
Epoch 23 (train) took 276.88s [processed 1487 samples/second]
Training Loss: 1.000849
Epoch 23 (valid) took 4.78s [processed 4807 samples/second]
Validation: Loss: 3.650825 | MRR: 0.476913
==== Epoch 24 ====
Epoch 24 (train) took 277.03s [processed 1487 samples/second]
Training Loss: 0.979800
Epoch 24 (valid) took 4.74s [processed 4851 samples/second]
Validation: Loss: 3.650053 | MRR: 0.477394
==== Epoch 25 ====
Epoch 25 (train) took 277.03s [processed 1487 samples/second]
Training Loss: 0.957916
Epoch 25 (valid) took 4.77s [processed 4819 samples/second]
Validation: Loss: 3.697258 | MRR: 0.477014
==== Epoch 26 ====
Epoch 26 (train) took 276.84s [processed 1488 samples/second]
Training Loss: 0.938178
Epoch 26 (valid) took 4.76s [processed 4830 samples/second]
Validation: Loss: 3.668215 | MRR: 0.479558
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-21-17-22-10_model_best.pkl.gz'.
==== Epoch 27 ====
Epoch 27 (train) took 276.89s [processed 1487 samples/second]
Training Loss: 0.918189
Epoch 27 (valid) took 4.76s [processed 4828 samples/second]
Validation: Loss: 3.768112 | MRR: 0.477481
==== Epoch 28 ====
Epoch 28 (train) took 276.90s [processed 1487 samples/second]
Training Loss: 0.900299
Epoch 28 (valid) took 4.74s [processed 4854 samples/second]
Validation: Loss: 3.710293 | MRR: 0.479380
==== Epoch 29 ====
Epoch 29 (train) took 276.63s [processed 1489 samples/second]
Training Loss: 0.887924
Epoch 29 (valid) took 4.75s [processed 4840 samples/second]
Validation: Loss: 3.726466 | MRR: 0.478964
==== Epoch 30 ====
Epoch 30 (train) took 277.06s [processed 1487 samples/second]
Training Loss: 0.867852
Epoch 30 (valid) took 4.75s [processed 4840 samples/second]
Validation: Loss: 3.687557 | MRR: 0.481004
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-21-17-22-10_model_best.pkl.gz'.
==== Epoch 31 ====
Epoch 31 (train) took 276.68s [processed 1489 samples/second]
Training Loss: 0.857545
Epoch 31 (valid) took 4.75s [processed 4846 samples/second]
Validation: Loss: 3.790133 | MRR: 0.478924
==== Epoch 32 ====
Epoch 32 (train) took 276.93s [processed 1487 samples/second]
Training Loss: 0.843977
Epoch 32 (valid) took 4.78s [processed 4809 samples/second]
Validation: Loss: 3.765147 | MRR: 0.479610
==== Epoch 33 ====
Epoch 33 (train) took 276.87s [processed 1488 samples/second]
Training Loss: 0.832431
Epoch 33 (valid) took 4.77s [processed 4819 samples/second]
Validation: Loss: 3.741030 | MRR: 0.479555
==== Epoch 34 ====
Epoch 34 (train) took 277.06s [processed 1487 samples/second]
Training Loss: 0.816323
Epoch 34 (valid) took 4.74s [processed 4851 samples/second]
Validation: Loss: 3.768274 | MRR: 0.479718
==== Epoch 35 ====
Epoch 35 (train) took 276.50s [processed 1490 samples/second]
Training Loss: 0.802595
Epoch 35 (valid) took 4.74s [processed 4851 samples/second]
Validation: Loss: 3.794832 | MRR: 0.479240
2020-04-21 20:22:44.829093: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-21 20:22:44.829151: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-21 20:22:44.829167: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-21 20:22:44.829178: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-21 20:22:44.829273: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: f389:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.538
FuncNameTest-All MRR (bs=1,000): 0.557
Validation-All MRR (bs=1,000): 0.493
Test-python MRR (bs=1,000): 0.538
FuncNameTest-python MRR (bs=1,000): 0.557
Validation-python MRR (bs=1,000): 0.493

wandb: Waiting for W&B process to finish, PID 232
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_runtime 10963.921169281006
wandb: train-mrr 0.8554288709325698
wandb: \_step 221
wandb: \_timestamp 1587500693.1832793
wandb: train-loss 0.8025954385984291
wandb: train-time-sec 276.50243616104126
wandb: val-time-sec 4.740976095199585
wandb: Waiting for W&B process to finish, PID 232
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_runtime 10963.921169281006
wandb: train-mrr 0.8554288709325698
wandb: \_step 221
wandb: \_timestamp 1587500693.1832793
wandb: train-loss 0.8025954385984291
wandb: train-time-sec 276.50243616104126
wandb: val-time-sec 4.740976095199585
wandb: val-mrr 0.47924025958517324
wandb: epoch 35
wandb: val-loss 3.7948320948559306
wandb: best_val_mrr_loss 3.687557075334632
wandb: best_val_mrr 0.4810040349545686
wandb: best_epoch 30
wandb: Test-All MRR (bs=1,000) 0.5383198097685118
wandb: FuncNameTest-All MRR (bs=1,000) 0.5565043419314503
wandb: Validation-All MRR (bs=1,000) 0.4930699855905518
wandb: Test-python MRR (bs=1,000) 0.5383198097685118
wandb: FuncNameTest-python MRR (bs=1,000) 0.5565043419314503
wandb: Validation-python MRR (bs=1,000) 0.4930699855905518
wandb: Syncing files in wandb/run-20200421_172210-0ibonvr0:
wandb: 1dcnn-2020-04-21-17-22-10-graph.pbtxt
wandb: 1dcnn-2020-04-21-17-22-10.train_log
wandb: 1dcnn-2020-04-21-17-22-10_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced 1dcnn-2020-04-21-17-22-10: https://app.wandb.ai/jianguda/CodeSearchNet/runs/0ibonvr0
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/0ibonvr0
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./1dcnn-2020-04-21-17-22-10_model_best.pkl.gz
2020-04-21 21:28:38.200106: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-21 21:28:42.929057: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: f389:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-21 21:28:42.929102: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-21 21:28:43.207095: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-21 21:28:43.207155: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-21 21:28:43.207172: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-21 21:28:43.207280: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: f389:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:10<00:00, 112215.64it/s]1156085it [00:21, 54547.98it/s]
Uploading predictions to W&B
NDCG Average: 0.204386158

# RNN

root@jian-csn:/home/dev/src# python train.py --model rnn ../resources/saved_models ../resources/data/python/final/jsonl/train ../resources/data/python/final/jsonl/valid ../resources/data/python/final/jsonl/test
wandb: W&B is a tool that helps track and visualize machine learning experiments
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 2
wandb: You chose 'Use an existing W&B account'
wandb: You can find your API key in your browser here: https://app.wandb.ai/authorize
wandb: Paste an API key from your profile and hit enter
wandb: Appending key for api.wandb.ai to your netrc file: /root/.netrc
wandb: Started W&B process version 0.8.12 with PID 27
wandb: Wandb version 0.8.32 is available! To upgrade, please run:
wandb: \$ pip install wandb --upgrade
wandb: Local directory: wandb/run-20200421_135718-51wy2d7z  
wandb: Syncing run rnn-2020-04-21-13-57-18: https://app.wandb.ai/jianguda/CodeSearchNet/runs/51wy2d7z
wandb: Run `wandb off` to turn off syncing.

2020-04-21 13:57:38.617314: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-21 13:57:38.697204: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: f389:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-21 13:57:38.697249: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-21 13:57:38.978150: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-21 13:57:38.978209: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-21 13:57:38.978223: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-21 13:57:38.978336: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: f389:00:00.0, compute capability: 3.7)
Tokenizing and building vocabulary for code snippets and queries. This step may take several hours.
Starting training run rnn-2020-04-21-13-57-18 of model RNNModel with following hypers:
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': True, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_rnn_num_layers': 2, 'code_rnn_hidden_dim': 64, 'code_rnn_cell_type': 'LSTM', 'code_rnn_is_bidirectional': True, 'code_rnn_dropout_keep_rate': 0.8, 'code_rnn_recurrent_dropout_keep_rate': 1.0, 'code_rnn_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_rnn_num_layers': 2, 'query_rnn_hidden_dim': 64, 'query_rnn_cell_type': 'LSTM', 'query_rnn_is_bidirectional': True, 'query_rnn_dropout_keep_rate': 0.8, 'query_rnn_recurrent_dropout_keep_rate': 1.0, 'query_rnn_pool_mode': 'weighted_mean', 'batch_size': 1000, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.01, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'softmax', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
Loading training and validation data.
Begin Training.
Training on 412178 python samples.
Validating on 23107 python samples.
==== Epoch 0 ====
Epoch 0 (train) took 444.57s [processed 926 samples/second]
Training Loss: 4.741099
Epoch 0 (valid) took 12.14s [processed 1894 samples/second]
Validation: Loss: 3.861075 | MRR: 0.345737
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-21-13-57-18_model_best.pkl.gz'.
==== Epoch 1 ====
Epoch 1 (train) took 443.46s [processed 929 samples/second]
Training Loss: 2.708128
Epoch 1 (valid) took 11.88s [processed 1935 samples/second]
Validation: Loss: 3.302661 | MRR: 0.434498
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-21-13-57-18_model_best.pkl.gz'.
==== Epoch 2 ====
Epoch 2 (train) took 443.95s [processed 928 samples/second]
Training Loss: 2.209600
Epoch 2 (valid) took 11.86s [processed 1939 samples/second]
Validation: Loss: 3.148307 | MRR: 0.464842
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-21-13-57-18_model_best.pkl.gz'.
==== Epoch 3 ====
Epoch 3 (train) took 443.16s [processed 929 samples/second]
Training Loss: 1.992659
Epoch 3 (valid) took 11.84s [processed 1942 samples/second]
Validation: Loss: 3.070878 | MRR: 0.480826
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-21-13-57-18_model_best.pkl.gz'.
==== Epoch 4 ====
Epoch 4 (train) took 442.74s [processed 930 samples/second]
Training Loss: 1.867726
Epoch 4 (valid) took 11.85s [processed 1940 samples/second]
Validation: Loss: 3.027033 | MRR: 0.490169
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-21-13-57-18_model_best.pkl.gz'.
==== Epoch 5 ====
Epoch 5 (train) took 443.42s [processed 929 samples/second]
Training Loss: 1.795169
Epoch 5 (valid) took 11.84s [processed 1941 samples/second]
Validation: Loss: 3.025865 | MRR: 0.490264
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-21-13-57-18_model_best.pkl.gz'.
==== Epoch 6 ====
Epoch 6 (train) took 443.58s [processed 928 samples/second]
Training Loss: 1.738102
Epoch 6 (valid) took 11.83s [processed 1943 samples/second]
Validation: Loss: 3.003330 | MRR: 0.498259
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-21-13-57-18_model_best.pkl.gz'.
==== Epoch 7 ====
Epoch 7 (train) took 442.72s [processed 930 samples/second]
Training Loss: 1.709271
Epoch 7 (valid) took 11.83s [processed 1944 samples/second]
Validation: Loss: 3.005734 | MRR: 0.497837
==== Epoch 8 ====
Epoch 8 (train) took 442.08s [processed 931 samples/second]
Training Loss: 1.671090
Epoch 8 (valid) took 11.82s [processed 1945 samples/second]
Validation: Loss: 2.988750 | MRR: 0.500252
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-21-13-57-18_model_best.pkl.gz'.
==== Epoch 9 ====
Epoch 9 (train) took 442.69s [processed 930 samples/second]
Training Loss: 1.647007
Epoch 9 (valid) took 11.81s [processed 1946 samples/second]
Validation: Loss: 2.996502 | MRR: 0.499790
==== Epoch 10 ====
Epoch 10 (train) took 443.14s [processed 929 samples/second]
Training Loss: 1.628592
Epoch 10 (valid) took 11.82s [processed 1946 samples/second]
Validation: Loss: 2.969010 | MRR: 0.503132
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-21-13-57-18_model_best.pkl.gz'.
==== Epoch 11 ====
Epoch 11 (train) took 442.37s [processed 931 samples/second]
Training Loss: 1.614697
Epoch 11 (valid) took 11.82s [processed 1945 samples/second]
Validation: Loss: 2.973725 | MRR: 0.503885
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-21-13-57-18_model_best.pkl.gz'.
==== Epoch 12 ====
Epoch 12 (train) took 441.85s [processed 932 samples/second]
Training Loss: 1.603524
Epoch 12 (valid) took 11.81s [processed 1947 samples/second]
Validation: Loss: 2.972117 | MRR: 0.502485
==== Epoch 13 ====
Epoch 13 (train) took 442.67s [processed 930 samples/second]
Training Loss: 1.594760
Epoch 13 (valid) took 11.83s [processed 1944 samples/second]
Validation: Loss: 2.957960 | MRR: 0.506116
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-21-13-57-18_model_best.pkl.gz'.
==== Epoch 14 ====
Epoch 14 (train) took 442.97s [processed 930 samples/second]
Training Loss: 1.585080
Epoch 14 (valid) took 11.79s [processed 1951 samples/second]
Validation: Loss: 2.975749 | MRR: 0.506920
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-21-13-57-18_model_best.pkl.gz'.
==== Epoch 15 ====
Epoch 15 (train) took 442.32s [processed 931 samples/second]
Training Loss: 1.582968
Epoch 15 (valid) took 11.84s [processed 1942 samples/second]
Validation: Loss: 2.959061 | MRR: 0.506381
==== Epoch 16 ====
Epoch 16 (train) took 441.59s [processed 932 samples/second]
Training Loss: 1.580715
Epoch 16 (valid) took 11.78s [processed 1953 samples/second]
Validation: Loss: 2.960841 | MRR: 0.507738
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-21-13-57-18_model_best.pkl.gz'.
==== Epoch 17 ====
Epoch 17 (train) took 442.53s [processed 931 samples/second]
Training Loss: 1.579178
Epoch 17 (valid) took 11.84s [processed 1942 samples/second]
Validation: Loss: 2.976293 | MRR: 0.507063
==== Epoch 18 ====
Epoch 18 (train) took 443.24s [processed 929 samples/second]
Training Loss: 1.582384
Epoch 18 (valid) took 11.85s [processed 1940 samples/second]
Validation: Loss: 2.977681 | MRR: 0.506112
==== Epoch 19 ====
Epoch 19 (train) took 442.31s [processed 931 samples/second]
Training Loss: 1.587162
Epoch 19 (valid) took 11.87s [processed 1938 samples/second]
Validation: Loss: 3.004034 | MRR: 0.503139
==== Epoch 20 ====
Epoch 20 (train) took 441.83s [processed 932 samples/second]
Training Loss: 1.602382
Epoch 20 (valid) took 11.81s [processed 1947 samples/second]
Validation: Loss: 2.987758 | MRR: 0.504780
==== Epoch 21 ====
Epoch 21 (train) took 442.65s [processed 930 samples/second]
Training Loss: 1.590139
Epoch 21 (valid) took 11.81s [processed 1946 samples/second]
Validation: Loss: 3.003621 | MRR: 0.504819
2020-04-21 16:55:04.487781: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-21 16:55:04.487846: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-21 16:55:04.487861: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-21 16:55:04.487873: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-21 16:55:04.487962: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: f389:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.643
FuncNameTest-All MRR (bs=1,000): 0.628
Validation-All MRR (bs=1,000): 0.595
Test-python MRR (bs=1,000): 0.643
FuncNameTest-python MRR (bs=1,000): 0.628
Validation-python MRR (bs=1,000): 0.595

wandb: Waiting for W&B process to finish, PID 27
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_runtime 10836.57086777687
wandb: train-mrr 0.7254979628775884
wandb: \_step 137
wandb: train-loss 1.5901389440286506
wandb: \_timestamp 1587488273.90377
wandb: val-loss 3.0036208007646645
wandb: val-time-sec 11.813484907150269
wandb: train-time-sec 442.65490984916687
wandb: epoch 21
wandb: val-mrr 0.5048191130264945
wandb: best_val_mrr_loss 2.96084071242291
wandb: best_val_mrr 0.5077379760742188
wandb: best_epoch 16
wandb: Test-All MRR (bs=1,000) 0.6432190565659188
wandb: FuncNameTest-All MRR (bs=1,000) 0.6280729017602099
wandb: val-mrr 0.5048191130264945
wandb: best_val_mrr_loss 2.96084071242291
wandb: best_val_mrr 0.5077379760742188
wandb: best_epoch 16
wandb: Test-All MRR (bs=1,000) 0.6432190565659188
wandb: FuncNameTest-All MRR (bs=1,000) 0.6280729017602099
wandb: Validation-All MRR (bs=1,000) 0.5949391241057485
wandb: Test-python MRR (bs=1,000) 0.6432190565659188
wandb: FuncNameTest-python MRR (bs=1,000) 0.6280729017602099
wandb: Validation-python MRR (bs=1,000) 0.5949391241057485
wandb: Syncing files in wandb/run-20200421_135718-51wy2d7z:
wandb: rnn-2020-04-21-13-57-18-graph.pbtxt
wandb: FuncNameTest-All MRR (bs=1,000) 0.6280729017602099
wandb: Validation-All MRR (bs=1,000) 0.5949391241057485
wandb: Test-python MRR (bs=1,000) 0.6432190565659188
wandb: FuncNameTest-python MRR (bs=1,000) 0.6280729017602099
wandb: Validation-python MRR (bs=1,000) 0.5949391241057485
wandb: Syncing files in wandb/run-20200421_135718-51wy2d7z:
wandb: rnn-2020-04-21-13-57-18-graph.pbtxt
wandb: rnn-2020-04-21-13-57-18.train_log
wandb: rnn-2020-04-21-13-57-18_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced rnn-2020-04-21-13-57-18: https://app.wandb.ai/jianguda/CodeSearchNet/runs/51wy2d7z
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/51wy2d7z  
wandb: Test-python MRR (bs=1,000) 0.6432190565659188
wandb: FuncNameTest-python MRR (bs=1,000) 0.6280729017602099
wandb: Validation-python MRR (bs=1,000) 0.5949391241057485
wandb: Syncing files in wandb/run-20200421_135718-51wy2d7z:
wandb: rnn-2020-04-21-13-57-18-graph.pbtxt
wandb: rnn-2020-04-21-13-57-18.train_log
wandb: rnn-2020-04-21-13-57-18_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced rnn-2020-04-21-13-57-18: https://app.wandb.ai/jianguda/CodeSearchNet/runs/51wy2d7z
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/51wy2d7z  
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./rnn-2020-04-21-13-57-18_model_best.pkl.gz
2020-04-21 17:00:37.568361: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-21 17:00:42.254967: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: f389:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-21 17:00:42.255013: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-21 17:00:42.541747: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-21 17:00:42.541812: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-21 17:00:42.541830: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-21 17:00:42.541942: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: f389:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 203602.98it/s]1156085it [00:19, 57871.84it/s]
Uploading predictions to W&B
NDCG Average: 0.184494158

# BERT

root@jian-csn:/home/dev/src# python train.py --model selfatt ../resources/saved_models ../resources/data/python/final/jsonl/train ../resources/data/python/final/jsonl/valid ../resources/data/python/final/jsonl/test
wandb: W&B is a tool that helps track and visualize machine learning experiments
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 2
wandb: You chose 'Use an existing W&B account'
wandb: You can find your API key in your browser here: https://app.wandb.ai/authorize
wandb: Paste an API key from your profile and hit enter
wandb: Appending key for api.wandb.ai to your netrc file: /root/.netrc
wandb: Started W&B process version 0.8.12 with PID 27
wandb: Wandb version 0.8.32 is available! To upgrade, please run:
wandb: \$ pip install wandb --upgrade
wandb: Local directory: wandb/run-20200420_072440-0dnxbbq4  
wandb: Syncing run selfatt-2020-04-20-07-24-40: https://app.wandb.ai/jianguda/CodeSearchNet/runs/0dnxbbq4
wandb: Run `wandb off` to turn off syncing.

2020-04-20 07:25:06.624840: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-20 07:25:06.785409: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: f389:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-20 07:25:06.785453: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-20 07:25:19.600832: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-20 07:25:19.600890: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-20 07:25:19.600909: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-20 07:25:19.601077: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: f389:00:00.0, compute capability: 3.7)
Tokenizing and building vocabulary for code snippets and queries. This step may take several hours.
Starting training run selfatt-2020-04-20-07-24-40 of model SelfAttentionModel with following hypers:
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': False, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_self_attention_activation': 'gelu', 'code_self_attention_hidden_size': 128, 'code_self_attention_intermediate_size': 512, 'code_self_attention_num_layers': 3, 'code_self_attention_num_heads': 8, 'code_self_attention_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_self_attention_activation': 'gelu', 'query_self_attention_hidden_size': 128, 'query_self_attention_intermediate_size': 512, 'query_self_attention_num_layers': 3, 'query_self_attention_num_heads': 8, 'query_self_attention_pool_mode': 'weighted_mean', 'batch_size': 450, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.0005, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'softmax', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
Loading training and validation data.
Begin Training.
Training on 412178 python samples.
Validating on 23107 python samples.
==== Epoch 0 ====
Epoch 0 (train) took 1290.72s [processed 319 samples/second]
Training Loss: 2.531904
Epoch 0 (valid) took 30.66s [processed 748 samples/second]
Validation: Loss: 2.879893 | MRR: 0.519129
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-20-07-24-40_model_best.pkl.gz'.
==== Epoch 1 ====
Epoch 1 (train) took 1294.97s [processed 317 samples/second]
Training Loss: 0.981238
Epoch 1 (valid) took 30.59s [processed 750 samples/second]
Validation: Loss: 2.699396 | MRR: 0.561203
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-20-07-24-40_model_best.pkl.gz'.
==== Epoch 2 ====
Epoch 2 (train) took 1293.48s [processed 318 samples/second]
Training Loss: 0.677104
Epoch 2 (valid) took 30.21s [processed 759 samples/second]
Validation: Loss: 2.725114 | MRR: 0.571176
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-20-07-24-40_model_best.pkl.gz'.
==== Epoch 3 ====
wandb: Network error resolved after 0:00:18.328419, resuming normal operation.r: 0.5098. MRR so far: 0.9126
wandb: Network error resolved after 0:00:01.397599, resuming normal operation.r: 0.5104. MRR so far: 0.9124
Epoch 3 (train) took 1293.89s [processed 318 samples/second]
Training Loss: 0.529656
Epoch 3 (valid) took 30.38s [processed 755 samples/second]
Validation: Loss: 2.770910 | MRR: 0.575010
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-20-07-24-40_model_best.pkl.gz'.
==== Epoch 4 ====
Epoch 4 (train) took 1293.77s [processed 318 samples/second]
Training Loss: 0.442653
Epoch 4 (valid) took 30.55s [processed 751 samples/second]
Validation: Loss: 2.861778 | MRR: 0.573883
==== Epoch 5 ====
Epoch 5 (train) took 1293.41s [processed 318 samples/second]
Training Loss: 0.380969
Epoch 5 (valid) took 30.28s [processed 757 samples/second]
Validation: Loss: 2.823424 | MRR: 0.569925
==== Epoch 6 ====
Epoch 6 (train) took 1294.91s [processed 317 samples/second]
Training Loss: 0.336186
Epoch 6 (valid) took 30.46s [processed 753 samples/second]
Validation: Loss: 2.898723 | MRR: 0.580194
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-20-07-24-40_model_best.pkl.gz'.
==== Epoch 7 ====
Epoch 7 (train) took 1293.78s [processed 318 samples/second]
Training Loss: 0.303143
Epoch 7 (valid) took 30.38s [processed 755 samples/second]
Validation: Loss: 2.995800 | MRR: 0.579054
==== Epoch 8 ====
Epoch 8 (train) took 1293.47s [processed 318 samples/second]
Training Loss: 0.277991
Epoch 8 (valid) took 30.32s [processed 757 samples/second]
Validation: Loss: 3.018289 | MRR: 0.577242
==== Epoch 9 ====
Epoch 9 (train) took 1294.39s [processed 318 samples/second]
Training Loss: 0.256967
Epoch 9 (valid) took 30.47s [processed 753 samples/second]
Validation: Loss: 3.068105 | MRR: 0.579269
==== Epoch 10 ====
Epoch 10 (train) took 1294.22s [processed 318 samples/second]
Training Loss: 0.240215
Epoch 10 (valid) took 30.35s [processed 756 samples/second]
Validation: Loss: 3.129440 | MRR: 0.583158
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-20-07-24-40_model_best.pkl.gz'.
==== Epoch 11 ====
Epoch 11 (train) took 1294.50s [processed 318 samples/second]
Training Loss: 0.225012
Epoch 11 (valid) took 30.38s [processed 755 samples/second]
Validation: Loss: 3.110826 | MRR: 0.582022
==== Epoch 12 ====
Epoch 12 (train) took 1295.41s [processed 317 samples/second]
Training Loss: 0.214434
Epoch 12 (valid) took 30.35s [processed 756 samples/second]
Validation: Loss: 3.161927 | MRR: 0.577164
==== Epoch 13 ====
Epoch 13 (train) took 1294.74s [processed 318 samples/second]
Training Loss: 0.204726
Epoch 13 (valid) took 30.29s [processed 757 samples/second]
Validation: Loss: 3.148747 | MRR: 0.581379
==== Epoch 14 ====
Epoch 14 (train) took 1294.97s [processed 317 samples/second]
Training Loss: 0.196432
Epoch 14 (valid) took 30.55s [processed 751 samples/second]
Validation: Loss: 3.237172 | MRR: 0.581070
==== Epoch 15 ====
Epoch 15 (train) took 1293.69s [processed 318 samples/second]
Training Loss: 0.187412
Epoch 15 (valid) took 30.29s [processed 757 samples/second]
Validation: Loss: 3.178661 | MRR: 0.581241
2020-04-20 13:29:06.973962: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-20 13:29:06.974021: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-20 13:29:06.974036: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-20 13:29:06.974048: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-20 13:29:06.974138: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: f389:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.677
FuncNameTest-All MRR (bs=1,000): 0.631
Validation-All MRR (bs=1,000): 0.630
Test-python MRR (bs=1,000): 0.677
FuncNameTest-python MRR (bs=1,000): 0.631
Validation-python MRR (bs=1,000): 0.630

wandb: Waiting for W&B process to finish, PID 27
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-mrr 0.96686544975832
wandb: \_timestamp 1587389580.6786277
wandb: \_runtime 22110.735265016556
wandb: \_step 181
wandb: train-loss 0.18741244672914673
wandb: Waiting for W&B process to finish, PID 27
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-mrr 0.96686544975832
wandb: \_timestamp 1587389580.6786277
wandb: \_runtime 22110.735265016556
wandb: \_step 181
wandb: train-loss 0.18741244672914673
wandb: val-mrr 0.5812412911874276
wandb: val-loss 3.178661213201635
wandb: train-time-sec 1293.6921164989471
wandb: val-time-sec 30.291332721710205
wandb: epoch 15
wandb: best_val_mrr_loss 3.1294403567033657
wandb: best_val_mrr 0.5831583467703759
wandb: best_epoch 10
wandb: Test-All MRR (bs=1,000) 0.676922503944287
wandb: FuncNameTest-All MRR (bs=1,000) 0.6307370092913241
wandb: Validation-All MRR (bs=1,000) 0.6303207590171781
wandb: Test-python MRR (bs=1,000) 0.676922503944287
wandb: FuncNameTest-python MRR (bs=1,000) 0.6307370092913241
wandb: Validation-python MRR (bs=1,000) 0.6303207590171781
wandb: Syncing files in wandb/run-20200420_072440-0dnxbbq4:
wandb: selfatt-2020-04-20-07-24-40-graph.pbtxt
wandb: selfatt-2020-04-20-07-24-40.train_log
wandb: selfatt-2020-04-20-07-24-40_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced selfatt-2020-04-20-07-24-40: https://app.wandb.ai/jianguda/CodeSearchNet/runs/0dnxbbq4
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/0dnxbbq4
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./selfatt-2020-04-20-07-24-40_model_best.pkl.gz
2020-04-20 14:25:23.779591: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-20 14:25:28.545699: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: f389:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-20 14:25:28.545743: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-20 14:25:28.822829: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-20 14:25:28.822892: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-20 14:25:28.822909: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-20 14:25:28.823016: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: f389:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 206768.49it/s]1156085it [00:19, 58151.77it/s]
Uploading predictions to W&B
NDCG Average: 0.139365385

# Tree (leaf-token) (RNN)

root@jian-csn:/home/dev/src# python train.py --model tree ../resources/saved_models ../resources/data/python/final/jsonl/train ../resources/data/python/final/jsonl/valid ../resources/data/python/final/jsonl/test

Epoch 24 (valid) took 11.95s [processed 1924 samples/second]
Validation: Loss: 2.867498 | MRR: 0.525917
2020-04-20 18:56:16.497837: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-20 18:56:16.497891: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-20 18:56:16.497905: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-20 18:56:16.497916: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-20 18:56:16.498014: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: f389:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.655
FuncNameTest-All MRR (bs=1,000): 0.651
Validation-All MRR (bs=1,000): 0.609
Test-python MRR (bs=1,000): 0.655
FuncNameTest-python MRR (bs=1,000): 0.651
Validation-python MRR (bs=1,000): 0.609

wandb: Waiting for W&B process to finish, PID 48
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_timestamp 1587409177.3122752
wandb: train-mrr 0.7283491450930105
wandb: train-loss 1.5997129782889654
wandb: \_step 155
wandb: \_runtime 12353.05219578743
wandb: val-time-sec 11.949178218841553
wandb: epoch 24
wandb: train-time-sec 446.96576023101807
wandb: val-mrr 0.5259173093049423
wandb: val-loss 2.867498491121375
wandb: best_val_mrr_loss 2.860019476517387
wandb: best_val_mrr 0.5281504065472147
wandb: best_epoch 19
wandb: Test-All MRR (bs=1,000) 0.6550104608198185
wandb: FuncNameTest-All MRR (bs=1,000) 0.651233484031534
wandb: Validation-All MRR (bs=1,000) 0.6093477965810784
wandb: Test-python MRR (bs=1,000) 0.6550104608198185
wandb: FuncNameTest-python MRR (bs=1,000) 0.651233484031534
wandb: Validation-python MRR (bs=1,000) 0.6093477965810784
wandb: Syncing files in wandb/run-20200420_153345-wkqffq2m:
wandb: tree-2020-04-20-15-33-45-graph.pbtxt
wandb: tree-2020-04-20-15-33-45.train_log
wandb: tree-2020-04-20-15-33-45_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced tree-2020-04-20-15-33-45: https://app.wandb.ai/jianguda/CodeSearchNet/runs/wkqffq2m
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/wkqffq2m
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./tree-2020-04-20-15-33-45_model_best.pkl.gz
2020-04-20 20:14:08.607057: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-20 20:14:13.433969: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: f389:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-20 20:14:13.434016: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-20 20:14:13.713644: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./tree-2020-04-20-15-33-45_model_best.pkl.gz
2020-04-20 20:14:08.607057: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-20 20:14:13.433969: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: f389:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-20 20:14:13.434016: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-20 20:14:13.713644: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-20 20:14:13.713706: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-20 20:14:13.713724: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-20 20:14:13.713831: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: f389:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 199093.93it/s]1156085it [00:19, 58022.59it/s]
Uploading predictions to W&B
NDCG Average: 0.182565464

# Tree (leaf-token) (RNN + attention)

Epoch 23 (valid) took 13.01s [processed 1767 samples/second]
Validation: Loss: 2.924728 | MRR: 0.518883
2020-04-22 02:47:57.516598: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-22 02:47:57.516660: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-22 02:47:57.516675: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-22 02:47:57.516687: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-22 02:47:57.516794: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: f389:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.655
FuncNameTest-All MRR (bs=1,000): 0.646
Validation-All MRR (bs=1,000): 0.608
Test-python MRR (bs=1,000): 0.655
FuncNameTest-python MRR (bs=1,000): 0.646
Validation-python MRR (bs=1,000): 0.608

wandb: Waiting for W&B process to finish, PID 49
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_timestamp 1587523880.2246943
wandb: train-loss 1.6241397932895179
wandb: train-mrr 0.724240914205903
wandb: \_runtime 12590.704892396927
wandb: \_step 149
wandb: val-loss 2.924727781959202
wandb: val-time-sec 13.012800216674805
wandb: epoch 23
wandb: train-time-sec 472.85001969337463
wandb: val-mrr 0.5188827859629755
wandb: best_val_mrr_loss 2.8994908332824707
wandb: best_val_mrr 0.520839259935462
wandb: best_epoch 18
wandb: Test-All MRR (bs=1,000) 0.6550869031526422
wandb: FuncNameTest-All MRR (bs=1,000) 0.6457047533912358
wandb: Validation-All MRR (bs=1,000) 0.6082310524279457
wandb: Test-python MRR (bs=1,000) 0.6550869031526422
wandb: FuncNameTest-python MRR (bs=1,000) 0.6457047533912358
wandb: Validation-python MRR (bs=1,000) 0.6082310524279457
wandb: Syncing files in wandb/run-20200421_232130-4hru2lxb:
wandb: tree-2020-04-21-23-21-30-graph.pbtxt
wandb: tree-2020-04-21-23-21-30.train_log
wandb: tree-2020-04-21-23-21-30_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced tree-2020-04-21-23-21-30: https://app.wandb.ai/jianguda/CodeSearchNet/runs/4hru2lxb
root@jian-csn:/home/dev/code# python predict.py -r github/CodeSearchNet/4hru2lxb
Fetching run from W&B...
ERROR: Problem querying W&B for wandb_run_id: github/CodeSearchNet/4hru2lxb
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/4hru2lxb
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./tree-2020-04-21-23-21-30_model_best.pkl.gz
2020-04-22 05:12:23.544214: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-22 05:12:28.366943: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: f389:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-22 05:12:28.366990: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-22 05:12:28.641838: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-22 05:12:28.641900: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-22 05:12:28.641917: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-22 05:12:28.642028: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: f389:00:00.0, compute capability: 3.7)
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/4hru2lxb
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./tree-2020-04-21-23-21-30_model_best.pkl.gz
2020-04-22 05:12:23.544214: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-22 05:12:28.366943: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: f389:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-22 05:12:28.366990: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-22 05:12:28.641838: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-22 05:12:28.641900: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-22 05:12:28.641917: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-22 05:12:28.642028: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: f389:00:00.0, compute capability: 3.7)
wandb: Synced tree-2020-04-21-23-21-30: https://app.wandb.ai/jianguda/CodeSearchNet/runs/4hru2lxb
root@jian-csn:/home/dev/code# python predict.py -r github/CodeSearchNet/4hru2lxb  
Fetching run from W&B...
ERROR: Problem querying W&B for wandb_run_id: github/CodeSearchNet/4hru2lxb
root@jian-csn:/home/dev/code# python predict.py -r jianguda/CodeSearchNet/4hru2lxb
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./tree-2020-04-21-23-21-30_model_best.pkl.gz
2020-04-22 05:12:23.544214: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-22 05:12:28.366943: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: f389:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-22 05:12:28.366990: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-22 05:12:28.641838: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-22 05:12:28.641900: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-22 05:12:28.641917: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-22 05:12:28.642028: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: f389:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|██████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:05<00:00, 193718.50it/s]1156085it [00:19, 57944.65it/s]
Uploading predictions to W&B
NDCG Average: 0.189511453
