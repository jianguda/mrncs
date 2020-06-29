python train.py --model neuralbow ../resources/saved_models ../resources/data/php/final/jsonl/train ../resources/data/php/final/jsonl/valid ../resources/data/php/final/jsonl/test

python predict.py -r jianguda/CodeSearchNet/0123456

# NBOW

root@jian-csn:/home/dev/src# python train.py --model neuralbow ../resources/saved_models ../resources/data/php/final/jsonl/train ../resources/data/php/final/jsonl/valid ../resources/data/php/final/jsonl/test
wandb: W&B is a tool that helps track and visualize machine learning experiments
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 2
wandb: You chose 'Use an existing W&B account'
wandb: You can find your API key in your browser here: https://app.wandb.ai/authorize
wandb: Paste an API key from your profile and hit enter
wandb: Appending key for api.wandb.ai to your netrc file: /root/.netrc
wandb: Started W&B process version 0.8.12 with PID 29
wandb: Wandb version 0.8.32 is available! To upgrade, please run:
wandb: \$ pip install wandb --upgrade
wandb: Local directory: wandb/run-20200424_121423-uhqzjbb0
wandb: Syncing run neuralbow-2020-04-24-12-14-23: https://app.wandb.ai/jianguda/CodeSearchNet/runs/uhqzjbb0
wandb: Run `wandb off` to turn off syncing.

2020-04-24 12:14:48.848896: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-24 12:14:49.028556: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fbae:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-24 12:14:49.028608: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-24 12:15:01.584702: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-24 12:15:01.584765: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-24 12:15:01.584783: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-24 12:15:01.584922: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fbae:00:00.0, compute capability: 3.7)
Tokenizing and building vocabulary for code snippets and queries. This step may take several hours.
WARNING:tensorflow:From /home/dev/src/models/model.py:299: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Starting training run neuralbow-2020-04-24-12-14-23 of model NeuralBoWModel with following hypers:
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': False, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_nbow_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_nbow_pool_mode': 'weighted_mean', 'batch_size': 1000, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.01, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'cosine', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
Loading training and validation data.
Begin Training.
Training on 523712 php samples.
Validating on 26015 php samples.
==== Epoch 0 ====
Epoch 0 (train) took 53.59s [processed 9758 samples/second]
Training Loss: 1.004832
Epoch 0 (valid) took 1.68s [processed 15458 samples/second]
Validation: Loss: 1.002159 | MRR: 0.300452
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-24-12-14-23_model_best.pkl.gz'.
==== Epoch 1 ====
Epoch 1 (train) took 50.95s [processed 10265 samples/second]
Training Loss: 0.999713
Epoch 1 (valid) took 1.64s [processed 15809 samples/second]
Validation: Loss: 1.017304 | MRR: 0.383607
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-24-12-14-23_model_best.pkl.gz'.
==== Epoch 2 ====
Epoch 2 (train) took 51.25s [processed 10205 samples/second]
Training Loss: 0.987593
Epoch 2 (valid) took 1.66s [processed 15693 samples/second]
Validation: Loss: 1.044374 | MRR: 0.434858
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-24-12-14-23_model_best.pkl.gz'.
==== Epoch 3 ====
Epoch 3 (train) took 51.32s [processed 10190 samples/second]
Training Loss: 0.967044
Epoch 3 (valid) took 1.67s [processed 15586 samples/second]
Validation: Loss: 1.053633 | MRR: 0.456122
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-24-12-14-23_model_best.pkl.gz'.
==== Epoch 4 ====
Epoch 4 (train) took 51.59s [processed 10137 samples/second]
Training Loss: 0.952197
Epoch 4 (valid) took 1.70s [processed 15306 samples/second]
Validation: Loss: 1.054976 | MRR: 0.466913
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-24-12-14-23_model_best.pkl.gz'.
==== Epoch 5 ====
Epoch 5 (train) took 51.49s [processed 10156 samples/second]
Training Loss: 0.941737
Epoch 5 (valid) took 1.62s [processed 16023 samples/second]
Validation: Loss: 1.054053 | MRR: 0.470760
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-24-12-14-23_model_best.pkl.gz'.
==== Epoch 6 ====
Epoch 6 (train) took 51.38s [processed 10179 samples/second]
Training Loss: 0.934098
Epoch 6 (valid) took 1.63s [processed 15979 samples/second]
Validation: Loss: 1.053151 | MRR: 0.473087
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-24-12-14-23_model_best.pkl.gz'.
==== Epoch 7 ====
Epoch 7 (train) took 51.33s [processed 10188 samples/second]
Training Loss: 0.928067
Epoch 7 (valid) took 1.63s [processed 15986 samples/second]
Validation: Loss: 1.051426 | MRR: 0.474269
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-24-12-14-23_model_best.pkl.gz'.
==== Epoch 8 ====
Epoch 8 (train) took 51.21s [processed 10212 samples/second]
Training Loss: 0.924130
Epoch 8 (valid) took 1.66s [processed 15699 samples/second]
Validation: Loss: 1.049363 | MRR: 0.476543
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-24-12-14-23_model_best.pkl.gz'.
==== Epoch 9 ====
Epoch 9 (train) took 50.78s [processed 10299 samples/second]
Training Loss: 0.920722
Epoch 9 (valid) took 1.64s [processed 15901 samples/second]
Validation: Loss: 1.049397 | MRR: 0.477425
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-24-12-14-23_model_best.pkl.gz'.
==== Epoch 10 ====
Epoch 10 (train) took 51.15s [processed 10224 samples/second]
Training Loss: 0.918114
Epoch 10 (valid) took 1.64s [processed 15844 samples/second]
Validation: Loss: 1.047223 | MRR: 0.478637
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-24-12-14-23_model_best.pkl.gz'.
==== Epoch 11 ====
Epoch 11 (train) took 51.45s [processed 10165 samples/second]
Training Loss: 0.915998
Epoch 11 (valid) took 1.69s [processed 15395 samples/second]
Validation: Loss: 1.047095 | MRR: 0.478045
==== Epoch 12 ====
Epoch 12 (train) took 51.10s [processed 10234 samples/second]
Training Loss: 0.913601
Epoch 12 (valid) took 1.60s [processed 16249 samples/second]
Validation: Loss: 1.046223 | MRR: 0.478803
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-24-12-14-23_model_best.pkl.gz'.
==== Epoch 13 ====
Epoch 13 (train) took 50.12s [processed 10434 samples/second]
Training Loss: 0.912056
Epoch 13 (valid) took 1.62s [processed 16098 samples/second]
Validation: Loss: 1.045956 | MRR: 0.477629
==== Epoch 14 ====
Epoch 14 (train) took 50.45s [processed 10366 samples/second]
Training Loss: 0.910586
Epoch 14 (valid) took 1.63s [processed 15979 samples/second]
Validation: Loss: 1.045023 | MRR: 0.479450
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-24-12-14-23_model_best.pkl.gz'.
==== Epoch 15 ====
Epoch 15 (train) took 50.02s [processed 10454 samples/second]
Training Loss: 0.909101
Epoch 15 (valid) took 1.60s [processed 16246 samples/second]
Validation: Loss: 1.044739 | MRR: 0.479402
==== Epoch 16 ====
Epoch 16 (train) took 50.65s [processed 10325 samples/second]
Training Loss: 0.908145
Epoch 16 (valid) took 1.61s [processed 16115 samples/second]
Validation: Loss: 1.044311 | MRR: 0.479038
==== Epoch 17 ====
Epoch 17 (train) took 50.91s [processed 10273 samples/second]
Training Loss: 0.907216
Epoch 17 (valid) took 1.65s [processed 15777 samples/second]
Validation: Loss: 1.043490 | MRR: 0.478598
==== Epoch 18 ====
Epoch 18 (train) took 50.46s [processed 10364 samples/second]
Training Loss: 0.906373
Epoch 18 (valid) took 1.61s [processed 16173 samples/second]
Validation: Loss: 1.043985 | MRR: 0.479708
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-24-12-14-23_model_best.pkl.gz'.
==== Epoch 19 ====
Epoch 19 (train) took 50.04s [processed 10450 samples/second]
Training Loss: 0.904824
Epoch 19 (valid) took 1.60s [processed 16215 samples/second]
Validation: Loss: 1.043523 | MRR: 0.478850
==== Epoch 20 ====
Epoch 20 (train) took 49.86s [processed 10490 samples/second]
Training Loss: 0.904180
Epoch 20 (valid) took 1.60s [processed 16221 samples/second]
Validation: Loss: 1.043409 | MRR: 0.478446
==== Epoch 21 ====
Epoch 21 (train) took 50.15s [processed 10428 samples/second]
Training Loss: 0.903238
Epoch 21 (valid) took 1.61s [processed 16182 samples/second]
Validation: Loss: 1.042428 | MRR: 0.478958
==== Epoch 22 ====
Epoch 22 (train) took 49.93s [processed 10475 samples/second]
Training Loss: 0.902308
Epoch 22 (valid) took 1.60s [processed 16223 samples/second]
Validation: Loss: 1.042910 | MRR: 0.477111
==== Epoch 23 ====
Epoch 23 (train) took 50.31s [processed 10394 samples/second]
Training Loss: 0.901471
Epoch 23 (valid) took 1.65s [processed 15764 samples/second]
Validation: Loss: 1.041549 | MRR: 0.479659
2020-04-24 12:47:10.305334: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-24 12:47:10.305406: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-24 12:47:10.305421: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-24 12:47:10.305432: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-24 12:47:10.305544: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fbae:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.568
FuncNameTest-All MRR (bs=1,000): 0.682
Validation-All MRR (bs=1,000): 0.577
Test-php MRR (bs=1,000): 0.568
FuncNameTest-php MRR (bs=1,000): 0.682
Validation-php MRR (bs=1,000): 0.577

wandb: Waiting for W&B process to finish, PID 29
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-mrr 0.7614874407620549
wandb: \_runtime 2101.676094532013
wandb: \_timestamp 1587732554.2902439
wandb: train-loss 0.9014708384956958
wandb: \_step 173
wandb: epoch 23
wandb: val-mrr 0.4796589637169471
wandb: train-time-sec 50.31468939781189
wandb: val-time-sec 1.6492671966552734
wandb: val-loss 1.0415492011950567
wandb: best_val_mrr_loss 1.043985467690688
wandb: best_val_mrr 0.4797075923039363
wandb: best_epoch 18
wandb: Test-All MRR (bs=1,000) 0.5678575278083237
wandb: FuncNameTest-All MRR (bs=1,000) 0.6816322369710296
wandb: Validation-All MRR (bs=1,000) 0.5772073119973169
wandb: Test-php MRR (bs=1,000) 0.5678575278083237
wandb: FuncNameTest-php MRR (bs=1,000) 0.6816322369710296
wandb: Validation-php MRR (bs=1,000) 0.5772073119973169
wandb: Syncing files in wandb/run-20200424_121423-uhqzjbb0:
wandb: neuralbow-2020-04-24-12-14-23-graph.pbtxt
wandb: neuralbow-2020-04-24-12-14-23.train_log
wandb: neuralbow-2020-04-24-12-14-23_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced neuralbow-2020-04-24-12-14-23: https://app.wandb.ai/jianguda/CodeSearchNet/runs/uhqzjbb0
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/uhqzjbb0
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./neuralbow-2020-04-24-12-14-23_model_best.pkl.gz
2020-04-24 12:49:41.100231: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-24 12:49:46.529869: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fbae:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-24 12:49:46.529935: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-24 12:49:46.826473: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-24 12:49:46.826543: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-24 12:49:46.826562: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-24 12:49:46.826688: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fbae:00:00.0, compute capability: 3.7)
WARNING:tensorflow:From /home/dev/src/models/model.py:299: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: php
100%|████████████████████████████████████████████████████████████████████████████████████| 977821/977821 [00:05<00:00, 181145.86it/s]977821it [00:17, 57388.85it/s]
Uploading predictions to W&B
NDCG Average: 0.149388907

# CNN

root@jian-csn:/home/dev/src# python train.py --model 1dcnn ../resources/saved_models ../resources/data/php/final/jsonl/train ../resources/data/php/final/jsonl/valid ../resources/data/php/final/jsonl/test
wandb: Started W&B process version 0.8.12 with PID 235
wandb: Wandb version 0.8.32 is available! To upgrade, please run:
wandb: \$ pip install wandb --upgrade
wandb: Local directory: wandb/run-20200424_130203-0lx6syno
wandb: Syncing run 1dcnn-2020-04-24-13-02-03: https://app.wandb.ai/jianguda/CodeSearchNet/runs/0lx6syno
wandb: Run `wandb off` to turn off syncing.

2020-04-24 13:02:08.873953: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-24 13:02:08.954086: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fbae:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-24 13:02:08.954135: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-24 13:02:09.231954: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-24 13:02:09.232022: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-24 13:02:09.232039: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-24 13:02:09.232158: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fbae:00:00.0, compute capability: 3.7)
Tokenizing and building vocabulary for code snippets and queries. This step may take several hours.
Starting training run 1dcnn-2020-04-24-13-02-03 of model ConvolutionalModel with following hypers:
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': False, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_1dcnn_position_encoding': 'learned', 'code_1dcnn_layer_list': [128, 128, 128], 'code_1dcnn_kernel_width': [16, 16, 16], 'code_1dcnn_add_residual_connections': True, 'code_1dcnn_activation': 'tanh', 'code_1dcnn_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_1dcnn_position_encoding': 'learned', 'query_1dcnn_layer_list': [128, 128, 128], 'query_1dcnn_kernel_width': [16, 16, 16], 'query_1dcnn_add_residual_connections': True, 'query_1dcnn_activation': 'tanh', 'query_1dcnn_pool_mode': 'weighted_mean', 'batch_size': 1000, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.0005, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'softmax', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
Loading training and validation data.
Begin Training.
Training on 523712 php samples.
Validating on 26015 php samples.
==== Epoch 0 ====
Epoch 0 (train) took 328.52s [processed 1591 samples/second]
Training Loss: 5.649869
Epoch 0 (valid) took 5.25s [processed 4951 samples/second]
Validation: Loss: 5.034023 | MRR: 0.197006
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-24-13-02-03_model_best.pkl.gz'.
==== Epoch 1 ====
Epoch 1 (train) took 325.58s [processed 1606 samples/second]
Training Loss: 4.035505
Epoch 1 (valid) took 5.18s [processed 5017 samples/second]
Validation: Loss: 4.218559 | MRR: 0.326438
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-24-13-02-03_model_best.pkl.gz'.
==== Epoch 2 ====
Epoch 2 (train) took 326.14s [processed 1603 samples/second]
Training Loss: 3.201373
Epoch 2 (valid) took 5.18s [processed 5014 samples/second]
Validation: Loss: 3.750084 | MRR: 0.399122
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-24-13-02-03_model_best.pkl.gz'.
==== Epoch 3 ====
Epoch 3 (train) took 326.18s [processed 1603 samples/second]
Training Loss: 2.717289
Epoch 3 (valid) took 5.20s [processed 5004 samples/second]
Validation: Loss: 3.471368 | MRR: 0.438775
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-24-13-02-03_model_best.pkl.gz'.
==== Epoch 4 ====
Epoch 4 (train) took 326.13s [processed 1603 samples/second]
Training Loss: 2.416217
Epoch 4 (valid) took 5.18s [processed 5024 samples/second]
Validation: Loss: 3.315552 | MRR: 0.461557
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-24-13-02-03_model_best.pkl.gz'.
==== Epoch 5 ====
Epoch 5 (train) took 325.75s [processed 1605 samples/second]
Training Loss: 2.210380
Epoch 5 (valid) took 5.19s [processed 5011 samples/second]
Validation: Loss: 3.235256 | MRR: 0.473398
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-24-13-02-03_model_best.pkl.gz'.
==== Epoch 6 ====
Epoch 6 (train) took 325.90s [processed 1604 samples/second]
Training Loss: 2.062749
Epoch 6 (valid) took 5.21s [processed 4993 samples/second]
Validation: Loss: 3.167178 | MRR: 0.484158
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-24-13-02-03_model_best.pkl.gz'.
==== Epoch 7 ====
Epoch 7 (train) took 325.41s [processed 1607 samples/second]
Training Loss: 1.949590
Epoch 7 (valid) took 5.17s [processed 5027 samples/second]
Validation: Loss: 3.114898 | MRR: 0.492983
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-24-13-02-03_model_best.pkl.gz'.
==== Epoch 8 ====
Epoch 8 (train) took 326.21s [processed 1603 samples/second]
Training Loss: 1.853074
Epoch 8 (valid) took 5.19s [processed 5012 samples/second]
Validation: Loss: 3.097766 | MRR: 0.500392
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-24-13-02-03_model_best.pkl.gz'.
==== Epoch 9 ====
Epoch 9 (train) took 325.65s [processed 1606 samples/second]
Training Loss: 1.774539
Epoch 9 (valid) took 5.19s [processed 5014 samples/second]
Validation: Loss: 3.094441 | MRR: 0.502702
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-24-13-02-03_model_best.pkl.gz'.
==== Epoch 10 ====
Epoch 10 (train) took 325.41s [processed 1607 samples/second]
Training Loss: 1.715676
Epoch 10 (valid) took 5.20s [processed 5004 samples/second]
Validation: Loss: 3.059560 | MRR: 0.506883
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-24-13-02-03_model_best.pkl.gz'.
==== Epoch 11 ====
Epoch 11 (train) took 325.38s [processed 1607 samples/second]
Training Loss: 1.656837
Epoch 11 (valid) took 5.16s [processed 5034 samples/second]
Validation: Loss: 3.097089 | MRR: 0.509120
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-24-13-02-03_model_best.pkl.gz'.
==== Epoch 12 ====
Epoch 12 (train) took 325.12s [processed 1608 samples/second]
Training Loss: 1.605212
Epoch 12 (valid) took 5.21s [processed 4990 samples/second]
Validation: Loss: 3.038987 | MRR: 0.512149
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-24-13-02-03_model_best.pkl.gz'.
==== Epoch 13 ====
Epoch 13 (train) took 324.73s [processed 1610 samples/second]
Training Loss: 1.562295
Epoch 13 (valid) took 5.16s [processed 5038 samples/second]
Validation: Loss: 3.032695 | MRR: 0.514080
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-24-13-02-03_model_best.pkl.gz'.
==== Epoch 14 ====
Epoch 14 (train) took 324.92s [processed 1609 samples/second]
Training Loss: 1.529552
Epoch 14 (valid) took 5.19s [processed 5012 samples/second]
Validation: Loss: 3.048865 | MRR: 0.515587
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-24-13-02-03_model_best.pkl.gz'.
==== Epoch 15 ====
Epoch 15 (train) took 324.56s [processed 1611 samples/second]
Training Loss: 1.491599
Epoch 15 (valid) took 5.17s [processed 5031 samples/second]
Validation: Loss: 3.063229 | MRR: 0.516141
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-24-13-02-03_model_best.pkl.gz'.
==== Epoch 16 ====
Epoch 16 (train) took 325.06s [processed 1608 samples/second]
Training Loss: 1.460215
Epoch 16 (valid) took 5.19s [processed 5011 samples/second]
Validation: Loss: 3.048771 | MRR: 0.519181
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-24-13-02-03_model_best.pkl.gz'.
==== Epoch 17 ====
Epoch 17 (train) took 325.46s [processed 1606 samples/second]
Training Loss: 1.429941
Epoch 17 (valid) took 5.15s [processed 5044 samples/second]
Validation: Loss: 3.064116 | MRR: 0.519943
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-24-13-02-03_model_best.pkl.gz'.
==== Epoch 18 ====
Epoch 18 (train) took 325.06s [processed 1608 samples/second]
Training Loss: 1.406350
Epoch 18 (valid) took 5.21s [processed 4991 samples/second]
Validation: Loss: 3.072068 | MRR: 0.521490
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-24-13-02-03_model_best.pkl.gz'.
==== Epoch 19 ====
Epoch 19 (train) took 325.45s [processed 1607 samples/second]
Training Loss: 1.383673
Epoch 19 (valid) took 5.18s [processed 5016 samples/second]
Validation: Loss: 3.070379 | MRR: 0.520915
==== Epoch 20 ====
Epoch 20 (train) took 325.77s [processed 1605 samples/second]
Training Loss: 1.363678
Epoch 20 (valid) took 5.17s [processed 5031 samples/second]
Validation: Loss: 3.063004 | MRR: 0.524775
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-24-13-02-03_model_best.pkl.gz'.
==== Epoch 21 ====
Epoch 21 (train) took 325.22s [processed 1608 samples/second]
Training Loss: 1.341817
Epoch 21 (valid) took 5.18s [processed 5020 samples/second]
Validation: Loss: 3.079959 | MRR: 0.525550
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-24-13-02-03_model_best.pkl.gz'.
==== Epoch 22 ====
Epoch 22 (train) took 325.18s [processed 1608 samples/second]
Training Loss: 1.324880
Epoch 22 (valid) took 5.18s [processed 5023 samples/second]
Validation: Loss: 3.072964 | MRR: 0.524468
==== Epoch 23 ====
Epoch 23 (train) took 325.23s [processed 1608 samples/second]
Training Loss: 1.306537
Epoch 23 (valid) took 5.17s [processed 5028 samples/second]
Validation: Loss: 3.075077 | MRR: 0.524443
==== Epoch 24 ====
Epoch 24 (train) took 325.46s [processed 1606 samples/second]
Training Loss: 1.288573
Epoch 24 (valid) took 5.19s [processed 5006 samples/second]
Validation: Loss: 3.061856 | MRR: 0.527164
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-24-13-02-03_model_best.pkl.gz'.
==== Epoch 25 ====
Epoch 25 (train) took 325.58s [processed 1606 samples/second]
Training Loss: 1.278454
Epoch 25 (valid) took 5.18s [processed 5021 samples/second]
Validation: Loss: 3.092406 | MRR: 0.526190
==== Epoch 26 ====
Epoch 26 (train) took 325.44s [processed 1607 samples/second]
Training Loss: 1.263743
Epoch 26 (valid) took 5.16s [processed 5039 samples/second]
Validation: Loss: 3.105943 | MRR: 0.526135
==== Epoch 27 ====
Epoch 27 (train) took 325.22s [processed 1608 samples/second]
Training Loss: 1.249401
Epoch 27 (valid) took 5.18s [processed 5018 samples/second]
Validation: Loss: 3.089560 | MRR: 0.524621
==== Epoch 28 ====
Epoch 28 (train) took 325.13s [processed 1608 samples/second]
Training Loss: 1.235497
Epoch 28 (valid) took 5.18s [processed 5024 samples/second]
Validation: Loss: 3.092625 | MRR: 0.526577
==== Epoch 29 ====
Epoch 29 (train) took 325.27s [processed 1607 samples/second]
Training Loss: 1.223122
Epoch 29 (valid) took 5.17s [processed 5028 samples/second]
Validation: Loss: 3.096417 | MRR: 0.527340
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-24-13-02-03_model_best.pkl.gz'.
==== Epoch 30 ====
Epoch 30 (train) took 325.24s [processed 1608 samples/second]
Training Loss: 1.212434
Epoch 30 (valid) took 5.17s [processed 5031 samples/second]
Validation: Loss: 3.115148 | MRR: 0.528858
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-24-13-02-03_model_best.pkl.gz'.
==== Epoch 31 ====
Epoch 31 (train) took 325.43s [processed 1607 samples/second]
Training Loss: 1.200284
Epoch 31 (valid) took 5.20s [processed 5002 samples/second]
Validation: Loss: 3.109105 | MRR: 0.529368
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-24-13-02-03_model_best.pkl.gz'.
==== Epoch 32 ====
Epoch 32 (train) took 325.34s [processed 1607 samples/second]
Training Loss: 1.192552
Epoch 32 (valid) took 5.18s [processed 5017 samples/second]
Validation: Loss: 3.123242 | MRR: 0.530511
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-24-13-02-03_model_best.pkl.gz'.
==== Epoch 33 ====
Epoch 33 (train) took 326.64s [processed 1601 samples/second]
Training Loss: 1.185192
Epoch 33 (valid) took 5.23s [processed 4974 samples/second]
Validation: Loss: 3.107278 | MRR: 0.529243
==== Epoch 34 ====
Epoch 34 (train) took 330.41s [processed 1582 samples/second]
Training Loss: 1.169454
Epoch 34 (valid) took 5.26s [processed 4940 samples/second]
Validation: Loss: 3.099416 | MRR: 0.530490
==== Epoch 35 ====
Epoch 35 (train) took 331.20s [processed 1579 samples/second]
Training Loss: 1.165248
Epoch 35 (valid) took 5.26s [processed 4943 samples/second]
Validation: Loss: 3.115224 | MRR: 0.530718
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-24-13-02-03_model_best.pkl.gz'.
==== Epoch 36 ====
Epoch 36 (train) took 331.45s [processed 1577 samples/second]
Training Loss: 1.158611
Epoch 36 (valid) took 5.24s [processed 4957 samples/second]
Validation: Loss: 3.122102 | MRR: 0.532821
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-24-13-02-03_model_best.pkl.gz'.
==== Epoch 37 ====
Epoch 37 (train) took 329.25s [processed 1588 samples/second]
Training Loss: 1.146165
Epoch 37 (valid) took 5.24s [processed 4961 samples/second]
Validation: Loss: 3.113867 | MRR: 0.530534
==== Epoch 38 ====
Epoch 38 (train) took 330.02s [processed 1584 samples/second]
Training Loss: 1.141395
Epoch 38 (valid) took 5.29s [processed 4911 samples/second]
Validation: Loss: 3.110316 | MRR: 0.530888
==== Epoch 39 ====
Epoch 39 (train) took 330.10s [processed 1584 samples/second]
Training Loss: 1.131622
Epoch 39 (valid) took 5.25s [processed 4950 samples/second]
Validation: Loss: 3.114089 | MRR: 0.532266
==== Epoch 40 ====
Epoch 40 (train) took 329.68s [processed 1586 samples/second]
Training Loss: 1.127524
Epoch 40 (valid) took 5.25s [processed 4952 samples/second]
Validation: Loss: 3.132702 | MRR: 0.530871
==== Epoch 41 ====
Epoch 41 (train) took 330.02s [processed 1584 samples/second]
Training Loss: 1.117428
Epoch 41 (valid) took 5.26s [processed 4941 samples/second]
Validation: Loss: 3.123530 | MRR: 0.531465
2020-04-24 17:05:31.987734: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-24 17:05:31.987807: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-24 17:05:31.987820: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-24 17:05:31.987831: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-24 17:05:31.987940: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fbae:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.543
FuncNameTest-All MRR (bs=1,000): 0.844
Validation-All MRR (bs=1,000): 0.542
Test-php MRR (bs=1,000): 0.543
FuncNameTest-php MRR (bs=1,000): 0.844
Validation-php MRR (bs=1,000): 0.542

wandb: Waiting for W&B process to finish, PID 235
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-mrr 0.8053808843492322
wandb: \_runtime 14755.741686820984
wandb: \_timestamp 1587748077.7571805
wandb: train-loss 1.117428192900882
wandb: \_step 299
wandb: train-time-sec 330.0238654613495
wandb: epoch 41
wandb: val-mrr 0.531464824969952
wandb: val-loss 3.1235298743614783
wandb: val-time-sec 5.26163125038147
wandb: best_val_mrr_loss 3.1221022514196544
wandb: best_val_mrr 0.5328205683781551
wandb: best_epoch 36
wandb: Test-All MRR (bs=1,000) 0.5429086591457103
wandb: FuncNameTest-All MRR (bs=1,000) 0.8436210781043124
wandb: Validation-All MRR (bs=1,000) 0.542186286557096
wandb: Test-php MRR (bs=1,000) 0.5429086591457103
wandb: FuncNameTest-php MRR (bs=1,000) 0.8436210781043124
wandb: Validation-php MRR (bs=1,000) 0.542186286557096
wandb: Syncing files in wandb/run-20200424_130203-0lx6syno:
wandb: 1dcnn-2020-04-24-13-02-03-graph.pbtxt
wandb: 1dcnn-2020-04-24-13-02-03.train_log
wandb: 1dcnn-2020-04-24-13-02-03_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced 1dcnn-2020-04-24-13-02-03: https://app.wandb.ai/jianguda/CodeSearchNet/runs/0lx6syno
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/0lx6syno
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./1dcnn-2020-04-24-13-02-03_model_best.pkl.gz
2020-04-24 17:08:19.787603: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-24 17:08:24.656265: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fbae:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-24 17:08:24.656319: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-24 17:08:24.955988: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-24 17:08:24.956058: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-24 17:08:24.956076: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-24 17:08:24.956219: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fbae:00:00.0, compute capability: 3.7)
Evaluating language: php
100%|████████████████████████████████████████████████████████████████████████████████████| 977821/977821 [00:05<00:00, 178673.97it/s]977821it [00:17, 57022.36it/s]
Uploading predictions to W&B
NDCG Average: 0.123752024

# RNN

root@jian-csn:/home/dev/src# python train.py --model rnn ../resources/saved_models ../resources/data/php/final/jsonl/train ../resources/data/php/final/jsonl/valid ../resources/data/php/final/jsonl/test
wandb: Started W&B process version 0.8.12 with PID 441
wandb: Wandb version 0.8.32 is available! To upgrade, please run:
wandb: \$ pip install wandb --upgrade
wandb: Local directory: wandb/run-20200424_172234-uu6qliml
wandb: Syncing run rnn-2020-04-24-17-22-34: https://app.wandb.ai/jianguda/CodeSearchNet/runs/uu6qliml
wandb: Run `wandb off` to turn off syncing.

2020-04-24 17:22:39.151564: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-24 17:22:40.479603: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fbae:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-24 17:22:40.479655: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-24 17:22:40.777158: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-24 17:22:40.777233: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-24 17:22:40.777243: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-24 17:22:40.777389: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fbae:00:00.0, compute capability: 3.7)
Tokenizing and building vocabulary for code snippets and queries. This step may take several hours.
Starting training run rnn-2020-04-24-17-22-34 of model RNNModel with following hypers:
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': True, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_rnn_num_layers': 2, 'code_rnn_hidden_dim': 64, 'code_rnn_cell_type': 'LSTM', 'code_rnn_is_bidirectional': True, 'code_rnn_dropout_keep_rate': 0.8, 'code_rnn_recurrent_dropout_keep_rate': 1.0, 'code_rnn_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_rnn_num_layers': 2, 'query_rnn_hidden_dim': 64, 'query_rnn_cell_type': 'LSTM', 'query_rnn_is_bidirectional': True, 'query_rnn_dropout_keep_rate': 0.8, 'query_rnn_recurrent_dropout_keep_rate': 1.0, 'query_rnn_pool_mode': 'weighted_mean', 'batch_size': 1000, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.01, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'softmax', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
Loading training and validation data.
Begin Training.
Training on 523712 php samples.
Validating on 26015 php samples.
==== Epoch 0 ====
Epoch 0 (train) took 569.63s [processed 918 samples/second]
Training Loss: 4.451806
Epoch 0 (valid) took 13.68s [processed 1900 samples/second]
Validation: Loss: 3.481787 | MRR: 0.418183
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-24-17-22-34_model_best.pkl.gz'.
==== Epoch 1 ====
Epoch 1 (train) took 565.25s [processed 925 samples/second]
Training Loss: 2.615667
Epoch 1 (valid) took 13.59s [processed 1913 samples/second]
Validation: Loss: 3.063198 | MRR: 0.481077
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-24-17-22-34_model_best.pkl.gz'.
==== Epoch 2 ====
Epoch 2 (train) took 568.07s [processed 920 samples/second]
Training Loss: 2.283362
Epoch 2 (valid) took 13.56s [processed 1917 samples/second]
Validation: Loss: 2.940262 | MRR: 0.504789
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-24-17-22-34_model_best.pkl.gz'.
==== Epoch 3 ====
Epoch 3 (train) took 568.58s [processed 919 samples/second]
Training Loss: 2.142615
Epoch 3 (valid) took 13.63s [processed 1907 samples/second]
Validation: Loss: 2.876679 | MRR: 0.515068
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-24-17-22-34_model_best.pkl.gz'.
==== Epoch 4 ====
Epoch 4 (train) took 565.63s [processed 924 samples/second]
Training Loss: 2.063101
Epoch 4 (valid) took 13.32s [processed 1952 samples/second]
Validation: Loss: 2.849073 | MRR: 0.519144
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-24-17-22-34_model_best.pkl.gz'.
==== Epoch 5 ====
Epoch 5 (train) took 561.91s [processed 930 samples/second]
Training Loss: 2.011029
Epoch 5 (valid) took 13.32s [processed 1952 samples/second]
Validation: Loss: 2.813089 | MRR: 0.522399
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-24-17-22-34_model_best.pkl.gz'.
==== Epoch 6 ====
Epoch 6 (train) took 559.68s [processed 934 samples/second]
Training Loss: 1.977048
Epoch 6 (valid) took 13.29s [processed 1955 samples/second]
Validation: Loss: 2.817913 | MRR: 0.525216
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-24-17-22-34_model_best.pkl.gz'.
==== Epoch 7 ====
Epoch 7 (train) took 560.38s [processed 933 samples/second]
Training Loss: 1.955678
Epoch 7 (valid) took 13.36s [processed 1946 samples/second]
Validation: Loss: 2.810381 | MRR: 0.527173
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-24-17-22-34_model_best.pkl.gz'.
==== Epoch 8 ====
Epoch 8 (train) took 558.65s [processed 936 samples/second]
Training Loss: 1.935237
Epoch 8 (valid) took 13.24s [processed 1963 samples/second]
Validation: Loss: 2.773020 | MRR: 0.529486
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-24-17-22-34_model_best.pkl.gz'.
==== Epoch 9 ====
Epoch 9 (train) took 559.82s [processed 934 samples/second]
Training Loss: 1.922916
Epoch 9 (valid) took 13.36s [processed 1945 samples/second]
Validation: Loss: 2.800936 | MRR: 0.527725
==== Epoch 10 ====
Epoch 10 (train) took 558.70s [processed 936 samples/second]
Training Loss: 1.919771
Epoch 10 (valid) took 13.28s [processed 1957 samples/second]
Validation: Loss: 2.787436 | MRR: 0.528441
==== Epoch 11 ====
Epoch 11 (train) took 560.22s [processed 933 samples/second]
Training Loss: 1.911294
Epoch 11 (valid) took 13.25s [processed 1961 samples/second]
Validation: Loss: 2.783622 | MRR: 0.531787
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-24-17-22-34_model_best.pkl.gz'.
==== Epoch 12 ====
Epoch 12 (train) took 558.51s [processed 936 samples/second]
Training Loss: 1.913282
Epoch 12 (valid) took 13.27s [processed 1959 samples/second]
Validation: Loss: 2.796530 | MRR: 0.528905
==== Epoch 13 ====
Epoch 13 (train) took 559.97s [processed 933 samples/second]
Training Loss: 1.907395
Epoch 13 (valid) took 13.24s [processed 1963 samples/second]
Validation: Loss: 2.791070 | MRR: 0.529206
==== Epoch 14 ====
Epoch 14 (train) took 558.22s [processed 936 samples/second]
Training Loss: 1.909475
Epoch 14 (valid) took 13.29s [processed 1956 samples/second]
Validation: Loss: 2.784692 | MRR: 0.530422
==== Epoch 15 ====
Epoch 15 (train) took 559.83s [processed 934 samples/second]
Training Loss: 1.912969
Epoch 15 (valid) took 13.28s [processed 1957 samples/second]
Validation: Loss: 2.790020 | MRR: 0.529495
==== Epoch 16 ====
Epoch 16 (train) took 558.35s [processed 936 samples/second]
Training Loss: 1.918077
Epoch 16 (valid) took 13.30s [processed 1955 samples/second]
Validation: Loss: 2.784462 | MRR: 0.530987
2020-04-24 20:16:31.887435: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-24 20:16:31.887507: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-24 20:16:31.887523: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-24 20:16:31.887535: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-24 20:16:31.887627: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fbae:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.601
FuncNameTest-All MRR (bs=1,000): 0.866
Validation-All MRR (bs=1,000): 0.604
Test-php MRR (bs=1,000): 0.601
FuncNameTest-php MRR (bs=1,000): 0.866
Validation-php MRR (bs=1,000): 0.604

wandb: Waiting for W&B process to finish, PID 441
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_timestamp 1587759583.4070237
wandb: \_runtime 10629.829368114471
wandb: \_step 124
wandb: train-mrr 0.6752753526968436
wandb: train-loss 1.9180774941727035
wandb: val-mrr 0.5309870816744291
wandb: val-loss 2.784462277705853
wandb: epoch 16
wandb: val-time-sec 13.296821355819702
wandb: train-time-sec 558.3516094684601
wandb: best_val_mrr_loss 2.7836222556921153
wandb: best_val_mrr 0.5317874391995944
wandb: best_epoch 11
wandb: Test-All MRR (bs=1,000) 0.6012475522164032
wandb: FuncNameTest-All MRR (bs=1,000) 0.8661563490233527
wandb: Validation-All MRR (bs=1,000) 0.60404175720039
wandb: Test-php MRR (bs=1,000) 0.6012475522164032
wandb: FuncNameTest-php MRR (bs=1,000) 0.8661563490233527
wandb: Validation-php MRR (bs=1,000) 0.60404175720039
wandb: Syncing files in wandb/run-20200424_172234-uu6qliml:
wandb: rnn-2020-04-24-17-22-34-graph.pbtxt
wandb: rnn-2020-04-24-17-22-34.train_log
wandb: rnn-2020-04-24-17-22-34_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced rnn-2020-04-24-17-22-34: https://app.wandb.ai/jianguda/CodeSearchNet/runs/uu6qliml
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/uu6qliml
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./rnn-2020-04-24-17-22-34_model_best.pkl.gz
2020-04-24 20:20:48.788867: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-24 20:20:53.634015: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fbae:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-24 20:20:53.634068: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-24 20:20:53.917774: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-24 20:20:53.917845: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-24 20:20:53.917862: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-24 20:20:53.917978: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fbae:00:00.0, compute capability: 3.7)
Evaluating language: php
100%|████████████████████████████████████████████████████████████████████████████████████| 977821/977821 [00:09<00:00, 100532.08it/s]977821it [00:17, 57253.44it/s]
Uploading predictions to W&B
NDCG Average: 0.097613787

# BERT

root@jian-csn:/home/dev/src# python train.py --model selfatt ../resources/saved_models ../resources/data/php/final/jsonl/train ../resources/data/php/final/jsonl/valid ../resources/data/php/final/jsonl/test
wandb: Started W&B process version 0.8.12 with PID 645
wandb: Wandb version 0.8.32 is available! To upgrade, please run:
wandb: \$ pip install wandb --upgrade
wandb: Local directory: wandb/run-20200424_203945-usymlfo5
wandb: Syncing run selfatt-2020-04-24-20-39-45: https://app.wandb.ai/jianguda/CodeSearchNet/runs/usymlfo5
wandb: Run `wandb off` to turn off syncing.

2020-04-24 20:39:51.302276: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-24 20:39:51.394890: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fbae:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-24 20:39:51.394942: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-24 20:39:51.687473: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-24 20:39:51.687538: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
Tokenizing and building vocabulary for code snippets and queries. This step may take several hours.
2020-04-24 20:39:51.687554: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-24 20:39:51.687672: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fbae:00:00.0, compute capability: 3.7)
Starting training run selfatt-2020-04-24-20-39-45 of model SelfAttentionModel with following hypers:
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': False, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_self_attention_activation': 'gelu', 'code_self_attention_hidden_size': 128, 'code_self_attention_intermediate_size': 512, 'code_self_attention_num_layers': 3, 'code_self_attention_num_heads': 8, 'code_self_attention_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_self_attention_activation': 'gelu', 'query_self_attention_hidden_size': 128, 'query_self_attention_intermediate_size': 512, 'query_self_attention_num_layers': 3, 'query_self_attention_num_heads': 8, 'query_self_attention_pool_mode': 'weighted_mean', 'batch_size': 450, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.0005, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'softmax', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
Loading training and validation data.
Begin Training.
Training on 523712 php samples.
Validating on 26015 php samples.
==== Epoch 0 ====
Epoch 0 (train) took 1599.13s [processed 327 samples/second]
Training Loss: 2.434418
Epoch 0 (valid) took 33.70s [processed 761 samples/second]
Validation: Loss: 2.531207 | MRR: 0.564815
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-24-20-39-45_model_best.pkl.gz'.
==== Epoch 1 ====
Epoch 1 (train) took 1598.57s [processed 327 samples/second]
Training Loss: 1.223568
Epoch 1 (valid) took 33.19s [processed 772 samples/second]
Validation: Loss: 2.410374 | MRR: 0.592059
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-24-20-39-45_model_best.pkl.gz'.
==== Epoch 2 ====
Epoch 2 (train) took 1598.37s [processed 327 samples/second]
Training Loss: 1.001109
Epoch 2 (valid) took 33.20s [processed 772 samples/second]
Validation: Loss: 2.394397 | MRR: 0.599368
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-24-20-39-45_model_best.pkl.gz'.
==== Epoch 3 ====
Epoch 3 (train) took 1597.01s [processed 327 samples/second]
Training Loss: 0.883754
Epoch 3 (valid) took 33.30s [processed 770 samples/second]
Validation: Loss: 2.371906 | MRR: 0.607986
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-24-20-39-45_model_best.pkl.gz'.
==== Epoch 4 ====
Epoch 4 (train) took 1599.53s [processed 327 samples/second]
Training Loss: 0.806238
Epoch 4 (valid) took 33.18s [processed 773 samples/second]
Validation: Loss: 2.355034 | MRR: 0.609687
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-24-20-39-45_model_best.pkl.gz'.
==== Epoch 5 ====
Epoch 5 (train) took 1597.83s [processed 327 samples/second]
Training Loss: 0.752897
Epoch 5 (valid) took 33.41s [processed 767 samples/second]
Validation: Loss: 2.403921 | MRR: 0.608149
==== Epoch 6 ====
Epoch 6 (train) took 1599.41s [processed 327 samples/second]
Training Loss: 0.714120
Epoch 6 (valid) took 33.37s [processed 768 samples/second]
Validation: Loss: 2.440947 | MRR: 0.608071
==== Epoch 7 ====
Epoch 7 (train) took 1599.30s [processed 327 samples/second]
Training Loss: 0.684192
Epoch 7 (valid) took 33.43s [processed 767 samples/second]
Validation: Loss: 2.398118 | MRR: 0.611491
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-24-20-39-45_model_best.pkl.gz'.
==== Epoch 8 ====
Epoch 8 (train) took 1601.23s [processed 326 samples/second]
Training Loss: 0.657275
Epoch 8 (valid) took 33.30s [processed 770 samples/second]
Validation: Loss: 2.461036 | MRR: 0.611322
==== Epoch 9 ====
Epoch 9 (train) took 1599.27s [processed 327 samples/second]
Training Loss: 0.635964
Epoch 9 (valid) took 33.44s [processed 766 samples/second]
Validation: Loss: 2.559199 | MRR: 0.611167
==== Epoch 10 ====
Epoch 10 (train) took 1599.75s [processed 327 samples/second]
Training Loss: 0.619211
Epoch 10 (valid) took 32.91s [processed 779 samples/second]
Validation: Loss: 2.516079 | MRR: 0.610861
==== Epoch 11 ====
Epoch 11 (train) took 1590.18s [processed 329 samples/second]
Training Loss: 0.602401
Epoch 11 (valid) took 33.03s [processed 776 samples/second]
Validation: Loss: 2.529945 | MRR: 0.612770
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-24-20-39-45_model_best.pkl.gz'.
==== Epoch 12 ====
Epoch 12 (train) took 1591.25s [processed 328 samples/second]
Training Loss: 0.587730
Epoch 12 (valid) took 33.33s [processed 769 samples/second]
Validation: Loss: 2.599908 | MRR: 0.609839
==== Epoch 13 ====
Epoch 13 (train) took 1590.80s [processed 328 samples/second]
Training Loss: 0.576585
Epoch 13 (valid) took 33.11s [processed 774 samples/second]
Validation: Loss: 2.560620 | MRR: 0.613617
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-24-20-39-45_model_best.pkl.gz'.
==== Epoch 14 ====
Epoch 14 (train) took 1591.77s [processed 328 samples/second]
Training Loss: 0.564060
Epoch 14 (valid) took 33.04s [processed 776 samples/second]
Validation: Loss: 2.593589 | MRR: 0.613016
==== Epoch 15 ====
Epoch 15 (train) took 1589.80s [processed 329 samples/second]
Training Loss: 0.554118
Epoch 15 (valid) took 33.18s [processed 772 samples/second]
Validation: Loss: 2.571424 | MRR: 0.613032
==== Epoch 16 ====
Epoch 16 (train) took 1591.86s [processed 328 samples/second]
Training Loss: 0.543905
Epoch 16 (valid) took 33.09s [processed 775 samples/second]
Validation: Loss: 2.602173 | MRR: 0.610589
==== Epoch 17 ====
Epoch 17 (train) took 1591.24s [processed 328 samples/second]
Training Loss: 0.535847
Epoch 17 (valid) took 32.92s [processed 779 samples/second]
Validation: Loss: 2.609410 | MRR: 0.612386
==== Epoch 18 ====
Epoch 18 (train) took 1592.47s [processed 328 samples/second]
Training Loss: 0.527801
Epoch 18 (valid) took 33.31s [processed 770 samples/second]
Validation: Loss: 2.649718 | MRR: 0.612090
2020-04-25 05:26:20.797025: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-25 05:26:20.797095: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-25 05:26:20.797111: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-25 05:26:20.797123: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-25 05:26:20.797228: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fbae:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.601
FuncNameTest-All MRR (bs=1,000): 0.876
Validation-All MRR (bs=1,000): 0.616
Test-php MRR (bs=1,000): 0.601
FuncNameTest-php MRR (bs=1,000): 0.876
Validation-php MRR (bs=1,000): 0.616

wandb: Waiting for W&B process to finish, PID 645
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_runtime 31859.485744953156
wandb: train-mrr 0.8987198863490053
wandb: train-loss 0.5278011737767726
wandb: \_step 252
wandb: \_timestamp 1587792643.289463
wandb: epoch 18
wandb: val-loss 2.649717690651877
wandb: val-mrr 0.6120895653440241
wandb: train-time-sec 1592.4712467193604
wandb: val-time-sec 33.30809783935547
wandb: best_val_mrr_loss 2.560619835267987
wandb: best_val_mrr 0.6136165965416743
wandb: best_epoch 13
wandb: Test-All MRR (bs=1,000) 0.6014210190124777
wandb: FuncNameTest-All MRR (bs=1,000) 0.8759801552141712
wandb: Validation-All MRR (bs=1,000) 0.6160001739236508
wandb: Test-php MRR (bs=1,000) 0.6014210190124777
wandb: FuncNameTest-php MRR (bs=1,000) 0.8759801552141712
wandb: Validation-php MRR (bs=1,000) 0.6160001739236508
wandb: Syncing files in wandb/run-20200424_203945-usymlfo5:
wandb: selfatt-2020-04-24-20-39-45-graph.pbtxt
wandb: selfatt-2020-04-24-20-39-45.train_log
wandb: selfatt-2020-04-24-20-39-45_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced selfatt-2020-04-24-20-39-45: https://app.wandb.ai/jianguda/CodeSearchNet/runs/usymlfo5
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/usymlfo5
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./selfatt-2020-04-24-20-39-45_model_best.pkl.gz
2020-04-25 06:58:07.009296: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-25 06:58:11.894340: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fbae:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-25 06:58:11.894391: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-25 06:58:12.173320: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-25 06:58:12.173394: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-25 06:58:12.173411: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-25 06:58:12.173525: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fbae:00:00.0, compute capability: 3.7)
Evaluating language: php
100%|████████████████████████████████████████████████████████████████████████████████████| 977821/977821 [00:09<00:00, 101152.06it/s]977821it [00:16, 58057.82it/s]
Uploading predictions to W&B
NDCG Average: 0.078482715
