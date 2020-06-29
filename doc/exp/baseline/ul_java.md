python train.py --model neuralbow ../resources/saved_models ../resources/data/java/final/jsonl/train ../resources/data/java/final/jsonl/valid ../resources/data/java/final/jsonl/test

python predict.py -r jianguda/CodeSearchNet/0123456

# NBOW

root@jian-csn:/home/dev/src# python train.py --model neuralbow ../resources/saved_models ../resources/data/java/final/jsonl/train ../resources/data/java/final/jsonl/valid ../resources/data/java/final/jsonl/test
wandb: Started W&B process version 0.8.12 with PID 1669
wandb: Wandb version 0.8.32 is available! To upgrade, please run:
wandb: \$ pip install wandb --upgrade
wandb: Local directory: wandb/run-20200423_190649-yyavkuve  
wandb: Syncing run neuralbow-2020-04-23-19-06-49: https://app.wandb.ai/jianguda/CodeSearchNet/runs/yyavkuve
wandb: Run `wandb off` to turn off syncing.

2020-04-23 19:06:55.258205: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-23 19:06:55.337617: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 42e4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-23 19:06:55.337659: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-23 19:06:55.614584: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-23 19:06:55.614645: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-23 19:06:55.614660: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-23 19:06:55.614767: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Tokenizing and building vocabulary for code snippets and queries. This step may take several hours.
WARNING:tensorflow:From /home/dev/src/models/model.py:299: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Starting training run neuralbow-2020-04-23-19-06-49 of model NeuralBoWModel with following hypers:
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': False, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_nbow_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_nbow_pool_mode': 'weighted_mean', 'batch_size': 1000, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.01, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'cosine', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
Loading training and validation data.
Begin Training.
Training on 454451 java samples.
Validating on 15328 java samples.
==== Epoch 0 ====
Epoch 0 (train) took 42.63s [processed 10649 samples/second]
Training Loss: 1.002380
Epoch 0 (valid) took 0.95s [processed 15854 samples/second]
Validation: Loss: 1.024066 | MRR: 0.310290
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-23-19-06-49_model_best.pkl.gz'.
==== Epoch 1 ====
Epoch 1 (train) took 41.60s [processed 10913 samples/second]
Training Loss: 0.960914
Epoch 1 (valid) took 0.92s [processed 16376 samples/second]
Validation: Loss: 1.069858 | MRR: 0.407025
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-23-19-06-49_model_best.pkl.gz'.
==== Epoch 2 ====
Epoch 2 (train) took 41.85s [processed 10848 samples/second]
Training Loss: 0.920222
Epoch 2 (valid) took 0.90s [processed 16630 samples/second]
Validation: Loss: 1.073160 | MRR: 0.427342
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-23-19-06-49_model_best.pkl.gz'.
==== Epoch 3 ====
Epoch 3 (train) took 41.62s [processed 10908 samples/second]
Training Loss: 0.901549
Epoch 3 (valid) took 0.91s [processed 16438 samples/second]
Validation: Loss: 1.070066 | MRR: 0.431971
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-23-19-06-49_model_best.pkl.gz'.
==== Epoch 4 ====
Epoch 4 (train) took 42.05s [processed 10796 samples/second]
Training Loss: 0.891935
Epoch 4 (valid) took 0.92s [processed 16348 samples/second]
Validation: Loss: 1.068360 | MRR: 0.436651
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-23-19-06-49_model_best.pkl.gz'.
==== Epoch 5 ====
Epoch 5 (train) took 41.68s [processed 10892 samples/second]
Training Loss: 0.885797
Epoch 5 (valid) took 0.91s [processed 16397 samples/second]
Validation: Loss: 1.067256 | MRR: 0.439177
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-23-19-06-49_model_best.pkl.gz'.
==== Epoch 6 ====
Epoch 6 (train) took 41.97s [processed 10818 samples/second]
Training Loss: 0.880793
Epoch 6 (valid) took 0.91s [processed 16551 samples/second]
Validation: Loss: 1.066834 | MRR: 0.436535
==== Epoch 7 ====
Epoch 7 (train) took 41.77s [processed 10870 samples/second]
Training Loss: 0.877188
Epoch 7 (valid) took 0.90s [processed 16686 samples/second]
Validation: Loss: 1.066010 | MRR: 0.437280
==== Epoch 8 ====
Epoch 8 (train) took 41.95s [processed 10822 samples/second]
Training Loss: 0.874185
Epoch 8 (valid) took 0.90s [processed 16592 samples/second]
Validation: Loss: 1.065217 | MRR: 0.435993
==== Epoch 9 ====
Epoch 9 (train) took 41.78s [processed 10867 samples/second]
Training Loss: 0.871647
Epoch 9 (valid) took 0.91s [processed 16411 samples/second]
Validation: Loss: 1.065019 | MRR: 0.435933
==== Epoch 10 ====
Epoch 10 (train) took 41.87s [processed 10842 samples/second]
Training Loss: 0.869475
Epoch 10 (valid) took 0.90s [processed 16630 samples/second]
Validation: Loss: 1.064582 | MRR: 0.435998
2020-04-23 19:24:05.335776: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-23 19:24:05.335841: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-23 19:24:05.335859: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-23 19:24:05.335872: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-23 19:24:05.335964: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.581
FuncNameTest-All MRR (bs=1,000): 0.666
Validation-All MRR (bs=1,000): 0.569
Test-java MRR (bs=1,000): 0.581
FuncNameTest-java MRR (bs=1,000): 0.666
Validation-java MRR (bs=1,000): 0.569

wandb: Waiting for W&B process to finish, PID 1669
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-loss 0.8694751447780542
wandb: \_timestamp 1587669945.3403664
wandb: \_runtime 1136.7251951694489
wandb: \_step 71
wandb: train-mrr 0.7939997164688446
wandb: val-time-sec 0.9019522666931152
wandb: val-loss 1.064582355817159
wandb: train-time-sec 41.8729190826416
wandb: epoch 10
wandb: val-mrr 0.435997666422526
wandb: best_val_mrr_loss 1.0672556479771933
wandb: best_val_mrr 0.43917693074544273
wandb: best_epoch 5
wandb: Test-All MRR (bs=1,000) 0.5808107778438159
wandb: FuncNameTest-All MRR (bs=1,000) 0.665763599151126
wandb: Validation-All MRR (bs=1,000) 0.5693588020579294
wandb: Test-java MRR (bs=1,000) 0.5808107778438159
wandb: FuncNameTest-java MRR (bs=1,000) 0.665763599151126
wandb: best_val_mrr 0.43917693074544273
wandb: best_epoch 5
wandb: Test-All MRR (bs=1,000) 0.5808107778438159
wandb: FuncNameTest-All MRR (bs=1,000) 0.665763599151126
wandb: Validation-All MRR (bs=1,000) 0.5693588020579294
wandb: Test-java MRR (bs=1,000) 0.5808107778438159
wandb: FuncNameTest-java MRR (bs=1,000) 0.665763599151126
wandb: Validation-java MRR (bs=1,000) 0.5693588020579294
wandb: Syncing files in wandb/run-20200423_190649-yyavkuve:
wandb: neuralbow-2020-04-23-19-06-49-graph.pbtxt
wandb: neuralbow-2020-04-23-19-06-49.train_log
wandb: neuralbow-2020-04-23-19-06-49_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced neuralbow-2020-04-23-19-06-49: https://app.wandb.ai/jianguda/CodeSearchNet/runs/yyavkuve
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/yyavkuve  
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./neuralbow-2020-04-23-19-06-49_model_best.pkl.gz
2020-04-23 19:26:47.007261: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-23 19:26:51.838503: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 42e4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-23 19:26:51.838549: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-23 19:26:52.128119: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-23 19:26:52.128184: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-23 19:26:52.128201: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-23 19:26:52.128313: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
WARNING:tensorflow:From /home/dev/src/models/model.py:299: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: java
100%|██████████████████████████████████████████████████████████████████████████████████| 1569889/1569889 [00:07<00:00, 198320.34it/s]1569889it [00:27, 57592.19it/s]
Uploading predictions to W&B
NDCG Average: 0.208864018

# CNN

root@jian-csn:/home/dev/src# python train.py --model 1dcnn ../resources/saved_models ../resources/data/java/final/jsonl/train ../resources/data/java/final/jsonl/valid ../resources/data/java/final/jsonl/test
wandb: Started W&B process version 0.8.12 with PID 1874
wandb: Wandb version 0.8.32 is available! To upgrade, please run:
wandb: \$ pip install wandb --upgrade
wandb: Local directory: wandb/run-20200423_194642-ls70soni
wandb: Syncing run 1dcnn-2020-04-23-19-46-42: https://app.wandb.ai/jianguda/CodeSearchNet/runs/ls70soni
wandb: Run `wandb off` to turn off syncing.

2020-04-23 19:46:47.960374: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-23 19:46:48.039264: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 42e4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-23 19:46:48.039308: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-23 19:46:48.318887: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-23 19:46:48.318968: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-23 19:46:48.318987: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-23 19:46:48.319100: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Tokenizing and building vocabulary for code snippets and queries. This step may take several hours.
Starting training run 1dcnn-2020-04-23-19-46-42 of model ConvolutionalModel with following hypers:
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': False, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_1dcnn_position_encoding': 'learned', 'code_1dcnn_layer_list': [128, 128, 128], 'code_1dcnn_kernel_width': [16, 16, 16], 'code_1dcnn_add_residual_connections': True, 'code_1dcnn_activation': 'tanh', 'code_1dcnn_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_1dcnn_position_encoding': 'learned', 'query_1dcnn_layer_list': [128, 128, 128], 'query_1dcnn_kernel_width': [16, 16, 16], 'query_1dcnn_add_residual_connections': True, 'query_1dcnn_activation': 'tanh', 'query_1dcnn_pool_mode': 'weighted_mean', 'batch_size': 1000, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.0005, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'softmax', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
Loading training and validation data.
Begin Training.
Training on 454451 java samples.
Validating on 15328 java samples.
==== Epoch 0 ====
Epoch 0 (train) took 286.20s [processed 1586 samples/second]
Training Loss: 5.620149
Epoch 0 (valid) took 3.11s [processed 4828 samples/second]
Validation: Loss: 5.526572 | MRR: 0.104187
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-19-46-42_model_best.pkl.gz'.
==== Epoch 1 ====
Epoch 1 (train) took 287.13s [processed 1581 samples/second]
Training Loss: 4.149888
Epoch 1 (valid) took 3.04s [processed 4940 samples/second]
Validation: Loss: 4.782327 | MRR: 0.215851
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-19-46-42_model_best.pkl.gz'.
==== Epoch 2 ====
Epoch 2 (train) took 287.96s [processed 1576 samples/second]
Training Loss: 3.244948
Epoch 2 (valid) took 3.04s [processed 4927 samples/second]
Validation: Loss: 4.274906 | MRR: 0.290196
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-19-46-42_model_best.pkl.gz'.
==== Epoch 3 ====
Epoch 3 (train) took 287.76s [processed 1577 samples/second]
Training Loss: 2.705284
Epoch 3 (valid) took 3.05s [processed 4911 samples/second]
Validation: Loss: 4.073514 | MRR: 0.325067
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-19-46-42_model_best.pkl.gz'.
==== Epoch 4 ====
Epoch 4 (train) took 287.98s [processed 1576 samples/second]
Training Loss: 2.380216
Epoch 4 (valid) took 3.03s [processed 4956 samples/second]
Validation: Loss: 3.961440 | MRR: 0.342892
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-19-46-42_model_best.pkl.gz'.
==== Epoch 5 ====
Epoch 5 (train) took 287.77s [processed 1577 samples/second]
Training Loss: 2.152004
Epoch 5 (valid) took 3.03s [processed 4955 samples/second]
Validation: Loss: 3.862712 | MRR: 0.364610
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-19-46-42_model_best.pkl.gz'.
==== Epoch 6 ====
Epoch 6 (train) took 287.77s [processed 1577 samples/second]
Training Loss: 1.972744
Epoch 6 (valid) took 3.06s [processed 4905 samples/second]
Validation: Loss: 3.816265 | MRR: 0.373870
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-19-46-42_model_best.pkl.gz'.
==== Epoch 7 ====
Epoch 7 (train) took 287.65s [processed 1578 samples/second]
Training Loss: 1.839305
Epoch 7 (valid) took 3.07s [processed 4882 samples/second]
Validation: Loss: 3.785008 | MRR: 0.379012
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-19-46-42_model_best.pkl.gz'.
==== Epoch 8 ====
Epoch 8 (train) took 287.81s [processed 1577 samples/second]
Training Loss: 1.738258
Epoch 8 (valid) took 3.07s [processed 4892 samples/second]
Validation: Loss: 3.725931 | MRR: 0.389824
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-19-46-42_model_best.pkl.gz'.
==== Epoch 9 ====
Epoch 9 (train) took 287.74s [processed 1577 samples/second]
Training Loss: 1.649375
Epoch 9 (valid) took 3.05s [processed 4911 samples/second]
Validation: Loss: 3.740898 | MRR: 0.390086
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-19-46-42_model_best.pkl.gz'.
==== Epoch 10 ====
Epoch 10 (train) took 287.63s [processed 1578 samples/second]
Training Loss: 1.570969
Epoch 10 (valid) took 3.02s [processed 4973 samples/second]
Validation: Loss: 3.719377 | MRR: 0.396120
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-19-46-42_model_best.pkl.gz'.
==== Epoch 11 ====
Epoch 11 (train) took 287.31s [processed 1580 samples/second]
Training Loss: 1.509127
Epoch 11 (valid) took 3.06s [processed 4908 samples/second]
Validation: Loss: 3.776039 | MRR: 0.399613
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-19-46-42_model_best.pkl.gz'.
==== Epoch 12 ====
Epoch 12 (train) took 287.57s [processed 1578 samples/second]
Training Loss: 1.447682
Epoch 12 (valid) took 3.04s [processed 4935 samples/second]
Validation: Loss: 3.775385 | MRR: 0.401346
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-19-46-42_model_best.pkl.gz'.
==== Epoch 13 ====
Epoch 13 (train) took 287.20s [processed 1580 samples/second]
Training Loss: 1.402141
Epoch 13 (valid) took 3.04s [processed 4937 samples/second]
Validation: Loss: 3.767275 | MRR: 0.404794
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-19-46-42_model_best.pkl.gz'.
==== Epoch 14 ====
Epoch 14 (train) took 287.37s [processed 1579 samples/second]
Training Loss: 1.358792
Epoch 14 (valid) took 3.03s [processed 4946 samples/second]
Validation: Loss: 3.800728 | MRR: 0.406087
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-19-46-42_model_best.pkl.gz'.
==== Epoch 15 ====
Epoch 15 (train) took 287.50s [processed 1579 samples/second]
Training Loss: 1.322321
Epoch 15 (valid) took 3.02s [processed 4960 samples/second]
Validation: Loss: 3.779871 | MRR: 0.407500
Epoch 14 (valid) took 3.03s [processed 4946 samples/second]
Validation: Loss: 3.800728 | MRR: 0.406087
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-19-46-42_model_best.pkl.gz'.
==== Epoch 15 ====
Epoch 15 (train) took 287.50s [processed 1579 samples/second]
Training Loss: 1.322321
Epoch 15 (valid) took 3.02s [processed 4960 samples/second]
Validation: Loss: 3.779871 | MRR: 0.407500
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-19-46-42_model_best.pkl.gz'.
==== Epoch 16 ====
Epoch 16 (train) took 287.48s [processed 1579 samples/second]
Training Loss: 1.286054
Epoch 16 (valid) took 3.02s [processed 4961 samples/second]
Validation: Loss: 3.780663 | MRR: 0.410333
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-19-46-42_model_best.pkl.gz'.
==== Epoch 17 ====
Epoch 17 (train) took 287.02s [processed 1581 samples/second]
Training Loss: 1.253645
Epoch 17 (valid) took 3.03s [processed 4949 samples/second]
Validation: Loss: 3.783805 | MRR: 0.408914
==== Epoch 18 ====
Epoch 18 (train) took 287.33s [processed 1580 samples/second]
Training Loss: 1.222682
Epoch 18 (valid) took 3.02s [processed 4974 samples/second]
Validation: Loss: 3.770249 | MRR: 0.411233
Training Loss: 1.253645
Epoch 17 (valid) took 3.03s [processed 4949 samples/second]
Validation: Loss: 3.783805 | MRR: 0.408914
==== Epoch 18 ====
Epoch 18 (train) took 287.33s [processed 1580 samples/second]
Training Loss: 1.222682
Epoch 18 (valid) took 3.02s [processed 4974 samples/second]
Validation: Loss: 3.770249 | MRR: 0.411233
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-19-46-42_model_best.pkl.gz'.
==== Epoch 19 ====
Epoch 19 (train) took 287.36s [processed 1579 samples/second]
Training Loss: 1.203888
Epoch 19 (valid) took 3.06s [processed 4903 samples/second]
Validation: Loss: 3.814675 | MRR: 0.410734
==== Epoch 20 ====
Epoch 20 (train) took 287.62s [processed 1578 samples/second]
Training Loss: 1.179031
Epoch 20 (valid) took 3.03s [processed 4949 samples/second]
Validation: Loss: 3.828565 | MRR: 0.414401
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-19-46-42_model_best.pkl.gz'.
==== Epoch 21 ====
Epoch 21 (train) took 287.38s [processed 1579 samples/second]
Training Loss: 1.156595
Epoch 21 (valid) took 3.03s [processed 4947 samples/second]
Validation: Loss: 3.859845 | MRR: 0.412491
==== Epoch 22 ====
Epoch 22 (train) took 287.45s [processed 1579 samples/second]
Training Loss: 1.137142
Epoch 22 (valid) took 3.05s [processed 4917 samples/second]
Validation: Loss: 3.821515 | MRR: 0.413610
==== Epoch 23 ====
Epoch 23 (train) took 287.34s [processed 1580 samples/second]
Training Loss: 1.115082
Epoch 23 (valid) took 3.04s [processed 4933 samples/second]
Validation: Loss: 3.883173 | MRR: 0.414502
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-19-46-42_model_best.pkl.gz'.
==== Epoch 24 ====
Epoch 24 (train) took 287.63s [processed 1578 samples/second]
Training Loss: 1.100578
Epoch 24 (valid) took 3.09s [processed 4861 samples/second]
Validation: Loss: 3.863960 | MRR: 0.414805
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-19-46-42_model_best.pkl.gz'.
==== Epoch 25 ====
Epoch 25 (train) took 287.14s [processed 1581 samples/second]
Training Loss: 1.086174
Epoch 25 (valid) took 3.03s [processed 4952 samples/second]
Validation: Loss: 3.920691 | MRR: 0.412895
==== Epoch 26 ====
Epoch 26 (train) took 287.32s [processed 1580 samples/second]
Training Loss: 1.068923
Epoch 26 (valid) took 3.05s [processed 4924 samples/second]
Validation: Loss: 3.900170 | MRR: 0.416624
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-19-46-42_model_best.pkl.gz'.
==== Epoch 27 ====
Epoch 27 (train) took 287.07s [processed 1581 samples/second]
Training Loss: 1.055352
Epoch 27 (valid) took 3.04s [processed 4935 samples/second]
Validation: Loss: 3.897430 | MRR: 0.416205
==== Epoch 28 ====
Epoch 28 (train) took 287.54s [processed 1578 samples/second]
Training Loss: 1.045937
Epoch 28 (valid) took 3.05s [processed 4923 samples/second]
Validation: Loss: 3.925543 | MRR: 0.412707
==== Epoch 29 ====
Epoch 29 (train) took 287.21s [processed 1580 samples/second]
Training Loss: 1.029287
Epoch 29 (valid) took 3.03s [processed 4953 samples/second]
Validation: Loss: 3.936552 | MRR: 0.413799
==== Epoch 30 ====
Epoch 30 (train) took 287.67s [processed 1578 samples/second]
Training Loss: 1.022569
Epoch 30 (valid) took 3.07s [processed 4891 samples/second]
Validation: Loss: 3.957137 | MRR: 0.415116
==== Epoch 31 ====
Epoch 31 (train) took 287.74s [processed 1577 samples/second]
Training Loss: 1.005778
Epoch 31 (valid) took 3.05s [processed 4918 samples/second]
Validation: Loss: 3.958731 | MRR: 0.418136
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-19-46-42_model_best.pkl.gz'.
==== Epoch 30 ====
Epoch 30 (train) took 287.67s [processed 1578 samples/second]
Training Loss: 1.022569
Epoch 30 (valid) took 3.07s [processed 4891 samples/second]
Validation: Loss: 3.957137 | MRR: 0.415116
==== Epoch 31 ====
Epoch 31 (train) took 287.74s [processed 1577 samples/second]
Training Loss: 1.005778
Epoch 31 (valid) took 3.05s [processed 4918 samples/second]
Validation: Loss: 3.958731 | MRR: 0.418136
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-19-46-42_model_best.pkl.gz'.
==== Epoch 32 ====
Epoch 32 (train) took 287.77s [processed 1577 samples/second]
Training Loss: 0.997660
Epoch 32 (valid) took 3.04s [processed 4936 samples/second]
Validation: Loss: 3.930026 | MRR: 0.417845
==== Epoch 33 ====
Epoch 33 (train) took 287.73s [processed 1577 samples/second]
Training Loss: 0.986192
Epoch 33 (valid) took 3.03s [processed 4943 samples/second]
Validation: Loss: 3.932226 | MRR: 0.415816
==== Epoch 34 ====
Epoch 34 (train) took 287.44s [processed 1579 samples/second]
Training Loss: 0.978175
Epoch 34 (valid) took 3.04s [processed 4938 samples/second]
Validation: Loss: 3.966798 | MRR: 0.418763
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-19-46-42_model_best.pkl.gz'.
==== Epoch 35 ====
Epoch 35 (train) took 287.87s [processed 1577 samples/second]
Training Loss: 0.968325
Epoch 35 (valid) took 3.04s [processed 4940 samples/second]
Validation: Loss: 3.974447 | MRR: 0.419604
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-19-46-42_model_best.pkl.gz'.
==== Epoch 36 ====
Epoch 36 (train) took 287.64s [processed 1578 samples/second]
Training Loss: 0.959590
Epoch 36 (valid) took 3.06s [processed 4905 samples/second]
Validation: Loss: 4.013772 | MRR: 0.415824
==== Epoch 37 ====
Epoch 37 (train) took 287.53s [processed 1578 samples/second]
Training Loss: 0.949264
Epoch 37 (valid) took 3.03s [processed 4950 samples/second]
Validation: Loss: 3.978953 | MRR: 0.417255
==== Epoch 38 ====
Epoch 38 (train) took 287.55s [processed 1578 samples/second]
Training Loss: 0.942061
Epoch 38 (valid) took 3.02s [processed 4967 samples/second]
Validation: Loss: 3.974341 | MRR: 0.419072
==== Epoch 39 ====
Epoch 39 (train) took 287.83s [processed 1577 samples/second]
Training Loss: 0.934510
Epoch 39 (valid) took 3.03s [processed 4954 samples/second]
Validation: Loss: 4.001976 | MRR: 0.420581
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-19-46-42_model_best.pkl.gz'.
==== Epoch 40 ====
Epoch 40 (train) took 287.45s [processed 1579 samples/second]
Training Loss: 0.925210
Epoch 40 (valid) took 3.01s [processed 4975 samples/second]
Validation: Loss: 4.042948 | MRR: 0.418588
==== Epoch 41 ====
Epoch 41 (train) took 287.66s [processed 1578 samples/second]
Training Loss: 0.919351
Epoch 41 (valid) took 3.03s [processed 4951 samples/second]
Validation: Loss: 4.050352 | MRR: 0.418831
==== Epoch 42 ====
Epoch 42 (train) took 287.39s [processed 1579 samples/second]
Training Loss: 0.913543
Epoch 42 (valid) took 3.01s [processed 4990 samples/second]
Validation: Loss: 4.043287 | MRR: 0.417997
==== Epoch 43 ====
Epoch 43 (train) took 287.48s [processed 1579 samples/second]
Training Loss: 0.906131
Epoch 43 (valid) took 3.03s [processed 4949 samples/second]
Validation: Loss: 4.081843 | MRR: 0.416814
==== Epoch 44 ====
Epoch 44 (train) took 287.70s [processed 1578 samples/second]
Training Loss: 0.899333
Epoch 44 (valid) took 3.03s [processed 4944 samples/second]
Validation: Loss: 4.106812 | MRR: 0.417197
2020-04-23 23:35:02.290008: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-23 23:35:02.290071: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-23 23:35:02.290087: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-23 23:35:02.290099: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-23 23:35:02.290197: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.530
FuncNameTest-All MRR (bs=1,000): 0.773
Validation-All MRR (bs=1,000): 0.504
Test-java MRR (bs=1,000): 0.530
FuncNameTest-java MRR (bs=1,000): 0.773
Validation-java MRR (bs=1,000): 0.504

wandb: Waiting for W&B process to finish, PID 1874
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_step 275
wandb: train-loss 0.8993327876282159
wandb: train-mrr 0.8306090454908195
wandb: \_timestamp 1587685021.216103
wandb: \_runtime 13819.981269836426
wandb: epoch 44
wandb: train-time-sec 287.69667768478394
wandb: val-time-sec 3.033641815185547
wandb: val-mrr 0.4171967793782552
wandb: val-loss 4.106812477111816
wandb: best_val_mrr_loss 4.001975933710734
wandb: best_val_mrr 0.42058077392578125
wandb: best_epoch 39
wandb: Test-All MRR (bs=1,000) 0.5302393033711135
wandb: FuncNameTest-All MRR (bs=1,000) 0.7729013480342846
wandb: Validation-All MRR (bs=1,000) 0.5038463380974542
wandb: Test-java MRR (bs=1,000) 0.5302393033711135
wandb: FuncNameTest-java MRR (bs=1,000) 0.7729013480342846
wandb: Validation-java MRR (bs=1,000) 0.5038463380974542
wandb: Syncing files in wandb/run-20200423_194642-ls70soni:
wandb: 1dcnn-2020-04-23-19-46-42-graph.pbtxt
wandb: 1dcnn-2020-04-23-19-46-42.train_log
wandb: 1dcnn-2020-04-23-19-46-42_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced 1dcnn-2020-04-23-19-46-42: https://app.wandb.ai/jianguda/CodeSearchNet/runs/ls70soni
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/ls70soni
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./1dcnn-2020-04-23-19-46-42_model_best.pkl.gz
2020-04-23 23:37:45.574248: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-23 23:37:50.610086: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 42e4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-23 23:37:50.610178: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-23 23:37:50.885493: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
2020-04-23 23:37:50.610086: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 42e4:00:00.0
wandb: 1dcnn-2020-04-23-19-46-42.train_log
wandb: 1dcnn-2020-04-23-19-46-42_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced 1dcnn-2020-04-23-19-46-42: https://app.wandb.ai/jianguda/CodeSearchNet/runs/ls70soni
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/ls70soni  
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./1dcnn-2020-04-23-19-46-42_model_best.pkl.gz
2020-04-23 23:37:45.574248: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-23 23:37:50.610086: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 42e4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-23 23:37:50.610178: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-23 23:37:50.885493: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-23 23:37:50.885557: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-23 23:37:50.885574: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-23 23:37:50.885683: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Evaluating language: java
100%|██████████████████████████████████████████████████████████████████████████████████| 1569889/1569889 [00:08<00:00, 196011.11it/s]1569889it [00:27, 57644.58it/s]
Uploading predictions to W&B
NDCG Average: 0.116348981

# RNN

root@jian-csn:/home/dev/src# python train.py --model rnn ../resources/saved_models ../resources/data/java/final/jsonl/train ../resources/data/java/final/jsonl/valid ../resources/data/java/final/jsonl/test
wandb: Started W&B process version 0.8.12 with PID 2383
wandb: Wandb version 0.8.32 is available! To upgrade, please run:
wandb: \$ pip install wandb --upgrade
wandb: Local directory: wandb/run-20200424_061222-adi1cax0
wandb: Syncing run rnn-2020-04-24-06-12-22: https://app.wandb.ai/jianguda/CodeSearchNet/runs/adi1cax0
wandb: Run `wandb off` to turn off syncing.

2020-04-24 06:12:27.596123: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-24 06:12:27.675191: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 42e4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-24 06:12:27.675238: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-24 06:12:27.955501: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-24 06:12:27.955567: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-24 06:12:27.955583: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-24 06:12:27.955694: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Tokenizing and building vocabulary for code snippets and queries. This step may take several hours.
Starting training run rnn-2020-04-24-06-12-22 of model RNNModel with following hypers:
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': True, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_rnn_num_layers': 2, 'code_rnn_hidden_dim': 64, 'code_rnn_cell_type': 'LSTM', 'code_rnn_is_bidirectional': True, 'code_rnn_dropout_keep_rate': 0.8, 'code_rnn_recurrent_dropout_keep_rate': 1.0, 'code_rnn_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_rnn_num_layers': 2, 'query_rnn_hidden_dim': 64, 'query_rnn_cell_type': 'LSTM', 'query_rnn_is_bidirectional': True, 'query_rnn_dropout_keep_rate': 0.8, 'query_rnn_recurrent_dropout_keep_rate': 1.0, 'query_rnn_pool_mode': 'weighted_mean', 'batch_size': 1000, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.01, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'softmax', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
Loading training and validation data.
Begin Training.
Training on 454451 java samples.
Validating on 15328 java samples.
==== Epoch 0 ====
Epoch 0 (train) took 490.75s [processed 925 samples/second]
Training Loss: 4.290682
Epoch 0 (valid) took 8.01s [processed 1872 samples/second]
Validation: Loss: 3.780313 | MRR: 0.348009
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-24-06-12-22_model_best.pkl.gz'.
==== Epoch 1 ====
Epoch 1 (train) took 488.30s [processed 929 samples/second]
Training Loss: 2.376959
Epoch 1 (valid) took 7.76s [processed 1933 samples/second]
Validation: Loss: 3.394579 | MRR: 0.411918
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-24-06-12-22_model_best.pkl.gz'.
==== Epoch 2 ====
Epoch 2 (train) took 487.71s [processed 930 samples/second]
Training Loss: 2.012066
Epoch 2 (valid) took 7.74s [processed 1937 samples/second]
Validation: Loss: 3.267137 | MRR: 0.431809
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-24-06-12-22_model_best.pkl.gz'.
==== Epoch 3 ====
Epoch 3 (train) took 488.71s [processed 928 samples/second]
Training Loss: 1.857256
Epoch 3 (valid) took 7.76s [processed 1933 samples/second]
Validation: Loss: 3.262681 | MRR: 0.435909
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-24-06-12-22_model_best.pkl.gz'.
==== Epoch 4 ====
Epoch 4 (train) took 488.96s [processed 928 samples/second]
Training Loss: 1.771898
Epoch 4 (valid) took 7.79s [processed 1926 samples/second]
Validation: Loss: 3.234054 | MRR: 0.442413
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-24-06-12-22_model_best.pkl.gz'.
==== Epoch 5 ====
Epoch 5 (train) took 488.17s [processed 930 samples/second]
Training Loss: 1.720917
Epoch 5 (valid) took 7.75s [processed 1935 samples/second]
Validation: Loss: 3.223739 | MRR: 0.446056
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-24-06-12-22_model_best.pkl.gz'.
==== Epoch 6 ====
Epoch 6 (train) took 487.40s [processed 931 samples/second]
Training Loss: 1.685628
Epoch 6 (valid) took 7.75s [processed 1935 samples/second]
Validation: Loss: 3.218857 | MRR: 0.448055
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-24-06-12-22_model_best.pkl.gz'.
==== Epoch 7 ====
Epoch 7 (train) took 488.07s [processed 930 samples/second]
Training Loss: 1.661936
Epoch 7 (valid) took 7.74s [processed 1939 samples/second]
Validation: Loss: 3.232906 | MRR: 0.448267
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-24-06-12-22_model_best.pkl.gz'.
==== Epoch 8 ====
Epoch 8 (train) took 488.44s [processed 929 samples/second]
Training Loss: 1.651766
Epoch 8 (valid) took 7.71s [processed 1944 samples/second]
Validation: Loss: 3.209340 | MRR: 0.448768
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-24-06-12-22_model_best.pkl.gz'.
==== Epoch 9 ====
Epoch 9 (train) took 487.51s [processed 931 samples/second]
Training Loss: 1.634113
Epoch 9 (valid) took 7.73s [processed 1940 samples/second]
Validation: Loss: 3.207217 | MRR: 0.448670
==== Epoch 10 ====
Epoch 10 (train) took 486.89s [processed 932 samples/second]
Training Loss: 1.627499
Epoch 10 (valid) took 7.72s [processed 1941 samples/second]
Validation: Loss: 3.209212 | MRR: 0.450691
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-24-06-12-22_model_best.pkl.gz'.
==== Epoch 11 ====
Epoch 11 (train) took 487.58s [processed 931 samples/second]
Training Loss: 1.622711
Epoch 11 (valid) took 7.79s [processed 1925 samples/second]
Validation: Loss: 3.232339 | MRR: 0.448353
==== Epoch 12 ====
Epoch 12 (train) took 488.20s [processed 929 samples/second]
Training Loss: 1.614263
Epoch 12 (valid) took 7.72s [processed 1943 samples/second]
Validation: Loss: 3.227369 | MRR: 0.448098
==== Epoch 13 ====
Epoch 13 (train) took 487.69s [processed 930 samples/second]
Training Loss: 1.616298
Epoch 13 (valid) took 7.75s [processed 1934 samples/second]
Validation: Loss: 3.210820 | MRR: 0.451538
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-24-06-12-22_model_best.pkl.gz'.
==== Epoch 14 ====
Epoch 14 (train) took 487.06s [processed 932 samples/second]
Training Loss: 1.613308
Epoch 14 (valid) took 7.73s [processed 1939 samples/second]
Validation: Loss: 3.220831 | MRR: 0.450837
==== Epoch 15 ====
Epoch 15 (train) took 487.89s [processed 930 samples/second]
Training Loss: 1.615268
Epoch 15 (valid) took 7.78s [processed 1929 samples/second]
Validation: Loss: 3.237167 | MRR: 0.448292
==== Epoch 16 ====
wandb: Network error resolved after 0:00:19.788720, resuming normal operation.far: 1.6174. MRR so far: 0.7163
wandb: Network error (HTTPError), entering retry loop. See /home/dev/src/wandb/debug.log for full traceback.60
wandb: Network error resolved after 0:00:15.808346, resuming normal operation.far: 1.6186. MRR so far: 0.7160
Epoch 16 (train) took 488.37s [processed 929 samples/second]
Training Loss: 1.623470
Epoch 16 (valid) took 7.74s [processed 1939 samples/second]
Validation: Loss: 3.228537 | MRR: 0.447674
==== Epoch 17 ====
Epoch 17 (train) took 487.72s [processed 930 samples/second]
Training Loss: 1.621259
Epoch 17 (valid) took 7.73s [processed 1940 samples/second]
Validation: Loss: 3.198973 | MRR: 0.450595
==== Epoch 18 ====
Epoch 18 (train) took 487.20s [processed 931 samples/second]
Training Loss: 1.628623
Epoch 18 (valid) took 7.72s [processed 1943 samples/second]
Validation: Loss: 3.215264 | MRR: 0.452194
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-24-06-12-22_model_best.pkl.gz'.
==== Epoch 19 ====
Epoch 19 (train) took 487.62s [processed 931 samples/second]
Training Loss: 1.636671
Epoch 19 (valid) took 7.72s [processed 1942 samples/second]
Validation: Loss: 3.247198 | MRR: 0.447986
==== Epoch 20 ====
Epoch 20 (train) took 488.18s [processed 929 samples/second]
Training Loss: 1.633719
Epoch 20 (valid) took 7.75s [processed 1935 samples/second]
Validation: Loss: 3.252065 | MRR: 0.448966
==== Epoch 21 ====
Epoch 21 (train) took 487.55s [processed 931 samples/second]
Training Loss: 1.652369
Epoch 21 (valid) took 7.73s [processed 1940 samples/second]
Validation: Loss: 3.241103 | MRR: 0.449975
==== Epoch 22 ====
Epoch 22 (train) took 487.00s [processed 932 samples/second]
Training Loss: 1.657303
Epoch 22 (valid) took 7.75s [processed 1934 samples/second]
Validation: Loss: 3.245459 | MRR: 0.447415
==== Epoch 23 ====
Epoch 23 (train) took 487.73s [processed 930 samples/second]
Training Loss: 1.665215
Epoch 23 (valid) took 7.74s [processed 1937 samples/second]
Validation: Loss: 3.254298 | MRR: 0.449035
2020-04-24 09:40:24.271735: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-24 09:40:24.271800: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-24 09:40:24.271816: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-24 09:40:24.271829: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-24 09:40:24.271932: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.581
FuncNameTest-All MRR (bs=1,000): 0.794
Validation-All MRR (bs=1,000): 0.561
Test-java MRR (bs=1,000): 0.581
FuncNameTest-java MRR (bs=1,000): 0.794
Validation-java MRR (bs=1,000): 0.561

wandb: Waiting for W&B process to finish, PID 2383
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-loss 1.6652152708973653
wandb: \_runtime 12646.390653848648
wandb: \_timestamp 1587721387.232695
wandb: train-mrr 0.7075645707588364
wandb: \_step 149
wandb: epoch 23
wandb: train-time-sec 487.72515320777893
wandb: val-mrr 0.44903487548828125
wandb: val-time-sec 7.741694927215576
wandb: val-loss 3.2542975107828775
wandb: best_val_mrr_loss 3.215264002482096
wandb: best_val_mrr 0.4521944742838542
wandb: best_epoch 18
wandb: Test-All MRR (bs=1,000) 0.5807536677380386
wandb: FuncNameTest-All MRR (bs=1,000) 0.7939039755690349
wandb: Validation-All MRR (bs=1,000) 0.5610670397170924
wandb: Test-java MRR (bs=1,000) 0.5807536677380386
wandb: FuncNameTest-java MRR (bs=1,000) 0.7939039755690349
wandb: Validation-java MRR (bs=1,000) 0.5610670397170924
wandb: Syncing files in wandb/run-20200424_061222-adi1cax0:
wandb: rnn-2020-04-24-06-12-22-graph.pbtxt
wandb: rnn-2020-04-24-06-12-22.train_log
wandb: rnn-2020-04-24-06-12-22_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced rnn-2020-04-24-06-12-22: https://app.wandb.ai/jianguda/CodeSearchNet/runs/adi1cax0
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/adi1cax0
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./rnn-2020-04-24-06-12-22_model_best.pkl.gz
2020-04-24 10:21:26.726912: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-24 10:21:31.667799: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 42e4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-24 10:21:31.667846: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-24 10:21:31.947476: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-24 10:21:31.947538: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-24 10:21:31.947556: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-24 10:21:31.947666: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Evaluating language: java
100%|██████████████████████████████████████████████████████████████████████████████████| 1569889/1569889 [00:14<00:00, 109533.44it/s]1569889it [00:27, 57208.99it/s]
Uploading predictions to W&B
NDCG Average: 0.122035273

# BERT

root@jian-csn:/home/dev/src# python train.py --model selfatt ../resources/saved_models ../resources/data/java/final/jsonl/train ../resources/data/java/final/jsonl/valid ../resources/data/java/final/jsonl/test
wandb: Started W&B process version 0.8.12 with PID 2173
wandb: Wandb version 0.8.32 is available! To upgrade, please run:
wandb: $ pip install wandb --upgrade
wandb: Local directory: wandb/run-20200424_002516-nritbbb8
wandb: Syncing run selfatt-2020-04-24-00-25-16: https://app.wandb.ai/jianguda/CodeSearchNet/runs/nritbbb8
root@jian-csn:/home/dev/src# python train.py --model selfatt ../resources/saved_models ../resources/data/java/final/jsonl/train ../resources/data/java/final/jsonl/valid ../resources/data/java/final/jsonl/test
wandb: Started W&B process version 0.8.12 with PID 2173
wandb: Wandb version 0.8.32 is available!  To upgrade, please run:
wandb:  $ pip install wandb --upgrade
wandb: Local directory: wandb/run-20200424_002516-nritbbb8
wandb: Syncing run selfatt-2020-04-24-00-25-16: https://app.wandb.ai/jianguda/CodeSearchNet/runs/nritbbb8
wandb: Run `wandb off` to turn off syncing.

2020-04-24 00:25:22.433862: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-24 00:25:22.514763: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 42e4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-24 00:25:22.514806: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-24 00:25:22.790857: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-24 00:25:22.790915: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-24 00:25:22.790929: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-24 00:25:22.791037: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Tokenizing and building vocabulary for code snippets and queries. This step may take several hours.
Starting training run selfatt-2020-04-24-00-25-16 of model SelfAttentionModel with following hypers:
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': False, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_self_attention_activation': 'gelu', 'code_self_attention_hidden_size': 128, 'code_self_attention_intermediate_size': 512, 'code_self_attention_num_layers': 3, 'code_self_attention_num_heads': 8, 'code_self_attention_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_self_attention_activation': 'gelu', 'query_self_attention_hidden_size': 128, 'query_self_attention_intermediate_size': 512, 'query_self_attention_num_layers': 3, 'query_self_attention_num_heads': 8, 'query_self_attention_pool_mode': 'weighted_mean', 'batch_size': 450, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.0005, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'softmax', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
Loading training and validation data.
Begin Training.
Training on 454451 java samples.
Validating on 15328 java samples.
==== Epoch 0 ====
Epoch 0 (train) took 1397.51s [processed 324 samples/second]
Training Loss: 2.331113
Epoch 0 (valid) took 20.77s [processed 736 samples/second]
Validation: Loss: 3.110438 | MRR: 0.451764
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-24-00-25-16_model_best.pkl.gz'.
==== Epoch 1 ====
Epoch 1 (train) took 1399.53s [processed 324 samples/second]
Training Loss: 1.089985
Epoch 1 (valid) took 20.01s [processed 764 samples/second]
Validation: Loss: 3.001767 | MRR: 0.474145
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-24-00-25-16_model_best.pkl.gz'.
==== Epoch 2 ====
Epoch 2 (train) took 1396.56s [processed 325 samples/second]
Training Loss: 0.852022
Epoch 2 (valid) took 20.32s [processed 752 samples/second]
Validation: Loss: 3.010332 | MRR: 0.486245
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-24-00-25-16_model_best.pkl.gz'.
==== Epoch 3 ====
Epoch 3 (train) took 1398.62s [processed 324 samples/second]
Training Loss: 0.734062
Epoch 3 (valid) took 20.04s [processed 763 samples/second]
Validation: Loss: 3.071282 | MRR: 0.491263
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-24-00-25-16_model_best.pkl.gz'.
==== Epoch 4 ====
Epoch 4 (train) took 1398.03s [processed 324 samples/second]
Training Loss: 0.658466
Epoch 4 (valid) took 20.18s [processed 758 samples/second]
Validation: Loss: 3.182527 | MRR: 0.490428
==== Epoch 5 ====
Epoch 5 (train) took 1400.09s [processed 324 samples/second]
Training Loss: 0.606719
Epoch 5 (valid) took 20.30s [processed 753 samples/second]
Validation: Loss: 3.183271 | MRR: 0.495917
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-24-00-25-16_model_best.pkl.gz'.
==== Epoch 6 ====
Epoch 6 (train) took 1398.13s [processed 324 samples/second]
Training Loss: 0.565752
Epoch 6 (valid) took 19.92s [processed 768 samples/second]
Validation: Loss: 3.214389 | MRR: 0.492818
==== Epoch 7 ====
Epoch 7 (train) took 1397.18s [processed 324 samples/second]
Training Loss: 0.534449
Epoch 7 (valid) took 20.11s [processed 760 samples/second]
Validation: Loss: 3.312335 | MRR: 0.491479
==== Epoch 8 ====
Epoch 8 (train) took 1398.43s [processed 324 samples/second]
Training Loss: 0.507047
Epoch 8 (valid) took 20.23s [processed 756 samples/second]
Validation: Loss: 3.359076 | MRR: 0.492518
==== Epoch 9 ====
Epoch 9 (train) took 1399.99s [processed 324 samples/second]
Training Loss: 0.484469
Epoch 9 (valid) took 20.39s [processed 750 samples/second]
Validation: Loss: 3.401904 | MRR: 0.495683
==== Epoch 10 ====
Epoch 10 (train) took 1399.79s [processed 324 samples/second]
Training Loss: 0.468273
Epoch 10 (valid) took 19.98s [processed 765 samples/second]
Validation: Loss: 3.446262 | MRR: 0.493793
2020-04-24 04:54:59.065418: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-24 04:54:59.065477: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-24 04:54:59.065493: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-24 04:54:59.065504: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-24 04:54:59.065599: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.551
FuncNameTest-All MRR (bs=1,000): 0.756
Validation-All MRR (bs=1,000): 0.532
Test-java MRR (bs=1,000): 0.551
FuncNameTest-java MRR (bs=1,000): 0.756
Validation-java MRR (bs=1,000): 0.532

wandb: Waiting for W&B process to finish, PID 2173
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-mrr 0.9031205749679748
wandb: train-loss 0.46827279375376857
wandb: \_runtime 16407.203567266464
wandb: \_timestamp 1587704322.868424
wandb: \_step 137
wandb: val-loss 3.44626241571763
wandb: train-time-sec 1399.7897834777832
wandb: val-mrr 0.49379305222455194
wandb: val-time-sec 19.97758150100708
wandb: epoch 10
wandb: best_val_mrr_loss 3.1832709242315853
wandb: best_val_mrr 0.49591727761661303
wandb: best_epoch 5
wandb: Test-All MRR (bs=1,000) 0.5509606841961511
wandb: FuncNameTest-All MRR (bs=1,000) 0.756333265724174
wandb: Validation-All MRR (bs=1,000) 0.5315781237178299
wandb: Test-java MRR (bs=1,000) 0.5509606841961511
wandb: FuncNameTest-java MRR (bs=1,000) 0.756333265724174
wandb: Validation-java MRR (bs=1,000) 0.5315781237178299
wandb: Syncing files in wandb/run-20200424_002516-nritbbb8:
wandb: selfatt-2020-04-24-00-25-16-graph.pbtxt
wandb: selfatt-2020-04-24-00-25-16.train_log
wandb: selfatt-2020-04-24-00-25-16_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced selfatt-2020-04-24-00-25-16: https://app.wandb.ai/jianguda/CodeSearchNet/runs/nritbbb8
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/nritbbb8
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./selfatt-2020-04-24-00-25-16_model_best.pkl.gz
2020-04-24 05:25:31.622664: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-24 05:25:36.478091: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 42e4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-24 05:25:36.478139: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-24 05:25:36.758897: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-24 05:25:36.758960: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-24 05:25:36.758979: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-24 05:25:36.759088: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Evaluating language: java
100%|██████████████████████████████████████████████████████████████████████████████████| 1569889/1569889 [00:14<00:00, 109632.01it/s]1569889it [00:27, 57265.04it/s]
Uploading predictions to W&B
NDCG Average: 0.095117461
