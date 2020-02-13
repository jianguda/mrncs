python train.py --model neuralbow ../resources/saved_models ../resources/data/go/final/jsonl/train ../resources/data/go/final/jsonl/valid ../resources/data/go/final/jsonl/test

python predict.py -r jianguda/CodeSearchNet/0123456

# NBOW

root@jian-csn:/home/dev/src# python train.py --model neuralbow ../resources/saved_models ../resources/data/go/final/jsonl/train ../resources/data/go/final/jsonl/valid ../resources/data/go/final/jsonl/test
wandb: W&B is a tool that helps track and visualize machine learning experiments
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 2
wandb: You chose 'Use an existing W&B account'
wandb: You can find your API key in your browser here: https://app.wandb.ai/authorize
wandb: Paste an API key from your profile and hit enter
wandb: Appending key for api.wandb.ai to your netrc file: /root/.netrc
wandb: Started W&B process version 0.8.12 with PID 28
wandb: Wandb version 0.8.32 is available! To upgrade, please run:
wandb: \$ pip install wandb --upgrade
wandb: Local directory: wandb/run-20200422_184412-yb7expi2
wandb: Syncing run neuralbow-2020-04-22-18-44-12: https://app.wandb.ai/jianguda/CodeSearchNet/runs/yb7expi2
wandb: Run `wandb off` to turn off syncing.

2020-04-22 18:44:33.020689: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-22 18:44:33.248382: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 42e4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-22 18:44:33.248426: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-22 18:44:46.780657: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-22 18:44:46.780703: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-22 18:44:46.780718: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-22 18:44:46.780830: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Tokenizing and building vocabulary for code snippets and queries. This step may take several hours.
WARNING:tensorflow:From /home/dev/src/models/model.py:299: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Starting training run neuralbow-2020-04-22-18-44-12 of model NeuralBoWModel with following hypers:
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': False, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_nbow_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_nbow_pool_mode': 'weighted_mean', 'batch_size': 1000, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.01, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'cosine', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
Loading training and validation data.
Begin Training.
Training on 317832 go samples.
Validating on 14242 go samples.
==== Epoch 0 ====
Epoch 0 (train) took 32.00s [processed 9904 samples/second]
Training Loss: 0.958089
Epoch 0 (valid) took 0.91s [processed 15369 samples/second]
Validation: Loss: 1.012580 | MRR: 0.606386
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-22-18-44-12_model_best.pkl.gz'.
==== Epoch 1 ====
Epoch 1 (train) took 29.86s [processed 10617 samples/second]
Training Loss: 0.840207
Epoch 1 (valid) took 0.87s [processed 16149 samples/second]
Validation: Loss: 1.000048 | MRR: 0.646282
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-22-18-44-12_model_best.pkl.gz'.
==== Epoch 2 ====
Epoch 2 (train) took 29.86s [processed 10617 samples/second]
Training Loss: 0.811081
Epoch 2 (valid) took 0.86s [processed 16234 samples/second]
Validation: Loss: 0.997912 | MRR: 0.653097
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-22-18-44-12_model_best.pkl.gz'.
==== Epoch 3 ====
Epoch 3 (train) took 29.82s [processed 10628 samples/second]
Training Loss: 0.797922
Epoch 3 (valid) took 0.87s [processed 16140 samples/second]
Validation: Loss: 0.996114 | MRR: 0.657370
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-22-18-44-12_model_best.pkl.gz'.
==== Epoch 4 ====
Epoch 4 (train) took 29.75s [processed 10656 samples/second]
Training Loss: 0.790061
Epoch 4 (valid) took 0.87s [processed 16130 samples/second]
Validation: Loss: 0.996054 | MRR: 0.657005
==== Epoch 5 ====
Epoch 5 (train) took 29.82s [processed 10630 samples/second]
Training Loss: 0.784133
Epoch 5 (valid) took 0.87s [processed 16062 samples/second]
Validation: Loss: 0.995720 | MRR: 0.659756
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-22-18-44-12_model_best.pkl.gz'.
==== Epoch 6 ====
Epoch 6 (train) took 29.70s [processed 10673 samples/second]
Training Loss: 0.780577
Epoch 6 (valid) took 0.86s [processed 16216 samples/second]
Validation: Loss: 0.995312 | MRR: 0.659564
==== Epoch 7 ====
Epoch 7 (train) took 29.80s [processed 10636 samples/second]
Training Loss: 0.777353
Epoch 7 (valid) took 0.86s [processed 16318 samples/second]
Validation: Loss: 0.995018 | MRR: 0.661302
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-22-18-44-12_model_best.pkl.gz'.
==== Epoch 8 ====
Epoch 8 (train) took 29.81s [processed 10632 samples/second]
Training Loss: 0.774863
Epoch 8 (valid) took 0.86s [processed 16273 samples/second]
Validation: Loss: 0.995267 | MRR: 0.661023
==== Epoch 9 ====
Epoch 9 (train) took 29.71s [processed 10670 samples/second]
Training Loss: 0.772703
Epoch 9 (valid) took 0.89s [processed 15771 samples/second]
Validation: Loss: 0.995752 | MRR: 0.659988
==== Epoch 10 ====
Epoch 10 (train) took 29.75s [processed 10654 samples/second]
Training Loss: 0.770818
Epoch 10 (valid) took 0.89s [processed 15696 samples/second]
Validation: Loss: 0.995291 | MRR: 0.659512
==== Epoch 11 ====
Epoch 11 (train) took 29.83s [processed 10627 samples/second]
Training Loss: 0.769610
Epoch 11 (valid) took 0.86s [processed 16236 samples/second]
Validation: Loss: 0.995235 | MRR: 0.659744
==== Epoch 12 ====
Epoch 12 (train) took 29.69s [processed 10675 samples/second]
Training Loss: 0.768074
Epoch 12 (valid) took 0.87s [processed 16011 samples/second]
Validation: Loss: 0.995526 | MRR: 0.662340
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-22-18-44-12_model_best.pkl.gz'.
==== Epoch 13 ====
Epoch 13 (train) took 29.67s [processed 10683 samples/second]
Training Loss: 0.767124
Epoch 13 (valid) took 0.85s [processed 16437 samples/second]
Validation: Loss: 0.994899 | MRR: 0.661213
==== Epoch 14 ====
Epoch 14 (train) took 29.70s [processed 10673 samples/second]
Training Loss: 0.765837
Epoch 14 (valid) took 0.86s [processed 16271 samples/second]
Validation: Loss: 0.995308 | MRR: 0.659850
==== Epoch 15 ====
Epoch 15 (train) took 29.71s [processed 10668 samples/second]
Training Loss: 0.764248
Epoch 15 (valid) took 0.87s [processed 16179 samples/second]
Validation: Loss: 0.996202 | MRR: 0.661289
==== Epoch 16 ====
Epoch 16 (train) took 29.69s [processed 10678 samples/second]
Training Loss: 0.763841
Epoch 16 (valid) took 0.86s [processed 16295 samples/second]
Validation: Loss: 0.995312 | MRR: 0.660340
==== Epoch 17 ====
Epoch 17 (train) took 29.70s [processed 10673 samples/second]
Training Loss: 0.763094
Epoch 17 (valid) took 0.87s [processed 16162 samples/second]
Validation: Loss: 0.995255 | MRR: 0.661203
2020-04-22 18:58:32.953000: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-22 18:58:32.953063: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-22 18:58:32.953079: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-22 18:58:32.953092: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-22 18:58:32.953181: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.668
FuncNameTest-All MRR (bs=1,000): 0.311
Validation-All MRR (bs=1,000): 0.799
Test-go MRR (bs=1,000): 0.668
FuncNameTest-go MRR (bs=1,000): 0.311
Validation-go MRR (bs=1,000): 0.799

wandb: Waiting for W&B process to finish, PID 28
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_runtime 923.4422557353973
wandb: \_timestamp 1587581965.3127446
wandb: train-loss 0.7630936029581618
wandb: \_step 95
wandb: train-mrr 0.9344640350823147
wandb: train-time-sec 29.70076084136963
wandb: val-time-sec 0.8662071228027344
wandb: val-loss 0.9952550232410431
wandb: val-mrr 0.6612027042933872
wandb: epoch 17
wandb: best_val_mrr_loss 0.995525666645595
wandb: best_val_mrr 0.6623400203159877
wandb: best_epoch 12
wandb: Test-All MRR (bs=1,000) 0.6681220598818597
wandb: FuncNameTest-All MRR (bs=1,000) 0.31059630248533837
wandb: Validation-All MRR (bs=1,000) 0.7985024391440415
wandb: Test-go MRR (bs=1,000) 0.6681220598818597
wandb: FuncNameTest-go MRR (bs=1,000) 0.31059630248533837
wandb: Validation-go MRR (bs=1,000) 0.7985024391440415
wandb: Syncing files in wandb/run-20200422_184412-yb7expi2:
wandb: neuralbow-2020-04-22-18-44-12-graph.pbtxt
wandb: neuralbow-2020-04-22-18-44-12.train_log
wandb: neuralbow-2020-04-22-18-44-12_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced neuralbow-2020-04-22-18-44-12: https://app.wandb.ai/jianguda/CodeSearchNet/runs/yb7expi2
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/yb7expi2
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./neuralbow-2020-04-22-18-44-12_model_best.pkl.gz
2020-04-22 19:03:15.707527: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-22 19:03:20.699078: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 42e4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-22 19:03:20.699123: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-22 19:03:20.969774: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-22 19:03:20.969835: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-22 19:03:20.969854: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-22 19:03:20.969962: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
WARNING:tensorflow:From /home/dev/src/models/model.py:299: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: go
100%|████████████████████████████████████████████████████████████████████████████████████| 726768/726768 [00:06<00:00, 120436.05it/s]726768it [00:12, 58344.78it/s]
Uploading predictions to W&B
NDCG Average: 0.116525132

# CNN

root@jian-csn:/home/dev/src# python train.py --model 1dcnn ../resources/saved_models ../resources/data/go/final/jsonl/train ../resources/data/go/final/jsonl/valid ../resources/data/go/final/jsonl/test
wandb: Started W&B process version 0.8.12 with PID 229
wandb: Wandb version 0.8.32 is available! To upgrade, please run:
wandb: \$ pip install wandb --upgrade
wandb: Local directory: wandb/run-20200422_191418-rkfw7wqq
wandb: Syncing run 1dcnn-2020-04-22-19-14-18: https://app.wandb.ai/jianguda/CodeSearchNet/runs/rkfw7wqq
wandb: Run `wandb off` to turn off syncing.

2020-04-22 19:14:24.424346: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-22 19:14:24.501677: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
ces/data/go/final/jsonl/valid ../resources/data/go/final/jsonl/test
wandb: Started W&B process version 0.8.12 with PID 229
wandb: Wandb version 0.8.32 is available! To upgrade, please run:
wandb: \$ pip install wandb --upgrade
wandb: Local directory: wandb/run-20200422_191418-rkfw7wqq
wandb: Syncing run 1dcnn-2020-04-22-19-14-18: https://app.wandb.ai/jianguda/CodeSearchNet/runs/rkfw7wqq
wandb: Run `wandb off` to turn off syncing.

2020-04-22 19:14:24.424346: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-22 19:14:24.501677: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 42e4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-22 19:14:24.501722: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-22 19:14:24.779041: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-22 19:14:24.779101: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-22 19:14:24.779120: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-22 19:14:24.779242: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Tokenizing and building vocabulary for code snippets and queries. This step may take several hours.
Starting training run 1dcnn-2020-04-22-19-14-18 of model ConvolutionalModel with following hypers:
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': False, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_1dcnn_position_encoding': 'learned', 'code_1dcnn_layer_list': [128, 128, 128], 'code_1dcnn_kernel_width': [16, 16, 16], 'code_1dcnn_add_residual_connections': True, 'code_1dcnn_activation': 'tanh', 'code_1dcnn_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_1dcnn_position_encoding': 'learned', 'query_1dcnn_layer_list': [128, 128, 128], 'query_1dcnn_kernel_width': [16, 16, 16], 'query_1dcnn_add_residual_connections': True, 'query_1dcnn_activation': 'tanh', 'query_1dcnn_pool_mode': 'weighted_mean', 'batch_size': 1000, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.0005, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'softmax', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
Loading training and validation data.
Begin Training.
Training on 317832 go samples.
Validating on 14242 go samples.
==== Epoch 0 ====
Epoch 0 (train) took 205.22s [processed 1544 samples/second]
Training Loss: 5.514914
Epoch 0 (valid) took 2.92s [processed 4796 samples/second]
Validation: Loss: 5.691059 | MRR: 0.076819
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-22-19-14-18_model_best.pkl.gz'.
==== Epoch 1 ====
Epoch 1 (train) took 199.90s [processed 1585 samples/second]
Training Loss: 3.748787
Epoch 1 (valid) took 2.84s [processed 4932 samples/second]
Validation: Loss: 4.446725 | MRR: 0.238668
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-22-19-14-18_model_best.pkl.gz'.
==== Epoch 2 ====
Epoch 2 (train) took 200.45s [processed 1581 samples/second]
Training Loss: 2.488946
Epoch 2 (valid) took 2.84s [processed 4923 samples/second]
Validation: Loss: 3.167598 | MRR: 0.451361
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-22-19-14-18_model_best.pkl.gz'.
==== Epoch 3 ====
Epoch 3 (train) took 200.97s [processed 1577 samples/second]
Training Loss: 1.666411
Epoch 3 (valid) took 2.86s [processed 4900 samples/second]
Validation: Loss: 2.545028 | MRR: 0.556795
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-22-19-14-18_model_best.pkl.gz'.
==== Epoch 4 ====
Epoch 4 (train) took 200.97s [processed 1577 samples/second]
Training Loss: 1.218475
Epoch 4 (valid) took 2.84s [processed 4928 samples/second]
Validation: Loss: 2.200507 | MRR: 0.614624
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-22-19-14-18_model_best.pkl.gz'.
==== Epoch 5 ====
Epoch 5 (train) took 200.88s [processed 1578 samples/second]
Training Loss: 0.961683
Epoch 5 (valid) took 2.87s [processed 4881 samples/second]
Validation: Loss: 2.024156 | MRR: 0.641297
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-22-19-14-18_model_best.pkl.gz'.
==== Epoch 6 ====
Epoch 6 (train) took 200.99s [processed 1577 samples/second]
Training Loss: 0.811831
Epoch 6 (valid) took 2.82s [processed 4959 samples/second]
Validation: Loss: 1.921403 | MRR: 0.660145
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-22-19-14-18_model_best.pkl.gz'.
==== Epoch 7 ====
Epoch 7 (train) took 200.87s [processed 1578 samples/second]
Training Loss: 0.707519
Epoch 7 (valid) took 2.86s [processed 4892 samples/second]
Validation: Loss: 1.861567 | MRR: 0.668707
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-22-19-14-18_model_best.pkl.gz'.
==== Epoch 8 ====
Epoch 8 (train) took 201.06s [processed 1576 samples/second]
Training Loss: 0.629128
Epoch 8 (valid) took 2.85s [processed 4916 samples/second]
Validation: Loss: 1.887404 | MRR: 0.672085
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-22-19-14-18_model_best.pkl.gz'.
==== Epoch 9 ====
Epoch 9 (train) took 201.03s [processed 1576 samples/second]
Training Loss: 0.568240
Epoch 9 (valid) took 2.83s [processed 4943 samples/second]
Validation: Loss: 1.878109 | MRR: 0.678510
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-22-19-14-18_model_best.pkl.gz'.
==== Epoch 10 ====
Epoch 10 (train) took 200.77s [processed 1578 samples/second]
Training Loss: 0.521045
Epoch 10 (valid) took 2.86s [processed 4894 samples/second]
Validation: Loss: 1.835592 | MRR: 0.688835
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-22-19-14-18_model_best.pkl.gz'.
==== Epoch 11 ====
Epoch 11 (train) took 200.72s [processed 1579 samples/second]
Training Loss: 0.484236
Epoch 11 (valid) took 2.85s [processed 4914 samples/second]
Validation: Loss: 1.866064 | MRR: 0.689957
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-22-19-14-18_model_best.pkl.gz'.
==== Epoch 12 ====
Epoch 12 (train) took 200.84s [processed 1578 samples/second]
Training Loss: 0.450887
Epoch 12 (valid) took 2.85s [processed 4904 samples/second]
Validation: Loss: 1.850771 | MRR: 0.694719
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-22-19-14-18_model_best.pkl.gz'.
==== Epoch 13 ====
Epoch 13 (train) took 200.91s [processed 1577 samples/second]
Training Loss: 0.426565
Epoch 13 (valid) took 2.87s [processed 4881 samples/second]
Validation: Loss: 1.819177 | MRR: 0.695148
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-22-19-14-18_model_best.pkl.gz'.
==== Epoch 14 ====
Epoch 14 (train) took 200.54s [processed 1580 samples/second]
Training Loss: 0.404458
Epoch 14 (valid) took 2.84s [processed 4926 samples/second]
Validation: Loss: 1.842125 | MRR: 0.694759
==== Epoch 15 ====
Epoch 15 (train) took 200.76s [processed 1578 samples/second]
Training Loss: 0.383426
Epoch 15 (valid) took 2.84s [processed 4921 samples/second]
Validation: Loss: 1.884267 | MRR: 0.692859
==== Epoch 16 ====
Epoch 16 (train) took 200.78s [processed 1578 samples/second]
Training Loss: 0.369524
Epoch 16 (valid) took 2.84s [processed 4934 samples/second]
Validation: Loss: 1.850165 | MRR: 0.696485
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-22-19-14-18_model_best.pkl.gz'.
==== Epoch 17 ====
Epoch 17 (train) took 200.86s [processed 1578 samples/second]
Training Loss: 0.355476
Epoch 17 (valid) took 2.83s [processed 4954 samples/second]
Validation: Loss: 1.890783 | MRR: 0.701780
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-22-19-14-18_model_best.pkl.gz'.
==== Epoch 18 ====
Epoch 18 (train) took 200.56s [processed 1580 samples/second]
Training Loss: 0.341922
Epoch 18 (valid) took 2.84s [processed 4933 samples/second]
Validation: Loss: 1.876547 | MRR: 0.700267
==== Epoch 19 ====
Epoch 19 (train) took 200.84s [processed 1578 samples/second]
Training Loss: 0.326680
Epoch 19 (valid) took 2.83s [processed 4951 samples/second]
Validation: Loss: 1.896224 | MRR: 0.700388
==== Epoch 20 ====
Epoch 20 (train) took 200.90s [processed 1577 samples/second]
Training Loss: 0.323352
Epoch 20 (valid) took 2.82s [processed 4963 samples/second]
Validation: Loss: 1.895066 | MRR: 0.701590
==== Epoch 21 ====
Epoch 21 (train) took 200.74s [processed 1579 samples/second]
Training Loss: 0.314301
Epoch 21 (valid) took 2.84s [processed 4922 samples/second]
Validation: Loss: 1.875647 | MRR: 0.702948
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-22-19-14-18_model_best.pkl.gz'.
==== Epoch 22 ====
Epoch 22 (train) took 200.56s [processed 1580 samples/second]
Training Loss: 0.303321
Epoch 22 (valid) took 2.86s [processed 4903 samples/second]
Validation: Loss: 1.903562 | MRR: 0.703065
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-22-19-14-18_model_best.pkl.gz'.
==== Epoch 23 ====
Epoch 23 (train) took 200.45s [processed 1581 samples/second]
Training Loss: 0.295601
Epoch 23 (valid) took 2.85s [processed 4915 samples/second]
Validation: Loss: 1.882415 | MRR: 0.704768
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-22-19-14-18_model_best.pkl.gz'.
==== Epoch 24 ====
Epoch 24 (train) took 200.51s [processed 1581 samples/second]
Training Loss: 0.286774
Epoch 24 (valid) took 2.82s [processed 4968 samples/second]
Validation: Loss: 1.879795 | MRR: 0.703278
==== Epoch 25 ====
Epoch 25 (train) took 200.39s [processed 1581 samples/second]
Training Loss: 0.281423
Epoch 25 (valid) took 2.83s [processed 4947 samples/second]
Validation: Loss: 1.909791 | MRR: 0.704187
==== Epoch 26 ====
Epoch 26 (train) took 200.67s [processed 1579 samples/second]
Training Loss: 0.275777
Epoch 26 (valid) took 2.86s [processed 4893 samples/second]
Validation: Loss: 1.881592 | MRR: 0.705325
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-22-19-14-18_model_best.pkl.gz'.
==== Epoch 27 ====
Epoch 27 (train) took 200.47s [processed 1581 samples/second]
Training Loss: 0.268867
Epoch 27 (valid) took 2.82s [processed 4958 samples/second]
Validation: Loss: 1.885901 | MRR: 0.709219
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-22-19-14-18_model_best.pkl.gz'.
==== Epoch 28 ====
Epoch 28 (train) took 200.71s [processed 1579 samples/second]
Training Loss: 0.265645
Epoch 28 (valid) took 2.87s [processed 4879 samples/second]
Validation: Loss: 1.893752 | MRR: 0.712528
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-22-19-14-18_model_best.pkl.gz'.
==== Epoch 29 ====
Epoch 29 (train) took 200.54s [processed 1580 samples/second]
Training Loss: 0.260628
Epoch 29 (valid) took 2.84s [processed 4928 samples/second]
Validation: Loss: 1.885954 | MRR: 0.709807
==== Epoch 30 ====
Epoch 30 (train) took 200.45s [processed 1581 samples/second]
Training Loss: 0.255794
Epoch 30 (valid) took 2.84s [processed 4926 samples/second]
Validation: Loss: 1.892028 | MRR: 0.708642
==== Epoch 31 ====
Epoch 31 (train) took 200.53s [processed 1580 samples/second]
Training Loss: 0.251220
Epoch 31 (valid) took 2.83s [processed 4955 samples/second]
Validation: Loss: 1.890955 | MRR: 0.712925
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-22-19-14-18_model_best.pkl.gz'.
==== Epoch 32 ====
Epoch 32 (train) took 200.62s [processed 1580 samples/second]
Training Loss: 0.248323
Epoch 32 (valid) took 2.86s [processed 4894 samples/second]
Validation: Loss: 1.870969 | MRR: 0.710726
==== Epoch 33 ====
Epoch 33 (train) took 200.61s [processed 1580 samples/second]
Training Loss: 0.244119
Epoch 33 (valid) took 2.88s [processed 4853 samples/second]
Validation: Loss: 1.898376 | MRR: 0.713527
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-22-19-14-18_model_best.pkl.gz'.
==== Epoch 34 ====
Epoch 34 (train) took 200.42s [processed 1581 samples/second]
Training Loss: 0.240188
Epoch 34 (valid) took 2.84s [processed 4934 samples/second]
Validation: Loss: 1.845867 | MRR: 0.711332
==== Epoch 35 ====
Epoch 35 (train) took 200.41s [processed 1581 samples/second]
Training Loss: 0.238381
Epoch 35 (valid) took 2.82s [processed 4965 samples/second]
Validation: Loss: 1.872724 | MRR: 0.714896
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-22-19-14-18_model_best.pkl.gz'.
==== Epoch 36 ====
Epoch 36 (train) took 200.52s [processed 1580 samples/second]
Training Loss: 0.235774
Epoch 36 (valid) took 2.85s [processed 4914 samples/second]
Validation: Loss: 1.892457 | MRR: 0.716111
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-22-19-14-18_model_best.pkl.gz'.
==== Epoch 37 ====
Epoch 37 (train) took 200.34s [processed 1582 samples/second]
Training Loss: 0.232595
Epoch 37 (valid) took 2.83s [processed 4948 samples/second]
Validation: Loss: 1.882250 | MRR: 0.714554
==== Epoch 38 ====
Epoch 38 (train) took 200.50s [processed 1581 samples/second]
Training Loss: 0.229333
Epoch 38 (valid) took 2.86s [processed 4894 samples/second]
Validation: Loss: 1.924967 | MRR: 0.715474
==== Epoch 39 ====
Epoch 39 (train) took 200.29s [processed 1582 samples/second]
Training Loss: 0.227970
Epoch 39 (valid) took 2.84s [processed 4937 samples/second]
Validation: Loss: 1.910120 | MRR: 0.714050
==== Epoch 40 ====
Epoch 40 (train) took 200.66s [processed 1579 samples/second]
Training Loss: 0.225386
Epoch 40 (valid) took 2.84s [processed 4932 samples/second]
Validation: Loss: 1.900043 | MRR: 0.715630
==== Epoch 41 ====
Epoch 41 (train) took 200.66s [processed 1579 samples/second]
Training Loss: 0.221824
Epoch 41 (valid) took 2.86s [processed 4890 samples/second]
Validation: Loss: 1.967844 | MRR: 0.712879
2020-04-22 21:42:27.126455: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-22 21:42:27.126512: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-22 21:42:27.126528: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-22 21:42:27.126540: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-22 21:42:27.126639: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.704
FuncNameTest-All MRR (bs=1,000): 0.100
Validation-All MRR (bs=1,000): 0.802
Test-go MRR (bs=1,000): 0.704
FuncNameTest-go MRR (bs=1,000): 0.100
Validation-go MRR (bs=1,000): 0.802

wandb: Waiting for W&B process to finish, PID 229
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_step 215
wandb: train-mrr 0.9518608384959705
wandb: train-loss 0.2218240488684892
wandb: \_timestamp 1587591811.6764138
wandb: \_runtime 8954.184934854507
wandb: val-mrr 0.7128791809082031
wandb: epoch 41
wandb: train-time-sec 200.65749263763428
wandb: val-time-sec 2.862886428833008
wandb: val-loss 1.9678442648478918
wandb: best_val_mrr_loss 1.8924570424216134
wandb: best_val_mrr 0.7161111624581473
wandb: best_epoch 36
wandb: Test-All MRR (bs=1,000) 0.7043914369465366
wandb: FuncNameTest-All MRR (bs=1,000) 0.09971611109999921
wandb: Validation-All MRR (bs=1,000) 0.8021590731734296
wandb: Test-go MRR (bs=1,000) 0.7043914369465366
wandb: Run summary:
wandb: \_step 215
wandb: train-mrr 0.9518608384959705
wandb: train-loss 0.2218240488684892
wandb: \_timestamp 1587591811.6764138
wandb: \_runtime 8954.184934854507
wandb: val-mrr 0.7128791809082031
wandb: epoch 41
wandb: train-time-sec 200.65749263763428
wandb: val-time-sec 2.862886428833008
wandb: val-loss 1.9678442648478918
wandb: best_val_mrr_loss 1.8924570424216134
wandb: best_val_mrr 0.7161111624581473
wandb: best_epoch 36
wandb: Test-All MRR (bs=1,000) 0.7043914369465366
wandb: FuncNameTest-All MRR (bs=1,000) 0.09971611109999921
wandb: Validation-All MRR (bs=1,000) 0.8021590731734296
wandb: Test-go MRR (bs=1,000) 0.7043914369465366
wandb: FuncNameTest-go MRR (bs=1,000) 0.09971611109999921
wandb: Validation-go MRR (bs=1,000) 0.8021590731734296
wandb: Syncing files in wandb/run-20200422_191418-rkfw7wqq:
wandb: 1dcnn-2020-04-22-19-14-18-graph.pbtxt
wandb: 1dcnn-2020-04-22-19-14-18.train_log
wandb: 1dcnn-2020-04-22-19-14-18_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced 1dcnn-2020-04-22-19-14-18: https://app.wandb.ai/jianguda/CodeSearchNet/runs/rkfw7wqq
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/rkfw7wqq  
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./1dcnn-2020-04-22-19-14-18_model_best.pkl.gz
2020-04-22 21:53:01.032846: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-22 21:53:05.924561: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 42e4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-22 21:53:05.924609: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-22 21:53:06.200075: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-22 21:53:06.200132: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-22 21:53:06.200150: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-22 21:53:06.200261: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Evaluating language: go
100%|████████████████████████████████████████████████████████████████████████████████████| 726768/726768 [00:06<00:00, 119507.91it/s]726768it [00:12, 58538.39it/s]
Uploading predictions to W&B
NDCG Average: 0.013920585

# RNN

root@jian-csn:/home/dev/src# python train.py --model rnn ../resources/saved_models ../resources/data/go/final/jsonl/train ../resources/data/go/final/jsonl/valid ../resources/data/go/final/jsonl/test
wandb: Started W&B process version 0.8.12 with PID 641
wandb: Wandb version 0.8.32 is available! To upgrade, please run:
wandb: \$ pip install wandb --upgrade
wandb: Local directory: wandb/run-20200423_082830-cxcb829t
wandb: Syncing run rnn-2020-04-23-08-28-30: https://app.wandb.ai/jianguda/CodeSearchNet/runs/cxcb829t
wandb: Run `wandb off` to turn off syncing.

2020-04-23 08:28:36.147449: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-23 08:28:36.224743: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 42e4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-23 08:28:36.224789: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-23 08:28:36.501810: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-23 08:28:36.501873: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-23 08:28:36.501889: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-23 08:28:36.501997: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Tokenizing and building vocabulary for code snippets and queries. This step may take several hours.
Starting training run rnn-2020-04-23-08-28-30 of model RNNModel with following hypers:
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': True, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_rnn_num_layers': 2, 'code_rnn_hidden_dim': 64, 'code_rnn_cell_type': 'LSTM', 'code_rnn_is_bidirectional': True, 'code_rnn_dropout_keep_rate': 0.8, 'code_rnn_recurrent_dropout_keep_rate': 1.0, 'code_rnn_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_rnn_num_layers': 2, 'query_rnn_hidden_dim': 64, 'query_rnn_cell_type': 'LSTM', 'query_rnn_is_bidirectional': True, 'query_rnn_dropout_keep_rate': 0.8, 'query_rnn_recurrent_dropout_keep_rate': 1.0, 'query_rnn_pool_mode': 'weighted_mean', 'batch_size': 1000, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.01, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'softmax', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
Loading training and validation data.
Begin Training.
Training on 317832 go samples.
Validating on 14242 go samples.
==== Epoch 0 ====
Epoch 0 (train) took 341.25s [processed 928 samples/second]
Training Loss: 3.830734
Epoch 0 (valid) took 7.24s [processed 1933 samples/second]
Validation: Loss: 2.646316 | MRR: 0.517714
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-23-08-28-30_model_best.pkl.gz'.
==== Epoch 1 ====
Epoch 1 (train) took 339.41s [processed 933 samples/second]
Training Loss: 1.194562
Epoch 1 (valid) took 6.95s [processed 2015 samples/second]
Validation: Loss: 1.906590 | MRR: 0.651152
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-23-08-28-30_model_best.pkl.gz'.
==== Epoch 2 ====
Epoch 2 (train) took 339.47s [processed 933 samples/second]
Training Loss: 0.810536
Epoch 2 (valid) took 6.96s [processed 2010 samples/second]
Validation: Loss: 1.801000 | MRR: 0.673394
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-23-08-28-30_model_best.pkl.gz'.
==== Epoch 3 ====
Epoch 3 (train) took 339.73s [processed 933 samples/second]
Training Loss: 0.661934
Epoch 3 (valid) took 6.97s [processed 2008 samples/second]
Validation: Loss: 1.700150 | MRR: 0.690456
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-23-08-28-30_model_best.pkl.gz'.
==== Epoch 4 ====
Epoch 4 (train) took 339.61s [processed 933 samples/second]
Training Loss: 0.580542
Epoch 4 (valid) took 6.96s [processed 2011 samples/second]
Validation: Loss: 1.693464 | MRR: 0.692342
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-23-08-28-30_model_best.pkl.gz'.
==== Epoch 5 ====
Epoch 5 (train) took 339.62s [processed 933 samples/second]
Training Loss: 0.530193
Epoch 5 (valid) took 6.94s [processed 2017 samples/second]
Validation: Loss: 1.650031 | MRR: 0.703549
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-23-08-28-30_model_best.pkl.gz'.
==== Epoch 6 ====
Epoch 6 (train) took 339.48s [processed 933 samples/second]
Training Loss: 0.497240
Epoch 6 (valid) took 6.98s [processed 2004 samples/second]
Validation: Loss: 1.663836 | MRR: 0.698348
==== Epoch 7 ====
Epoch 7 (train) took 339.25s [processed 934 samples/second]
Training Loss: 0.473460
Epoch 7 (valid) took 6.95s [processed 2013 samples/second]
Validation: Loss: 1.654139 | MRR: 0.703792
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-23-08-28-30_model_best.pkl.gz'.
==== Epoch 8 ====
Epoch 8 (train) took 339.29s [processed 934 samples/second]
Training Loss: 0.451851
Epoch 8 (valid) took 6.94s [processed 2016 samples/second]
Validation: Loss: 1.684258 | MRR: 0.700488
==== Epoch 9 ====
Epoch 9 (train) took 339.14s [processed 934 samples/second]
Training Loss: 0.439435
Epoch 9 (valid) took 6.94s [processed 2017 samples/second]
Validation: Loss: 1.683779 | MRR: 0.701209
==== Epoch 10 ====
Epoch 10 (train) took 339.12s [processed 934 samples/second]
Training Loss: 0.433020
Epoch 10 (valid) took 6.95s [processed 2013 samples/second]
Validation: Loss: 1.667694 | MRR: 0.700779
==== Epoch 11 ====
Epoch 11 (train) took 339.37s [processed 934 samples/second]
Training Loss: 0.429866
Epoch 11 (valid) took 6.92s [processed 2021 samples/second]
Validation: Loss: 1.655147 | MRR: 0.701751
==== Epoch 12 ====
Epoch 12 (train) took 339.41s [processed 933 samples/second]
Training Loss: 0.425577
Epoch 12 (valid) took 6.95s [processed 2014 samples/second]
Validation: Loss: 1.661920 | MRR: 0.703699
2020-04-23 09:48:22.604830: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-23 09:48:22.604894: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-23 09:48:22.604909: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-23 09:48:22.604921: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-23 09:48:22.605085: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.708
FuncNameTest-All MRR (bs=1,000): 0.139
Validation-All MRR (bs=1,000): 0.792
Test-go MRR (bs=1,000): 0.708
FuncNameTest-go MRR (bs=1,000): 0.139
Validation-go MRR (bs=1,000): 0.792

wandb: Waiting for W&B process to finish, PID 641
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-loss 0.42557731035756013
wandb: \_timestamp 1587635392.676421
wandb: train-mrr 0.9170707147159035
wandb: \_step 70
wandb: \_runtime 4883.371154546738
wandb: val-time-sec 6.950692653656006
wandb: val-mrr 0.7036994149344308
wandb: train-time-sec 339.41486716270447
wandb: epoch 12
wandb: val-loss 1.6619200663907188
wandb: best_val_mrr_loss 1.6541389184338706
wandb: best_val_mrr 0.7037921447753906
wandb: best_epoch 7
wandb: Test-All MRR (bs=1,000) 0.7082008740117465
wandb: FuncNameTest-All MRR (bs=1,000) 0.13937100834525776
wandb: Validation-All MRR (bs=1,000) 0.7917052388675451
wandb: Test-go MRR (bs=1,000) 0.7082008740117465
wandb: FuncNameTest-go MRR (bs=1,000) 0.13937100834525776
wandb: Validation-go MRR (bs=1,000) 0.7917052388675451
wandb: Syncing files in wandb/run-20200423_082830-cxcb829t:
wandb: rnn-2020-04-23-08-28-30-graph.pbtxt
wandb: rnn-2020-04-23-08-28-30.train_log
wandb: rnn-2020-04-23-08-28-30_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced rnn-2020-04-23-08-28-30: https://app.wandb.ai/jianguda/CodeSearchNet/runs/cxcb829t
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/cxcb829t
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./rnn-2020-04-23-08-28-30_model_best.pkl.gz
2020-04-23 09:50:54.369524: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-23 09:50:59.215748: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 42e4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-23 09:50:59.215800: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-23 09:50:59.490126: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-23 09:50:59.490189: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-23 09:50:59.490206: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-23 09:50:59.490316: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Evaluating language: go
100%|████████████████████████████████████████████████████████████████████████████████████| 726768/726768 [00:03<00:00, 214015.59it/s]726768it [00:12, 58171.77it/s]
Uploading predictions to W&B
NDCG Average: 0.031064525

# BERT

root@jian-csn:/home/dev/src# python train.py --model selfatt ../resources/saved_models ../resources/data/go/final/jsonl/train ../resources/data/go/final/jsonl/valid ../resources/data/go/final/jsonl/test
wandb: Started W&B process version 0.8.12 with PID 435
wandb: Wandb version 0.8.32 is available! To upgrade, please run:
wandb: \$ pip install wandb --upgrade
wandb: Local directory: wandb/run-20200422_220903-1eer61o9
wandb: Syncing run selfatt-2020-04-22-22-09-03: https://app.wandb.ai/jianguda/CodeSearchNet/runs/1eer61o9
wandb: Run `wandb off` to turn off syncing.

2020-04-22 22:09:08.816791: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-22 22:09:08.896046: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 42e4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-22 22:09:08.896091: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-22 22:09:09.169175: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-22 22:09:09.169238: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-22 22:09:09.169254: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-22 22:09:09.169361: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Tokenizing and building vocabulary for code snippets and queries. This step may take several hours.
Starting training run selfatt-2020-04-22-22-09-03 of model SelfAttentionModel with following hypers:
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': False, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_self_attention_activation': 'gelu', 'code_self_attention_hidden_size': 128, 'code_self_attention_intermediate_size': 512, 'code_self_attention_num_layers': 3, 'code_self_attention_num_heads': 8, 'code_self_attention_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_self_attention_activation': 'gelu', 'query_self_attention_hidden_size': 128, 'query_self_attention_intermediate_size': 512, 'query_self_attention_num_layers': 3, 'query_self_attention_num_heads': 8, 'query_self_attention_pool_mode': 'weighted_mean', 'batch_size': 450, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.0005, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'softmax', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': False, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_self_attention_activation': 'gelu', 'code_self_attention_hidden_size': 128, 'code_self_attention_intermediate_size': 512, 'code_self_attention_num_layers': 3, 'code_self_attention_num_heads': 8, 'code_self_attention_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_self_attention_activation': 'gelu', 'query_self_attention_hidden_size': 128, 'query_self_attention_intermediate_size': 512, 'query_self_attention_num_layers': 3, 'query_self_attention_num_heads': 8, 'query_self_attention_pool_mode': 'weighted_mean', 'batch_size': 450, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.0005, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'softmax', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
Starting training run selfatt-2020-04-22-22-09-03 of model SelfAttentionModel with following hypers:
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': False, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_self_attention_activation': 'gelu', 'code_self_attention_hidden_size': 128, 'code_self_attention_intermediate_size': 512, 'code_self_attention_num_layers': 3, 'code_self_attention_num_heads': 8, 'code_self_attention_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_self_attention_activation': 'gelu', 'query_self_attention_hidden_size': 128, 'query_self_attention_intermediate_size': 512, 'query_self_attention_num_layers': 3, 'query_self_attention_num_heads': 8, 'query_self_attention_pool_mode': 'weighted_mean', 'batch_size': 450, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.0005, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'softmax', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
Loading training and validation data.
Begin Training.
Training on 317832 go samples.
Validating on 14242 go samples.
==== Epoch 0 ====
Epoch 0 (train) took 982.06s [processed 323 samples/second]
Training Loss: 1.496670
Epoch 0 (valid) took 18.75s [processed 743 samples/second]
Validation: Loss: 1.704320 | MRR: 0.685982
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-22-22-09-03_model_best.pkl.gz'.
==== Epoch 1 ====
Epoch 1 (train) took 978.97s [processed 324 samples/second]
Training Loss: 0.342322
Epoch 1 (valid) took 18.43s [processed 757 samples/second]
Validation: Loss: 1.513606 | MRR: 0.720284
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-22-22-09-03_model_best.pkl.gz'.
==== Epoch 2 ====
Epoch 2 (train) took 979.01s [processed 324 samples/second]
Training Loss: 0.222831
Epoch 2 (valid) took 18.25s [processed 764 samples/second]
Validation: Loss: 1.508091 | MRR: 0.724418
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-22-22-09-03_model_best.pkl.gz'.
==== Epoch 3 ====
Epoch 3 (train) took 977.73s [processed 324 samples/second]
Training Loss: 0.173991
Epoch 3 (valid) took 18.27s [processed 763 samples/second]
Validation: Loss: 1.467845 | MRR: 0.729135
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-22-22-09-03_model_best.pkl.gz'.
==== Epoch 4 ====
Epoch 4 (train) took 980.19s [processed 324 samples/second]
Training Loss: 0.149989
Epoch 4 (valid) took 18.47s [processed 755 samples/second]
Validation: Loss: 1.507893 | MRR: 0.734176
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-22-22-09-03_model_best.pkl.gz'.
==== Epoch 5 ====
Epoch 5 (train) took 979.13s [processed 324 samples/second]
Training Loss: 0.132590
Epoch 5 (valid) took 18.39s [processed 758 samples/second]
Validation: Loss: 1.472356 | MRR: 0.741572
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-22-22-09-03_model_best.pkl.gz'.
==== Epoch 6 ====
Epoch 6 (train) took 979.57s [processed 324 samples/second]
Training Loss: 0.121505
Epoch 6 (valid) took 18.36s [processed 759 samples/second]
Validation: Loss: 1.479756 | MRR: 0.738594
==== Epoch 7 ====
Epoch 7 (train) took 979.23s [processed 324 samples/second]
Training Loss: 0.114204
Epoch 7 (valid) took 18.31s [processed 761 samples/second]
Validation: Loss: 1.435729 | MRR: 0.748752
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-22-22-09-03_model_best.pkl.gz'.
==== Epoch 8 ====
Epoch 8 (train) took 980.29s [processed 324 samples/second]
Training Loss: 0.106238
Epoch 8 (valid) took 18.49s [processed 754 samples/second]
Validation: Loss: 1.487149 | MRR: 0.748585
==== Epoch 9 ====
Epoch 9 (train) took 978.04s [processed 324 samples/second]
Training Loss: 0.100526
Epoch 9 (valid) took 18.40s [processed 758 samples/second]
Validation: Loss: 1.450532 | MRR: 0.754171
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-22-22-09-03_model_best.pkl.gz'.
==== Epoch 10 ====
Epoch 10 (train) took 979.68s [processed 324 samples/second]
Training Loss: 0.099382
Epoch 10 (valid) took 18.09s [processed 771 samples/second]
Validation: Loss: 1.410719 | MRR: 0.755293
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-22-22-09-03_model_best.pkl.gz'.
==== Epoch 11 ====
Epoch 11 (train) took 977.56s [processed 324 samples/second]
Training Loss: 0.092718
Epoch 11 (valid) took 18.23s [processed 765 samples/second]
Validation: Loss: 1.443578 | MRR: 0.761298
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-22-22-09-03_model_best.pkl.gz'.
==== Epoch 12 ====
Epoch 12 (train) took 978.49s [processed 324 samples/second]
Training Loss: 0.090337
Epoch 12 (valid) took 18.27s [processed 763 samples/second]
Validation: Loss: 1.448263 | MRR: 0.758336
==== Epoch 13 ====
Epoch 13 (train) took 976.88s [processed 325 samples/second]
Training Loss: 0.088651
Epoch 13 (valid) took 18.45s [processed 756 samples/second]
Validation: Loss: 1.423731 | MRR: 0.760683
==== Epoch 14 ====
Epoch 14 (train) took 979.86s [processed 324 samples/second]
Training Loss: 0.085491
Epoch 14 (valid) took 18.54s [processed 752 samples/second]
Validation: Loss: 1.461432 | MRR: 0.754603
==== Epoch 15 ====
Epoch 15 (train) took 977.76s [processed 324 samples/second]
Training Loss: 0.082098
Epoch 15 (valid) took 18.11s [processed 770 samples/second]
Validation: Loss: 1.398380 | MRR: 0.765131
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-22-22-09-03_model_best.pkl.gz'.
==== Epoch 16 ====
Epoch 16 (train) took 978.25s [processed 324 samples/second]
Training Loss: 0.080387
Epoch 16 (valid) took 18.40s [processed 758 samples/second]
Validation: Loss: 1.466565 | MRR: 0.753708
==== Epoch 17 ====
Epoch 17 (train) took 978.36s [processed 324 samples/second]
Training Loss: 0.079293
Epoch 17 (valid) took 18.22s [processed 765 samples/second]
Validation: Loss: 1.442179 | MRR: 0.762669
==== Epoch 18 ====
Epoch 18 (train) took 980.19s [processed 324 samples/second]
Training Loss: 0.077282
Epoch 18 (valid) took 18.18s [processed 767 samples/second]
Validation: Loss: 1.430964 | MRR: 0.767753
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-22-22-09-03_model_best.pkl.gz'.
==== Epoch 19 ====
Epoch 19 (train) took 978.65s [processed 324 samples/second]
Training Loss: 0.076456
Epoch 19 (valid) took 18.44s [processed 756 samples/second]
Validation: Loss: 1.439917 | MRR: 0.763548
==== Epoch 20 ====
Epoch 20 (train) took 980.12s [processed 324 samples/second]
Training Loss: 0.074436
Epoch 20 (valid) took 18.23s [processed 765 samples/second]
Validation: Loss: 1.402227 | MRR: 0.767207
==== Epoch 21 ====
Epoch 21 (train) took 979.39s [processed 324 samples/second]
Training Loss: 0.072989
Epoch 21 (valid) took 18.29s [processed 762 samples/second]
Validation: Loss: 1.371054 | MRR: 0.770736
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-22-22-09-03_model_best.pkl.gz'.
==== Epoch 22 ====
Epoch 22 (train) took 979.46s [processed 324 samples/second]
Training Loss: 0.072036
Epoch 22 (valid) took 18.48s [processed 754 samples/second]
Validation: Loss: 1.422109 | MRR: 0.768679
==== Epoch 23 ====
Epoch 23 (train) took 978.56s [processed 324 samples/second]
Training Loss: 0.070368
Epoch 23 (valid) took 18.31s [processed 762 samples/second]
Validation: Loss: 1.406291 | MRR: 0.769711
==== Epoch 24 ====
Epoch 24 (train) took 980.28s [processed 324 samples/second]
Training Loss: 0.069606
Epoch 24 (valid) took 18.37s [processed 759 samples/second]
Validation: Loss: 1.381876 | MRR: 0.770518
==== Epoch 25 ====
Epoch 25 (train) took 979.28s [processed 324 samples/second]
Training Loss: 0.069582
Epoch 25 (valid) took 18.33s [processed 761 samples/second]
Validation: Loss: 1.417345 | MRR: 0.770681
==== Epoch 26 ====
Epoch 26 (train) took 979.63s [processed 324 samples/second]
Training Loss: 0.067652
Epoch 26 (valid) took 18.22s [processed 765 samples/second]
Validation: Loss: 1.376251 | MRR: 0.772837
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-22-22-09-03_model_best.pkl.gz'.
==== Epoch 27 ====
Epoch 27 (train) took 978.55s [processed 324 samples/second]
Training Loss: 0.067534
Epoch 27 (valid) took 18.33s [processed 761 samples/second]
Validation: Loss: 1.437043 | MRR: 0.770955
==== Epoch 28 ====
Epoch 28 (train) took 978.78s [processed 324 samples/second]
Training Loss: 0.067088
Epoch 28 (valid) took 18.23s [processed 765 samples/second]
Validation: Loss: 1.342551 | MRR: 0.776474
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-22-22-09-03_model_best.pkl.gz'.
==== Epoch 29 ====
Epoch 29 (train) took 978.23s [processed 324 samples/second]
Training Loss: 0.067645
Epoch 29 (valid) took 18.28s [processed 763 samples/second]
Validation: Loss: 1.392350 | MRR: 0.773276
==== Epoch 30 ====
Epoch 30 (train) took 980.65s [processed 323 samples/second]
Training Loss: 0.064681
Epoch 30 (valid) took 18.38s [processed 758 samples/second]
Validation: Loss: 1.405393 | MRR: 0.767468
==== Epoch 31 ====
Epoch 31 (train) took 979.79s [processed 324 samples/second]
Training Loss: 0.064309
Epoch 31 (valid) took 18.28s [processed 763 samples/second]
Validation: Loss: 1.434681 | MRR: 0.768110
==== Epoch 32 ====
wandb: Network error resolved after 0:00:17.911427, resuming normal operation.ar: 0.0667. MRR so far: 0.9832
Epoch 32 (train) took 979.80s [processed 324 samples/second]
Training Loss: 0.065777
Epoch 32 (valid) took 18.21s [processed 766 samples/second]
Validation: Loss: 1.443195 | MRR: 0.761259
==== Epoch 33 ====
Epoch 33 (train) took 977.27s [processed 325 samples/second]
Training Loss: 0.064123
Epoch 33 (valid) took 18.75s [processed 744 samples/second]
Validation: Loss: 1.359924 | MRR: 0.774776
2020-04-23 07:39:32.380389: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-23 07:39:32.380451: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-23 07:39:32.380467: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-23 07:39:32.380479: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-23 07:39:32.380569: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.726
FuncNameTest-All MRR (bs=1,000): 0.159
Validation-All MRR (bs=1,000): 0.800
Test-go MRR (bs=1,000): 0.726
FuncNameTest-go MRR (bs=1,000): 0.159
Validation-go MRR (bs=1,000): 0.800

wandb: Waiting for W&B process to finish, PID 435
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_timestamp 1587627701.4675248
wandb: \_step 311
wandb: train-mrr 0.983661585571558
wandb: \_runtime 34359.38560938835
wandb: train-loss 0.06412325226424774
wandb: epoch 33
wandb: train-time-sec 977.2724525928497
wandb: val-time-sec 18.749594688415527
wandb: val-loss 1.35992423565157
wandb: val-mrr 0.7747763772506441
wandb: best_val_mrr_loss 1.3425512333070078
wandb: best_val_mrr 0.7764740712155578
wandb: best_epoch 28
wandb: Test-All MRR (bs=1,000) 0.7256812492941721
wandb: FuncNameTest-All MRR (bs=1,000) 0.15864312289572563
wandb: Validation-All MRR (bs=1,000) 0.799873386290268
wandb: Test-go MRR (bs=1,000) 0.7256812492941721
wandb: FuncNameTest-go MRR (bs=1,000) 0.15864312289572563
wandb: val-mrr 0.7747763772506441
wandb: best_val_mrr_loss 1.3425512333070078
wandb: best_val_mrr 0.7764740712155578
wandb: best_epoch 28
wandb: Test-All MRR (bs=1,000) 0.7256812492941721
wandb: FuncNameTest-All MRR (bs=1,000) 0.15864312289572563
wandb: Validation-All MRR (bs=1,000) 0.799873386290268
wandb: Test-go MRR (bs=1,000) 0.7256812492941721
wandb: FuncNameTest-go MRR (bs=1,000) 0.15864312289572563
wandb: Validation-go MRR (bs=1,000) 0.799873386290268
wandb: Syncing files in wandb/run-20200422_220903-1eer61o9:
wandb: selfatt-2020-04-22-22-09-03-graph.pbtxt
wandb: Validation-go MRR (bs=1,000) 0.799873386290268
wandb: Syncing files in wandb/run-20200422_220903-1eer61o9:
wandb: selfatt-2020-04-22-22-09-03-graph.pbtxt
wandb: train-time-sec 977.2724525928497
wandb: val-time-sec 18.749594688415527
wandb: val-loss 1.35992423565157
wandb: val-mrr 0.7747763772506441
wandb: best_val_mrr_loss 1.3425512333070078
wandb: best_val_mrr 0.7764740712155578
wandb: best_epoch 28
wandb: Test-All MRR (bs=1,000) 0.7256812492941721
wandb: FuncNameTest-All MRR (bs=1,000) 0.15864312289572563
wandb: Validation-All MRR (bs=1,000) 0.799873386290268
wandb: Test-go MRR (bs=1,000) 0.7256812492941721
wandb: FuncNameTest-go MRR (bs=1,000) 0.15864312289572563
wandb: Validation-go MRR (bs=1,000) 0.799873386290268
wandb: Syncing files in wandb/run-20200422_220903-1eer61o9:
wandb: selfatt-2020-04-22-22-09-03-graph.pbtxt
wandb: selfatt-2020-04-22-22-09-03.train_log
wandb: selfatt-2020-04-22-22-09-03_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced selfatt-2020-04-22-22-09-03: https://app.wandb.ai/jianguda/CodeSearchNet/runs/1eer61o9
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/1eer61o9
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./selfatt-2020-04-22-22-09-03_model_best.pkl.gz
2020-04-23 08:01:23.883462: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-23 08:01:28.742270: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 42e4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-23 08:01:28.742316: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-23 08:01:29.015973: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-23 08:01:29.016028: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/1eer61o9
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./selfatt-2020-04-22-22-09-03_model_best.pkl.gz
2020-04-23 08:01:23.883462: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-23 08:01:28.742270: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 42e4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-23 08:01:28.742316: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-23 08:01:29.015973: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-23 08:01:29.016028: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-23 08:01:29.016045: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-23 08:01:29.016157: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Evaluating language: go
wandb:
wandb: Synced selfatt-2020-04-22-22-09-03: https://app.wandb.ai/jianguda/CodeSearchNet/runs/1eer61o9
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/1eer61o9  
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./selfatt-2020-04-22-22-09-03_model_best.pkl.gz
2020-04-23 08:01:23.883462: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-23 08:01:28.742270: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 42e4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-23 08:01:28.742316: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-23 08:01:29.015973: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-23 08:01:29.016028: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-23 08:01:29.016045: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-23 08:01:29.016157: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Evaluating language: go
100%|████████████████████████████████████████████████████████████████████████████████████| 726768/726768 [00:03<00:00, 215579.65it/s]726768it [00:12, 58620.07it/s]
Uploading predictions to W&B
NDCG Average: 0.036671457
