root@jian-csn:/home/dev/src# python train.py --model 1dcnn
wandb: W&B is a tool that helps track and visualize machine learning experiments
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 2
wandb: You chose 'Use an existing W&B account'
wandb: You can find your API key in your browser here: https://app.wandb.ai/authorize
wandb: Paste an API key from your profile and hit enter
wandb: Appending key for api.wandb.ai to your netrc file: /root/.netrc
wandb: Started W&B process version 0.8.12 with PID 93
wandb: Wandb version 0.8.31 is available! To upgrade, please run:
wandb: \$ pip install wandb --upgrade
wandb: Local directory: wandb/run-20200329_213006-52tlfk4e
wandb: Syncing run 1dcnn-2020-03-29-21-30-06: https://app.wandb.ai/jianguda/CodeSearchNet/runs/52tlfk4e
wandb: Run `wandb off` to turn off syncing.

2020-03-29 21:30:42.438877: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-03-29 21:30:42.583693: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:  
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 5a1b:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-03-29 21:30:42.583738: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-03-29 21:30:55.875624: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with
strength 1 edge matrix:
2020-03-29 21:30:55.875681: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-03-29 21:30:55.875699: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-03-29 21:30:55.875833: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 5a1b:00:00.0, compute capability: 3.7)
Tokenizing and building vocabulary for code snippets and queries. This step may take several hours.
Starting training run 1dcnn-2020-03-29-21-30-06 of model ConvolutionalModel with following hypers:
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': False, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_1dcnn_position_encoding': 'learned', 'code_1dcnn_layer_list': [128, 128, 128], 'code_1dcnn_kernel_width': [16, 16, 16], 'code_1dcnn_add_residual_connections': True, 'code_1dcnn_activation': 'tanh', 'code_1dcnn_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_1dcnn_position_encoding': 'learned', 'query_1dcnn_layer_list': [128, 128, 128], 'query_1dcnn_kernel_width': [16, 16, 16], 'query_1dcnn_add_residual_connections': True, 'query_1dcnn_activation': 'tanh', 'query_1dcnn_pool_mode': 'weighted_mean', 'batch_size': 1000, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.0005, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'softmax', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
Loading training and validation data.
Begin Training.
Training on 523712 php, 454451 java, 317832 go, 412178 python, 123889 javascript, 48791 ruby samples.
Validating on 2209 ruby, 8253 javascript, 23107 python, 26015 php, 14242 go, 15328 java samples.
==== Epoch 0 ====
Epoch 0 (train) took 1263.70s [processed 1487 samples/second]
Training Loss: 4.267302
Epoch 0 (valid) took 20.16s [processed 4415 samples/second]
Validation: Loss: 3.799882 | MRR: 0.316951
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 1 ====
Epoch 1 (train) took 1225.68s [processed 1533 samples/second]
Training Loss: 2.628207
Epoch 1 (valid) took 19.33s [processed 4604 samples/second]
Validation: Loss: 3.250573 | MRR: 0.414342
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 2 ====
Epoch 2 (train) took 1223.77s [processed 1536 samples/second]
Training Loss: 2.135355
Epoch 2 (valid) took 19.30s [processed 4610 samples/second]
Validation: Loss: 3.046889 | MRR: 0.452496
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 3 ====
Epoch 3 (train) took 1223.17s [processed 1536 samples/second]
Training Loss: 1.882441
Epoch 3 (valid) took 19.29s [processed 4613 samples/second]
Validation: Loss: 2.943138 | MRR: 0.472844
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 4 ====
Epoch 4 (train) took 1222.33s [processed 1538 samples/second]
Training Loss: 1.720046
Epoch 4 (valid) took 19.22s [processed 4630 samples/second]
Validation: Loss: 2.886071 | MRR: 0.487348
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 5 ====
Epoch 5 (train) took 1222.91s [processed 1537 samples/second]
Training Loss: 1.600738
Epoch 5 (valid) took 19.26s [processed 4620 samples/second]
Validation: Loss: 2.852080 | MRR: 0.500428
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 6 ====
Epoch 6 (train) took 1220.91s [processed 1539 samples/second]
Training Loss: 1.515285
Epoch 6 (valid) took 19.21s [processed 4633 samples/second]
Validation: Loss: 2.835959 | MRR: 0.499408
==== Epoch 7 ====
Epoch 7 (train) took 1220.69s [processed 1540 samples/second]
Training Loss: 1.443713
Epoch 7 (valid) took 19.19s [processed 4636 samples/second]
Validation: Loss: 2.827495 | MRR: 0.511503
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 8 ====
Epoch 8 (train) took 1220.75s [processed 1540 samples/second]
Training Loss: 1.385510
Epoch 8 (valid) took 19.18s [processed 4641 samples/second]
Validation: Loss: 2.848391 | MRR: 0.510608
==== Epoch 9 ====
Epoch 9 (train) took 1221.22s [processed 1539 samples/second]
Training Loss: 1.335213
Epoch 9 (valid) took 19.27s [processed 4617 samples/second]
Validation: Loss: 2.840503 | MRR: 0.517464
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 10 ====
Epoch 10 (train) took 1219.69s [processed 1541 samples/second]
Training Loss: 1.292338
Epoch 10 (valid) took 19.13s [processed 4651 samples/second]
Validation: Loss: 2.828362 | MRR: 0.522368
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 11 ====
Epoch 11 (train) took 1219.92s [processed 1541 samples/second]
Training Loss: 1.253534
Epoch 11 (valid) took 19.15s [processed 4647 samples/second]
Validation: Loss: 2.848869 | MRR: 0.523907
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 12 ====
Epoch 12 (train) took 1219.95s [processed 1541 samples/second]
Training Loss: 1.222664
Epoch 12 (valid) took 19.13s [processed 4652 samples/second]
Validation: Loss: 2.840689 | MRR: 0.523859
==== Epoch 13 ====
Epoch 13 (train) took 1219.30s [processed 1541 samples/second]
Training Loss: 1.193528
Epoch 13 (valid) took 19.18s [processed 4640 samples/second]
Validation: Loss: 2.878104 | MRR: 0.526082
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 14 ====
Epoch 14 (train) took 1218.75s [processed 1542 samples/second]
Training Loss: 1.168973
Epoch 14 (valid) took 19.21s [processed 4631 samples/second]
Validation: Loss: 2.861891 | MRR: 0.530876
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 15 ====
Epoch 15 (train) took 1219.34s [processed 1541 samples/second]
Training Loss: 1.143648
Epoch 15 (valid) took 19.25s [processed 4624 samples/second]
Validation: Loss: 2.865064 | MRR: 0.532015
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 16 ====
Epoch 16 (train) took 1218.95s [processed 1542 samples/second]
Training Loss: 1.124707
Epoch 16 (valid) took 19.19s [processed 4637 samples/second]
Validation: Loss: 2.863290 | MRR: 0.531388
==== Epoch 17 ====
Epoch 17 (train) took 1218.44s [processed 1542 samples/second]
Training Loss: 1.102456
Epoch 17 (valid) took 19.24s [processed 4625 samples/second]
Validation: Loss: 2.899552 | MRR: 0.532325
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 18 ====
Epoch 18 (train) took 1217.88s [processed 1543 samples/second]
Training Loss: 1.085785
Epoch 18 (valid) took 19.15s [processed 4648 samples/second]
Validation: Loss: 2.866586 | MRR: 0.534342
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 19 ====
Epoch 19 (train) took 1218.68s [processed 1542 samples/second]
Training Loss: 1.068502
Epoch 19 (valid) took 19.19s [processed 4636 samples/second]
Validation: Loss: 2.882681 | MRR: 0.535876
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 20 ====
Epoch 20 (train) took 1217.60s [processed 1544 samples/second]
Training Loss: 1.053301
Epoch 20 (valid) took 19.18s [processed 4640 samples/second]
Validation: Loss: 2.887900 | MRR: 0.535388
==== Epoch 21 ====
Epoch 21 (train) took 1217.05s [processed 1544 samples/second]
Training Loss: 1.038602
Epoch 21 (valid) took 19.17s [processed 4642 samples/second]
Validation: Loss: 2.885874 | MRR: 0.537413
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 22 ====
Epoch 22 (train) took 1216.91s [processed 1544 samples/second]
Training Loss: 1.027730
Epoch 22 (valid) took 19.20s [processed 4634 samples/second]
Validation: Loss: 2.913107 | MRR: 0.538446
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 23 ====
Epoch 23 (train) took 1217.37s [processed 1544 samples/second]
Training Loss: 1.013303
Epoch 23 (valid) took 19.21s [processed 4633 samples/second]
Validation: Loss: 2.940238 | MRR: 0.537282
==== Epoch 24 ====
Epoch 24 (train) took 1216.83s [processed 1544 samples/second]
Training Loss: 1.002886
Epoch 24 (valid) took 19.14s [processed 4649 samples/second]
Validation: Loss: 2.930059 | MRR: 0.537730
==== Epoch 25 ====
Epoch 25 (train) took 1216.98s [processed 1544 samples/second]
Training Loss: 0.989385
Epoch 25 (valid) took 19.26s [processed 4620 samples/second]
Validation: Loss: 2.963648 | MRR: 0.537623
==== Epoch 26 ====
Epoch 26 (train) took 1217.17s [processed 1544 samples/second]
Training Loss: 0.981461
Epoch 26 (valid) took 19.30s [processed 4610 samples/second]
Validation: Loss: 2.955070 | MRR: 0.538329
==== Epoch 27 ====
Epoch 27 (train) took 1217.05s [processed 1544 samples/second]
Training Loss: 0.970055
Epoch 27 (valid) took 19.17s [processed 4642 samples/second]
Validation: Loss: 2.962602 | MRR: 0.540404
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 28 ====
Epoch 28 (train) took 1216.83s [processed 1545 samples/second]
Training Loss: 0.959461
Epoch 28 (valid) took 19.19s [processed 4638 samples/second]
Validation: Loss: 2.963511 | MRR: 0.538524
==== Epoch 29 ====
Epoch 29 (train) took 1216.51s [processed 1545 samples/second]
Training Loss: 0.952494
Epoch 29 (valid) took 19.25s [processed 4623 samples/second]
Validation: Loss: 3.008753 | MRR: 0.542697
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 30 ====
Epoch 30 (train) took 1216.31s [processed 1545 samples/second]
Training Loss: 0.944475
Epoch 30 (valid) took 19.23s [processed 4629 samples/second]
Validation: Loss: 3.020233 | MRR: 0.540062
==== Epoch 31 ====
wandb: Network error resolved after 0:00:13.325445, resuming normal operation. far: 0.9287. MRR so far: 0.7995
Epoch 31 (train) took 1216.28s [processed 1545 samples/second]
Training Loss: 0.935855
Epoch 31 (valid) took 19.15s [processed 4648 samples/second]
Validation: Loss: 3.010576 | MRR: 0.541098
==== Epoch 32 ====
Epoch 32 (train) took 1216.48s [processed 1545 samples/second]
Training Loss: 0.929831
Epoch 32 (valid) took 19.20s [processed 4634 samples/second]
Validation: Loss: 3.018528 | MRR: 0.542684
==== Epoch 33 ====
Epoch 33 (train) took 1216.75s [processed 1545 samples/second]
Training Loss: 0.920860
Epoch 33 (valid) took 19.24s [processed 4624 samples/second]
Validation: Loss: 2.998966 | MRR: 0.542972
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 34 ====
Epoch 34 (train) took 1223.89s [processed 1536 samples/second]
Training Loss: 0.915143
Epoch 34 (valid) took 19.30s [processed 4612 samples/second]
Validation: Loss: 3.013871 | MRR: 0.543809
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 35 ====
Epoch 35 (train) took 1223.66s [processed 1536 samples/second]
Training Loss: 0.908783
Epoch 35 (valid) took 19.29s [processed 4614 samples/second]
Validation: Loss: 3.008378 | MRR: 0.541879
==== Epoch 36 ====
Epoch 36 (train) took 1222.96s [processed 1537 samples/second]
Training Loss: 0.901446
Epoch 36 (valid) took 19.32s [processed 4605 samples/second]
Validation: Loss: 3.031430 | MRR: 0.543915
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 37 ====
Epoch 37 (train) took 1224.07s [processed 1535 samples/second]
Training Loss: 0.896093
Epoch 37 (valid) took 19.27s [processed 4619 samples/second]
Validation: Loss: 3.037847 | MRR: 0.544203
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 38 ====
Epoch 38 (train) took 1222.80s [processed 1537 samples/second]
Training Loss: 0.890972
Epoch 38 (valid) took 19.35s [processed 4598 samples/second]
Validation: Loss: 3.061255 | MRR: 0.543175
==== Epoch 39 ====
Epoch 39 (train) took 1223.19s [processed 1536 samples/second]
Training Loss: 0.883220
Epoch 39 (valid) took 19.35s [processed 4599 samples/second]
Validation: Loss: 3.020819 | MRR: 0.543492
==== Epoch 40 ====
Epoch 40 (train) took 1221.90s [processed 1538 samples/second]
Training Loss: 0.877495
Epoch 40 (valid) took 19.26s [processed 4620 samples/second]
Validation: Loss: 3.022971 | MRR: 0.543786
==== Epoch 41 ====
Epoch 41 (train) took 1221.64s [processed 1538 samples/second]
Training Loss: 0.871741
Epoch 41 (valid) took 19.24s [processed 4626 samples/second]
Validation: Loss: 3.053709 | MRR: 0.545208
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 42 ====
Epoch 42 (train) took 1221.93s [processed 1538 samples/second]
Training Loss: 0.869109
Epoch 42 (valid) took 19.39s [processed 4589 samples/second]
Validation: Loss: 3.047029 | MRR: 0.544208
==== Epoch 43 ====
Epoch 43 (train) took 1222.14s [processed 1538 samples/second]
Training Loss: 0.864229
Epoch 43 (valid) took 19.24s [processed 4624 samples/second]
Validation: Loss: 3.034899 | MRR: 0.544888
==== Epoch 44 ====
Epoch 44 (train) took 1222.28s [processed 1538 samples/second]
Training Loss: 0.858894
Epoch 44 (valid) took 19.30s [processed 4610 samples/second]
Validation: Loss: 3.085050 | MRR: 0.544166
==== Epoch 45 ====
Epoch 45 (train) took 1220.99s [processed 1539 samples/second]
Training Loss: 0.853162
Epoch 45 (valid) took 19.26s [processed 4621 samples/second]
Validation: Loss: 3.051922 | MRR: 0.543932
==== Epoch 46 ====
Epoch 46 (train) took 1221.71s [processed 1538 samples/second]
Training Loss: 0.850667
Epoch 46 (valid) took 19.32s [processed 4606 samples/second]
Validation: Loss: 3.051301 | MRR: 0.545838
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 47 ====
Epoch 47 (train) took 1221.48s [processed 1539 samples/second]
Training Loss: 0.843915
Epoch 47 (valid) took 19.36s [processed 4598 samples/second]
Validation: Loss: 3.075633 | MRR: 0.546551
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 48 ====
Epoch 48 (train) took 1222.16s [processed 1538 samples/second]
Epoch 47 (train) took 1221.48s [processed 1539 samples/second]
Training Loss: 0.843915
Epoch 47 (valid) took 19.36s [processed 4598 samples/second]
Validation: Loss: 3.075633 | MRR: 0.546551
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.  
==== Epoch 48 ====
Epoch 46 (valid) took 19.32s [processed 4606 samples/second]
Validation: Loss: 3.051301 | MRR: 0.545838
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 47 ====
Epoch 47 (train) took 1221.48s [processed 1539 samples/second]
Training Loss: 0.843915
Epoch 47 (valid) took 19.36s [processed 4598 samples/second]
Validation: Loss: 3.075633 | MRR: 0.546551
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 48 ====
Epoch 48 (train) took 1222.16s [processed 1538 samples/second]
Training Loss: 0.841737
Epoch 48 (valid) took 19.20s [processed 4635 samples/second]
Validation: Loss: 3.107280 | MRR: 0.544544
==== Epoch 49 ====
Epoch 49 (train) took 1221.77s [processed 1538 samples/second]
Training Loss: 0.837624
Epoch 49 (valid) took 19.21s [processed 4632 samples/second]
Validation: Loss: 3.091261 | MRR: 0.547000
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 50 ====
Epoch 50 (train) took 1221.74s [processed 1538 samples/second]
Training Loss: 0.832534
Epoch 50 (valid) took 19.30s [processed 4610 samples/second]
Validation: Loss: 3.094565 | MRR: 0.545025
==== Epoch 51 ====
Epoch 51 (train) took 1220.41s [processed 1540 samples/second]
Training Loss: 0.829696
Epoch 51 (valid) took 19.22s [processed 4629 samples/second]
Validation: Loss: 3.098206 | MRR: 0.546183
==== Epoch 52 ====
Epoch 52 (train) took 1219.53s [processed 1541 samples/second]
Training Loss: 0.825173
Epoch 52 (valid) took 19.13s [processed 4652 samples/second]
Validation: Loss: 3.079948 | MRR: 0.547051
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 53 ====
Epoch 53 (train) took 1220.81s [processed 1539 samples/second]
Training Loss: 0.822558
Epoch 53 (valid) took 19.25s [processed 4624 samples/second]
Validation: Loss: 3.079359 | MRR: 0.546234
==== Epoch 54 ====
Epoch 54 (train) took 1220.20s [processed 1540 samples/second]
Training Loss: 0.816716
Epoch 54 (valid) took 19.26s [processed 4620 samples/second]
Validation: Loss: 3.092103 | MRR: 0.547524
Best result so far -- saved model as '/home/dev/resources/saved_models/1dcnn-2020-03-29-21-30-06_model_best.pkl.gz'.
==== Epoch 55 ====
Epoch 55 (train) took 1220.47s [processed 1540 samples/second]
Training Loss: 0.814657
Epoch 55 (valid) took 19.23s [processed 4628 samples/second]
Validation: Loss: 3.124124 | MRR: 0.546443
==== Epoch 56 ====
Epoch 56 (train) took 1221.09s [processed 1539 samples/second]
Training Loss: 0.812077
Epoch 56 (valid) took 19.23s [processed 4628 samples/second]
Validation: Loss: 3.128747 | MRR: 0.545031
==== Epoch 57 ====
Epoch 57 (train) took 1221.35s [processed 1539 samples/second]
Training Loss: 0.810347
Epoch 57 (valid) took 19.14s [processed 4650 samples/second]
Validation: Loss: 3.126130 | MRR: 0.546952
==== Epoch 58 ====
Epoch 58 (train) took 1220.01s [processed 1540 samples/second]
Training Loss: 0.805103
Epoch 58 (valid) took 19.25s [processed 4622 samples/second]
Validation: Loss: 3.136955 | MRR: 0.546613
==== Epoch 59 ====
Epoch 59 (train) took 1221.08s [processed 1539 samples/second]
Training Loss: 0.801382
Epoch 59 (valid) took 19.33s [processed 4603 samples/second]
Validation: Loss: 3.124764 | MRR: 0.546984
2020-03-30 18:51:37.373964: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-03-30 18:51:37.374021: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:  
2020-03-30 18:51:37.374036: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-03-30 18:51:37.374047: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-03-30 18:51:37.374137: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 5a1b:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.626
FuncNameTest-All MRR (bs=1,000): 0.653
Validation-All MRR (bs=1,000): 0.628
Test-php MRR (bs=1,000): 0.538
FuncNameTest-php MRR (bs=1,000): 0.794
Validation-php MRR (bs=1,000): 0.543
Test-python MRR (bs=1,000): 0.577
FuncNameTest-python MRR (bs=1,000): 0.553
Validation-python MRR (bs=1,000): 0.525
Test-go MRR (bs=1,000): 0.636
FuncNameTest-go MRR (bs=1,000): 0.152
Validation-go MRR (bs=1,000): 0.734
Test-java MRR (bs=1,000): 0.523
FuncNameTest-java MRR (bs=1,000): 0.734
Validation-java MRR (bs=1,000): 0.488
Test-javascript MRR (bs=1,000): 0.352
FuncNameTest-javascript MRR (bs=1,000): 0.090
Validation-javascript MRR (bs=1,000): 0.349
Test-ruby MRR (bs=1,000): 0.258
FuncNameTest-ruby MRR (bs=1,000): 0.380
Validation-ruby MRR (bs=1,000): 0.313

wandb: Waiting for W&B process to finish, PID 93
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-mrr 0.8213118291976604
wandb: \_timestamp 1585594908.5329769
wandb: \_runtime 77512.50928974152
wandb: \_step 1220
wandb: train-loss 0.8013818985287179
wandb: val-loss 3.124764364756895
wandb: train-time-sec 1221.0846650600433
wandb: val-time-sec 19.334858179092407
wandb: epoch 59
wandb: val-mrr 0.5469843791147296
wandb: best_val_mrr_loss 3.09210284372394
wandb: best_val_mrr 0.547523699128226
wandb: best_epoch 54
wandb: Test-All MRR (bs=1,000) 0.625992742244369
wandb: FuncNameTest-All MRR (bs=1,000) 0.6531568896796881
wandb: Validation-All MRR (bs=1,000) 0.6278785193722158
wandb: Test-php MRR (bs=1,000) 0.5384011149053582
wandb: FuncNameTest-php MRR (bs=1,000) 0.7944014350424062
wandb: Validation-php MRR (bs=1,000) 0.5432616727286491
wandb: Test-python MRR (bs=1,000) 0.5771667575192423
wandb: FuncNameTest-python MRR (bs=1,000) 0.5532754483550751
wandb: Validation-python MRR (bs=1,000) 0.5249272177698441
wandb: Test-go MRR (bs=1,000) 0.6362897599653291
wandb: FuncNameTest-go MRR (bs=1,000) 0.15195907021330263
wandb: Validation-go MRR (bs=1,000) 0.733675980546229
wandb: Test-java MRR (bs=1,000) 0.5226257751223657
wandb: FuncNameTest-java MRR (bs=1,000) 0.7339308885133465
wandb: Validation-java MRR (bs=1,000) 0.4875599531682362
wandb: Test-javascript MRR (bs=1,000) 0.3523213372585854
wandb: FuncNameTest-javascript MRR (bs=1,000) 0.08961784398417168
wandb: Validation-javascript MRR (bs=1,000) 0.3488643106259676
wandb: Test-ruby MRR (bs=1,000) 0.2582596311582359
wandb: FuncNameTest-ruby MRR (bs=1,000) 0.38011964698948203
wandb: Validation-ruby MRR (bs=1,000) 0.3129914773017939
wandb: Syncing files in wandb/run-20200329_213006-52tlfk4e:
wandb: 1dcnn-2020-03-29-21-30-06-graph.pbtxt
wandb: 1dcnn-2020-03-29-21-30-06.train_log
wandb: 1dcnn-2020-03-29-21-30-06_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced 1dcnn-2020-03-29-21-30-06: https://app.wandb.ai/jianguda/CodeSearchNet/runs/52tlfk4e

root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/52tlfk4e
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./1dcnn-2020-03-29-21-30-06_model_best.pkl.gz
2020-03-30 20:15:47.735109: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-03-30 20:15:52.583827: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 5a1b:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-03-30 20:15:52.583875: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-03-30 20:15:52.852549: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-03-30 20:15:52.852610: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-03-30 20:15:52.852627: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-03-30 20:15:52.852740: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 5a1b:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:10<00:00, 113626.43it/s]1156085it [00:19, 58040.14it/s]
Evaluating language: go
2020-03-30 20:15:52.852740: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 5a1b:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:10<00:00, 113626.43it/s]1156085it [00:19, 58040.14it/s]
Evaluating language: go
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 726768/726768 [00:06<00:00, 108957.44it/s]726768it [00:13, 55691.63it/s]
Evaluating language: javascript
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 1857835/1857835 [00:12<00:00, 146239.31it/s]1857835it [00:32, 56320.45it/s]
Evaluating language: java
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 1569889/1569889 [00:15<00:00, 101092.60it/s]1569889it [00:28, 55424.68it/s]
Evaluating language: php
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 977821/977821 [00:10<00:00, 92534.13it/s]977821it [00:18, 53623.82it/s]
Evaluating language: ruby
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 164048/164048 [00:00<00:00, 330221.26it/s]164048it [00:03, 43110.50it/s]
Uploading predictions to W&B
