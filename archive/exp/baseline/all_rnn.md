root@jian-csn:/home/dev/src# python train.py --model rnn
wandb: Started W&B process version 0.8.12 with PID 170
wandb: Wandb version 0.8.29 is available! To upgrade, please run:
wandb: \$ pip install wandb --upgrade
wandb: Local directory: wandb/run-20200309_212933-cz83ren4
wandb: Syncing run rnn-2020-03-09-21-29-33: https://app.wandb.ai/jianguda/CodeSearchNet/runs/cz83ren4
wandb: Run `wandb off` to turn off syncing.

2020-03-09 21:29:38.983881: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-03-09 21:29:39.063482: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 9e51:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-03-09 21:29:39.063525: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-03-09 21:29:39.342490: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-03-09 21:29:39.342545: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-03-09 21:29:39.342560: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-03-09 21:29:39.342670: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 9e51:00:00.0, compute capability: 3.7)
Tokenizing and building vocabulary for code snippets and queries. This step may take several hours.
Starting training run rnn-2020-03-09-21-29-33 of model RNNModel with following hypers:
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': True, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_rnn_num_layers': 2, 'code_rnn_hidden_dim': 64, 'code_rnn_cell_type': 'LSTM', 'code_rnn_is_bidirectional': True, 'code_rnn_dropout_keep_rate': 0.8, 'code_rnn_recurrent_dropout_keep_rate': 1.0, 'code_rnn_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_rnn_num_layers': 2, 'query_rnn_hidden_dim': 64, 'query_rnn_cell_type': 'LSTM', 'query_rnn_is_bidirectional': True, 'query_rnn_dropout_keep_rate': 0.8, 'query_rnn_recurrent_dropout_keep_rate': 1.0, 'query_rnn_pool_mode': 'weighted_mean', 'batch_size': 1000, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.01, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'softmax', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
Loading training and validation data.
Begin Training.
Training on 523712 php, 454451 java, 317832 go, 412178 python, 123889 javascript, 48791 ruby samples.
Validating on 2209 ruby, 8253 javascript, 23107 python, 26015 php, 14242 go, 15328 java samples.
==== Epoch 0 ====
Epoch 0 (train) took 7500.90s [processed 250 samples/second]
Training Loss: 4.086505
Epoch 0 (valid) took 145.01s [processed 613 samples/second]
Validation: Loss: 3.782689 | MRR: 0.321196
Best result so far -- saved model as '/home/dev/resources/saved_models/rnn-2020-03-09-21-29-33_model_best.pkl.gz'.
==== Epoch 1 ====
Epoch 1 (train) took 7306.86s [processed 257 samples/second]
Training Loss: 2.914890
Epoch 1 (valid) took 142.82s [processed 623 samples/second]
Validation: Loss: 3.461232 | MRR: 0.369655
Best result so far -- saved model as '/home/dev/resources/saved_models/rnn-2020-03-09-21-29-33_model_best.pkl.gz'.
==== Epoch 2 ====
Epoch 2 (train) took 6954.13s [processed 270 samples/second]
Training Loss: 2.696498
Epoch 2 (valid) took 143.12s [processed 621 samples/second]
Validation: Loss: 3.389836 | MRR: 0.389361
Best result so far -- saved model as '/home/dev/resources/saved_models/rnn-2020-03-09-21-29-33_model_best.pkl.gz'.
==== Epoch 3 ====
Epoch 3 (train) took 6949.24s [processed 270 samples/second]
Training Loss: 2.632974
Epoch 3 (valid) took 142.78s [processed 623 samples/second]
Validation: Loss: 3.347341 | MRR: 0.394334
Best result so far -- saved model as '/home/dev/resources/saved_models/rnn-2020-03-09-21-29-33_model_best.pkl.gz'.
==== Epoch 4 ====
Epoch 4 (train) took 6952.13s [processed 270 samples/second]
Training Loss: 2.609830
Epoch 4 (valid) took 143.52s [processed 620 samples/second]
Validation: Loss: 3.324655 | MRR: 0.394037
==== Epoch 5 ====
Epoch 5 (train) took 6958.76s [processed 270 samples/second]
Training Loss: 2.625237
Epoch 5 (valid) took 143.13s [processed 621 samples/second]
Validation: Loss: 3.350345 | MRR: 0.391914
==== Epoch 6 ====
Epoch 6 (train) took 6972.66s [processed 269 samples/second]
Training Loss: 2.682649
Epoch 6 (valid) took 143.31s [processed 621 samples/second]
Validation: Loss: 3.410244 | MRR: 0.380128
==== Epoch 7 ====
Epoch 7 (train) took 7001.51s [processed 268 samples/second]
Training Loss: 2.777345
Epoch 7 (valid) took 143.38s [processed 620 samples/second]
Validation: Loss: 3.478265 | MRR: 0.362878
==== Epoch 8 ====
Epoch 8 (train) took 6991.03s [processed 268 samples/second]
Training Loss: 2.930490
Epoch 8 (valid) took 143.42s [processed 620 samples/second]
Validation: Loss: 3.604990 | MRR: 0.333064
2020-03-10 16:07:50.682817: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-03-10 16:07:50.682881: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-03-10 16:07:50.682898: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-03-10 16:07:50.682910: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-03-10 16:07:50.683011: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 9e51:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.428
FuncNameTest-All MRR (bs=1,000): 0.515
Validation-All MRR (bs=1,000): 0.427
Test-java MRR (bs=1,000): 0.288
FuncNameTest-java MRR (bs=1,000): 0.523
Validation-java MRR (bs=1,000): 0.271
Test-php MRR (bs=1,000): 0.297
FuncNameTest-php MRR (bs=1,000): 0.620
Validation-php MRR (bs=1,000): 0.296
Test-javascript MRR (bs=1,000): 0.151
FuncNameTest-javascript MRR (bs=1,000): 0.059
Validation-javascript MRR (bs=1,000): 0.146
Test-ruby MRR (bs=1,000): 0.066
FuncNameTest-ruby MRR (bs=1,000): 0.222
Validation-ruby MRR (bs=1,000): 0.071
Test-python MRR (bs=1,000): 0.284
FuncNameTest-python MRR (bs=1,000): 0.377
Validation-python MRR (bs=1,000): 0.242
Test-go MRR (bs=1,000): 0.440
FuncNameTest-go MRR (bs=1,000): 0.057
Validation-go MRR (bs=1,000): 0.502

wandb: Waiting for W&B process to finish, PID 170
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-loss 2.9304903262473165
wandb: train-mrr 0.4501743075593989
wandb: \_runtime 68214.62910890579
wandb: \_step 200
wandb: \_timestamp 1583857586.8305802
wandb: val-mrr 0.33306416835141983
wandb: epoch 8
wandb: train-time-sec 6991.026738405228
wandb: val-time-sec 143.42358827590942
wandb: val-loss 3.604989935842793
wandb: best_val_mrr_loss 3.3473411249310785
wandb: best_val_mrr 0.39433413319105515
wandb: best_epoch 3
wandb: Test-All MRR (bs=1,000) 0.42795423010603045
wandb: FuncNameTest-All MRR (bs=1,000) 0.514547890144966
wandb: Validation-All MRR (bs=1,000) 0.4268070640722169
wandb: Test-java MRR (bs=1,000) 0.28812292156083985
wandb: FuncNameTest-java MRR (bs=1,000) 0.522779172471362
wandb: Validation-java MRR (bs=1,000) 0.27092829711375804
wandb: Test-php MRR (bs=1,000) 0.2974205664117266
wandb: FuncNameTest-php MRR (bs=1,000) 0.6200338475968559
wandb: Validation-php MRR (bs=1,000) 0.2956254088445787
wandb: FuncNameTest-All MRR (bs=1,000) 0.514547890144966
wandb: Validation-All MRR (bs=1,000) 0.4268070640722169
wandb: Test-java MRR (bs=1,000) 0.28812292156083985
wandb: FuncNameTest-java MRR (bs=1,000) 0.522779172471362
wandb: Validation-java MRR (bs=1,000) 0.27092829711375804
wandb: Test-php MRR (bs=1,000) 0.2974205664117266
wandb: FuncNameTest-php MRR (bs=1,000) 0.6200338475968559
wandb: Validation-php MRR (bs=1,000) 0.2956254088445787
wandb: Test-javascript MRR (bs=1,000) 0.1508673152268327
wandb: FuncNameTest-javascript MRR (bs=1,000) 0.05941195761022323
wandb: Validation-javascript MRR (bs=1,000) 0.14596613365156796
wandb: Test-ruby MRR (bs=1,000) 0.06603304820463166
wandb: FuncNameTest-All MRR (bs=1,000) 0.514547890144966
wandb: Validation-All MRR (bs=1,000) 0.4268070640722169
wandb: Test-java MRR (bs=1,000) 0.28812292156083985
wandb: FuncNameTest-java MRR (bs=1,000) 0.522779172471362
wandb: Validation-java MRR (bs=1,000) 0.27092829711375804
wandb: Test-php MRR (bs=1,000) 0.2974205664117266
wandb: FuncNameTest-php MRR (bs=1,000) 0.6200338475968559
wandb: Validation-php MRR (bs=1,000) 0.2956254088445787
wandb: Test-javascript MRR (bs=1,000) 0.1508673152268327
wandb: FuncNameTest-javascript MRR (bs=1,000) 0.05941195761022323
wandb: Validation-javascript MRR (bs=1,000) 0.14596613365156796
wandb: Test-ruby MRR (bs=1,000) 0.06603304820463166
wandb: best_val_mrr 0.39433413319105515
wandb: best_epoch 3
wandb: Test-All MRR (bs=1,000) 0.42795423010603045
wandb: FuncNameTest-All MRR (bs=1,000) 0.514547890144966
wandb: Validation-All MRR (bs=1,000) 0.4268070640722169
wandb: Test-java MRR (bs=1,000) 0.28812292156083985
wandb: FuncNameTest-java MRR (bs=1,000) 0.522779172471362
wandb: Validation-java MRR (bs=1,000) 0.27092829711375804
wandb: Test-php MRR (bs=1,000) 0.2974205664117266
wandb: FuncNameTest-php MRR (bs=1,000) 0.6200338475968559
wandb: Validation-php MRR (bs=1,000) 0.2956254088445787
wandb: Test-javascript MRR (bs=1,000) 0.1508673152268327
wandb: FuncNameTest-javascript MRR (bs=1,000) 0.05941195761022323
wandb: Validation-javascript MRR (bs=1,000) 0.14596613365156796
wandb: Test-ruby MRR (bs=1,000) 0.06603304820463166
wandb: FuncNameTest-ruby MRR (bs=1,000) 0.2215034348952883
wandb: Validation-ruby MRR (bs=1,000) 0.07053172092976104
wandb: Test-python MRR (bs=1,000) 0.2841846881504547
wandb: FuncNameTest-python MRR (bs=1,000) 0.3770075799110158
wandb: Validation-python MRR (bs=1,000) 0.24202309733526486
wandb: Test-go MRR (bs=1,000) 0.43957281140854076
wandb: FuncNameTest-go MRR (bs=1,000) 0.057487610874484694
wandb: Validation-go MRR (bs=1,000) 0.5021354414662135
wandb: Syncing files in wandb/run-20200309_212933-cz83ren4:
wandb: rnn-2020-03-09-21-29-33-graph.pbtxt
wandb: rnn-2020-03-09-21-29-33.train_log
wandb: rnn-2020-03-09-21-29-33_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced rnn-2020-03-09-21-29-33: https://app.wandb.ai/jianguda/CodeSearchNet/runs/cz83ren4

root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/cz83ren4
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./rnn-2020-03-09-21-29-33_model_best.pkl.gz
2020-03-10 17:13:08.559617: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-03-10 17:13:13.383663: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 9e51:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-03-10 17:13:13.383711: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-03-10 17:13:13.655098: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-03-10 17:13:13.655177: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-03-10 17:13:13.655190: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-03-10 17:13:13.655303: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 9e51:00:00.0, compute capability: 3.7)
Evaluating language: python
100%|█████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:10<00:00, 111771.41it/s]1156085it [00:20, 57371.21it/s]
Evaluating language: go
100%|███████████████████████████████████████████████████████████████████████████████| 726768/726768 [00:06<00:00, 107950.02it/s]726768it [00:13, 54167.36it/s]
Evaluating language: javascript
100%|█████████████████████████████████████████████████████████████████████████████| 1857835/1857835 [00:12<00:00, 143912.53it/s]1857835it [00:32, 56386.54it/s]
Evaluating language: java
100%|█████████████████████████████████████████████████████████████████████████████| 1569889/1569889 [00:01<00:00, 822139.60it/s]1569889it [00:28, 55577.04it/s]
Evaluating language: php
100%|███████████████████████████████████████████████████████████████████████████████| 977821/977821 [00:01<00:00, 670556.07it/s]977821it [00:18, 54104.68it/s]
Evaluating language: ruby
100%|███████████████████████████████████████████████████████████████████████████████| 164048/164048 [00:00<00:00, 368497.59it/s]164048it [00:03, 44736.48it/s]
Uploading predictions to W&B
