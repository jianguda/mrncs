root@jian-csn:/home/dev/src# python train.py --model neuralbow
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
wandb: Local directory: wandb/run-20200418_111638-583w3dvb
wandb: Syncing run neuralbow-2020-04-18-11-16-38: https://app.wandb.ai/jianguda/CodeSearchNet/runs/583w3dvb
wandb: Run `wandb off` to turn off syncing.

2020-04-18 11:17:28.275704: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-04-18 11:17:28.551945: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 06bb:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-18 11:17:28.551990: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-18 11:17:42.051097: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-04-18 11:17:42.051148: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-18 11:17:42.051161: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-18 11:17:42.051287: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 06bb:00:00.0, compute capability: 3.7)
Tokenizing and building vocabulary for code snippets and queries. This step may take several hours.
WARNING:tensorflow:From /home/dev/src/models/model.py:299: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is
deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Starting training run neuralbow-2020-04-18-11-16-38 of model NeuralBoWModel with following hypers:
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': False, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_nbow_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_nbow_pool_mode': 'weighted_mean', 'batch_size': 1000, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.01, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'cosine', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
Loading training and validation data.
Begin Training.
Training on 523712 php, 454451 java, 317832 go, 412178 python, 123889 javascript, 48791 ruby samples.
Validating on 2209 ruby, 8253 javascript, 23107 python, 26015 php, 14242 go, 15328 java samples.
==== Epoch 0 ====
Epoch 0 (train) took 206.41s [processed 9108 samples/second]
Training Loss: 0.991448
Epoch 0 (valid) took 6.24s [processed 14263 samples/second]
Validation: Loss: 1.023524 | MRR: 0.379480
Best result so far -- saved model as '/home/dev/resources/saved_models/neuralbow-2020-04-18-11-16-38_model_best.pkl.gz'.
==== Epoch 1 ====
Epoch 1 (train) took 202.24s [processed 9296 samples/second]
Training Loss: 0.942981
Epoch 1 (valid) took 6.16s [processed 14445 samples/second]
Validation: Loss: 1.037274 | MRR: 0.471512
Best result so far -- saved model as '/home/dev/resources/saved_models/neuralbow-2020-04-18-11-16-38_model_best.pkl.gz'.
==== Epoch 2 ====
Epoch 2 (train) took 202.39s [processed 9289 samples/second]
Training Loss: 0.905479
Epoch 2 (valid) took 6.10s [processed 14591 samples/second]
Validation: Loss: 1.035770 | MRR: 0.492151
Best result so far -- saved model as '/home/dev/resources/saved_models/neuralbow-2020-04-18-11-16-38_model_best.pkl.gz'.
==== Epoch 3 ====
Epoch 3 (train) took 201.95s [processed 9309 samples/second]
Training Loss: 0.888690
Epoch 3 (valid) took 6.21s [processed 14329 samples/second]
Validation: Loss: 1.034022 | MRR: 0.498460
Best result so far -- saved model as '/home/dev/resources/saved_models/neuralbow-2020-04-18-11-16-38_model_best.pkl.gz'.
==== Epoch 4 ====
Epoch 4 (train) took 202.02s [processed 9305 samples/second]
Training Loss: 0.879329
Epoch 4 (valid) took 6.16s [processed 14449 samples/second]
Validation: Loss: 1.033800 | MRR: 0.502859
Best result so far -- saved model as '/home/dev/resources/saved_models/neuralbow-2020-04-18-11-16-38_model_best.pkl.gz'.
==== Epoch 5 ====
Epoch 5 (train) took 202.94s [processed 9263 samples/second]
Training Loss: 0.873041
Epoch 5 (valid) took 6.15s [processed 14476 samples/second]
Validation: Loss: 1.032582 | MRR: 0.508723
Best result so far -- saved model as '/home/dev/resources/saved_models/neuralbow-2020-04-18-11-16-38_model_best.pkl.gz'.
==== Epoch 6 ====
Epoch 6 (train) took 201.88s [processed 9312 samples/second]
Training Loss: 0.868361
Epoch 6 (valid) took 6.16s [processed 14448 samples/second]
Validation: Loss: 1.031853 | MRR: 0.509234
Best result so far -- saved model as '/home/dev/resources/saved_models/neuralbow-2020-04-18-11-16-38_model_best.pkl.gz'.
==== Epoch 7 ====
Epoch 7 (train) took 201.71s [processed 9320 samples/second]
Training Loss: 0.864532
Epoch 7 (valid) took 6.20s [processed 14366 samples/second]
Validation: Loss: 1.031773 | MRR: 0.510089
Best result so far -- saved model as '/home/dev/resources/saved_models/neuralbow-2020-04-18-11-16-38_model_best.pkl.gz'.
==== Epoch 8 ====
Epoch 8 (train) took 201.99s [processed 9307 samples/second]
Training Loss: 0.861596
Epoch 8 (valid) took 6.15s [processed 14476 samples/second]
Validation: Loss: 1.031758 | MRR: 0.509728
==== Epoch 9 ====
Epoch 9 (train) took 201.61s [processed 9325 samples/second]
Training Loss: 0.859169
Epoch 9 (valid) took 6.11s [processed 14567 samples/second]
Validation: Loss: 1.031374 | MRR: 0.510043
==== Epoch 10 ====
Epoch 10 (train) took 201.55s [processed 9327 samples/second]
Training Loss: 0.857122
Epoch 10 (valid) took 6.15s [processed 14473 samples/second]
Validation: Loss: 1.030795 | MRR: 0.512182
==== Epoch 8 ====
Epoch 8 (train) took 201.99s [processed 9307 samples/second]
Training Loss: 0.861596
Epoch 8 (valid) took 6.15s [processed 14476 samples/second]
Validation: Loss: 1.031758 | MRR: 0.509728
==== Epoch 9 ====
Epoch 9 (train) took 201.61s [processed 9325 samples/second]
Training Loss: 0.859169
Epoch 9 (valid) took 6.11s [processed 14567 samples/second]
Validation: Loss: 1.031374 | MRR: 0.510043
==== Epoch 10 ====
Epoch 10 (train) took 201.55s [processed 9327 samples/second]
Training Loss: 0.857122
Epoch 10 (valid) took 6.15s [processed 14473 samples/second]
Validation: Loss: 1.030795 | MRR: 0.512182
Best result so far -- saved model as '/home/dev/resources/saved_models/neuralbow-2020-04-18-11-16-38_model_best.pkl.gz'.  
==== Epoch 11 ====
Epoch 11 (train) took 203.05s [processed 9258 samples/second]
Training Loss: 0.855601
Epoch 11 (valid) took 6.13s [processed 14521 samples/second]
Validation: Loss: 1.031908 | MRR: 0.511163
==== Epoch 12 ====
Epoch 12 (train) took 201.80s [processed 9316 samples/second]
Training Loss: 0.853856
Epoch 12 (valid) took 6.13s [processed 14519 samples/second]
Validation: Loss: 1.031139 | MRR: 0.511874
==== Epoch 13 ====
Epoch 13 (train) took 201.96s [processed 9308 samples/second]
Training Loss: 0.852388
Epoch 13 (valid) took 6.08s [processed 14630 samples/second]
Validation: Loss: 1.031788 | MRR: 0.512328
Best result so far -- saved model as '/home/dev/resources/saved_models/neuralbow-2020-04-18-11-16-38_model_best.pkl.gz'.
==== Epoch 14 ====
Epoch 14 (train) took 201.91s [processed 9311 samples/second]
Training Loss: 0.851257
Epoch 14 (valid) took 6.09s [processed 14618 samples/second]
Validation: Loss: 1.031084 | MRR: 0.513033
Best result so far -- saved model as '/home/dev/resources/saved_models/neuralbow-2020-04-18-11-16-38_model_best.pkl.gz'.
==== Epoch 15 ====
Epoch 15 (train) took 201.47s [processed 9331 samples/second]
Training Loss: 0.850120
Epoch 15 (valid) took 6.21s [processed 14330 samples/second]
Validation: Loss: 1.031527 | MRR: 0.514622
Best result so far -- saved model as '/home/dev/resources/saved_models/neuralbow-2020-04-18-11-16-38_model_best.pkl.gz'.
==== Epoch 16 ====
Epoch 16 (train) took 203.32s [processed 9246 samples/second]
Training Loss: 0.849152
Epoch 16 (valid) took 6.10s [processed 14579 samples/second]
Validation: Loss: 1.031024 | MRR: 0.514212
==== Epoch 17 ====
Epoch 17 (train) took 202.62s [processed 9278 samples/second]
Training Loss: 0.848046
Epoch 17 (valid) took 6.15s [processed 14479 samples/second]
Validation: Loss: 1.030928 | MRR: 0.512442
==== Epoch 18 ====
Epoch 18 (train) took 202.48s [processed 9284 samples/second]
Training Loss: 0.847354
Epoch 18 (valid) took 6.12s [processed 14536 samples/second]
Validation: Loss: 1.031363 | MRR: 0.514236
==== Epoch 19 ====
Epoch 19 (train) took 202.13s [processed 9301 samples/second]
Training Loss: 0.846595
Epoch 19 (valid) took 6.14s [processed 14501 samples/second]
Validation: Loss: 1.030955 | MRR: 0.513813
==== Epoch 20 ====
Epoch 20 (train) took 202.25s [processed 9295 samples/second]
Training Loss: 0.845798
Epoch 20 (valid) took 6.12s [processed 14535 samples/second]
Validation: Loss: 1.030727 | MRR: 0.514492
2020-04-18 13:08:11.878445: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-18 13:08:11.878511: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-04-18 13:08:11.878527: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-18 13:08:11.878537: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-18 13:08:11.878630: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 06bb:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.617
FuncNameTest-All MRR (bs=1,000): 0.566
Validation-All MRR (bs=1,000): 0.634
Test-go MRR (bs=1,000): 0.641
FuncNameTest-go MRR (bs=1,000): 0.309
Validation-go MRR (bs=1,000): 0.773
Test-javascript MRR (bs=1,000): 0.457
FuncNameTest-javascript MRR (bs=1,000): 0.231
Validation-javascript MRR (bs=1,000): 0.452
Test-java MRR (bs=1,000): 0.522
Validation: Loss: 1.030727 | MRR: 0.514492
2020-04-18 13:08:11.878445: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-18 13:08:11.878511: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-04-18 13:08:11.878527: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-18 13:08:11.878537: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-18 13:08:11.878630: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 06bb:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.617
FuncNameTest-All MRR (bs=1,000): 0.566
Validation-All MRR (bs=1,000): 0.634
Test-go MRR (bs=1,000): 0.641
FuncNameTest-go MRR (bs=1,000): 0.309
Validation-go MRR (bs=1,000): 0.773
Test-javascript MRR (bs=1,000): 0.457
FuncNameTest-javascript MRR (bs=1,000): 0.231
Validation-javascript MRR (bs=1,000): 0.452
Test-java MRR (bs=1,000): 0.522
FuncNameTest-java MRR (bs=1,000): 0.613
Validation-java MRR (bs=1,000): 0.501
Test-php MRR (bs=1,000): 0.482
FuncNameTest-php MRR (bs=1,000): 0.580
Validation-php MRR (bs=1,000): 0.489
Test-ruby MRR (bs=1,000): 0.431
FuncNameTest-ruby MRR (bs=1,000): 0.407
Validation-ruby MRR (bs=1,000): 0.484
Test-python MRR (bs=1,000): 0.571
FuncNameTest-python MRR (bs=1,000): 0.471
Validation-python MRR (bs=1,000): 0.546

wandb: Waiting for W&B process to finish, PID 27
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_step 440
wandb: \_timestamp 1587215824.2971852
wandb: \_runtime 7236.970227241516
wandb: train-mrr 0.7866807060404026
wandb: train-loss 0.8457981236437534
wandb: val-mrr 0.5144922382483321
wandb: train-time-sec 202.24954462051392
wandb: val-time-sec 6.122960567474365
wandb: epoch 20
wandb: val-loss 1.030726556027873
wandb: best_val_mrr_loss 1.0315266954764892
wandb: best_val_mrr 0.5146219410414107
wandb: best_epoch 15
wandb: Test-All MRR (bs=1,000) 0.6174381304856793
wandb: FuncNameTest-All MRR (bs=1,000) 0.5657067243982508
wandb: Validation-All MRR (bs=1,000) 0.6338415983592811
wandb: Test-go MRR (bs=1,000) 0.6406472389662792
wandb: FuncNameTest-go MRR (bs=1,000) 0.30897124931741343
wandb: Validation-go MRR (bs=1,000) 0.7731982634531713
wandb: Test-javascript MRR (bs=1,000) 0.4572665518600904
wandb: FuncNameTest-javascript MRR (bs=1,000) 0.23129809937014023
wandb: Validation-javascript MRR (bs=1,000) 0.4515928128826386
wandb: Test-java MRR (bs=1,000) 0.5220118683725677
wandb: FuncNameTest-java MRR (bs=1,000) 0.6127223103402382
wandb: Validation-java MRR (bs=1,000) 0.5014251759619901
wandb: Test-php MRR (bs=1,000) 0.48203245415089985
wandb: FuncNameTest-php MRR (bs=1,000) 0.5796434336562468
wandb: Validation-php MRR (bs=1,000) 0.48874628599972847
wandb: Test-ruby MRR (bs=1,000) 0.4314329520917035
wandb: FuncNameTest-ruby MRR (bs=1,000) 0.4072770982290759
wandb: Validation-ruby MRR (bs=1,000) 0.48390229624171
wandb: Test-python MRR (bs=1,000) 0.5708904747724953
wandb: FuncNameTest-python MRR (bs=1,000) 0.4713447233062718
wandb: Validation-python MRR (bs=1,000) 0.545988597922657
wandb: Syncing files in wandb/run-20200418_111638-583w3dvb:
wandb: neuralbow-2020-04-18-11-16-38-graph.pbtxt
wandb: neuralbow-2020-04-18-11-16-38.train_log
wandb: neuralbow-2020-04-18-11-16-38_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced neuralbow-2020-04-18-11-16-38: https://app.wandb.ai/jianguda/CodeSearchNet/runs/583w3dvb

root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/583w3dvb
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./neuralbow-2020-04-18-11-16-38_model_best.pkl.gz
2020-04-18 13:21:50.259658: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-04-18 13:21:55.383111: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 06bb:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-18 13:21:55.383160: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-18 13:21:55.663361: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
Fetching run files from W&B...
Restoring model from ./neuralbow-2020-04-18-11-16-38_model_best.pkl.gz
2020-04-18 13:21:50.259658: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-04-18 13:21:55.383111: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 06bb:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-18 13:21:55.383160: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-18 13:21:55.663361: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
wandb: Synced neuralbow-2020-04-18-11-16-38: https://app.wandb.ai/jianguda/CodeSearchNet/runs/583w3dvb
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/583w3dvb
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./neuralbow-2020-04-18-11-16-38_model_best.pkl.gz
2020-04-18 13:21:50.259658: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-04-18 13:21:55.383111: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 06bb:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-18 13:21:55.383160: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-18 13:21:55.663361: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-04-18 13:21:55.663429: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/583w3dvb
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./neuralbow-2020-04-18-11-16-38_model_best.pkl.gz
2020-04-18 13:21:50.259658: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-04-18 13:21:55.383111: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 06bb:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-18 13:21:55.383160: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-18 13:21:55.663361: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-04-18 13:21:55.663429: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-18 13:21:55.663448: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-18 13:21:55.663569: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 06bb:00:00.0, compute capability: 3.7)
WARNING:tensorflow:From /home/dev/src/models/model.py:299: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is
deprecated and will be removed in a future version.
Instructions for updating:
2020-04-18 13:21:55.663569: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 06bb:00:00.0, compute capability: 3.7)
WARNING:tensorflow:From /home/dev/src/models/model.py:299: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is
deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|███████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:10<00:00, 114586.59it/s]1156085it [00:19, 58074.49it/s]
Evaluating language: go
100%|█████████████████████████████████████████████████████████████████████████████████| 726768/726768 [00:06<00:00, 110919.80it/s]726768it [00:13, 55752.47it/s]
Evaluating language: javascript
2020-04-18 13:21:55.663361: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-04-18 13:21:55.663429: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-18 13:21:55.663448: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-18 13:21:55.663569: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 06bb:00:00.0, compute capability: 3.7)
WARNING:tensorflow:From /home/dev/src/models/model.py:299: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is
deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: python
100%|███████████████████████████████████████████████████████████████████████████████| 1156085/1156085 [00:10<00:00, 114586.59it/s]1156085it [00:19, 58074.49it/s]
Evaluating language: go
100%|█████████████████████████████████████████████████████████████████████████████████| 726768/726768 [00:06<00:00, 110919.80it/s]726768it [00:13, 55752.47it/s]
Evaluating language: javascript
100%|███████████████████████████████████████████████████████████████████████████████| 1857835/1857835 [00:12<00:00, 148145.89it/s]1857835it [00:32, 56380.07it/s]
Evaluating language: java
100%|███████████████████████████████████████████████████████████████████████████████| 1569889/1569889 [00:15<00:00, 102064.23it/s]1569889it [00:28, 54263.12it/s]
Evaluating language: php
100%|██████████████████████████████████████████████████████████████████████████████████| 977821/977821 [00:10<00:00, 92744.80it/s]977821it [00:18, 53046.31it/s]
Evaluating language: ruby
100%|█████████████████████████████████████████████████████████████████████████████████| 164048/164048 [00:00<00:00, 335410.69it/s]164048it [00:03, 42636.55it/s]
Uploading predictions to W&B
