python train.py --model neuralbow ../resources/saved_models ../resources/data/javascript/final/jsonl/train ../resources/data/javascript/final/jsonl/valid ../resources/data/javascript/final/jsonl/test

python predict.py -r jianguda/CodeSearchNet/0123456

# NBOW

root@jian-csn:/home/dev/src# python train.py --model neuralbow ../resources/saved_models ../resources/data/javascript/final/jsonl/train ../resources/data/javascript/final/jsonl/valid ../resources/data/javascript/final/jsonl/test
wandb: Started W&B process version 0.8.12 with PID 850
wandb: Wandb version 0.8.32 is available! To upgrade, please run:
wandb: \$ pip install wandb --upgrade
wandb: Local directory: wandb/run-20200423_102448-dqilr06w
wandb: Syncing run neuralbow-2020-04-23-10-24-48: https://app.wandb.ai/jianguda/CodeSearchNet/runs/dqilr06w
wandb: Run `wandb off` to turn off syncing.

2020-04-23 10:24:53.980753: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-23 10:24:54.060650: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 42e4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-23 10:24:54.060694: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-23 10:24:54.336170: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-23 10:24:54.336230: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-23 10:24:54.336245: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-23 10:24:54.336354: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Tokenizing and building vocabulary for code snippets and queries. This step may take several hours.
WARNING:tensorflow:From /home/dev/src/models/model.py:299: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Starting training run neuralbow-2020-04-23-10-24-48 of model NeuralBoWModel with following hypers:
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': False, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_nbow_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_nbow_pool_mode': 'weighted_mean', 'batch_size': 1000, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.01, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'cosine', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
Loading training and validation data.
Begin Training.
Training on 123889 javascript samples.
Validating on 8253 javascript samples.
==== Epoch 0 ====
Epoch 0 (train) took 11.86s [processed 10374 samples/second]
Training Loss: 1.013560
Epoch 0 (valid) took 0.56s [processed 14381 samples/second]
Validation: Loss: 1.003280 | MRR: 0.134883
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-23-10-24-48_model_best.pkl.gz'.
==== Epoch 1 ====
Epoch 1 (train) took 11.50s [processed 10692 samples/second]
Training Loss: 1.003117
Epoch 1 (valid) took 0.51s [processed 15685 samples/second]
Validation: Loss: 1.002909 | MRR: 0.194579
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-23-10-24-48_model_best.pkl.gz'.
==== Epoch 2 ====
Epoch 2 (train) took 11.54s [processed 10656 samples/second]
Training Loss: 1.001778
Epoch 2 (valid) took 0.51s [processed 15536 samples/second]
Validation: Loss: 1.003500 | MRR: 0.247841
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-23-10-24-48_model_best.pkl.gz'.
==== Epoch 3 ====
Epoch 3 (train) took 11.50s [processed 10693 samples/second]
Training Loss: 1.000488
Epoch 3 (valid) took 0.51s [processed 15587 samples/second]
Validation: Loss: 1.008266 | MRR: 0.280811
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-23-10-24-48_model_best.pkl.gz'.
==== Epoch 4 ====
Epoch 4 (train) took 11.45s [processed 10744 samples/second]
Training Loss: 0.997958
Epoch 4 (valid) took 0.52s [processed 15480 samples/second]
Validation: Loss: 1.021799 | MRR: 0.304887
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-23-10-24-48_model_best.pkl.gz'.
==== Epoch 5 ====
Epoch 5 (train) took 11.57s [processed 10628 samples/second]
Training Loss: 0.986640
Epoch 5 (valid) took 0.51s [processed 15685 samples/second]
Validation: Loss: 1.065436 | MRR: 0.327715
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-23-10-24-48_model_best.pkl.gz'.
==== Epoch 6 ====
Epoch 6 (train) took 11.48s [processed 10716 samples/second]
Training Loss: 0.950845
Epoch 6 (valid) took 0.55s [processed 14627 samples/second]
Validation: Loss: 1.093420 | MRR: 0.354092
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-23-10-24-48_model_best.pkl.gz'.
==== Epoch 7 ====
Epoch 7 (train) took 11.44s [processed 10755 samples/second]
Training Loss: 0.918169
Epoch 7 (valid) took 0.51s [processed 15791 samples/second]
Validation: Loss: 1.095462 | MRR: 0.361111
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-23-10-24-48_model_best.pkl.gz'.
==== Epoch 8 ====
Epoch 8 (train) took 11.46s [processed 10730 samples/second]
Training Loss: 0.898603
Epoch 8 (valid) took 0.51s [processed 15714 samples/second]
Validation: Loss: 1.096992 | MRR: 0.362614
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-23-10-24-48_model_best.pkl.gz'.
==== Epoch 9 ====
Epoch 9 (train) took 11.46s [processed 10730 samples/second]
Training Loss: 0.886465
Epoch 9 (valid) took 0.51s [processed 15588 samples/second]
Validation: Loss: 1.096859 | MRR: 0.360976
==== Epoch 10 ====
Epoch 10 (train) took 11.46s [processed 10735 samples/second]
Training Loss: 0.877422
Epoch 10 (valid) took 0.50s [processed 15844 samples/second]
Validation: Loss: 1.096516 | MRR: 0.359092
==== Epoch 11 ====
Epoch 11 (train) took 11.50s [processed 10699 samples/second]
Training Loss: 0.870955
Epoch 11 (valid) took 0.50s [processed 15906 samples/second]
Validation: Loss: 1.096330 | MRR: 0.358251
==== Epoch 12 ====
Epoch 12 (train) took 11.52s [processed 10675 samples/second]
Training Loss: 0.865355
Epoch 12 (valid) took 0.51s [processed 15704 samples/second]
Validation: Loss: 1.096590 | MRR: 0.358399
==== Epoch 13 ====
Epoch 13 (train) took 11.47s [processed 10719 samples/second]
Training Loss: 0.861088
Epoch 13 (valid) took 0.51s [processed 15766 samples/second]
Validation: Loss: 1.095683 | MRR: 0.357773
2020-04-23 10:31:51.756640: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-23 10:31:51.756702: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-23 10:31:51.756717: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-23 10:31:51.756728: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-23 10:31:51.756826: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.427
FuncNameTest-All MRR (bs=1,000): 0.195
Validation-All MRR (bs=1,000): 0.410
Test-javascript MRR (bs=1,000): 0.427
FuncNameTest-javascript MRR (bs=1,000): 0.195
Validation-javascript MRR (bs=1,000): 0.410

wandb: Waiting for W&B process to finish, PID 850
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-loss 0.8610876580563988
wandb: \_runtime 459.69758200645447
wandb: \_timestamp 1587637946.9750588
wandb: \_step 47
wandb: train-mrr 0.8842883762266578
wandb: train-time-sec 11.474740982055664
wandb: val-mrr 0.3577732582092285
wandb: val-loss 1.09568253159523
wandb: val-time-sec 0.5073943138122559
wandb: epoch 13
wandb: train-mrr 0.8842883762266578
wandb: train-time-sec 11.474740982055664
wandb: val-mrr 0.3577732582092285
wandb: val-loss 1.09568253159523
wandb: val-time-sec 0.5073943138122559
wandb: epoch 13
wandb: best_val_mrr_loss 1.0969923734664917
wandb: best_val_mrr 0.36261447525024415
wandb: best_epoch 8
wandb: Test-All MRR (bs=1,000) 0.42676054709171374
wandb: FuncNameTest-All MRR (bs=1,000) 0.19487572101091757
wandb: Validation-All MRR (bs=1,000) 0.40997349541934974
wandb: Test-javascript MRR (bs=1,000) 0.42676054709171374
wandb: FuncNameTest-javascript MRR (bs=1,000) 0.19487572101091757
wandb: Validation-javascript MRR (bs=1,000) 0.40997349541934974
wandb: Syncing files in wandb/run-20200423_102448-dqilr06w:
wandb: neuralbow-2020-04-23-10-24-48-graph.pbtxt
wandb: neuralbow-2020-04-23-10-24-48.train_log
wandb: neuralbow-2020-04-23-10-24-48_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced neuralbow-2020-04-23-10-24-48: https://app.wandb.ai/jianguda/CodeSearchNet/runs/dqilr06w
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/dqilr06w
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./neuralbow-2020-04-23-10-24-48_model_best.pkl.gz
2020-04-23 10:38:09.684867: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-23 10:38:14.738260: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 42e4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-23 10:38:14.738307: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-23 10:38:15.013271: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-23 10:38:15.013328: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-23 10:38:15.013343: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-23 10:38:15.013455: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
WARNING:tensorflow:From /home/dev/src/models/model.py:299: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: javascript
100%|███████████████████████████████████████████████████████████████████████████████████| 1857835/1857835 [00:19<00:00, 97218.60it/s]1857835it [00:32, 56744.99it/s]
Uploading predictions to W&B
NDCG Average: 0.065279179

# CNN

root@jian-csn:/home/dev/src# python train.py --model 1dcnn ../resources/saved_models ../resources/data/javascript/final/jsonl/train ../resources/data/javascript/final/jsonl/valid ../resources/data/javascript/final/jsonl/test
wandb: Started W&B process version 0.8.12 with PID 1052
wandb: Wandb version 0.8.32 is available! To upgrade, please run:
wandb: \$ pip install wandb --upgrade
wandb: Local directory: wandb/run-20200423_111516-o2qy00vv
wandb: Syncing run 1dcnn-2020-04-23-11-15-16: https://app.wandb.ai/jianguda/CodeSearchNet/runs/o2qy00vv
wandb: Run `wandb off` to turn off syncing.

2020-04-23 11:15:22.479712: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-23 11:15:22.558665: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 42e4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-23 11:15:22.558709: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-23 11:15:22.833914: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-23 11:15:22.833969: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-23 11:15:22.833985: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-23 11:15:22.834097: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Tokenizing and building vocabulary for code snippets and queries. This step may take several hours.
Starting training run 1dcnn-2020-04-23-11-15-16 of model ConvolutionalModel with following hypers:
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': False, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_1dcnn_position_encoding': 'learned', 'code_1dcnn_layer_list': [128, 128, 128], 'code_1dcnn_kernel_width': [16, 16, 16], 'code_1dcnn_add_residual_connections': True, 'code_1dcnn_activation': 'tanh', 'code_1dcnn_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_1dcnn_position_encoding': 'learned', 'query_1dcnn_layer_list': [128, 128, 128], 'query_1dcnn_kernel_width': [16, 16, 16], 'query_1dcnn_add_residual_connections': True, 'query_1dcnn_activation': 'tanh', 'query_1dcnn_pool_mode': 'weighted_mean', 'batch_size': 1000, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.0005, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'softmax', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
Loading training and validation data.
Begin Training.
Training on 123889 javascript samples.
Validating on 8253 javascript samples.
==== Epoch 0 ====
Epoch 0 (train) took 78.51s [processed 1566 samples/second]
Training Loss: 6.610626
Epoch 0 (valid) took 1.70s [processed 4716 samples/second]
Validation: Loss: 6.327774 | MRR: 0.028948
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-11-15-16_model_best.pkl.gz'.
==== Epoch 1 ====
Epoch 1 (train) took 77.16s [processed 1593 samples/second]
Training Loss: 6.012784
Epoch 1 (valid) took 1.65s [processed 4854 samples/second]
Validation: Loss: 6.144703 | MRR: 0.045763
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-11-15-16_model_best.pkl.gz'.
==== Epoch 2 ====
Epoch 2 (train) took 77.59s [processed 1585 samples/second]
Training Loss: 5.600365
Epoch 2 (valid) took 1.63s [processed 4903 samples/second]
Validation: Loss: 5.836975 | MRR: 0.069050
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-11-15-16_model_best.pkl.gz'.
==== Epoch 3 ====
Epoch 3 (train) took 77.68s [processed 1583 samples/second]
Training Loss: 5.273057
Epoch 3 (valid) took 1.64s [processed 4883 samples/second]
Validation: Loss: 5.704672 | MRR: 0.086236
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-11-15-16_model_best.pkl.gz'.
==== Epoch 4 ====
Epoch 4 (train) took 77.98s [processed 1577 samples/second]
Training Loss: 4.965838
Epoch 4 (valid) took 1.64s [processed 4868 samples/second]
Validation: Loss: 5.514788 | MRR: 0.114179
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-11-15-16_model_best.pkl.gz'.
==== Epoch 5 ====
Epoch 5 (train) took 77.96s [processed 1577 samples/second]
Training Loss: 4.688487
Epoch 5 (valid) took 1.64s [processed 4870 samples/second]
Validation: Loss: 5.424552 | MRR: 0.134745
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-11-15-16_model_best.pkl.gz'.
==== Epoch 6 ====
Epoch 6 (train) took 78.07s [processed 1575 samples/second]
Training Loss: 4.395355
Epoch 6 (valid) took 1.69s [processed 4738 samples/second]
Validation: Loss: 5.338802 | MRR: 0.151949
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-11-15-16_model_best.pkl.gz'.
==== Epoch 7 ====
Epoch 7 (train) took 78.12s [processed 1574 samples/second]
Training Loss: 4.139434
Epoch 7 (valid) took 1.65s [processed 4839 samples/second]
Validation: Loss: 5.217942 | MRR: 0.172057
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-11-15-16_model_best.pkl.gz'.
==== Epoch 8 ====
Epoch 8 (train) took 78.09s [processed 1575 samples/second]
Training Loss: 3.912483
Epoch 8 (valid) took 1.65s [processed 4853 samples/second]
Validation: Loss: 5.179908 | MRR: 0.183262
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-11-15-16_model_best.pkl.gz'.
==== Epoch 9 ====
Epoch 9 (train) took 77.97s [processed 1577 samples/second]
Training Loss: 3.704238
Epoch 9 (valid) took 1.65s [processed 4849 samples/second]
Validation: Loss: 5.201503 | MRR: 0.192575
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-11-15-16_model_best.pkl.gz'.
==== Epoch 10 ====
Epoch 10 (train) took 78.01s [processed 1576 samples/second]
Training Loss: 3.518434
Epoch 10 (valid) took 1.65s [processed 4843 samples/second]
Validation: Loss: 5.206944 | MRR: 0.201628
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-11-15-16_model_best.pkl.gz'.
==== Epoch 11 ====
Epoch 11 (train) took 78.04s [processed 1576 samples/second]
Training Loss: 3.366565
Epoch 11 (valid) took 1.65s [processed 4836 samples/second]
Validation: Loss: 5.191791 | MRR: 0.204877
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-11-15-16_model_best.pkl.gz'.
==== Epoch 12 ====
Epoch 12 (train) took 78.11s [processed 1574 samples/second]
Training Loss: 3.220203
Epoch 12 (valid) took 1.66s [processed 4821 samples/second]
Validation: Loss: 5.340349 | MRR: 0.214519
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-11-15-16_model_best.pkl.gz'.
==== Epoch 13 ====
Epoch 13 (train) took 77.93s [processed 1578 samples/second]
Training Loss: 3.050248
Epoch 13 (valid) took 1.67s [processed 4801 samples/second]
Validation: Loss: 5.218050 | MRR: 0.219738
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-11-15-16_model_best.pkl.gz'.
==== Epoch 14 ====
Epoch 14 (train) took 77.89s [processed 1579 samples/second]
Training Loss: 2.925753
Epoch 14 (valid) took 1.66s [processed 4815 samples/second]
Validation: Loss: 5.273759 | MRR: 0.225814
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-11-15-16_model_best.pkl.gz'.
==== Epoch 15 ====
Epoch 15 (train) took 77.87s [processed 1579 samples/second]
Training Loss: 2.782702
Epoch 15 (valid) took 1.63s [processed 4894 samples/second]
Validation: Loss: 5.280099 | MRR: 0.236459
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-11-15-16_model_best.pkl.gz'.
==== Epoch 16 ====
Epoch 16 (train) took 77.75s [processed 1581 samples/second]
Training Loss: 2.658102
Epoch 16 (valid) took 1.65s [processed 4855 samples/second]
Validation: Loss: 5.402630 | MRR: 0.232467
==== Epoch 17 ====
Epoch 17 (train) took 77.91s [processed 1578 samples/second]
Training Loss: 2.544314
Epoch 17 (valid) took 1.64s [processed 4889 samples/second]
Validation: Loss: 5.397550 | MRR: 0.239800
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-11-15-16_model_best.pkl.gz'.
==== Epoch 18 ====
Epoch 18 (train) took 78.05s [processed 1575 samples/second]
Training Loss: 2.444119
Epoch 18 (valid) took 1.65s [processed 4853 samples/second]
Validation: Loss: 5.442584 | MRR: 0.244527
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-11-15-16_model_best.pkl.gz'.
==== Epoch 19 ====
Epoch 19 (train) took 77.89s [processed 1579 samples/second]
Training Loss: 2.335569
Epoch 19 (valid) took 1.65s [processed 4851 samples/second]
Validation: Loss: 5.536970 | MRR: 0.244382
==== Epoch 20 ====
Epoch 20 (train) took 77.88s [processed 1579 samples/second]
Training Loss: 2.253812
Epoch 20 (valid) took 1.68s [processed 4772 samples/second]
Validation: Loss: 5.589247 | MRR: 0.243512
==== Epoch 21 ====
Epoch 21 (train) took 77.95s [processed 1578 samples/second]
Training Loss: 2.159553
Epoch 21 (valid) took 1.66s [processed 4817 samples/second]
Validation: Loss: 5.648034 | MRR: 0.246148
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-11-15-16_model_best.pkl.gz'.
==== Epoch 22 ====
Epoch 22 (train) took 78.02s [processed 1576 samples/second]
Training Loss: 2.084846
Epoch 22 (valid) took 1.65s [processed 4860 samples/second]
Validation: Loss: 5.702461 | MRR: 0.249497
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-11-15-16_model_best.pkl.gz'.
==== Epoch 23 ====
Epoch 23 (train) took 77.92s [processed 1578 samples/second]
Training Loss: 2.031494
Epoch 23 (valid) took 1.63s [processed 4903 samples/second]
Validation: Loss: 5.816080 | MRR: 0.248037
==== Epoch 24 ====
Epoch 24 (train) took 77.91s [processed 1578 samples/second]
Training Loss: 1.960611
Epoch 24 (valid) took 1.62s [processed 4924 samples/second]
Validation: Loss: 5.818700 | MRR: 0.247326
==== Epoch 25 ====
Epoch 25 (train) took 78.01s [processed 1576 samples/second]
Training Loss: 1.879736
Epoch 25 (valid) took 1.63s [processed 4893 samples/second]
Validation: Loss: 5.945212 | MRR: 0.253470
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-11-15-16_model_best.pkl.gz'.
==== Epoch 26 ====
Epoch 26 (train) took 78.14s [processed 1574 samples/second]
Training Loss: 1.815079
Epoch 26 (valid) took 1.64s [processed 4880 samples/second]
Validation: Loss: 5.986954 | MRR: 0.252485
==== Epoch 27 ====
Epoch 27 (train) took 77.91s [processed 1578 samples/second]
Training Loss: 1.757260
Epoch 27 (valid) took 1.64s [processed 4882 samples/second]
Validation: Loss: 6.025797 | MRR: 0.249986
==== Epoch 28 ====
Epoch 28 (train) took 77.98s [processed 1577 samples/second]
Training Loss: 1.710408
Epoch 28 (valid) took 1.64s [processed 4877 samples/second]
Validation: Loss: 6.122263 | MRR: 0.252509
==== Epoch 29 ====
Epoch 29 (train) took 77.99s [processed 1577 samples/second]
Training Loss: 1.656230
Epoch 29 (valid) took 1.65s [processed 4863 samples/second]
Validation: Loss: 6.226599 | MRR: 0.255565
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-23-11-15-16_model_best.pkl.gz'.
==== Epoch 30 ====
Epoch 30 (train) took 78.06s [processed 1575 samples/second]
Training Loss: 1.598889
Epoch 30 (valid) took 1.67s [processed 4788 samples/second]
Validation: Loss: 6.235690 | MRR: 0.253889
==== Epoch 31 ====
Epoch 31 (train) took 77.90s [processed 1578 samples/second]
Training Loss: 1.549763
Epoch 31 (valid) took 1.64s [processed 4892 samples/second]
Validation: Loss: 6.341252 | MRR: 0.253354
==== Epoch 32 ====
Epoch 32 (train) took 78.04s [processed 1576 samples/second]
Training Loss: 1.512762
Epoch 32 (valid) took 1.63s [processed 4901 samples/second]
Validation: Loss: 6.428675 | MRR: 0.250108
==== Epoch 33 ====
Epoch 33 (train) took 78.07s [processed 1575 samples/second]
Training Loss: 1.475208
Epoch 33 (valid) took 1.66s [processed 4820 samples/second]
Validation: Loss: 6.464026 | MRR: 0.254337
==== Epoch 34 ====
Epoch 34 (train) took 78.14s [processed 1574 samples/second]
Training Loss: 1.431017
Epoch 34 (valid) took 1.64s [processed 4876 samples/second]
Validation: Loss: 6.479836 | MRR: 0.252235
2020-04-23 12:06:44.769457: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-23 12:06:44.769516: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-23 12:06:44.769532: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-23 12:06:44.769545: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-23 12:06:44.769627: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.230
FuncNameTest-All MRR (bs=1,000): 0.051
Validation-All MRR (bs=1,000): 0.234
Test-javascript MRR (bs=1,000): 0.230
FuncNameTest-javascript MRR (bs=1,000): 0.051
Validation-javascript MRR (bs=1,000): 0.234

wandb: Waiting for W&B process to finish, PID 1052
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-mrr 0.7417553328847497
wandb: train-loss 1.4310169859630306
wandb: \_step 110
wandb: \_timestamp 1587643646.035694
wandb: \_runtime 3130.344263076782
wandb: val-time-sec 1.6405537128448486
wandb: train-time-sec 78.13866257667542
wandb: val-mrr 0.25223456382751464
wandb: epoch 34
wandb: val-loss 6.479835510253906
wandb: best_val_mrr_loss 6.226599037647247
wandb: best_val_mrr 0.25556498336791994
wandb: best_epoch 29
wandb: Test-All MRR (bs=1,000) 0.22981019790691393
wandb: FuncNameTest-All MRR (bs=1,000) 0.05093934802298782
wandb: Validation-All MRR (bs=1,000) 0.23396666746547617
wandb: Test-javascript MRR (bs=1,000) 0.22981019790691393
wandb: FuncNameTest-javascript MRR (bs=1,000) 0.05093934802298782
wandb: Validation-javascript MRR (bs=1,000) 0.23396666746547617
wandb: Syncing files in wandb/run-20200423_111516-o2qy00vv:
wandb: 1dcnn-2020-04-23-11-15-16-graph.pbtxt
wandb: 1dcnn-2020-04-23-11-15-16.train_log
wandb: 1dcnn-2020-04-23-11-15-16_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
Fetching run from W&B...
Fetching run files from W&B...
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/o2qy00vv
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./1dcnn-2020-04-23-11-15-16_model_best.pkl.gz
2020-04-23 12:14:51.470747: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-23 12:14:56.239037: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 42e4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-23 12:14:56.239084: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-23 12:14:56.516079: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-23 12:14:56.516134: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-23 12:14:56.516151: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-23 12:14:56.516262: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Evaluating language: javascript
100%|███████████████████████████████████████████████████████████████████████████████████| 1857835/1857835 [00:19<00:00, 97364.16it/s]1857835it [00:32, 57153.33it/s]
Uploading predictions to W&B
NDCG Average: 0.009823176

# RNN

root@jian-csn:/home/dev/src# python train.py --model rnn ../resources/saved_models ../resources/data/javascript/final/jsonl/train ../resources/data/javascript/final/jsonl/valid ../resources/data/javascript/final/jsonl/test
wandb: Started W&B process version 0.8.12 with PID 1257
wandb: Wandb version 0.8.32 is available! To upgrade, please run:
wandb: \$ pip install wandb --upgrade
wandb: Local directory: wandb/run-20200423_125326-fwfy3qaf
wandb: Syncing run rnn-2020-04-23-12-53-26: https://app.wandb.ai/jianguda/CodeSearchNet/runs/fwfy3qaf
wandb: Run `wandb off` to turn off syncing.

2020-04-23 12:53:31.724828: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-23 12:53:32.080257: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 42e4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-23 12:53:32.080304: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-23 12:53:32.355731: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-23 12:53:32.355787: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-23 12:53:32.355804: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-23 12:53:32.355914: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Tokenizing and building vocabulary for code snippets and queries. This step may take several hours.
Starting training run rnn-2020-04-23-12-53-26 of model RNNModel with following hypers:
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': True, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_rnn_num_layers': 2, 'code_rnn_hidden_dim': 64, 'code_rnn_cell_type': 'LSTM', 'code_rnn_is_bidirectional': True, 'code_rnn_dropout_keep_rate': 0.8, 'code_rnn_recurrent_dropout_keep_rate': 1.0, 'code_rnn_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_rnn_num_layers': 2, 'query_rnn_hidden_dim': 64, 'query_rnn_cell_type': 'LSTM', 'query_rnn_is_bidirectional': True, 'query_rnn_dropout_keep_rate': 0.8, 'query_rnn_recurrent_dropout_keep_rate': 1.0, 'query_rnn_pool_mode': 'weighted_mean', 'batch_size': 1000, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.01, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'softmax', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
Loading training and validation data.
Begin Training.
Training on 123889 javascript samples.
Validating on 8253 javascript samples.
==== Epoch 0 ====
Epoch 0 (train) took 133.94s [processed 918 samples/second]
Training Loss: 6.532283
Epoch 0 (valid) took 4.42s [processed 1811 samples/second]
Validation: Loss: 6.088582 | MRR: 0.051492
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-23-12-53-26_model_best.pkl.gz'.
==== Epoch 1 ====
Epoch 1 (train) took 131.98s [processed 931 samples/second]
Training Loss: 5.284877
Epoch 1 (valid) took 4.11s [processed 1944 samples/second]
Validation: Loss: 5.045206 | MRR: 0.173334
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-23-12-53-26_model_best.pkl.gz'.
==== Epoch 2 ====
Epoch 2 (train) took 132.08s [processed 931 samples/second]
Training Loss: 4.272918
Epoch 2 (valid) took 4.13s [processed 1935 samples/second]
Validation: Loss: 4.540858 | MRR: 0.252602
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-23-12-53-26_model_best.pkl.gz'.
==== Epoch 3 ====
Epoch 3 (train) took 132.11s [processed 931 samples/second]
Training Loss: 3.644676
Epoch 3 (valid) took 4.12s [processed 1943 samples/second]
Validation: Loss: 4.314516 | MRR: 0.296239
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-23-12-53-26_model_best.pkl.gz'.
==== Epoch 4 ====
Epoch 4 (train) took 131.96s [processed 932 samples/second]
Training Loss: 3.237185
Epoch 4 (valid) took 4.12s [processed 1943 samples/second]
Validation: Loss: 4.227133 | MRR: 0.313316
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-23-12-53-26_model_best.pkl.gz'.
==== Epoch 5 ====
Epoch 5 (train) took 132.07s [processed 931 samples/second]
Training Loss: 2.939514
Epoch 5 (valid) took 4.14s [processed 1934 samples/second]
Validation: Loss: 4.156968 | MRR: 0.333086
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-23-12-53-26_model_best.pkl.gz'.
==== Epoch 6 ====
Epoch 6 (train) took 132.08s [processed 931 samples/second]
Training Loss: 2.724303
Epoch 6 (valid) took 4.13s [processed 1937 samples/second]
Validation: Loss: 4.128531 | MRR: 0.337727
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-23-12-53-26_model_best.pkl.gz'.
==== Epoch 7 ====
Epoch 7 (train) took 132.13s [processed 930 samples/second]
Training Loss: 2.553450
Epoch 7 (valid) took 4.11s [processed 1947 samples/second]
Validation: Loss: 4.148622 | MRR: 0.345545
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-23-12-53-26_model_best.pkl.gz'.
==== Epoch 8 ====
Epoch 8 (train) took 131.98s [processed 931 samples/second]
Training Loss: 2.426270
Epoch 8 (valid) took 4.11s [processed 1945 samples/second]
Validation: Loss: 4.150024 | MRR: 0.348404
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-23-12-53-26_model_best.pkl.gz'.
==== Epoch 9 ====
Epoch 9 (train) took 131.96s [processed 932 samples/second]
Training Loss: 2.312955
Epoch 9 (valid) took 4.12s [processed 1941 samples/second]
Validation: Loss: 4.157904 | MRR: 0.348671
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-23-12-53-26_model_best.pkl.gz'.
==== Epoch 10 ====
Epoch 10 (train) took 131.87s [processed 932 samples/second]
Training Loss: 2.221908
Epoch 10 (valid) took 4.12s [processed 1940 samples/second]
Validation: Loss: 4.192291 | MRR: 0.350252
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-23-12-53-26_model_best.pkl.gz'.
==== Epoch 11 ====
Epoch 11 (train) took 131.95s [processed 932 samples/second]
Training Loss: 2.160076
Epoch 11 (valid) took 4.13s [processed 1938 samples/second]
Validation: Loss: 4.222370 | MRR: 0.348797
==== Epoch 12 ====
Epoch 12 (train) took 131.89s [processed 932 samples/second]
Training Loss: 2.086901
Epoch 12 (valid) took 4.12s [processed 1942 samples/second]
Validation: Loss: 4.232814 | MRR: 0.353332
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-23-12-53-26_model_best.pkl.gz'.
==== Epoch 13 ====
Epoch 13 (train) took 131.93s [processed 932 samples/second]
Training Loss: 2.023536
Epoch 13 (valid) took 4.12s [processed 1940 samples/second]
Validation: Loss: 4.286648 | MRR: 0.345685
==== Epoch 14 ====
Epoch 14 (train) took 131.92s [processed 932 samples/second]
Training Loss: 1.975643
Epoch 14 (valid) took 4.12s [processed 1941 samples/second]
Validation: Loss: 4.309800 | MRR: 0.349738
==== Epoch 15 ====
Epoch 15 (train) took 131.89s [processed 932 samples/second]
Training Loss: 1.936729
Epoch 15 (valid) took 4.11s [processed 1945 samples/second]
Validation: Loss: 4.311899 | MRR: 0.349728
==== Epoch 16 ====
Epoch 16 (train) took 131.89s [processed 932 samples/second]
Training Loss: 1.905408
Epoch 16 (valid) took 4.11s [processed 1945 samples/second]
Validation: Loss: 4.293936 | MRR: 0.353301
==== Epoch 17 ====
Epoch 17 (train) took 131.93s [processed 932 samples/second]
Training Loss: 1.868000
Epoch 17 (valid) took 4.15s [processed 1929 samples/second]
Validation: Loss: 4.369351 | MRR: 0.351131
2020-04-23 13:38:49.432743: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-23 13:38:49.432804: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-23 13:38:49.432815: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-23 13:38:49.432827: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-23 13:38:49.432924: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.369
FuncNameTest-All MRR (bs=1,000): 0.129
Validation-All MRR (bs=1,000): 0.366
Test-javascript MRR (bs=1,000): 0.369
FuncNameTest-javascript MRR (bs=1,000): 0.129
Validation-javascript MRR (bs=1,000): 0.366

wandb: Waiting for W&B process to finish, PID 1257
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_step 59
wandb: \_timestamp 1587649183.4661582
wandb: train-loss 1.8680000062880477
wandb: \_runtime 2778.0294713974
wandb: train-mrr 0.672340961735423
wandb: val-time-sec 4.146760702133179
wandb: val-loss 4.369350880384445
wandb: train-time-sec 131.9253442287445
wandb: val-mrr 0.35113095474243167
wandb: epoch 17
wandb: best_val_mrr_loss 4.232814341783524
wandb: best_val_mrr 0.35333225631713866
wandb: best_epoch 12
wandb: Test-All MRR (bs=1,000) 0.3685089757507097
wandb: FuncNameTest-All MRR (bs=1,000) 0.12888260483634953
wandb: Validation-All MRR (bs=1,000) 0.3663947796020599
wandb: Test-javascript MRR (bs=1,000) 0.3685089757507097
wandb: FuncNameTest-javascript MRR (bs=1,000) 0.12888260483634953
wandb: Validation-javascript MRR (bs=1,000) 0.3663947796020599
wandb: Syncing files in wandb/run-20200423_125326-fwfy3qaf:
wandb: rnn-2020-04-23-12-53-26-graph.pbtxt
wandb: rnn-2020-04-23-12-53-26.train_log
wandb: rnn-2020-04-23-12-53-26_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced rnn-2020-04-23-12-53-26: https://app.wandb.ai/jianguda/CodeSearchNet/runs/fwfy3qaf
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/fwfy3qaf
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./rnn-2020-04-23-12-53-26_model_best.pkl.gz
2020-04-23 13:54:04.706219: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-23 13:54:09.726428: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 42e4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-23 13:54:09.726475: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-23 13:54:10.006850: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-23 13:54:10.006912: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-23 13:54:10.006929: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-23 13:54:10.007032: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Evaluating language: javascript
100%|██████████████████████████████████████████████████████████████████████████████████| 1857835/1857835 [00:10<00:00, 177863.34it/s]1857835it [00:32, 57241.59it/s]
Uploading predictions to W&B
NDCG Average: 0.025343692

# BERT

root@jian-csn:/home/dev/src# python train.py --model selfatt ../resources/saved_models ../resources/data/javascript/final/jsonl/train ../resources/data/javascript/final/jsonl/valid ../resources/data/javascript/final/jsonl/test
wandb: Started W&B process version 0.8.12 with PID 1460
wandb: Wandb version 0.8.32 is available! To upgrade, please run:
wandb: \$ pip install wandb --upgrade
wandb: Local directory: wandb/run-20200423_151444-rnntsrkw
wandb: Syncing run selfatt-2020-04-23-15-14-44: https://app.wandb.ai/jianguda/CodeSearchNet/runs/rnntsrkw
wandb: Run `wandb off` to turn off syncing.

2020-04-23 15:14:48.885091: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-23 15:14:49.733464: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 42e4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-23 15:14:49.733508: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-23 15:14:50.012839: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-23 15:14:50.012896: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-23 15:14:50.012911: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-23 15:14:50.013022: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Tokenizing and building vocabulary for code snippets and queries. This step may take several hours.
Starting training run selfatt-2020-04-23-15-14-44 of model SelfAttentionModel with following hypers:
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': False, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_self_attention_activation': 'gelu', 'code_self_attention_hidden_size': 128, 'code_self_attention_intermediate_size': 512, 'code_self_attention_num_layers': 3, 'code_self_attention_num_heads': 8, 'code_self_attention_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_self_attention_activation': 'gelu', 'query_self_attention_hidden_size': 128, 'query_self_attention_intermediate_size': 512, 'query_self_attention_num_layers': 3, 'query_self_attention_num_heads': 8, 'query_self_attention_pool_mode': 'weighted_mean', 'batch_size': 450, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.0005, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'softmax', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
Loading training and validation data.
Begin Training.
Training on 123889 javascript samples.
Validating on 8253 javascript samples.
==== Epoch 0 ====
Epoch 0 (train) took 381.80s [processed 324 samples/second]
Training Loss: 4.676234
Epoch 0 (valid) took 10.94s [processed 740 samples/second]
Validation: Loss: 4.073948 | MRR: 0.307569
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-23-15-14-44_model_best.pkl.gz'.
==== Epoch 1 ====
Epoch 1 (train) took 382.45s [processed 323 samples/second]
Training Loss: 2.453642
Epoch 1 (valid) took 10.69s [processed 758 samples/second]
Validation: Loss: 3.632816 | MRR: 0.400290
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-23-15-14-44_model_best.pkl.gz'.
==== Epoch 2 ====
Epoch 2 (train) took 381.34s [processed 324 samples/second]
Training Loss: 1.571470
Epoch 2 (valid) took 10.45s [processed 775 samples/second]
Validation: Loss: 3.553188 | MRR: 0.430076
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-23-15-14-44_model_best.pkl.gz'.
==== Epoch 3 ====
Epoch 3 (train) took 381.28s [processed 324 samples/second]
Training Loss: 1.098106
Epoch 3 (valid) took 10.55s [processed 767 samples/second]
Validation: Loss: 3.690110 | MRR: 0.433115
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-23-15-14-44_model_best.pkl.gz'.
==== Epoch 4 ====
Epoch 4 (train) took 381.04s [processed 324 samples/second]
Training Loss: 0.817429
Epoch 4 (valid) took 10.44s [processed 775 samples/second]
Validation: Loss: 3.872365 | MRR: 0.431865
==== Epoch 5 ====
Epoch 5 (train) took 380.64s [processed 325 samples/second]
Training Loss: 0.648503
Epoch 5 (valid) took 10.43s [processed 776 samples/second]
Validation: Loss: 4.087818 | MRR: 0.430704
==== Epoch 6 ====
Epoch 6 (train) took 381.13s [processed 324 samples/second]
Training Loss: 0.535564
Epoch 6 (valid) took 10.57s [processed 766 samples/second]
Validation: Loss: 4.136629 | MRR: 0.428756
==== Epoch 7 ====
Epoch 7 (train) took 380.91s [processed 324 samples/second]
Training Loss: 0.458685
Epoch 7 (valid) took 10.55s [processed 767 samples/second]
Validation: Loss: 4.341251 | MRR: 0.425043
==== Epoch 8 ====
Epoch 8 (train) took 381.70s [processed 324 samples/second]
Training Loss: 0.403333
Epoch 8 (valid) took 10.55s [processed 767 samples/second]
Validation: Loss: 4.442404 | MRR: 0.422090
2020-04-23 16:17:51.935399: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-23 16:17:51.935457: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-23 16:17:51.935473: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-23 16:17:51.935484: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-23 16:17:51.935578: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.417
FuncNameTest-All MRR (bs=1,000): 0.113
Validation-All MRR (bs=1,000): 0.414
Test-javascript MRR (bs=1,000): 0.417
FuncNameTest-javascript MRR (bs=1,000): 0.113
Validation-javascript MRR (bs=1,000): 0.414

wandb: Waiting for W&B process to finish, PID 1460
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_step 41
wandb: train-loss 0.4033332296934995
wandb: train-mrr 0.9318099747474747
wandb: \_runtime 3860.9670586586
wandb: \_timestamp 1587658744.3886657
wandb: val-time-sec 10.554514408111572
wandb: val-loss 4.442403581407335
wandb: val-mrr 0.42208955270272713
wandb: train-time-sec 381.7013738155365
wandb: epoch 8
wandb: best_val_mrr_loss 3.6901098489761353
wandb: best_val_mrr 0.43311541145230514
wandb: best_epoch 3
wandb: Test-All MRR (bs=1,000) 0.41709353941651117
wandb: FuncNameTest-All MRR (bs=1,000) 0.11260266227365223
wandb: Validation-All MRR (bs=1,000) 0.41416733443207165
wandb: Test-javascript MRR (bs=1,000) 0.41709353941651117
wandb: FuncNameTest-javascript MRR (bs=1,000) 0.11260266227365223
wandb: Validation-javascript MRR (bs=1,000) 0.41416733443207165
wandb: Syncing files in wandb/run-20200423_151444-rnntsrkw:
wandb: selfatt-2020-04-23-15-14-44-graph.pbtxt
wandb: selfatt-2020-04-23-15-14-44.train_log
wandb: selfatt-2020-04-23-15-14-44_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced selfatt-2020-04-23-15-14-44: https://app.wandb.ai/jianguda/CodeSearchNet/runs/rnntsrkw
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/rnntsrkw
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./selfatt-2020-04-23-15-14-44_model_best.pkl.gz
2020-04-23 18:04:01.268662: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-23 18:04:06.079054: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 42e4:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-23 18:04:06.079110: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-23 18:04:06.360461: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-23 18:04:06.360524: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-23 18:04:06.360540: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-23 18:04:06.360650: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 42e4:00:00.0, compute capability: 3.7)
Evaluating language: javascript
100%|██████████████████████████████████████████████████████████████████████████████████| 1857835/1857835 [00:10<00:00, 178730.31it/s]1857835it [00:32, 57217.48it/s]
Uploading predictions to W&B
NDCG Average: 0.026145095
