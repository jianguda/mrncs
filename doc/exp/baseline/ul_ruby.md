python train.py --model neuralbow ../resources/saved_models ../resources/data/ruby/final/jsonl/train ../resources/data/ruby/final/jsonl/valid ../resources/data/ruby/final/jsonl/test

python predict.py -r jianguda/CodeSearchNet/0123456

# NBOW

root@jian-csn:/home/dev/src# python train.py --model neuralbow ../resources/saved_models ../resources/data/ruby/final/jsonl/train ../resources/data/ruby/final/jsonl/valid ../resources/data/ruby/final/jsonl/test
wandb: Started W&B process version 0.8.12 with PID 1481
wandb: Wandb version 0.8.33 is available! To upgrade, please run:
wandb: \$ pip install wandb --upgrade
wandb: Local directory: wandb/run-20200425_093922-csym74wu
wandb: Syncing run neuralbow-2020-04-25-09-39-22: https://app.wandb.ai/jianguda/CodeSearchNet/runs/csym74wu
wandb: Run `wandb off` to turn off syncing.

2020-04-25 09:39:27.997639: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-25 09:39:28.077706: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fbae:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-25 09:39:28.077755: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-25 09:39:28.355560: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-25 09:39:28.355629: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-25 09:39:28.355645: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-25 09:39:28.355758: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fbae:00:00.0, compute capability: 3.7)
Tokenizing and building vocabulary for code snippets and queries. This step may take several hours.
WARNING:tensorflow:From /home/dev/src/models/model.py:299: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Starting training run neuralbow-2020-04-25-09-39-22 of model NeuralBoWModel with following hypers:
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': False, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_nbow_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_nbow_pool_mode': 'weighted_mean', 'batch_size': 1000, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.01, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'cosine', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
Loading training and validation data.
Begin Training.
Training on 48791 ruby samples.
Validating on 2209 ruby samples.
==== Epoch 0 ====
Epoch 0 (train) took 4.92s [processed 9747 samples/second]
Training Loss: 1.029991
Epoch 0 (valid) took 0.17s [processed 11597 samples/second]
Validation: Loss: 1.007114 | MRR: 0.076647
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-25-09-39-22_model_best.pkl.gz'.
==== Epoch 1 ====
Epoch 1 (train) took 4.54s [processed 10578 samples/second]
Training Loss: 1.003564
Epoch 1 (valid) took 0.13s [processed 15813 samples/second]
Validation: Loss: 1.005268 | MRR: 0.136583
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-25-09-39-22_model_best.pkl.gz'.
==== Epoch 2 ====
Epoch 2 (train) took 4.58s [processed 10489 samples/second]
Training Loss: 0.999224
Epoch 2 (valid) took 0.13s [processed 15757 samples/second]
Validation: Loss: 1.007194 | MRR: 0.204185
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-25-09-39-22_model_best.pkl.gz'.
==== Epoch 3 ====
Epoch 3 (train) took 4.58s [processed 10472 samples/second]
Training Loss: 0.988066
Epoch 3 (valid) took 0.13s [processed 15081 samples/second]
Validation: Loss: 1.034760 | MRR: 0.244344
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-25-09-39-22_model_best.pkl.gz'.
==== Epoch 4 ====
Epoch 4 (train) took 4.60s [processed 10437 samples/second]
Training Loss: 0.958079
Epoch 4 (valid) took 0.13s [processed 15074 samples/second]
Validation: Loss: 1.077281 | MRR: 0.290427
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-25-09-39-22_model_best.pkl.gz'.
==== Epoch 5 ====
Epoch 5 (train) took 4.57s [processed 10510 samples/second]
Training Loss: 0.898285
Epoch 5 (valid) took 0.13s [processed 15513 samples/second]
Validation: Loss: 1.106834 | MRR: 0.325121
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-25-09-39-22_model_best.pkl.gz'.
==== Epoch 6 ====
Epoch 6 (train) took 4.59s [processed 10458 samples/second]
Training Loss: 0.849897
Epoch 6 (valid) took 0.13s [processed 15290 samples/second]
Validation: Loss: 1.106181 | MRR: 0.332177
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-25-09-39-22_model_best.pkl.gz'.
==== Epoch 7 ====
Epoch 7 (train) took 4.59s [processed 10450 samples/second]
Training Loss: 0.820557
Epoch 7 (valid) took 0.13s [processed 15169 samples/second]
Validation: Loss: 1.107057 | MRR: 0.325388
==== Epoch 8 ====
Epoch 8 (train) took 4.57s [processed 10512 samples/second]
Training Loss: 0.800728
Epoch 8 (valid) took 0.13s [processed 14949 samples/second]
Validation: Loss: 1.106633 | MRR: 0.333024
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-25-09-39-22_model_best.pkl.gz'.
==== Epoch 9 ====
Epoch 9 (train) took 4.59s [processed 10453 samples/second]
Training Loss: 0.787405
Epoch 9 (valid) took 0.13s [processed 15449 samples/second]
Validation: Loss: 1.106807 | MRR: 0.334451
Best result so far -- saved model as '../resources/saved_models/neuralbow-2020-04-25-09-39-22_model_best.pkl.gz'.
==== Epoch 10 ====
Epoch 10 (train) took 4.60s [processed 10442 samples/second]
Training Loss: 0.776875
Epoch 10 (valid) took 0.13s [processed 14844 samples/second]
Validation: Loss: 1.106732 | MRR: 0.329752
==== Epoch 11 ====
Epoch 11 (train) took 4.55s [processed 10544 samples/second]
Training Loss: 0.767458
Epoch 11 (valid) took 0.13s [processed 15045 samples/second]
Validation: Loss: 1.107420 | MRR: 0.331327
==== Epoch 12 ====
wandb: Network error resolved after 0:00:17.567933, resuming normal operation.ar: 0.7584. MRR so far: 0.9512
Epoch 12 (train) took 4.58s [processed 10479 samples/second]
Training Loss: 0.761184
Epoch 12 (valid) took 0.13s [processed 15494 samples/second]
Validation: Loss: 1.107177 | MRR: 0.327440
==== Epoch 13 ====
Epoch 13 (train) took 4.59s [processed 10452 samples/second]
Training Loss: 0.755417
Epoch 13 (valid) took 0.13s [processed 14974 samples/second]
Validation: Loss: 1.104693 | MRR: 0.331553
==== Epoch 14 ====
Epoch 14 (train) took 4.55s [processed 10550 samples/second]
Training Loss: 0.750164
Epoch 14 (valid) took 0.13s [processed 15658 samples/second]
Validation: Loss: 1.106024 | MRR: 0.329118
2020-04-25 09:42:13.194418: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-25 09:42:13.194485: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-25 09:42:13.194501: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-25 09:42:13.194513: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-25 09:42:13.194630: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fbae:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.321
FuncNameTest-All MRR (bs=1,000): 0.301
Validation-All MRR (bs=1,000): 0.354
Test-ruby MRR (bs=1,000): 0.321
FuncNameTest-ruby MRR (bs=1,000): 0.301
Validation-ruby MRR (bs=1,000): 0.354

wandb: Waiting for W&B process to finish, PID 1481
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-mrr 0.9537247428894043
wandb: \_step 35
wandb: \_runtime 180.3738398551941
wandb: \_timestamp 1587807741.58082
wandb: train-loss 0.7501635725299517
wandb: val-time-sec 0.12772345542907715
wandb: val-loss 1.1060237884521484
wandb: train-time-sec 4.549360990524292
wandb: val-mrr 0.3291182098388672
wandb: epoch 14
wandb: best_val_mrr_loss 1.1068071722984314
wandb: best_val_mrr 0.3344512786865234
wandb: best_epoch 9
wandb: Test-All MRR (bs=1,000) 0.3210054056677428
wandb: FuncNameTest-All MRR (bs=1,000) 0.3008278867279238
wandb: Validation-All MRR (bs=1,000) 0.3535395486634525
wandb: Test-ruby MRR (bs=1,000) 0.3210054056677428
wandb: FuncNameTest-ruby MRR (bs=1,000) 0.3008278867279238
wandb: Validation-ruby MRR (bs=1,000) 0.3535395486634525
wandb: Syncing files in wandb/run-20200425_093922-csym74wu:
wandb: neuralbow-2020-04-25-09-39-22-graph.pbtxt
wandb: neuralbow-2020-04-25-09-39-22.train_log
wandb: neuralbow-2020-04-25-09-39-22_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced neuralbow-2020-04-25-09-39-22: https://app.wandb.ai/jianguda/CodeSearchNet/runs/csym74wu
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/csym74wu
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./neuralbow-2020-04-25-09-39-22_model_best.pkl.gz
2020-04-25 09:45:19.851538: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-25 09:45:24.700143: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fbae:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-25 09:45:24.700193: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-25 09:45:24.983696: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-25 09:45:24.983769: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-25 09:45:24.983787: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-25 09:45:24.983922: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fbae:00:00.0, compute capability: 3.7)
WARNING:tensorflow:From /home/dev/src/models/model.py:299: calling norm (from tensorflow.python.ops.linalg_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
Evaluating language: ruby
100%|████████████████████████████████████████████████████████████████████████████████████| 164048/164048 [00:00<00:00, 198999.43it/s]164048it [00:02, 58918.51it/s]
Uploading predictions to W&B
NDCG Average: 0.129447305

# CNN

root@jian-csn:/home/dev/src# python train.py --model 1dcnn ../resources/saved_models ../resources/data/ruby/final/jsonl/train ../resources/data/ruby/final/jsonl/valid ../resources/data/ruby/final/jsonl/test
wandb: Started W&B process version 0.8.12 with PID 1273
wandb: Wandb version 0.8.33 is available! To upgrade, please run:
wandb: \$ pip install wandb --upgrade
wandb: Local directory: wandb/run-20200425_085920-pd7zutcw
wandb: Syncing run 1dcnn-2020-04-25-08-59-20: https://app.wandb.ai/jianguda/CodeSearchNet/runs/pd7zutcw
wandb: Run `wandb off` to turn off syncing.

2020-04-25 08:59:26.437398: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-25 08:59:26.519354: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fbae:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-25 08:59:26.519403: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-25 08:59:26.798676: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-25 08:59:26.798746: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-25 08:59:26.798764: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-25 08:59:26.798884: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fbae:00:00.0, compute capability: 3.7)
Tokenizing and building vocabulary for code snippets and queries. This step may take several hours.
Starting training run 1dcnn-2020-04-25-08-59-20 of model ConvolutionalModel with following hypers:
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': False, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_1dcnn_position_encoding': 'learned', 'code_1dcnn_layer_list': [128, 128, 128], 'code_1dcnn_kernel_width': [16, 16, 16], 'code_1dcnn_add_residual_connections': True, 'code_1dcnn_activation': 'tanh', 'code_1dcnn_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_1dcnn_position_encoding': 'learned', 'query_1dcnn_layer_list': [128, 128, 128], 'query_1dcnn_kernel_width': [16, 16, 16], 'query_1dcnn_add_residual_connections': True, 'query_1dcnn_activation': 'tanh', 'query_1dcnn_pool_mode': 'weighted_mean', 'batch_size': 1000, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.0005, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'softmax', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
Loading training and validation data.
Begin Training.
Training on 48791 ruby samples.
Validating on 2209 ruby samples.
==== Epoch 0 ====
Epoch 0 (train) took 30.95s [processed 1551 samples/second]
Training Loss: 6.726075
Epoch 0 (valid) took 0.50s [processed 4023 samples/second]
Validation: Loss: 6.838372 | MRR: 0.010934
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 1 ====
Epoch 1 (train) took 29.35s [processed 1635 samples/second]
Training Loss: 6.415437
Epoch 1 (valid) took 0.43s [processed 4670 samples/second]
Validation: Loss: 6.693928 | MRR: 0.016185
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 2 ====
Epoch 2 (train) took 29.38s [processed 1633 samples/second]
Training Loss: 6.149328
Epoch 2 (valid) took 0.41s [processed 4927 samples/second]
Validation: Loss: 6.603335 | MRR: 0.019112
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 3 ====
Epoch 3 (train) took 29.41s [processed 1632 samples/second]
Training Loss: 5.932328
Epoch 3 (valid) took 0.42s [processed 4802 samples/second]
Validation: Loss: 6.513342 | MRR: 0.026499
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 4 ====
Epoch 4 (train) took 29.45s [processed 1629 samples/second]
Training Loss: 5.718227
Epoch 4 (valid) took 0.40s [processed 4977 samples/second]
Validation: Loss: 6.514266 | MRR: 0.026658
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 5 ====
Epoch 5 (train) took 29.53s [processed 1625 samples/second]
Training Loss: 5.575029
Epoch 5 (valid) took 0.42s [processed 4808 samples/second]
Validation: Loss: 6.459483 | MRR: 0.032563
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 6 ====
Epoch 6 (train) took 29.55s [processed 1624 samples/second]
Training Loss: 5.406704
Epoch 6 (valid) took 0.42s [processed 4773 samples/second]
Validation: Loss: 6.462878 | MRR: 0.037360
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 7 ====
Epoch 7 (train) took 29.59s [processed 1621 samples/second]
Training Loss: 5.252764
Epoch 7 (valid) took 0.40s [processed 4977 samples/second]
Validation: Loss: 6.453270 | MRR: 0.040564
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 8 ====
Epoch 8 (train) took 29.65s [processed 1618 samples/second]
Training Loss: 5.096955
Epoch 8 (valid) took 0.42s [processed 4815 samples/second]
Validation: Loss: 6.446691 | MRR: 0.047052
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 9 ====
Epoch 9 (train) took 29.75s [processed 1613 samples/second]
Training Loss: 4.959157
Epoch 9 (valid) took 0.42s [processed 4791 samples/second]
Validation: Loss: 6.442641 | MRR: 0.050359
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 10 ====
Epoch 10 (train) took 29.69s [processed 1616 samples/second]
Training Loss: 4.821751
Epoch 10 (valid) took 0.42s [processed 4755 samples/second]
Validation: Loss: 6.381736 | MRR: 0.053741
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 11 ====
Epoch 11 (train) took 29.68s [processed 1617 samples/second]
Training Loss: 4.618516
Epoch 11 (valid) took 0.40s [processed 4939 samples/second]
Validation: Loss: 6.428420 | MRR: 0.060216
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 12 ====
Epoch 12 (train) took 29.77s [processed 1612 samples/second]
Training Loss: 4.473898
Epoch 12 (valid) took 0.41s [processed 4837 samples/second]
Validation: Loss: 6.495464 | MRR: 0.062447
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 13 ====
Epoch 13 (train) took 29.76s [processed 1613 samples/second]
Training Loss: 4.309914
Epoch 13 (valid) took 0.42s [processed 4763 samples/second]
Validation: Loss: 6.537596 | MRR: 0.062121
==== Epoch 14 ====
Epoch 14 (train) took 29.80s [processed 1610 samples/second]
Training Loss: 4.168978
Epoch 14 (valid) took 0.41s [processed 4839 samples/second]
Validation: Loss: 6.565465 | MRR: 0.073020
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 15 ====
Epoch 15 (train) took 29.85s [processed 1607 samples/second]
Training Loss: 4.053414
Epoch 15 (valid) took 0.42s [processed 4727 samples/second]
Validation: Loss: 6.688949 | MRR: 0.071926
==== Epoch 16 ====
Epoch 16 (train) took 29.87s [processed 1607 samples/second]
Training Loss: 3.905768
Epoch 16 (valid) took 0.43s [processed 4666 samples/second]
Validation: Loss: 6.775934 | MRR: 0.074832
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 17 ====
Epoch 17 (train) took 29.86s [processed 1607 samples/second]
Training Loss: 3.774059
Epoch 17 (valid) took 0.42s [processed 4766 samples/second]
Validation: Loss: 6.830378 | MRR: 0.080177
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 18 ====
Epoch 18 (train) took 29.92s [processed 1604 samples/second]
Training Loss: 3.640647
Epoch 18 (valid) took 0.42s [processed 4775 samples/second]
Validation: Loss: 6.859082 | MRR: 0.084535
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 19 ====
Epoch 19 (train) took 29.86s [processed 1607 samples/second]
Training Loss: 3.493481
Epoch 19 (valid) took 0.43s [processed 4690 samples/second]
Validation: Loss: 6.924100 | MRR: 0.088054
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 20 ====
Epoch 20 (train) took 29.87s [processed 1607 samples/second]
Training Loss: 3.358816
Epoch 20 (valid) took 0.41s [processed 4850 samples/second]
Validation: Loss: 7.068056 | MRR: 0.094276
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 21 ====
Epoch 21 (train) took 29.83s [processed 1608 samples/second]
Training Loss: 3.246293
Epoch 21 (valid) took 0.41s [processed 4933 samples/second]
Validation: Loss: 7.282237 | MRR: 0.097670
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 22 ====
Epoch 22 (train) took 29.87s [processed 1606 samples/second]
Training Loss: 3.112498
Epoch 22 (valid) took 0.42s [processed 4796 samples/second]
Validation: Loss: 7.247936 | MRR: 0.092643
==== Epoch 23 ====
Epoch 23 (train) took 29.84s [processed 1608 samples/second]
Training Loss: 3.008053
Epoch 23 (valid) took 0.41s [processed 4884 samples/second]
Validation: Loss: 7.402079 | MRR: 0.101221
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 24 ====
Epoch 24 (train) took 29.93s [processed 1603 samples/second]
Training Loss: 2.910231
Epoch 24 (valid) took 0.41s [processed 4882 samples/second]
Validation: Loss: 7.459923 | MRR: 0.102466
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 25 ====
Epoch 25 (train) took 29.82s [processed 1609 samples/second]
Training Loss: 2.831823
Epoch 25 (valid) took 0.42s [processed 4743 samples/second]
Validation: Loss: 7.719288 | MRR: 0.104471
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 26 ====
Epoch 26 (train) took 29.91s [processed 1605 samples/second]
Training Loss: 2.731586
Epoch 26 (valid) took 0.42s [processed 4788 samples/second]
Validation: Loss: 7.794290 | MRR: 0.107904
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 27 ====
Epoch 27 (train) took 29.87s [processed 1607 samples/second]
Training Loss: 2.651340
Epoch 27 (valid) took 0.40s [processed 4959 samples/second]
Validation: Loss: 7.944121 | MRR: 0.101304
==== Epoch 28 ====
Epoch 28 (train) took 29.85s [processed 1608 samples/second]
Training Loss: 2.547943
Epoch 28 (valid) took 0.41s [processed 4855 samples/second]
Validation: Loss: 8.060939 | MRR: 0.104970
==== Epoch 29 ====
Epoch 29 (train) took 29.85s [processed 1607 samples/second]
Training Loss: 2.478790
Epoch 29 (valid) took 0.40s [processed 4957 samples/second]
Validation: Loss: 8.220294 | MRR: 0.104694
==== Epoch 30 ====
Epoch 30 (train) took 29.88s [processed 1606 samples/second]
Training Loss: 2.390168
Epoch 30 (valid) took 0.41s [processed 4848 samples/second]
Validation: Loss: 8.295368 | MRR: 0.109407
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 31 ====
Epoch 31 (train) took 29.93s [processed 1603 samples/second]
Training Loss: 2.337081
Epoch 31 (valid) took 0.41s [processed 4864 samples/second]
Validation: Loss: 8.291552 | MRR: 0.112177
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 32 ====
Epoch 32 (train) took 29.88s [processed 1606 samples/second]
Training Loss: 2.265881
Epoch 32 (valid) took 0.41s [processed 4903 samples/second]
Validation: Loss: 8.590224 | MRR: 0.115690
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 33 ====
Epoch 33 (train) took 29.90s [processed 1605 samples/second]
Training Loss: 2.179763
Epoch 33 (valid) took 0.41s [processed 4887 samples/second]
Validation: Loss: 8.531707 | MRR: 0.117916
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 34 ====
Epoch 34 (train) took 29.83s [processed 1609 samples/second]
Training Loss: 2.109611
Epoch 34 (valid) took 0.42s [processed 4737 samples/second]
Validation: Loss: 8.710139 | MRR: 0.120414
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 35 ====
Epoch 35 (train) took 29.79s [processed 1611 samples/second]
Training Loss: 2.058121
Epoch 35 (valid) took 0.42s [processed 4729 samples/second]
Validation: Loss: 8.751064 | MRR: 0.122179
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 36 ====
Epoch 36 (train) took 29.79s [processed 1611 samples/second]
Training Loss: 1.991883
Epoch 36 (valid) took 0.42s [processed 4812 samples/second]
Validation: Loss: 8.834703 | MRR: 0.122503
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 37 ====
Epoch 37 (train) took 29.85s [processed 1607 samples/second]
Training Loss: 1.928599
Epoch 37 (valid) took 0.41s [processed 4820 samples/second]
Validation: Loss: 8.988107 | MRR: 0.122730
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 38 ====
Epoch 38 (train) took 29.86s [processed 1607 samples/second]
Training Loss: 1.873780
Epoch 38 (valid) took 0.42s [processed 4765 samples/second]
Validation: Loss: 9.056296 | MRR: 0.125138
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 39 ====
Epoch 39 (train) took 29.82s [processed 1609 samples/second]
Training Loss: 1.834929
Epoch 39 (valid) took 0.42s [processed 4767 samples/second]
Validation: Loss: 9.139081 | MRR: 0.123300
==== Epoch 40 ====
Epoch 40 (train) took 29.82s [processed 1609 samples/second]
Training Loss: 1.773483
Epoch 40 (valid) took 0.41s [processed 4856 samples/second]
Validation: Loss: 9.194609 | MRR: 0.120965
==== Epoch 41 ====
Epoch 41 (train) took 30.01s [processed 1599 samples/second]
Training Loss: 1.728069
Epoch 41 (valid) took 0.42s [processed 4797 samples/second]
Validation: Loss: 9.162970 | MRR: 0.123880
==== Epoch 42 ====
Epoch 42 (train) took 29.95s [processed 1602 samples/second]
Training Loss: 1.688042
Epoch 42 (valid) took 0.42s [processed 4720 samples/second]
Validation: Loss: 9.293426 | MRR: 0.125985
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 43 ====
Epoch 43 (train) took 29.90s [processed 1605 samples/second]
Training Loss: 1.637239
Epoch 43 (valid) took 0.41s [processed 4829 samples/second]
Validation: Loss: 9.482204 | MRR: 0.121706
==== Epoch 44 ====
Epoch 44 (train) took 29.81s [processed 1610 samples/second]
Training Loss: 1.600069
Epoch 44 (valid) took 0.42s [processed 4713 samples/second]
Validation: Loss: 9.530779 | MRR: 0.124652
==== Epoch 45 ====
Epoch 45 (train) took 29.84s [processed 1608 samples/second]
Training Loss: 1.528621
Epoch 45 (valid) took 0.42s [processed 4732 samples/second]
Validation: Loss: 9.777125 | MRR: 0.126816
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 46 ====
Epoch 46 (train) took 29.87s [processed 1607 samples/second]
Training Loss: 1.520127
Epoch 46 (valid) took 0.42s [processed 4773 samples/second]
Validation: Loss: 9.615557 | MRR: 0.130300
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 47 ====
Epoch 47 (train) took 29.89s [processed 1605 samples/second]
Training Loss: 1.462104
Epoch 47 (valid) took 0.41s [processed 4854 samples/second]
Validation: Loss: 9.801299 | MRR: 0.136234
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 48 ====
Epoch 48 (train) took 29.82s [processed 1609 samples/second]
Training Loss: 1.422998
Epoch 48 (valid) took 0.41s [processed 4902 samples/second]
Validation: Loss: 9.782894 | MRR: 0.137975
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 49 ====
Epoch 49 (train) took 29.87s [processed 1607 samples/second]
Training Loss: 1.391230
Epoch 49 (valid) took 0.41s [processed 4882 samples/second]
Validation: Loss: 9.815516 | MRR: 0.135232
==== Epoch 50 ====
Epoch 50 (train) took 29.86s [processed 1607 samples/second]
Training Loss: 1.345164
Epoch 50 (valid) took 0.41s [processed 4904 samples/second]
Validation: Loss: 10.104353 | MRR: 0.136744
==== Epoch 51 ====
Epoch 51 (train) took 29.87s [processed 1606 samples/second]
Training Loss: 1.313980
Epoch 51 (valid) took 0.41s [processed 4903 samples/second]
Validation: Loss: 10.172393 | MRR: 0.139389
Best result so far -- saved model as '../resources/saved_models/1dcnn-2020-04-25-08-59-20_model_best.pkl.gz'.
==== Epoch 52 ====
Epoch 52 (train) took 29.94s [processed 1603 samples/second]
Training Loss: 1.281119
Epoch 52 (valid) took 0.42s [processed 4802 samples/second]
Validation: Loss: 10.175857 | MRR: 0.133629
==== Epoch 53 ====
Epoch 53 (train) took 29.85s [processed 1607 samples/second]
Training Loss: 1.248108
Epoch 53 (valid) took 0.42s [processed 4719 samples/second]
Validation: Loss: 10.228742 | MRR: 0.138702
==== Epoch 54 ====
Epoch 54 (train) took 29.88s [processed 1606 samples/second]
Training Loss: 1.219873
Epoch 54 (valid) took 0.42s [processed 4751 samples/second]
Validation: Loss: 10.284327 | MRR: 0.134649
==== Epoch 55 ====
Epoch 55 (train) took 29.88s [processed 1606 samples/second]
Training Loss: 1.201789
Epoch 55 (valid) took 0.41s [processed 4881 samples/second]
Validation: Loss: 10.225861 | MRR: 0.137300
==== Epoch 56 ====
Epoch 56 (train) took 29.91s [processed 1604 samples/second]
Training Loss: 1.152978
Epoch 56 (valid) took 0.42s [processed 4797 samples/second]
Validation: Loss: 10.332220 | MRR: 0.134808
2020-04-25 09:31:09.579787: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-25 09:31:09.579847: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-25 09:31:09.579863: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-25 09:31:09.579874: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-25 09:31:09.579981: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fbae:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.117
FuncNameTest-All MRR (bs=1,000): 0.151
Validation-All MRR (bs=1,000): 0.117
Test-ruby MRR (bs=1,000): 0.117
FuncNameTest-ruby MRR (bs=1,000): 0.151
Validation-ruby MRR (bs=1,000): 0.117

wandb: Waiting for W&B process to finish, PID 1273
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_timestamp 1587807079.7655425
wandb: train-loss 1.1529782563447952
wandb: \_runtime 1920.1446933746338
wandb: train-mrr 0.790405881245931
wandb: \_step 119
wandb: epoch 56
wandb: val-loss 10.332220077514648
wandb: val-time-sec 0.4168815612792969
wandb: train-time-sec 29.912850856781006
wandb: val-mrr 0.1348078384399414
wandb: best_val_mrr_loss 10.172393321990967
wandb: best_val_mrr 0.1393888168334961
wandb: best_epoch 51
wandb: Test-All MRR (bs=1,000) 0.11652631478199921
wandb: FuncNameTest-All MRR (bs=1,000) 0.1508902292559331
wandb: Validation-All MRR (bs=1,000) 0.11696748565738571
wandb: Test-ruby MRR (bs=1,000) 0.11652631478199921
wandb: FuncNameTest-ruby MRR (bs=1,000) 0.1508902292559331
wandb: Validation-ruby MRR (bs=1,000) 0.11696748565738571
wandb: Syncing files in wandb/run-20200425_085920-pd7zutcw:
wandb: 1dcnn-2020-04-25-08-59-20-graph.pbtxt
wandb: 1dcnn-2020-04-25-08-59-20.train_log
wandb: 1dcnn-2020-04-25-08-59-20_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced 1dcnn-2020-04-25-08-59-20: https://app.wandb.ai/jianguda/CodeSearchNet/runs/pd7zutcw
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/pd7zutcw
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./1dcnn-2020-04-25-08-59-20_model_best.pkl.gz
2020-04-25 09:35:04.116162: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-25 09:35:08.985148: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fbae:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-25 09:35:08.985200: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-25 09:35:09.264874: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-25 09:35:09.264946: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-25 09:35:09.264966: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-25 09:35:09.265098: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fbae:00:00.0, compute capability: 3.7)
Evaluating language: ruby
100%|████████████████████████████████████████████████████████████████████████████████████| 164048/164048 [00:00<00:00, 200047.39it/s]164048it [00:02, 59386.32it/s]
Uploading predictions to W&B
NDCG Average: 0.039529858

# RNN

root@jian-csn:/home/dev/src# python train.py --model rnn ../resources/saved_models ../resources/data/ruby/final/jsonl/train ../resources/data/ruby/final/jsonl/valid ../resources/data/ruby/final/jsonl/test
wandb: Started W&B process version 0.8.12 with PID 1062
wandb: Wandb version 0.8.33 is available! To upgrade, please run:
wandb: \$ pip install wandb --upgrade
wandb: Local directory: wandb/run-20200425_082143-y3anhd89
wandb: Syncing run rnn-2020-04-25-08-21-43: https://app.wandb.ai/jianguda/CodeSearchNet/runs/y3anhd89
wandb: Run `wandb off` to turn off syncing.

2020-04-25 08:21:48.982084: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-25 08:21:49.062665: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fbae:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-25 08:21:49.062715: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-25 08:21:49.342071: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-25 08:21:49.342141: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-25 08:21:49.342157: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-25 08:21:49.342276: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fbae:00:00.0, compute capability: 3.7)
Tokenizing and building vocabulary for code snippets and queries. This step may take several hours.
Starting training run rnn-2020-04-25-08-21-43 of model RNNModel with following hypers:
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': True, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_rnn_num_layers': 2, 'code_rnn_hidden_dim': 64, 'code_rnn_cell_type': 'LSTM', 'code_rnn_is_bidirectional': True, 'code_rnn_dropout_keep_rate': 0.8, 'code_rnn_recurrent_dropout_keep_rate': 1.0, 'code_rnn_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_rnn_num_layers': 2, 'query_rnn_hidden_dim': 64, 'query_rnn_cell_type': 'LSTM', 'query_rnn_is_bidirectional': True, 'query_rnn_dropout_keep_rate': 0.8, 'query_rnn_recurrent_dropout_keep_rate': 1.0, 'query_rnn_pool_mode': 'weighted_mean', 'batch_size': 1000, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.01, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'softmax', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
Loading training and validation data.
Begin Training.
Training on 48791 ruby samples.
Validating on 2209 ruby samples.
==== Epoch 0 ====
Epoch 0 (train) took 53.49s [processed 897 samples/second]
Training Loss: 6.595758
Epoch 0 (valid) took 1.32s [processed 1509 samples/second]
Validation: Loss: 6.696951 | MRR: 0.012947
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-25-08-21-43_model_best.pkl.gz'.
==== Epoch 1 ====
Epoch 1 (train) took 51.68s [processed 928 samples/second]
Training Loss: 6.022021
Epoch 1 (valid) took 1.03s [processed 1940 samples/second]
Validation: Loss: 6.362419 | MRR: 0.036473
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-25-08-21-43_model_best.pkl.gz'.
==== Epoch 2 ====
Epoch 2 (train) took 51.49s [processed 932 samples/second]
Training Loss: 5.339310
Epoch 2 (valid) took 1.04s [processed 1931 samples/second]
Validation: Loss: 5.943691 | MRR: 0.073437
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-25-08-21-43_model_best.pkl.gz'.
==== Epoch 3 ====
Epoch 3 (train) took 51.68s [processed 928 samples/second]
Training Loss: 4.636522
Epoch 3 (valid) took 1.04s [processed 1921 samples/second]
Validation: Loss: 5.565829 | MRR: 0.118089
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-25-08-21-43_model_best.pkl.gz'.
==== Epoch 4 ====
Epoch 4 (train) took 51.52s [processed 931 samples/second]
Training Loss: 3.982798
Epoch 4 (valid) took 1.03s [processed 1939 samples/second]
Validation: Loss: 5.369571 | MRR: 0.157176
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-25-08-21-43_model_best.pkl.gz'.
==== Epoch 5 ====
Epoch 5 (train) took 51.72s [processed 928 samples/second]
Training Loss: 3.445299
Epoch 5 (valid) took 1.03s [processed 1936 samples/second]
Validation: Loss: 5.197875 | MRR: 0.194046
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-25-08-21-43_model_best.pkl.gz'.
==== Epoch 6 ====
Epoch 6 (train) took 51.57s [processed 930 samples/second]
Training Loss: 3.033279
Epoch 6 (valid) took 1.04s [processed 1929 samples/second]
Validation: Loss: 5.123431 | MRR: 0.215224
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-25-08-21-43_model_best.pkl.gz'.
==== Epoch 7 ====
Epoch 7 (train) took 51.72s [processed 928 samples/second]
Training Loss: 2.693869
Epoch 7 (valid) took 1.04s [processed 1929 samples/second]
Validation: Loss: 5.203980 | MRR: 0.224877
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-25-08-21-43_model_best.pkl.gz'.
==== Epoch 8 ====
Epoch 8 (train) took 51.54s [processed 931 samples/second]
Training Loss: 2.398006
Epoch 8 (valid) took 1.03s [processed 1940 samples/second]
Validation: Loss: 5.209411 | MRR: 0.223185
==== Epoch 9 ====
Epoch 9 (train) took 51.72s [processed 928 samples/second]
Training Loss: 2.181608
Epoch 9 (valid) took 1.03s [processed 1935 samples/second]
Validation: Loss: 5.242259 | MRR: 0.237201
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-25-08-21-43_model_best.pkl.gz'.
==== Epoch 10 ====
Epoch 10 (train) took 51.52s [processed 931 samples/second]
Training Loss: 1.966747
Epoch 10 (valid) took 1.03s [processed 1947 samples/second]
Validation: Loss: 5.384676 | MRR: 0.240982
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-25-08-21-43_model_best.pkl.gz'.
==== Epoch 11 ====
Epoch 11 (train) took 51.68s [processed 928 samples/second]
Training Loss: 1.788478
Epoch 11 (valid) took 1.03s [processed 1938 samples/second]
Validation: Loss: 5.340765 | MRR: 0.254778
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-25-08-21-43_model_best.pkl.gz'.
==== Epoch 12 ====
Epoch 12 (train) took 51.51s [processed 931 samples/second]
Training Loss: 1.650575
Epoch 12 (valid) took 1.04s [processed 1928 samples/second]
Validation: Loss: 5.481176 | MRR: 0.249502
==== Epoch 13 ====
Epoch 13 (train) took 51.67s [processed 928 samples/second]
Training Loss: 1.527207
Epoch 13 (valid) took 1.03s [processed 1940 samples/second]
Validation: Loss: 5.584269 | MRR: 0.252956
==== Epoch 14 ====
Epoch 14 (train) took 51.52s [processed 931 samples/second]
Training Loss: 1.424841
Epoch 14 (valid) took 1.03s [processed 1942 samples/second]
Validation: Loss: 5.638798 | MRR: 0.257153
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-25-08-21-43_model_best.pkl.gz'.
==== Epoch 15 ====
Epoch 15 (train) took 51.61s [processed 929 samples/second]
Training Loss: 1.330373
Epoch 15 (valid) took 1.04s [processed 1917 samples/second]
Validation: Loss: 5.660293 | MRR: 0.260757
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-25-08-21-43_model_best.pkl.gz'.
==== Epoch 16 ====
Epoch 16 (train) took 51.45s [processed 933 samples/second]
Training Loss: 1.253612
Epoch 16 (valid) took 1.04s [processed 1929 samples/second]
Validation: Loss: 5.776386 | MRR: 0.259184
==== Epoch 17 ====
Epoch 17 (train) took 51.65s [processed 929 samples/second]
Training Loss: 1.194502
Epoch 17 (valid) took 1.03s [processed 1940 samples/second]
Validation: Loss: 5.808317 | MRR: 0.258833
==== Epoch 18 ====
Epoch 18 (train) took 51.47s [processed 932 samples/second]
Training Loss: 1.129615
Epoch 18 (valid) took 1.03s [processed 1935 samples/second]
Validation: Loss: 5.756117 | MRR: 0.263298
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-25-08-21-43_model_best.pkl.gz'.
==== Epoch 19 ====
Epoch 19 (train) took 51.69s [processed 928 samples/second]
Training Loss: 1.073149
Epoch 19 (valid) took 1.04s [processed 1929 samples/second]
Validation: Loss: 6.104710 | MRR: 0.256955
==== Epoch 20 ====
Epoch 20 (train) took 51.49s [processed 932 samples/second]
Training Loss: 1.022618
Epoch 20 (valid) took 1.03s [processed 1938 samples/second]
Validation: Loss: 6.029381 | MRR: 0.259372
==== Epoch 21 ====
Epoch 21 (train) took 51.61s [processed 930 samples/second]
Training Loss: 0.981010
Epoch 21 (valid) took 1.03s [processed 1939 samples/second]
Validation: Loss: 5.924418 | MRR: 0.269697
Best result so far -- saved model as '../resources/saved_models/rnn-2020-04-25-08-21-43_model_best.pkl.gz'.
==== Epoch 22 ====
Epoch 22 (train) took 51.41s [processed 933 samples/second]
Training Loss: 0.943145
Epoch 22 (valid) took 1.03s [processed 1949 samples/second]
Validation: Loss: 6.102213 | MRR: 0.258836
==== Epoch 23 ====
Epoch 23 (train) took 51.59s [processed 930 samples/second]
Training Loss: 0.915597
Epoch 23 (valid) took 1.03s [processed 1933 samples/second]
Validation: Loss: 6.054564 | MRR: 0.262293
==== Epoch 24 ====
Epoch 24 (train) took 51.38s [processed 934 samples/second]
Training Loss: 0.904264
Epoch 24 (valid) took 1.03s [processed 1939 samples/second]
Validation: Loss: 6.180969 | MRR: 0.257403
==== Epoch 25 ====
Epoch 25 (train) took 51.60s [processed 930 samples/second]
Training Loss: 0.876520
Epoch 25 (valid) took 1.04s [processed 1930 samples/second]
Validation: Loss: 6.249434 | MRR: 0.260129
==== Epoch 26 ====
Epoch 26 (train) took 51.38s [processed 934 samples/second]
Training Loss: 0.847172
Epoch 26 (valid) took 1.03s [processed 1940 samples/second]
Validation: Loss: 6.230485 | MRR: 0.265289
2020-04-25 08:47:26.187615: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-25 08:47:26.187680: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-25 08:47:26.187695: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-25 08:47:26.187710: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-25 08:47:26.187811: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fbae:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.223
FuncNameTest-All MRR (bs=1,000): 0.387
Validation-All MRR (bs=1,000): 0.261
Test-ruby MRR (bs=1,000): 0.223
FuncNameTest-ruby MRR (bs=1,000): 0.387
Validation-ruby MRR (bs=1,000): 0.261

wandb: Waiting for W&B process to finish, PID 1062
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-loss 0.8471718666454157
wandb: \_runtime 1559.0122203826904
wandb: train-mrr 0.8485706520080566
wandb: \_step 59
wandb: \_timestamp 1587804461.1470537
wandb: train-time-sec 51.378010749816895
wandb: val-time-sec 1.0305602550506592
wandb: val-mrr 0.2652894592285156
wandb: val-loss 6.230484962463379
wandb: epoch 26
wandb: best_val_mrr_loss 5.924418210983276
wandb: best_val_mrr 0.26969671630859376
wandb: best_epoch 21
wandb: Test-All MRR (bs=1,000) 0.2226205764721421
wandb: FuncNameTest-All MRR (bs=1,000) 0.38668080925628634
wandb: Validation-All MRR (bs=1,000) 0.2605755959421592
wandb: Test-ruby MRR (bs=1,000) 0.2226205764721421
wandb: FuncNameTest-ruby MRR (bs=1,000) 0.38668080925628634
wandb: Validation-ruby MRR (bs=1,000) 0.2605755959421592
wandb: Syncing files in wandb/run-20200425_082143-y3anhd89:
wandb: rnn-2020-04-25-08-21-43-graph.pbtxt
wandb: rnn-2020-04-25-08-21-43.train_log
wandb: rnn-2020-04-25-08-21-43_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced rnn-2020-04-25-08-21-43: https://app.wandb.ai/jianguda/CodeSearchNet/runs/y3anhd89
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/y3anhd89
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./rnn-2020-04-25-08-21-43_model_best.pkl.gz
2020-04-25 08:48:17.595620: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-25 08:48:22.404621: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fbae:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-25 08:48:22.404673: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-25 08:48:22.694765: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-25 08:48:22.694837: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-25 08:48:22.694853: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-25 08:48:22.694976: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fbae:00:00.0, compute capability: 3.7)
Evaluating language: ruby
100%|████████████████████████████████████████████████████████████████████████████████████| 164048/164048 [00:00<00:00, 194222.06it/s]164048it [00:02, 58981.85it/s]
Uploading predictions to W&B
NDCG Average: 0.055163338

# BERT

root@jian-csn:/home/dev/src# python train.py --model selfatt ../resources/saved_models ../resources/data/ruby/final/jsonl/train ../resources/data/ruby/final/jsonl/valid ../resources/data/ruby/final/jsonl/test
wandb: Started W&B process version 0.8.12 with PID 851
wandb: Wandb version 0.8.33 is available! To upgrade, please run:
wandb: \$ pip install wandb --upgrade
wandb: Local directory: wandb/run-20200425_073143-7jfp9lyl  
wandb: Syncing run selfatt-2020-04-25-07-31-43: https://app.wandb.ai/jianguda/CodeSearchNet/runs/7jfp9lyl
wandb: Run `wandb off` to turn off syncing.

2020-04-25 07:31:49.016027: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-25 07:31:49.095957: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fbae:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-25 07:31:49.096005: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0  
2020-04-25 07:31:49.370871: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-25 07:31:49.370938: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-25 07:31:49.370953: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-25 07:31:49.371073: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fbae:00:00.0, compute capability: 3.7)
Tokenizing and building vocabulary for code snippets and queries. This step may take several hours.
Starting training run selfatt-2020-04-25-07-31-43 of model SelfAttentionModel with following hypers:
{'code_token_vocab_size': 10000, 'code_token_vocab_count_threshold': 10, 'code_token_embedding_size': 128, 'code_use_subtokens': False, 'code_mark_subtoken_end': False, 'code_max_num_tokens': 200, 'code_use_bpe': True, 'code_pct_bpe': 0.5, 'code_self_attention_activation': 'gelu', 'code_self_attention_hidden_size': 128, 'code_self_attention_intermediate_size': 512, 'code_self_attention_num_layers': 3, 'code_self_attention_num_heads': 8, 'code_self_attention_pool_mode': 'weighted_mean', 'query_token_vocab_size': 10000, 'query_token_vocab_count_threshold': 10, 'query_token_embedding_size': 128, 'query_use_subtokens': False, 'query_mark_subtoken_end': False, 'query_max_num_tokens': 30, 'query_use_bpe': True, 'query_pct_bpe': 0.5, 'query_self_attention_activation': 'gelu', 'query_self_attention_hidden_size': 128, 'query_self_attention_intermediate_size': 512, 'query_self_attention_num_layers': 3, 'query_self_attention_num_heads': 8, 'query_self_attention_pool_mode': 'weighted_mean', 'batch_size': 450, 'optimizer': 'Adam', 'seed': 0, 'dropout_keep_rate': 0.9, 'learning_rate': 0.0005, 'learning_rate_code_scale_factor': 1.0, 'learning_rate_query_scale_factor': 1.0, 'learning_rate_decay': 0.98, 'momentum': 0.85, 'gradient_clip': 1, 'loss': 'softmax', 'margin': 1, 'max_epochs': 300, 'patience': 5, 'fraction_using_func_name': 0.1, 'min_len_func_name_for_query': 12, 'query_random_token_frequency': 0.0}
Loading training and validation data.
Begin Training.
Training on 48791 ruby samples.
Validating on 2209 ruby samples.
==== Epoch 0 ====
Epoch 0 (train) took 148.27s [processed 327 samples/second]
Training Loss: 5.409651
Epoch 0 (valid) took 2.63s [processed 684 samples/second]
Validation: Loss: 5.256474 | MRR: 0.138274
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-25-07-31-43_model_best.pkl.gz'.
==== Epoch 1 ====
Epoch 1 (train) took 146.54s [processed 331 samples/second]
Training Loss: 3.041944
Epoch 1 (valid) took 2.30s [processed 781 samples/second]
Validation: Loss: 4.130856 | MRR: 0.336261
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-25-07-31-43_model_best.pkl.gz'.
==== Epoch 2 ====
Epoch 2 (train) took 146.99s [processed 330 samples/second]
Training Loss: 1.610682
Epoch 2 (valid) took 2.31s [processed 780 samples/second]
Validation: Loss: 3.880517 | MRR: 0.380329
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-25-07-31-43_model_best.pkl.gz'.
==== Epoch 3 ====
Epoch 3 (train) took 147.29s [processed 329 samples/second]
Training Loss: 0.953049
Epoch 3 (valid) took 2.31s [processed 778 samples/second]
Validation: Loss: 3.973366 | MRR: 0.396153
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-25-07-31-43_model_best.pkl.gz'.
==== Epoch 4 ====
Epoch 4 (train) took 147.37s [processed 329 samples/second]
Training Loss: 0.610886
Epoch 4 (valid) took 2.32s [processed 774 samples/second]
Validation: Loss: 4.039880 | MRR: 0.404104
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-25-07-31-43_model_best.pkl.gz'.
==== Epoch 5 ====
Epoch 5 (train) took 147.38s [processed 329 samples/second]
Training Loss: 0.437324
Epoch 5 (valid) took 2.31s [processed 780 samples/second]
Validation: Loss: 4.124913 | MRR: 0.399352
==== Epoch 6 ====
Epoch 6 (train) took 147.54s [processed 329 samples/second]
Training Loss: 0.348149
Epoch 6 (valid) took 2.32s [processed 774 samples/second]
Validation: Loss: 4.281767 | MRR: 0.412711
Best result so far -- saved model as '../resources/saved_models/selfatt-2020-04-25-07-31-43_model_best.pkl.gz'.
==== Epoch 7 ====
Epoch 7 (train) took 147.51s [processed 329 samples/second]
Training Loss: 0.293062
Epoch 7 (valid) took 2.31s [processed 778 samples/second]
Validation: Loss: 4.378719 | MRR: 0.408091
==== Epoch 8 ====
Epoch 8 (train) took 147.56s [processed 329 samples/second]
Training Loss: 0.257615
Epoch 8 (valid) took 2.31s [processed 779 samples/second]
Validation: Loss: 4.519779 | MRR: 0.396948
==== Epoch 9 ====
Epoch 9 (train) took 147.64s [processed 329 samples/second]
Training Loss: 0.231291
Epoch 9 (valid) took 2.32s [processed 777 samples/second]
Validation: Loss: 4.471228 | MRR: 0.408025
==== Epoch 10 ====
Epoch 10 (train) took 147.59s [processed 329 samples/second]
Training Loss: 0.221564
Epoch 10 (valid) took 2.32s [processed 774 samples/second]
Validation: Loss: 4.439744 | MRR: 0.406252
==== Epoch 11 ====
Epoch 11 (train) took 147.54s [processed 329 samples/second]
Training Loss: 0.200222
Epoch 11 (valid) took 2.35s [processed 767 samples/second]
Validation: Loss: 4.647786 | MRR: 0.403177
2020-04-25 08:03:30.643929: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-25 08:03:30.644001: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-25 08:03:30.644017: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-25 08:03:30.644030: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-25 08:03:30.644134: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fbae:00:00.0, compute capability: 3.7)
Test-All MRR (bs=1,000): 0.351
FuncNameTest-All MRR (bs=1,000): 0.526
Validation-All MRR (bs=1,000): 0.394
Test-ruby MRR (bs=1,000): 0.351
FuncNameTest-ruby MRR (bs=1,000): 0.526
Validation-ruby MRR (bs=1,000): 0.394

wandb: Waiting for W&B process to finish, PID 851
wandb: Program ended successfully.
wandb: Run summary:
wandb: train-mrr 0.96702880859375
wandb: \_timestamp 1587801831.3228285
wandb: \_step 41
wandb: train-loss 0.20022226759680994
wandb: \_runtime 1929.0970742702484
wandb: val-time-sec 2.3460564613342285
wandb: val-mrr 0.40317674424913197
wandb: epoch 11
wandb: val-loss 4.647786259651184
wandb: train-time-sec 147.54106950759888
wandb: best_val_mrr_loss 4.281766891479492
wandb: best_val_mrr 0.41271131727430554
wandb: best_epoch 6
wandb: Test-All MRR (bs=1,000) 0.350636802020858
wandb: FuncNameTest-All MRR (bs=1,000) 0.5262124932295927
wandb: Validation-All MRR (bs=1,000) 0.3936782432045811
wandb: Test-ruby MRR (bs=1,000) 0.350636802020858
wandb: FuncNameTest-ruby MRR (bs=1,000) 0.5262124932295927
wandb: Validation-ruby MRR (bs=1,000) 0.3936782432045811
wandb: Syncing files in wandb/run-20200425_073143-7jfp9lyl:
wandb: selfatt-2020-04-25-07-31-43-graph.pbtxt
wandb: selfatt-2020-04-25-07-31-43.train_log
wandb: selfatt-2020-04-25-07-31-43_model_best.pkl.gz
wandb: plus 9 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced selfatt-2020-04-25-07-31-43: https://app.wandb.ai/jianguda/CodeSearchNet/runs/7jfp9lyl
root@jian-csn:/home/dev/src# python predict.py -r jianguda/CodeSearchNet/7jfp9lyl
Fetching run from W&B...
Fetching run files from W&B...
Restoring model from ./selfatt-2020-04-25-07-31-43_model_best.pkl.gz
2020-04-25 08:15:11.202059: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow
binary was not compiled to use: AVX2 FMA
2020-04-25 08:15:16.020162: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: fbae:00:00.0
totalMemory: 11.17GiB freeMemory: 11.11GiB
2020-04-25 08:15:16.020212: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2020-04-25 08:15:16.302958: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength
1 edge matrix:
2020-04-25 08:15:16.303023: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988] 0
2020-04-25 08:15:16.303041: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0: N
2020-04-25 08:15:16.303156: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10762 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: fbae:00:00.0, compute capability: 3.7)
Evaluating language: ruby
100%|████████████████████████████████████████████████████████████████████████████████████| 164048/164048 [00:00<00:00, 197822.38it/s]164048it [00:02, 58460.18it/s]
Uploading predictions to W&B
NDCG Average: 0.115235428
