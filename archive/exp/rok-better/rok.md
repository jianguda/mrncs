# rok-raw 2020-05-15

wandb: Waiting for W&B process to finish, PID 2605
wandb: Program ended successfully.
wandb: Run summary:
wandb: loss 0.5782262828342731
wandb: \_step 62
wandb: \_runtime 4116.103474617004
wandb: epoch 11
wandb: \_timestamp 1589558722.7746432
wandb: php_valid_mrr 0.5807396016725876
wandb: ruby_valid_mrr 0.4260285436740947
wandb: javascript_test_mrr 0.4723616442725182
wandb: javascript_valid_mrr 0.44908999581619224
wandb: python_test_mrr 0.668680053122934
wandb: test_mean_mrr 0.5641920678610788
wandb: python_valid_mrr 0.6370209572069467
wandb: ruby_test_mrr 0.368550230203695
wandb: go_test_mrr 0.698753132887966
wandb: java_valid_mrr 0.5939793524100658
wandb: go_valid_mrr 0.825111387632352
wandb: php_test_mrr 0.5708819744602321
wandb: java_test_mrr 0.6059253722191272
wandb: valid_mean_mrr 0.5853283064020399
wandb: Syncing files in wandb/run-20200515_145647-3gtby7ob:
wandb: code/code_search/train_model.py
wandb: model_predictions.csv
wandb: plus 8 W&B file(s) and 1 media file(s)
wandb:
wandb: Synced distinctive-dawn-170: https://app.wandb.ai/jianguda/CodeSearchNet/runs/3gtby7ob
(env) jian@jian-rok:/datadrive/codesnippetsearch/code_search\$ python evaluate_model.py -r jianguda/CodeSearchNet/3gtby7ob
2020-05-15 16:54:00.525955: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer.so.6'; dlerror: libnvinfer.so.6: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: :/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/extras/CUPTI/lib64
2020-05-15 16:54:00.526080: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer_plugin.so.6'; dlerror: libnvinfer_plugin.so.6: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: :/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/extras/CUPTI/lib64
2020-05-15 16:54:00.526099: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:30] Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
Using TensorFlow backend.
Fetching run from W&B...
Uploading predictions to W&B
