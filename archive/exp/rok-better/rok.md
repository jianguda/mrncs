# rok-raw (all)

Restoring model weights from the end of the best epoch
Epoch 00011: early stopping
Evaluating go - Valid Mean MRR: 0.823909558181, Test Mean MRR: 0.6963660907047619
Evaluating java - Valid Mean MRR: 0.5948656227043544, Test Mean MRR: 0.6087932094319645
Evaluating javascript - Valid Mean MRR: 0.4526421443768543, Test Mean MRR: 0.4699129365572518
Evaluating php - Valid Mean MRR: 0.5807359635037714, Test Mean MRR: 0.5676996776175864
Evaluating python - Valid Mean MRR: 0.63822232958513, Test Mean MRR: 0.6678135876242807
Evaluating ruby - Valid Mean MRR: 0.4275033754978549, Test Mean MRR: 0.3691721512250925
All languages - Valid Mean MRR: 0.5863131656414942, Test Mean MRR: 0.5632929421934896
Building go code embeddings
Building java code embeddings
Building javascript code embeddings
Building php code embeddings
Building python code embeddings
Building ruby code embeddings
Evaluating go
Evaluating java
Evaluating javascript
Evaluating php
Evaluating python
Evaluating ruby

wandb: Waiting for W&B process to finish, PID 7506
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_timestamp 1589664968.516636
wandb: epoch 10
wandb: \_step 60
wandb: \_runtime 5087.102079153061
wandb: loss 0.5871380963089362
wandb: php_valid_mrr 0.5807359635037714
wandb: test_mean_mrr 0.5632929421934896
wandb: python_test_mrr 0.6678135876242807
wandb: go_test_mrr 0.6963660907047619
wandb: java_valid_mrr 0.5948656227043544
wandb: go_valid_mrr 0.823909558181
wandb: javascript_valid_mrr 0.4526421443768543
wandb: javascript_test_mrr 0.4699129365572518
wandb: java_test_mrr 0.6087932094319645
wandb: python_valid_mrr 0.63822232958513
wandb: ruby_valid_mrr 0.4275033754978549
wandb: valid_mean_mrr 0.5863131656414942
wandb: ruby_test_mrr 0.3691721512250925
wandb: php_test_mrr 0.5676996776175864
wandb: Syncing files in wandb/run-20200516_201122-16uf81rn:
wandb: code/rok/train_model.py
wandb: model_predictions.csv
wandb: javascript_test_mrr 0.4699129365572518
wandb: java_test_mrr 0.6087932094319645
wandb: python_valid_mrr 0.63822232958513
wandb: ruby_valid_mrr 0.4275033754978549
wandb: valid_mean_mrr 0.5863131656414942
wandb: ruby_test_mrr 0.3691721512250925
wandb: php_test_mrr 0.5676996776175864
wandb: Syncing files in wandb/run-20200516_201122-16uf81rn:
wandb: code/rok/train_model.py
wandb: model_predictions.csv
wandb: plus 8 W&B file(s) and 1 media file(s)
wandb:
wandb: Synced hearty-sun-175: https://app.wandb.ai/jianguda/CodeSearchNet/runs/16uf81rn
(env) jian@jian-rok:/datadrive/codesnippetsearch/rok\$ python3 evaluate_model.py -r jianguda/CodeSearchNet/16uf81rn
2020-05-16 21:54:12.144068: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer.so.6'; dlerror: libnvinfer.so.6: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: :/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/extras/CUPTI/lib64
2020-05-16 21:54:12.144191: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer_plugin.so.6'; dlerror: libnvinfer_plugin.so.6: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: :/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/extras/CUPTI/lib64
2020-05-16 21:54:12.144211: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:30] Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
Using TensorFlow backend.
Fetching run from W&B...
Uploading predictions to W&B
NDCG Average: 0.287438887

# rok-raw (python)

Mean MRR: 0.6370916644403227
Restoring model weights from the end of the best epoch
Epoch 00010: early stopping
Evaluating python - Valid Mean MRR: 0.6395251337091707, Test Mean MRR: 0.6679720027757647
All languages - Valid Mean MRR: 0.6395251337091707, Test Mean MRR: 0.6679720027757647
Building python code embeddings
Evaluating python

wandb: Waiting for W&B process to finish, PID 5788
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_step 10
wandb: \_runtime 932.78076338768
NDCG Average: 0.436770339

# investigate

NDCG Average: 0.183757510

wandb: Waiting for W&B process to finish, PID 4273
wandb: Program ended successfully.
wandb: Run summary:
wandb: loss 0.794597789055305
wandb: \_timestamp 1589828509.4911835
wandb: epoch 9
wandb: \_runtime 938.096200466156
wandb: \_step 10
wandb: python_test_mrr 0.6660733305832783
wandb: valid_mean_mrr 0.6380853681944435
wandb: python_valid_mrr 0.6380853681944435
wandb: test_mean_mrr 0.6660733305832783
wandb: Syncing files in wandb/run-20200518_184612-2ac2310h:
wandb: code/rok/train_model.py
wandb: model_predictions.csv
wandb: plus 8 W&B file(s) and 1 media file(s)
wandb:
wandb: Synced dazzling-voice-239: https://app.wandb.ai/jianguda/CodeSearchNet/runs/2ac2310h
NDCG Average: 0.426382069

Mean MRR: 0.6041023280638096
Restoring model weights from the end of the best epoch
Epoch 00009: early stopping
Evaluating python - Valid Mean MRR: 0.6081112587730109, Test Mean MRR: 0.6421409209496697
All languages - Valid Mean MRR: 0.6081112587730109, Test Mean MRR: 0.6421409209496697
Building python code embeddings
Evaluating python

wandb: Waiting for W&B process to finish, PID 3471
wandb: Program ended successfully.
wandb: Run summary:
wandb: \_timestamp 1589833474.0580506
wandb: loss 0.8254537528907713
wandb: \_runtime 431.7965040206909
wandb: epoch 8
wandb: \_step 9
wandb: test_mean_mrr 0.6421409209496697
wandb: python_valid_mrr 0.6081112587730109
wandb: valid_mean_mrr 0.6081112587730109
wandb: python_test_mrr 0.6421409209496697
wandb: Syncing files in wandb/run-20200518_201722-3570nlbo:
wandb: code/rok/train_model.py
wandb: model_predictions.csv
wandb: plus 8 W&B file(s) and 1 media file(s)
wandb:
wandb: Synced scarlet-puddle-242: https://app.wandb.ai/jianguda/CodeSearchNet/runs/3570nlbo
NDCG Average: 0.217951688

Mean MRR: 0.604826949862068
Restoring model weights from the end of the best epoch
Epoch 00011: early stopping
Evaluating python - Valid Mean MRR: 0.6068308051204107, Test Mean MRR: 0.6393532876534658
All languages - Valid Mean MRR: 0.6068308051204107, Test Mean MRR: 0.6393532876534658
Building python code embeddings
Evaluating python
NDCG Average: 0.408769285

# tmp

python - Valid Mean MRR: 0.6512005183234959, Test Mean MRR: 0.6663939745875966
All languages - Valid Mean MRR: 0.6512005183234959, Test Mean MRR: 0.6663939745875966
Evaluating python
building python code embeddings
building python query embeddings

wandb: Waiting for W&B process to finish, PID 14000
wandb: Program ended successfully.
wandb: Run summary:
wandb: valid_mean_mrr 0.6512005183234959
wandb: \_runtime 17.266382694244385
wandb: python_valid_mrr 0.6512005183234959
wandb: \_step 0
wandb: test_mean_mrr 0.6663939745875966
wandb: python_test_mrr 0.6663939745875966
wandb: \_timestamp 1590073093.9967086
wandb: Syncing files in wandb/run-20200521_145759-vzp2ooo7:
wandb: model_predictions.csv
wandb: plus 7 W&B file(s) and 0 media file(s)
wandb:
wandb: Synced kth-nbow_annoy-2020-05-21: https://app.wandb.ai/jianguda/CodeSearchNet/runs/vzp2ooo7
NDCG Average: 0.200732492

# log

SBT concat
NDCG Average: 0.413983817
