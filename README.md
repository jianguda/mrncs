# MRNCS

Because of the issue in my implementation, I usually run following commands to get results. The results produced by the first run is not correct, so the second run is required. When I have time, I would go to correct this.

```
python3 run.py train evaluate
python3 run.py evaluate
```

## About

```markdown
**core** scripts to prepare the data, train the language models and save the embeddings
the Implementation
**resources** raw data and preprocessed data
----**caches** intermediate objects during training (docs, vocabularies, models, embeddings etc.)
----**data** raw data
**doc** documentations
----**exp** experimental results
----**setup** how to prepare the Azure VM
----`@tree.md` guidance for run experiments
```

## How to Reproduce Results

1. prepare the Azure VM (following the TensorFlow2 part at `doc/setup/README`)
2. check CodeSnippetSearch [README](https://github.com/novoselrok/codesnippetsearch)
3. override with our implementation by following the guidance at `doc/@tree.md`

## References

The partial implementations are with references to the following projects:

- [github/CodeSearchNet](https://github.com/github/CodeSearchNet)
- [novoselrok/codesnippetsearch](https://github.com/novoselrok/codesnippetsearch)

## Model description

We are using BPE encoding to encode both code strings and query strings (docstrings are used as a proxy for queries). 
Code strings are padded and encoded to a length of 30 tokens and query strings are padded and encoded to a length of 200 tokens. 
Embedding size is set to 256. Token embeddings are masked and then an unweighted mean is performed to get 256-length vectors for code strings and query strings.
Finally, cosine similarity is calculated between the code vectors and the query vectors and "cosine loss" is calculated 
(the loss function is documented in code_search/train_model.py#cosine_loss).
Further details can be found on the [WANDB run](https://app.wandb.ai/roknovosel/glorified-code-search/runs/21hzzq1h/overview).

## Data

We are using the data from the CodeSearchNet project. Run the following commands to download the required data:

- `$ mkdir -p resources/data; cd resources/data`
- `$ wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{python,java,go,php,javascript,ruby}.zip`

This will download around 20GB of data. Overview of the data structure is listed [here](https://github.com/github/CodeSearchNet/tree/master/resources).

## Training the models

If you can, you should be performing these steps inside a virtual environment.
To install the required dependencies run: `$ pip install -r requirements.txt`

### Preparing the data

Data preparation step is separate from the training step because it is time and memory consuming. We will prepare all the
necessary data needed for training. This includes preprocessing code docs, building vocabularies, and encoding sequences.

The first step is to convert evaluation code documents (`*_dedupe_definitions_v2.pkl` files) from a `pickle` format to `jsonl` format. We will be using the jsonl
format throughout the project, since we can read the file line by line and keep the memory footprint minimal. Reading the
evaluation docs requires **more** than 16GB of memory, because the entire file has to be read in memory (largest is `javascript_dedupe_definitions_v2.pkl` at 6.6GB).
If you do not have this kind of horsepower, I suggest renting a cloud server with >16GB of memory and running this step on there. After you are done,
just download the jsonl files to your local machine. Subsequent preparation and training steps should not take more than 16GB of memory.

To convert ruby evaluation docs to `jsonl` format move inside the `code_search` directory run the following command:
`$ python parse_dedupe_definitions.py ruby`. Run this command for the remaining 5 languages: `python`, `java`, `go`, `php` and `javascript`.

To prepare the data for training run: `$ python prepare_data.py --prepare-all`. It uses the Python multiprocessing
module to take advantage of multiple cores. If you encounter memory errors or slow performance you can tweak the number of
processes by changing the parameter passed to `multiprocessing.Pool`.

### Training and evaluation

You start the training by running: `$ python train_model.py`. This will train separate models for each language, build code embeddings
and evaluate them according to MRR (Mean Reciprocal Rank) and output `model_predictions.csv`. These will be evaluated by Github & WANDB 
using NDCG (Normalized Discounted cumulative gain) metric to rank the submissions.

### Query the trained models

Run `$ python search.py "read file lines"` and it will output 3 best ranked results for each language.
