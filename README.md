# CSN

Because of the issue in my implementation, I usually run following commands to get results. The results produced by the first run is not correct, so the second run is required. When I have time, I would go to correct this.

```
python3 run.py train evaluate
python3 run.py evaluate
```

## About

```markdown
**code** the TensorFlow Implementation
**core** the Keras Implementation
**doc** documentations
----**exp** experimental results
----**fig** figures for thesis writing
----**setup** how to prepare the Azure VM
----`@csn.md` guidance for run experiments with TensorFlow Implementation
----`@tree.md` guidance for run experiments with Keras Implementation
```

## How to Reproduce Results

### For the TensorFlow implementation

1. prepare the Azure VM by following the TensorFlow1 part at `doc/setup/README`
2. check CodeSearchNet [QuickStart](https://github.com/github/CodeSearchNet#quickstart)
3. override with our implementation by following the guidance at `doc/@csn.md`
4. run the `treeleaf`, `treepath` or `treeraw` model

### For the Keras implementation

1. prepare the Azure VM (following the TensorFlow2 part at `doc/setup/README`)
2. check CodeSnippetSearch [README](https://github.com/novoselrok/codesnippetsearch)
3. override with our implementation by following the guidance at `doc/@tree.md`
4. run models

## References

- [CodeSearchNet](https://github.com/github/CodeSearchNet)
- [CodeSnippetSearch](https://github.com/novoselrok/codesnippetsearch)
