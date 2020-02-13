# MRNCS

This repo is the artifact for the paper entitled `Multimodel Representation for Neural Code Search`, accepted by `ICSME 2021`

## Repo Structure

```markdown
**code** implementations to support the experiments
----**build** the tree-sitter code parser
----**tree** the code package of the implementations
----**vendor** the Git submodules for build the code parser
**resources** raw data and preprocessed data
----**cache** preprocessed data
----**data** raw data
**doc** documentations
----**exp** experimental results
----**info** supplementary snippets
----`@guide.md` guidance for run experiments
```

## How to Reproduce Experiments

Please go to check `doc/@guide.md`

## References

The partial implementations are with references to the following projects, during the evolutionary history of this repo

- [github/CodeSearchNet](https://github.com/github/CodeSearchNet)
- [novoselrok/codesnippetsearch](https://github.com/novoselrok/codesnippetsearch)
