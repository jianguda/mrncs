from scripts.run import run4corpus, run4demo


if __name__ == '__main__':
    # JGD checklist
    #  done baseline for src-token
    #  done baseline(att) for leaf-token
    #  done baseline(att) for tree-path
    #  done check the att part
    #  todo (preprocessing docstring tokens)
    #  todo nbow (leaf-token) + rnn (tree-path)
    #  todo compare code2vec and multi-modal model
    #  todo subtoken?
    #  ......
    #  leaf + path V.S. treeLSTM
    #  Siamese NN
    # run4corpus(locally=True)
    run4corpus()
    # run4demo()

# alon_encoder will be replaced by tree_encoder
#  * check L508 at alon_encoder (by learning the implementation of self_att_encoder)
#  * or check the dimension info (recorded at `@.md`)

# cd /mnt/c/Users/jian/PycharmProjects/csn/code
# python3 wow.py
