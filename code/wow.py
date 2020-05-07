from scripts.run import check_data, run4corpus, run4demo


if __name__ == '__main__':
    # JGD checklist
    #  done baseline for src-token
    #  done baseline (att) for leaf-token
    #  done (preprocessing tokens)
    #  todo investigate the loss function
    #  todo run on the whole corpus (multi-lang)
    #  todo baseline (att) for tree-path
    #  todo check leaf-token and tree-path
    #  todo (leaf-token) + (tree-path)
    #  todo improve model (FC or extra abstract layers)
    #  todo improve model (the so-called devout-resonance)
    #  todo improve model (Siamese NN)
    #  todo compare code2vec and multi-modal model ?
    #  ......
    #  leaf + path V.S. treeLSTM
    # run4corpus(locally=True)
    # run4corpus()
    # run4demo()
    check_data()

# alon_encoder will be replaced by tree_encoder
#  * check L508 at alon_encoder (by learning the implementation of self_att_encoder)
#  * or check the dimension info (recorded at `@.md`)

# cd /mnt/c/Users/jian/PycharmProjects/csn/code
# python3 wow.py
