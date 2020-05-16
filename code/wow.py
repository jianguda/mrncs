from scripts.run import check_data, run4corpus, run4demo


if __name__ == '__main__':
    # JGD checklist
    #  done baseline for src-token
    #  done (preprocessing tokens)
    #  todo check leaf-token and tree-path
    #  todo baseline (att) for leaf-token
    #  todo baseline (att) for tree-path
    #  todo (leaf-token) + (tree-path)
    # JGD
    #  done preprocessing
    #  todo multi-modal (semantic info + syntactic info) + attention (and FC or extra abstract layers)
    #  todo metric learning / loss function (Siamese NN) Deep Metric Learning: A Survey
    #  todo CAT https://arxiv.org/abs/2005.06980
    # JGD why Rok
    #  They say "We are using BPE encoding to encode both code strings and query strings". The string padding and cosine loss is also interesting to look deeper at.
    # ​ SGTM, just make sure that you can make them all into one "story" and have a red thread between them.
    #  ......
    #  leaf + path V.S. treeLSTM
    # run4corpus(locally=True)
    # run4corpus()
    # run4demo()
    # JGD
    check_data()

# alon_encoder will be replaced by tree_encoder
#  * check L508 at alon_encoder (by learning the implementation of self_att_encoder)
#  * or check the dimension info (recorded at `@.md`)

# cd /mnt/c/Users/jian/PycharmProjects/csn/code
# python3 wow.py
