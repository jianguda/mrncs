from scripts.run import run4corpus, run4demo


if __name__ == '__main__':
    # JGD todo
    #  run experiments and debug
    #  * check L508 at alon_encoder (by learning the implementation of self_att_encoder)
    #  * or check the dimension info (recorded at `@.md`)
    #  HPT -> ReducedAST & ReducedSPT
    #  check the implementation of path-length, path-width, K, M
    #  ......
    # run4corpus(locally=True)
    run4corpus()
    # run4demo()

# cd /mnt/c/Users/jian/PycharmProjects/csn/code
# python3 wow.py
