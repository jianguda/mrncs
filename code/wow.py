from scripts.run import check_data, run4corpus, run4demo


if __name__ == '__main__':
    # run4corpus(locally=True)
    # run4corpus()
    run4demo()
    # check_data()

# alon_encoder will be replaced by tree_encoder
#  * check L508 at alon_encoder (by learning the implementation of self_att_encoder)
#  * or check the dimension info (recorded at `@.md`)

# cd /mnt/c/Users/jian/PycharmProjects/csn/code
# python3 wow.py
