import train_utils as train_utils

if __name__ == '__main__':
    config = train_utils.make_config(train_utils.define_config())
    from la_mbda.la_mbda import LAMBDA

    train_utils.train(config, LAMBDA)
