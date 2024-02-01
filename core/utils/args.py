from argparse import ArgumentParser, Namespace

def get_train_args() -> Namespace:
    
    parser = ArgumentParser()
    parser.add_argument('--train_batch_size')