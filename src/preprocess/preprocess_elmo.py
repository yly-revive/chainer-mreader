import chainer

from bilm import Elmo
from bilm import TokenBatcher
from bilm import dump_token_embeddings

from argparse import ArgumentParser
import json
from progressbar import ProgressBar


def main():
    parser = ArgumentParser()

    parser.add_argument('--options-file', '-o', type=str, default="", help="elmo option file")
    parser.add_argument('--weight-file', '-w', type=str, default="", help="elmo weight file")
    parser.add_argument('--train-file', '-t', type=str, default="", help="training data")
    parser.add_argument('--dev-file', '-d', type=str, default="", help="dev data")
    parser.add_argument('--gpu', '-g', type=int, default="-1", help="gpu")
    parser.add_argument('--vocab-file', '-v', type=str, default="", help="vocab file")
    parser.add_argument('--token-embedding-file', '-e', type=str, default="", help="embedding file")

    args = parser.parse_args()

    # -o ../../../test_elmo/src/elmo-chainer/elmo_2x4096_512_2048cnn_2xhighway_options.json -w ../../../test_elmo/src/elmo-chainer/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 -t ../../data/datasets/SQuAD-train-v1.1-processed-spacy.txt -d ../../data/datasets/SQuAD-dev-v1.1-processed-spacy.txt -g=5 -v ../../data/embeddings/elmo/vocab_squad_1_1.txt -e ../../data/embeddings/elmo/token_embedding_squad_1_1.hdf5

    all_tokens = ['<S>', '</S>']

    with open(args.train_file) as f:

        json_list = [json.loads(line) for line in f]

        pbar = ProgressBar()

        for json_item in pbar(json_list):

            for token in json_item["document"]:
                if token not in all_tokens:
                    all_tokens.append(token)

            for token in json_item["question"]:
                if token not in all_tokens:
                    all_tokens.append(token)

    with open(args.dev_file) as f:

        json_list = [json.loads(line) for line in f]

        pbar = ProgressBar()

        for json_item in pbar(json_list):

            for token in json_item["document"]:
                if token not in all_tokens:
                    all_tokens.append(token)

            for token in json_item["question"]:
                if token not in all_tokens:
                    all_tokens.append(token)

    # vocab_file = 'vocab_squad1_1.txt'

    with open(args.vocab_file, 'w') as fout:
        fout.write('\n'.join(all_tokens))

    # Location of pretrained LM.
    # options_file = 'elmo_2x4096_512_2048cnn_2xhighway_options.json'
    # weight_file = 'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'

    # Dump the token embeddings to a file. Run this once for your dataset.
    # token_embedding_file = 'elmo_token_embeddings_squad1_1.hdf5'

    # gpu id
    # if you want to use cpu, set gpu=-1
    # gpu = -1
    # batchsize
    # encoding each token is inefficient
    # encoding too many tokens is difficult due to memory
    batchsize = 64

    dump_token_embeddings(
        args.vocab_file, args.options_file, args.weight_file, args.token_embedding_file,
        gpu=args.gpu, batchsize=batchsize
    )


if __name__ == '__main__':
    main()
