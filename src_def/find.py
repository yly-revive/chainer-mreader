import numpy as np
from argparse import ArgumentParser
import json
from progressbar import ProgressBar
from sklearn.metrics.pairwise import cosine_similarity


def cal_oov_dict(vocab, dict_vocab, text):
    tokens = text.strip().split("', '")
    vocab_count = 0
    dict_vocab_count = 0
    dict_vocab_count_2 = 0
    for token in tokens:
        if token.lower() not in vocab:
            vocab_count += 1
            if token.lower() not in dict_vocab:
                dict_vocab_count += 1
        # if token not in dict_vocab:
        #    dict_vocab_count+=1
        if token.lower() not in dict_vocab:
            dict_vocab_count_2 += 1

    return len(tokens), vocab_count, dict_vocab_count, dict_vocab_count_2


def main(args):
    vocab_l = []

    with open(args.vocab_file) as vf:
        for line in vf:
            vocab_l.append(line.strip())

    dict_vocab = []
    with open(args.dict_file) as df:
        data = json.load(df)

        for vocab in data:
            dict_vocab.append(vocab)

    listt = []
    with open(args.in_dict_file) as dif, \
            open(args.in_file) as infi:
        index = []
        for i, line in enumerate(dif):

            """
            if "em:True" in line and "em:False" in infi[i]:
                text = infi[i - 3][4:-2]
                a, b, c = cal_oov_dict(vocab_l, dict_vocab, text)
                listt.append(" ".join([i, a, b, c]))
            """
            if "em:True" in line:
                index.append(i)

        text = ""
        for j, line in enumerate(infi):
            if "t:[" in line:
                text = line[4:-2]
            if "em:False" in line and j in index:
                # text = infi[i - 3][4:-2]
                a, b, c, d = cal_oov_dict(vocab_l, dict_vocab, text)
                listt.append(" ".join([str(j), str(a), str(b), str(c), str(d)]))

    with open(args.output_file, "w") as of:

        for item in listt:
            of.write(item)
            of.write("\n")


def main_2(args):
    vocab_l = []

    with open(args.vocab_file) as vf:
        for line in vf:
            vocab_l.append(line.strip())

    dict_vocab = []
    with open(args.dict_file) as df:
        data = json.load(df)

        for vocab in data:
            dict_vocab.append(vocab)

    listt = []
    with open(args.in_dict_file) as dif, \
            open(args.in_file) as infi:
        index = []
        for i, line in enumerate(infi):

            """
            if "em:True" in line and "em:False" in infi[i]:
                text = infi[i - 3][4:-2]
                a, b, c = cal_oov_dict(vocab_l, dict_vocab, text)
                listt.append(" ".join([i, a, b, c]))
            """
            if "em:True" in line:
                index.append(i)

        text = ""
        for j, line in enumerate(dif):
            if "t:[" in line:
                text = line[4:-2]
            if "em:False" in line and j in index:
                # text = infi[i - 3][4:-2]
                a, b, c, d = cal_oov_dict(vocab_l, dict_vocab, text)
                listt.append(" ".join([str(j), str(a), str(b), str(c), str(d)]))

    with open(args.output_file + "_r", "w") as of:

        for item in listt:
            of.write(item)
            of.write("\n")


def main_3(args):
    vocab_l = []

    with open(args.vocab_file) as vf:
        for line in vf:
            vocab_l.append(line.strip())

    dict_vocab = []
    with open(args.dict_file) as df:
        data = json.load(df)

        for vocab in data:
            dict_vocab.append(vocab)

    listt = []
    with open(args.in_dict_file) as dif, \
            open(args.in_file) as infi:
        index = []
        for i, line in enumerate(infi):

            """
            if "em:True" in line and "em:False" in infi[i]:
                text = infi[i - 3][4:-2]
                a, b, c = cal_oov_dict(vocab_l, dict_vocab, text)
                listt.append(" ".join([i, a, b, c]))
            """
            if "em:False" in line:
                index.append(i)

        text = ""
        for j, line in enumerate(dif):
            if "t:[" in line:
                text = line[4:-2]
            if "em:False" in line and j in index:
                # text = infi[i - 3][4:-2]
                a, b, c, d = cal_oov_dict(vocab_l, dict_vocab, text)
                listt.append(" ".join([str(j), str(a), str(b), str(c), str(d)]))

    with open(args.output_file + "_aw", "w") as of:

        for item in listt:
            of.write(item)
            of.write("\n")


def main_4(args):
    vocab_l = []

    with open(args.vocab_file) as vf:
        for line in vf:
            vocab_l.append(line.strip())

    dict_vocab = []
    with open(args.dict_file) as df:
        data = json.load(df)

        for vocab in data:
            dict_vocab.append(vocab)

    listt = []
    with open(args.in_dict_file) as dif, \
            open(args.in_file) as infi:
        index = []
        for i, line in enumerate(infi):

            """
            if "em:True" in line and "em:False" in infi[i]:
                text = infi[i - 3][4:-2]
                a, b, c = cal_oov_dict(vocab_l, dict_vocab, text)
                listt.append(" ".join([i, a, b, c]))
            """
            if "em:True" in line:
                index.append(i)

        text = ""
        for j, line in enumerate(dif):
            if "t:[" in line:
                text = line[4:-2]
            if "em:True" in line and j in index:
                # text = infi[i - 3][4:-2]
                a, b, c, d = cal_oov_dict(vocab_l, dict_vocab, text)
                listt.append(" ".join([str(j), str(a), str(b), str(c), str(d)]))

    with open(args.output_file + "_at", "w") as of:

        for item in listt:
            of.write(item)
            of.write("\n")


def get_vocab_file(emb_file, vocab_file):
    vocab_list = []
    with open(emb_file, 'r') as f:

        pbar = ProgressBar()
        # pbar = tqdm(f)
        for line in pbar(f):
            # for line in pbar:
            vector = line.rstrip().split(' ')
            word = vector[0]
            vocab_list.append(word)

    with open(vocab_file, "w") as of:
        for item in vocab_list:
            of.write(item)
            of.write("\n")


def compare_word_vectors(emb_file, word1, word2):
    # vocab_list = []
    word1_emb = []
    word2_emb = []
    word1_exist = False
    word2_exist = False
    with open(emb_file, 'r') as f:

        pbar = ProgressBar()
        # pbar = tqdm(f)
        for line in pbar(f):
            # for line in pbar:
            vector = line.rstrip().split(' ')
            word = vector[0]
            # vocab_list.append(word)
            if word.lower() == word1.lower():
                word1_emb = vector[1:]
                word1_exist = True
            if word.lower() == word2.lower():
                word2_emb = vector[1:]
                word2_exist = True

    if word1_exist is False:
        print("word={} is not exist!\n".format(word1))

    if word2_exist is False:
        print("word={} is not exist!\n".format(word2))

    if word1_exist and word2_exist:
        print(
            "word1={}, word2={}, similarity={}".format(word1, word2, cosine_similarity([word1_emb], [word2_emb])[0][0]))


def calc_oov_for_text(args):
    vocab_l = []

    with open(args.vocab_file) as vf:
        for line in vf:
            vocab_l.append(line.strip())

    dict_vocab = []
    with open(args.dict_file) as df:
        data = json.load(df)

        for vocab in data:
            dict_vocab.append(vocab)

    tokens = args.text.strip().split()
    vocab_count = 0
    dict_vocab_count = 0
    dict_vocab_count_2 = 0
    for token in tokens:
        #print(token)
        if token.lower() not in vocab_l:
            print("not in vocab:{}".format(token.lower()))
            vocab_count += 1
            if token.lower() not in dict_vocab:
                dict_vocab_count += 1
        # if token not in dict_vocab:
        #    dict_vocab_count+=1
        if token.lower() not in dict_vocab:

            print("not in dict_vocab:{}".format(token.lower()))
            dict_vocab_count_2 += 1

    #return len(tokens), vocab_count, dict_vocab_count, dict_vocab_count_2
    #a, b, c, d = cal_oov_dict(vocab_l, dict_vocab, args.text)

    print("{}".format(" ".join([str(len(tokens)), str(vocab_count), str(dict_vocab_count), str(dict_vocab_count_2)])))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in-dict-file", type=str, default="")
    parser.add_argument("--in-file", type=str, default="")

    parser.add_argument("--vocab-file", type=str, default="")
    parser.add_argument("--dict-file", type=str, default="")
    parser.add_argument("--embedding-file", type=str, default="")

    parser.add_argument("--output-file", type=str, default="")
    parser.add_argument("--word1", type=str, default="")
    parser.add_argument("--word2", type=str, default="")

    args = parser.parse_args()

    # get_vocab_file(args.embedding_file, args.vocab_file)
    # main(args)
    # main_4(args)
    """
    compare_word_vectors(args.embedding_file, args.word1, args.word2)
    #args.text = "What year was the Carolina Panthers franchise founded?"
    """
    args.text = "What was the last Super Bowl the Broncos participated in ? "
    calc_oov_for_text(args)
