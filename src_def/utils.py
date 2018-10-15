import numpy as np
import json
from enum import Enum
from progressbar import ProgressBar
# import tqdm
import os

from bilm import Batcher

'''
class QuestionType(IntEnum):
    WHAT = 1
    HOW = 2
    WHO = 3
    WHEN = 4
    WHICH = 5
    WHERE = 6
    WHY = 7
    BE = 8
    WHOSE = 9
    WHOM = 10
    OTHER = 11
'''


class QuestionType(object):
    WHAT = 0
    HOW = 1
    WHO = 2
    WHEN = 3
    WHICH = 4
    WHERE = 5
    WHY = 6
    BE = 7
    WHOSE = 8
    WHOM = 9
    OTHER = 10

    def __init__(self):
        super(QuestionType, self).__init__()

    @staticmethod
    def get_keys():
        return ["WHAT", "HOW", "WHO",
                "WHEN", "WHICH", "WHERE", "WHY", "BE", "WHOSE", "WHOM", "OTHER"]

    @staticmethod
    def be_types():
        return ["is", "am", "are"]


'''
class NoValue(Enum):
    def __repr__(self):
        return '<%s.%s>' % (self.__class__.__name__, self.name)
'''


class DataType(object):
    CONTEXT = 'document'
    QUESTION = 'question'

    def __init__(self):
        super(DataType, self).__init__()


class DataUtils(object):
    word_dict = {}
    char_dict = {}

    MAX_CHAR_PER_WORD = 40

    # IS_DEBUG = True
    IS_DEBUG = False

    MAX_DOC_LENGTH = 0
    MAX_Q_LENGTH = 0

    batcher = None
    use_elmo = False

    dictionary = None
    dict_embedding = None
    use_dict = False

    def __init__(self):
        super(DataUtils, self).__init__()

    @staticmethod
    def load_elmo_batcher(vocab_file):
        DataUtils.batcher = Batcher(vocab_file, 50)
        DataUtils.use_elmo = True

    @staticmethod
    def load_data(file):
        # Load JSON file
        # json_data = json.load(open(file, 'r'))
        if DataUtils.IS_DEBUG:
            print("function(load_data):start...")

        with open(file, 'r') as f:
            json_list = [json.loads(line) for line in f]

            for json_data in json_list:
                '''
                # debug
                if len(json_data["answers"]) <= 0:
                    continue
                '''

                DataUtils.add_em_feature(json_data)
                DataUtils.add_question_feature(json_data)
            # return json_list

            # debug
            # json_list_ret = [json_item for json_item in json_list if len(json_item["answers"]) > 0 ]
            json_list_ret = []

            for json_item in json_list:
                if len(json_item["answers"]) > 0:
                    delete = False
                    for answer in json_item["answers"]:
                        if answer[1] >= 300:
                            delete = True
                            break
                    if delete:
                        continue
                    else:
                        json_item["document"] = json_item["document"][:300]
                        json_list_ret.append(json_item)

            return json_list_ret

    @staticmethod
    def add_em_feature(data):

        # if DataUtils.IS_DEBUG:
        #    print("function(add_em_feature):start...")
        # document = data['document']
        # question = data['question']
        document = data['document_lower']

        question = data['question_lower']

        d_em_feature = []
        q_em_feature = []

        for word in document:
            if word.lower() in question:
                d_em_feature.append(1)
            else:
                d_em_feature.append(0)

        for word in question:
            if word.lower() in document:
                q_em_feature.append(1)
            else:
                q_em_feature.append(0)

        data['d_em_feature'] = d_em_feature
        data['q_em_feature'] = q_em_feature

    @staticmethod
    def add_question_feature(data):

        # if DataUtils.IS_DEBUG:
        #    print("function(add_question_feature):start...")

        question = data['question']

        keys = QuestionType.get_keys()

        for tok in question:
            if tok.lower() in QuestionType.be_types():
                data['q_type'] = QuestionType.BE
            elif tok.upper() in keys:
                data['q_type'] = getattr(QuestionType, tok.upper())
            else:
                data['q_type'] = QuestionType.OTHER

    @staticmethod
    def transform_pos_feature(all_data):

        if DataUtils.IS_DEBUG:
            print("function(transform_pos_feature):start...")

        pos_set = set()

        # get all possible pos
        for data in all_data:
            document_pos = data['cpos']
            question_pos = data['qpos']

            pos_set |= set(document_pos)
            pos_set |= set(question_pos)

        # len_pos = len(pos_set)

        # use dictionary to match pos tag with its index
        pos_dict = {}
        for i, pos in enumerate(pos_set):
            pos_dict[pos] = i

        for data in all_data:
            document_pos = data['cpos']
            question_pos = data['qpos']

            document_pos_idx = [pos_dict[i] for i in document_pos]
            question_pos_idx = [pos_dict[i] for i in question_pos]

            data['cpos_idx'] = document_pos_idx
            data['qpos_idx'] = question_pos_idx

        return len(pos_dict)

    @staticmethod
    def transform_ner_feature(all_data):

        if DataUtils.IS_DEBUG:
            print("function(transform_ner_feature):start...")

        ner_set = set()

        # get all possible ner
        for data in all_data:
            document_ner = data['cner']
            question_ner = data['qner']

            ner_set |= set(document_ner)
            ner_set |= set(question_ner)

        # len_ner = len(ner_set)

        # use dictionary to match ner with its index
        ner_dict = {}
        for i, ner in enumerate(ner_set):
            ner_dict[ner] = i

        for data in all_data:
            document_ner = data['cner']
            question_ner = data['qner']

            document_ner_idx = [ner_dict[i] for i in document_ner]
            question_ner_idx = [ner_dict[i] for i in question_ner]

            data['cner_idx'] = document_ner_idx
            data['qner_idx'] = question_ner_idx

        return len(ner_dict)

    @staticmethod
    def get_max_question_len(data):

        if DataUtils.IS_DEBUG:
            print("function(get_max_question_len):start...")

        max_q_len = -1

        for item in data:

            question = item['question']

            q_seq_len = len(question)

            if q_seq_len > max_q_len:
                max_q_len = q_seq_len

        return max_q_len

    @staticmethod
    def get_max_len(data, data_type):

        if DataUtils.IS_DEBUG:
            print("function(get_max_len):start...")

        max_len = -1

        for item in data:

            data = item[data_type]

            seq_len = len(data)

            if seq_len > max_len:
                max_len = seq_len

        return max_len

    @staticmethod
    def cal_mask(data, max_len, max_q_len):

        if DataUtils.IS_DEBUG:
            print("function(cal_mask):start...")

        for item in data:
            document = item['document']
            question = item['question']

            d_seq_len = len(document)
            q_seq_len = len(question)

            d_ones = np.ones(d_seq_len)
            q_ones = np.ones(q_seq_len)

            d_mask = np.zeros(max_len)
            q_mask = np.zeros(max_q_len)

            d_mask[:d_seq_len] = d_ones
            q_mask[:q_seq_len] = q_ones

            item['d_mask'] = d_mask
            item['q_mask'] = q_mask

    @staticmethod
    def convert_data(data, max_len, max_q_len):

        if DataUtils.IS_DEBUG:
            print("function(convert_data):start...")

        if len(DataUtils.word_dict) <= 0 or len(DataUtils.char_dict) <= 0:
            raise RuntimeError('Embedding hasn\'t been loaded.')

        q_list = []
        q_char_list = []
        q_feature_list = []
        q_mask_list = []
        d_list = []
        d_char_list = []
        d_feature_list = []
        d_mask_list = []
        target_list = []

        for item in data:
            q = np.ones(max_q_len) * (-1)
            for i, word in enumerate(item['question']):
                q[i] = DataUtils.word_dict[word.lower()]

            d = np.ones(max_len) * (-1)
            for i, word in enumerate(item['document']):
                d[i] = DataUtils.word_dict[word.lower()]

            q_char = np.ones((max_q_len, DataUtils.MAX_CHAR_PER_WORD)) * (-1)
            for i, word_level in enumerate(item['question_char']):
                for j, char_item in enumerate(word_level):
                    q_char[i][j] = DataUtils.char_dict[char_item.lower()]

            d_char = np.ones((max_len, DataUtils.MAX_CHAR_PER_WORD)) * (-1)
            for i, word_level in enumerate(item['document_char']):
                for j, char_item in enumerate(word_level):
                    d_char[i][j] = DataUtils.char_dict[char_item.lower()]

            q_feature = np.ones((max_q_len, 4)) * (-1)
            for i, tok in enumerate(item['question']):
                q_feature[i][0] = item['qpos_idx'][i]
                q_feature[i][1] = item['qner_idx'][i]
                q_feature[i][2] = item['q_em_feature'][i]
                q_feature[i][3] = item['q_type']

            d_feature = np.ones((max_len, 4)) * (-1)
            for i, tok in enumerate(item['document']):
                d_feature[i][0] = item['cpos_idx'][i]
                d_feature[i][1] = item['cner_idx'][i]
                d_feature[i][2] = item['d_em_feature'][i]
                d_feature[i][3] = QuestionType.OTHER

            q_mask = item['q_mask']
            d_mask = item['d_mask']
            target = item['answers']

            q_list.append(q)
            q_char_list.append(q_char)
            q_feature_list.append(q_feature)
            q_mask_list.append(q_mask)
            d_list.append(d)
            d_char_list.append(d_char)
            d_feature_list.append(d_feature)
            d_mask_list.append(d_mask)
            target_list.append(target)

        if DataUtils.IS_DEBUG:
            print("len of q_list = %d" % len(q_list))
            print("len of q_char_list = %d" % len(q_char_list))
            print("len of q_feature_list = %d" % len(q_feature_list))
            print("len of q_mask_list = %d" % len(q_mask_list))
            print("len of d_list = %d" % len(d_list))
            print("len of d_char_list = %d" % len(d_char_list))
            print("len of d_feature_list = %d" % len(d_feature_list))
            print("len of d_mask_list = %d" % len(d_mask_list))
            print("len of target_list = %d" % len(target_list))

        return np.array(d_list), np.array(d_char_list), np.array(d_feature_list), np.array(d_mask_list), \
               np.array(q_list), np.array(q_char_list), np.array(q_feature_list), np.array(q_mask_list), np.array(
            target_list)

    @staticmethod
    def load_embedding(data, file, dimension, pretrained_embedding_file=None, pretrained_index_file=None,
                       overwrite=False, pretrain=True):

        if DataUtils.IS_DEBUG:
            print("function(load_embedding):start...")

        if (overwrite is False) and (pretrained_index_file and pretrained_embedding_file and os.path.isfile(
                pretrained_index_file) and os.path.isfile(pretrained_embedding_file)):
            embedding_matrix = np.array(np.load(pretrained_embedding_file))
            with open(pretrained_index_file) as f:
                for line in f:
                    strings = line.rstrip().split(' ')
                    DataUtils.word_dict[strings[0]] = int(strings[1])
                return embedding_matrix

        word_list = set()
        # char_list = set()

        for item in data:
            word_list |= set([i.lower() for i in item['document']])
            word_list |= set([i.lower() for i in item['question']])

            '''
            for char_item in item['document_char']:
                char_list |= set(char_item)
    
            for char_item in item['question_char']:
                char_list |= set(char_item)
         '''

        embedding_dict = {}
        with open(file, 'r') as f:
            if DataUtils.IS_DEBUG:
                print("read embedding : start...")

            pbar = ProgressBar()
            # pbar = tqdm(f)
            for line in pbar(f):
                # for line in pbar:
                vector = line.rstrip().split(' ')
                word = vector[0]
                # pbar.set_description("Processing %s" % word)
                embedding = vector[1:]
                # assert (len(embedding) == 300)
                assert (len(embedding) == 100)
                embedding_dict[word.lower()] = embedding

        # embedding_matrix = np.zeros((len(word_list)+1, dimension))
        # embedding_matrix[0] = np.random_normal(0, 1, dimension)
        embedding_matrix = np.zeros((len(word_list), dimension))

        index = 0
        if DataUtils.IS_DEBUG:
            print("load embedding : start...")

        pbar = ProgressBar()
        for word in pbar(word_list):
            # pbar = tqdm(word_list)
            # for word in pbar:

            # pbar.set_description("Processing %s" % word)

            if word.lower() in embedding_dict:
                embedding_matrix[index] = embedding_dict[word.lower()]
            else:
                embedding_matrix[index] = np.random.normal(0, 1, dimension)

            DataUtils.word_dict[word.lower()] = index
            index += 1

        if DataUtils.IS_DEBUG:
            print("read embedding : finished...")
        if pretrain:
            embedding_matrix.dump(pretrained_embedding_file)
            with open(pretrained_index_file, 'w') as f:
                for i, item in enumerate(DataUtils.word_dict):
                    f.write(item)
                    f.write(' ')
                    f.write(str(i))
                    f.write('\n')

        return embedding_matrix

    @staticmethod
    def load_char_embedding(data, file, dimension, pretrained_embedding_file=None, pretrained_index_file=None,
                            overwrite=False, pretrain=True):
        if DataUtils.IS_DEBUG:
            print("function(load_char_embedding):start...")

        if (overwrite is False) and (pretrained_index_file and pretrained_embedding_file and os.path.isfile(
                pretrained_index_file) and os.path.isfile(pretrained_embedding_file)):
            embedding_matrix = np.array(np.load(pretrained_embedding_file))
            with open(pretrained_index_file) as f:
                for line in f:
                    strings = line.rstrip().split(' ')
                    DataUtils.char_dict[strings[0]] = int(strings[1])
                return embedding_matrix

        char_list = set()

        for item in data:

            for char_item in item['document_char']:
                char_list |= set([i.lower() for i in char_item])

            for char_item in item['question_char']:
                char_list |= set([i.lower() for i in char_item])

        embedding_dict = {}
        with open(file, 'r') as f:
            if DataUtils.IS_DEBUG:
                print("read character embedding : start...")

            pbar = ProgressBar()
            for line in pbar(f):
                # pbar = tqdm(f)
                # for line in pbar:
                vector = line.rstrip().split(' ')
                char = vector[0]

                # pbar.set_description("Processing %s" % char)

                embedding = vector[1:]
                embedding_dict[char.lower()] = embedding

        # embedding_matrix = np.zeros((len(word_list)+1, dimension))
        # embedding_matrix[0] = np.random_normal(0, 1, dimension)
        embedding_matrix = np.zeros((len(char_list), dimension))

        index = 0
        if DataUtils.IS_DEBUG:
            print("load character embedding : start...")

        pbar = ProgressBar()
        for char in pbar(char_list):
            # pbar = tqdm()
            # for char in pbar:

            # pbar.set_description("Processing %s" % char)

            if char.lower() in embedding_dict:
                embedding_matrix[index] = embedding_dict[char.lower()]
            else:
                embedding_matrix[index] = np.random.normal(0, 1, dimension)

            DataUtils.char_dict[char.lower()] = index
            index += 1

        if pretrain:
            embedding_matrix.dump(pretrained_embedding_file)
            with open(pretrained_index_file, 'w') as f:
                for i, item in enumerate(DataUtils.char_dict):
                    f.write(item)
                    f.write(' ')
                    f.write(str(i))
                    f.write('\n')
        return embedding_matrix

    @staticmethod
    def convert_item(item):

        if DataUtils.IS_DEBUG:
            print("function(convert_data):start...")

        assert (DataUtils.MAX_DOC_LENGTH > 0 and DataUtils.MAX_Q_LENGTH > 0)

        if len(DataUtils.word_dict) <= 0 or len(DataUtils.char_dict) <= 0:
            raise RuntimeError('Embedding hasn\'t been loaded.')

        # for item in data:
        q = np.ones(DataUtils.MAX_Q_LENGTH, dtype=int) * (-1)
        for i, word in enumerate(item['question']):
            q[i] = DataUtils.word_dict[word.lower()]

        d = np.ones(DataUtils.MAX_DOC_LENGTH, dtype=int) * (-1)
        for i, word in enumerate(item['document']):
            d[i] = DataUtils.word_dict[word.lower()]

        q_char = np.ones((DataUtils.MAX_Q_LENGTH, DataUtils.MAX_CHAR_PER_WORD), dtype=int) * (-1)
        for i, word_level in enumerate(item['question_char']):
            for j, char_item in enumerate(word_level):
                q_char[i][j] = DataUtils.char_dict[char_item.lower()]

        d_char = np.ones((DataUtils.MAX_DOC_LENGTH, DataUtils.MAX_CHAR_PER_WORD), dtype=int) * (-1)
        for i, word_level in enumerate(item['document_char']):
            if i >= 300:
                break
            for j, char_item in enumerate(word_level):
                d_char[i][j] = DataUtils.char_dict[char_item.lower()]

        q_feature = np.ones((DataUtils.MAX_Q_LENGTH, 4), dtype=int) * (-1)
        for i, tok in enumerate(item['question']):
            q_feature[i][0] = item['qpos_idx'][i]
            q_feature[i][1] = item['qner_idx'][i]
            q_feature[i][2] = item['q_em_feature'][i]
            q_feature[i][3] = item['q_type']

        d_feature = np.ones((DataUtils.MAX_DOC_LENGTH, 4), dtype=int) * (-1)
        for i, tok in enumerate(item['document']):
            d_feature[i][0] = item['cpos_idx'][i]
            d_feature[i][1] = item['cner_idx'][i]
            d_feature[i][2] = item['d_em_feature'][i]
            d_feature[i][3] = QuestionType.OTHER

        q_mask = item['q_mask']
        d_mask = item['d_mask']
        # target = item['answers']
        # assert((len(item['answers']) == 3) or (len(item['answers']) == 1))
        # answer_a = [np.asarray(i) for i in item['answers'] if len(i) == 2]
        answer_a = [np.asarray(i) for i in item['answers']]
        # target = np.asarray(item['answers'])
        target = np.asarray(answer_a)

        ret = (d,)
        ret += (d_char,)
        ret += (d_feature,)
        ret += (np.asarray(d_mask, dtype=int),)
        ret += (q,)
        ret += (q_char,)
        ret += (q_feature,)
        ret += (np.asarray(q_mask, dtype=int),)

        if DataUtils.use_elmo:
            # add elmo id
            context_ids = DataUtils.batcher.batch_sentences([item["document"]], add_bos_eos=False)
            question_ids = DataUtils.batcher.batch_sentences([item["question"]], add_bos_eos=False)

            c_ids = np.zeros((DataUtils.MAX_DOC_LENGTH, 50), dtype=np.int32)
            q_ids = np.zeros((DataUtils.MAX_Q_LENGTH, 50), dtype=np.int32)
            c_ids[:len(context_ids[0]), :] = context_ids[0]
            q_ids[:len(question_ids[0]), :] = question_ids[0]

            ret += (c_ids,)
            ret += (q_ids,)

        if DataUtils.use_dict:
            d_has_gloss = np.zeros(DataUtils.MAX_DOC_LENGTH, dtype=np.int32)
            q_has_gloss = np.zeros(DataUtils.MAX_Q_LENGTH, dtype=np.int32)
            d_glosses = np.zeros((DataUtils.MAX_DOC_LENGTH, 10, 512), dtype=np.float32)
            q_glosses = np.zeros((DataUtils.MAX_Q_LENGTH, 10, 512), dtype=np.float32)
            for i, tok in enumerate(item['document']):
                if tok in DataUtils.dictionary:
                    d_has_gloss[i] = 1
                    for j, d_item in enumerate(DataUtils.dictionary[tok]):
                        # make definition 10 at most
                        if j >= 10:
                            break
                        # gloss = [int(did) for did in d_item["definition_ids"].strip().split()]

                        d_glosses[i, j] = DataUtils.dict_embedding[d_item["definition"]]

            for i, tok in enumerate(item['question']):
                if tok in DataUtils.dictionary:
                    q_has_gloss[i] = 1
                    for j, d_item in enumerate(DataUtils.dictionary[tok]):
                        # make definition 10 at most
                        if j >= 10:
                            break
                        q_glosses[i, j] = DataUtils.dict_embedding[d_item["definition"]]

            ret += (d_has_gloss,)
            ret += (d_glosses,)
            ret += (q_has_gloss,)
            ret += (q_glosses,)

        ret += (target,)

        # debug
        m_sum = np.sum(np.asarray(d_mask, dtype=np.int32))
        for t in target:
            if m_sum <= t[0] or m_sum <= t[1]:
                print("m_sum={}, target[0]:{},target[1]:{},document={}, question={}", m_sum, t[0], t[1],
                      " ".join(item["document"]), " ".join(item["question"]))
        # debug
        # if target[0][1] == 216:
        #    print(item['document'])

        if len(item['answers']) == 0:
            print("id:\n")
            print(item["id"])
            print("doc:\n")
            print(item["document"])
            print("question:\n")
            print(item["question"])

        return ret

    @staticmethod
    def convert_item_dev(item):

        # d = np.ones(DataUtils.MAX_DOC_LENGTH, dtype=int) * (-1)
        d = []
        for i, word in enumerate(item['document']):
            # d[i] = DataUtils.word_dict[word.lower()]
            d.append(word)

        ret = DataUtils.convert_item(item)

        ret += (np.asarray(d),)

        return ret

    @staticmethod
    def load_dictionary(file, embedding_file=None):

        json_data = json.load(open(file))

        """
        converted_dict = {}

        for key in json_data:

            # for debug when using part of data
            if key.lower() not in DataUtils.word_dict:
                continue
            converted_dict[DataUtils.word_dict[key.lower()]] = json_data[key]
        """
        # return converted_dict
        # DataUtils.dictionary = converted_dict
        DataUtils.dictionary = json_data
        DataUtils.dict_embedding = np.load(embedding_file)
        DataUtils.use_dict = True
