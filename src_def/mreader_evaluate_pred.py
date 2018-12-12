import chainer
from chainer.backends import cuda
import chainer.functions as F
from evaluation import *
from utils import *
import six
from chainer.dataset import convert
import os
import chainer.computational_graph as c


class MReaderEvaluatorPred():
    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, model, test_data, f1_key, em_key,
                 batch_size=100, device=-1, max_length=100, dot_file=None, file_p = ""):
        self.model = model
        self.test_data = test_data
        self.f1_key = f1_key
        self.em_key = em_key
        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length
        self.dot_file = dot_file
        self.output_file = file_p

    def __call__(self):

        with chainer.no_backprop_mode():
            f1 = AverageMeter()
            exact_match = AverageMeter()

            with open(self.output_file, "w") as of:
                # Run through examples
                examples = 0
                for i in six.moves.range(0, len(self.test_data), self.batch_size):
                    # ex_id, batch_size = ex[-1], ex[0].size(0)
                    # create batch
                    batch = []
                    data_size = self.batch_size if (i + self.batch_size <= len(self.test_data) - 1) else (
                            len(self.test_data) - i)
                    for j in six.moves.range(data_size):
                        item = DataUtils.convert_item_dev(self.test_data[i + j])
                        input_item = item[:-2]
                        batch.append(input_item)

                    input_item_gpu = convert.concat_examples(batch, self.device)
                    # input_item_gpu = cuda.to_gpu(input_item, self.device)
                    pred_s, pred_e = self.model.forward(*input_item_gpu)

                    for j, (s, e) in enumerate(zip(pred_s, pred_e)):

                        max_s = 0
                        max_e = 0

                        max_val = 0
                        # argmax_j1 = 0
                        start = cuda.to_cpu(s.data)
                        end = cuda.to_cpu(e.data)
                        for k in six.moves.range(len(start)):

                            val1 = start[max_s]
                            if start[k] > val1:
                                val1 = start[k]
                                max_s = k

                                # 20180927 add start
                                # reset max_val
                                # max_val = 0
                                # 20180927 add end

                            val2 = end[k]
                            if val1 * val2 > max_val:
                                max_e = k
                                max_val = val1 * val2

                        start_p = max_s
                        end_p = max_e

                        text = self.test_data[i + j]['document']
                        prediction = ' '.join(text[start_p:end_p + 1])

                        ground_truths = self.test_data[i + j]['answers_text']
                        '''
                        target = self.test_data[i + j]['answers_text']
                        ground_truths = []
                        for truth in target:
                            # ground_truths.extend(c[i][truth[0]:truth[1]])
                            ground_truths.append(''.join(text[truth[0]:truth[1]]))
                        '''
                        exact_match_tmp = metric_max_over_ground_truths(
                            exact_match_score, prediction, ground_truths)
                        exact_match.update(exact_match_tmp)

                        f1_tmp = metric_max_over_ground_truths(
                            f1_score, prediction, ground_truths)
                        f1.update(f1_tmp)
                        of.write("t:{}\n".format(text))
                        of.write("q:{}\n".format(self.test_data[i+j]["question"]))
                        of.write("p:{}\n".format(prediction))
                        of.write("g:{}\n".format(ground_truths))
                        of.write("f1:{}, em:{}\n".format(f1_tmp, exact_match_tmp))
                        of.write("\n")

                '''
                prediction = item[0][pred_s:pred_e] # get predicted answer
                ground_truths = ex['answers_text']
                
                exact_match.update(utils.metric_max_over_ground_truths(
                        utils.exact_match_score, prediction, ground_truths))
                f1.update(utils.metric_max_over_ground_truths(
                        utils.f1_score, prediction, ground_truths))
                '''
            print("f1:{0}, em:{1}".format(f1.avg, exact_match.avg))
        chainer.report({self.f1_key: f1.avg, self.em_key: exact_match.avg})
