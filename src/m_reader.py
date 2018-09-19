import chainer
from chainer import Variable
from chainer import Parameter
import chainer.functions as F
import chainer.links as L
import six
from linkers import *
from evaluation import *
from progressbar import ProgressBar
import math
import os
import chainer.computational_graph as CG


class MReader(chainer.Chain):

    def __init__(self, args):
        super(MReader, self).__init__()

        with self.init_scope():
            self.args = args
            # word embedding layer
            self.w_embedding = L.EmbedID(self.args.vocab_size,
                                         self.args.embedding_dim, initialW=self.args.w_embeddings, ignore_label=-1)

            # character embedding layer
            self.char_embedding = L.EmbedID(self.args.char_size,
                                            self.args.char_embedding_dim, initialW=self.args.char_embeddings,
                                            ignore_label=-1)

            ## feature embedding layer
            # pos feature embedding
            self.pos_embedding = L.EmbedID(self.args.pos_size, self.args.pos_size, ignore_label=-1)
            # ner feature embedding
            self.ner_embedding = L.EmbedID(self.args.ner_size, self.args.ner_size, ignore_label=-1)
            # question type embedding used as a feature
            self.q_type_embedding = L.EmbedID(self.args.qtype_size, self.args.qtype_size, ignore_label=-1)
            # self.em_embedding = L.EmbedID(self.args.em_size, self.args.pos_size, ignore_label=-1)

            # bilstm for character encoding
            self.char_bilstm = L.NStepBiLSTM(n_layers=1, in_size=args.char_embedding_dim,
                                             out_size=args.char_hidden_size, dropout=args.char_dropout)

            # encoder
            # encoder_input_size = args.embedding_dim + args.char_hidden_size * 2 + args.num_features
            encoder_input_size = args.embedding_dim + args.char_hidden_size * 2 + args.pos_size + args.ner_size + args.qtype_size + 1
            self.encoder_bilstm = L.NStepBiLSTM(n_layers=1, in_size=encoder_input_size,
                                                out_size=args.encoder_hidden_size, dropout=args.encoder_dropout)

            # Interactive aligning, self aligning and aggregating
            self.interactive_aligners = chainer.ChainList()
            self.interactive_SFUs = chainer.ChainList()
            self.self_aligners = chainer.ChainList()
            self.self_SFUs = chainer.ChainList()
            self.aggregate_rnns = chainer.ChainList()

            context_hidden_size = 2 * args.encoder_hidden_size

            for _ in six.moves.range(args.hops):
                # Iterative Aligner
                self.interactive_aligners.append(InteractiveAligner())
                self.interactive_SFUs.append(SFU(context_hidden_size, 3 * context_hidden_size))

                # Self Aligner
                self.self_aligners.append(SelfAttnAligner())
                self.self_SFUs.append(SFU(context_hidden_size, 3 * context_hidden_size))
                self.aggregate_rnns.append(L.NStepBiLSTM(n_layers=1, in_size=context_hidden_size,
                                                         out_size=args.encoder_hidden_size,
                                                         dropout=args.encoder_dropout))

            self.mem_ans_ptr = MemAnsPtr(self.args)

            # self.f1 = AverageMeter()
            # self.exact_match = AverageMeter()

    def forward(self, c, c_char, c_feature, c_mask, q, q_char, q_feature, q_mask):
        """Inputs:
        c = document word indices             [batch * len_d]
        c_char = document char indices           [batch * len_d * len_c]
        c_feature = document word features indices  [batch * len_d * nfeat * feature_dims(various)]
        c_mask = document padding mask        [batch * len_d]
        q = question word indices             [batch * len_q]
        q_char = document char indices           [batch * len_d]
        q_feature = document word features indices  [batch * len_d * nfeat * feature_dims(various)]
        q_mask = question padding mask        [batch * len_q]
        """

        # Embed word embedding 
        c_emb = self.w_embedding(c)
        q_emb = self.w_embedding(q)

        # fix word embedding matrix
        self.w_embedding.disable_update()

        # Embed character embedding
        c_char_emb = self.char_embedding(c_char)
        q_char_emb = self.char_embedding(q_char)

        # fix character embedding matrix
        self.char_embedding.disable_update()

        # reshape context for feeding into BiLSTM
        batch_c_size, max_c_seq_len, max_c_char_len, char_embedding_size = c_char_emb.shape
        c_char_bilstm_input = F.reshape(c_char_emb, (batch_c_size * max_c_seq_len, max_c_char_len, char_embedding_size))
        # split_c_char_bilstm_input = self.xp.vsplit(c_char_bilstm_input, batch_c_size * max_c_seq_len)
        split_c_char_bilstm_input = F.split_axis(c_char_bilstm_input, batch_c_size * max_c_seq_len, axis=0)

        # reshape context for feeding into BiLSTM
        batch_q_size, max_q_seq_len, max_q_char_len, char_embedding_size = q_char_emb.shape
        q_char_bilstm_input = F.reshape(q_char_emb, (batch_q_size * max_q_seq_len, max_q_char_len, char_embedding_size))
        # split_q_char_bilstm_input = self.xp.vsplit(q_char_bilstm_input, batch_q_size * max_q_seq_len)
        split_q_char_bilstm_input = F.split_axis(q_char_bilstm_input, batch_q_size * max_q_seq_len, axis=0)

        # [batch_size, seq_len, dims] -> tuple of [seq_len, dims]
        split_c_char_bilstm_input = [F.squeeze(i) for i in split_c_char_bilstm_input]

        # [batch_size, seq_len, dims] -> tuple of [seq_len, dims]
        split_q_char_bilstm_input = [F.squeeze(i) for i in split_q_char_bilstm_input]

        # c_char_hidden, _, _ = self.char_bilstm(None, None, c_char_emb)
        # q_char_hidden, _, _ = self.char_bilstm(None, None, q_char_emb)
        c_char_hidden, _, _ = self.char_bilstm(None, None, split_c_char_bilstm_input)
        q_char_hidden, _, _ = self.char_bilstm(None, None, split_q_char_bilstm_input)

        # concat forward and backward representation
        c_char_hidden = F.concat([c_char_hidden[0], c_char_hidden[1]], axis=1)
        q_char_hidden = F.concat([q_char_hidden[0], q_char_hidden[1]], axis=1)

        # back to original size
        c_char_hidden = F.reshape(c_char_hidden, (batch_c_size, max_c_seq_len, -1))
        q_char_hidden = F.reshape(q_char_hidden, (batch_q_size, max_q_seq_len, -1))

        c_input = [c_emb, c_char_hidden]
        q_input = [q_emb, q_char_hidden]

        # Add additional features
        if self.args.num_features > 0:
            c_pos_feat, c_ner_feat, c_em_feat, c_qtype_feat = F.split_axis(c_feature, 4, axis=2)

            ## feature embedding
            c_pos_input = self.pos_embedding(c_pos_feat)
            c_ner_input = self.ner_embedding(c_ner_feat)
            c_qtype_input = self.q_type_embedding(c_qtype_feat)

            c_pos_input = F.squeeze(c_pos_input)
            c_ner_input = F.squeeze(c_ner_input)
            # c_em_feat_input = F.squeeze(c_em_feat).data.astype(self.xp.float32)
            c_em_feat_input = c_em_feat.data.astype(self.xp.float32)
            c_qtype_input = F.squeeze(c_qtype_input)

            c_input.append(c_pos_input)
            c_input.append(c_ner_input)
            c_input.append(c_em_feat_input)
            c_input.append(c_qtype_input)

            q_pos_feat, q_ner_feat, q_em_feat, q_qtype_feat = F.split_axis(q_feature, 4, axis=2)
            q_pos_input = self.pos_embedding(q_pos_feat)
            q_ner_input = self.ner_embedding(q_ner_feat)
            q_qtype_input = self.q_type_embedding(q_qtype_feat)

            q_pos_input = F.squeeze(q_pos_input)
            q_ner_input = F.squeeze(q_ner_input)
            # q_em_feat_input = F.squeeze(q_em_feat).data.astype(self.xp.float32)
            q_em_feat_input = q_em_feat.data.astype(self.xp.float32)
            q_qtype_input = F.squeeze(q_qtype_input)

            q_input.append(q_pos_input)
            q_input.append(q_ner_input)
            q_input.append(q_em_feat_input)
            q_input.append(q_qtype_input)

            # c_input.append(c_feature)
            # q_input.append(q_feature)

        # Encode context with bi-lstm
        c_input = F.concat(c_input, axis=2)
        c_input_bilstm = [i for i in c_input]

        # _, _, context = self.encoder_bilstm(None, None, F.concat(c_input, axis=2))
        _, _, context = self.encoder_bilstm(None, None, c_input_bilstm)

        # Encode question with bi-lstm
        q_input = F.concat(q_input, axis=2)
        q_input_bilstm = [i for i in q_input]
        # _, _, question = self.encoder_bilstm(None, None, F.concat(q_input, axis=2))
        _, _, question = self.encoder_bilstm(None, None, q_input_bilstm)

        # Align and aggregate
        c_check = context

        c_check = F.stack(c_check, axis=0)
        question = F.stack(question, axis=0)

        for i in six.moves.range(self.args.hops):
            '''
            q_tide, _ = self.interactive_aligners[i].forward(c_check, question, q_mask)
            
            c_bar = self.interactive_SFUs[i].forward(c_check,
                                                     F.concat([q_tide, c_check * q_tide, c_check - q_tide], axis=2))

            c_tide, _ = self.self_aligners[i].forward(c_bar, c_mask)
            c_hat = self.self_SFUs[i].forward(c_bar, F.concat([c_tide, c_bar * c_tide, c_bar - c_tide], axis=2))
            # c_check = self.aggregate_rnns[i].forward(c_hat, c_mask)

            # batch to tuple
            c_hat_input = [F.squeeze(item) for item in c_hat]
            # _, _, c_check = self.aggregate_rnns[i](None, None, c_hat)
            _, _, c_check = self.aggregate_rnns[i](None, None, c_hat_input)
            c_check = F.stack(c_check, axis=0)
            '''
            q_tide, _ = self.interactive_aligners[i](c_check, question, q_mask)

            c_bar = self.interactive_SFUs[i](c_check,
                                             F.concat([q_tide, c_check * q_tide, c_check - q_tide], axis=2))

            c_tide, _ = self.self_aligners[i](c_bar, c_mask)
            c_hat = self.self_SFUs[i](c_bar, F.concat([c_tide, c_bar * c_tide, c_bar - c_tide], axis=2))
            # c_check = self.aggregate_rnns[i].forward(c_hat, c_mask)

            # batch to tuple
            c_hat_input = [F.squeeze(item) for item in c_hat]
            # _, _, c_check = self.aggregate_rnns[i](None, None, c_hat)
            _, _, c_check = self.aggregate_rnns[i](None, None, c_hat_input)
            c_check = F.stack(c_check, axis=0)

        # start_scores, end_scores = self.mem_ans_ptr.forward(c_check, question, c_mask, q_mask)
        start_scores, end_scores = self.mem_ans_ptr(c_check, question, c_mask, q_mask)

        return start_scores, end_scores

    def get_loss_function(self):

        def loss_f(c, c_char, c_feature, c_mask, q, q_char, q_feature, q_mask, target):
            # rec_loss = 0
            start_scores, end_scores = self.forward(c, c_char, c_feature, c_mask, q, q_char, q_feature, q_mask)

            # start_losses = F.bernoulli_nll(start_scores, target[:, 0, 0].astype(self.xp.float32))
            # end_losses = F.bernoulli_nll(end_scores, target[:, 0, 1].astype(self.xp.float32))

            # start_losses = F.bernoulli_nll(target[:, 0, 0].astype(self.xp.float32), start_scores)
            # end_losses = F.bernoulli_nll(target[:, 0, 1].astype(self.xp.float32), end_scores)
            start_losses = self.neg_loglikelihood_fun(target[:, 0, 0], start_scores)

            end_losses = self.neg_loglikelihood_fun(target[:, 0, 1], end_scores, c_mask)

            rec_loss = start_losses + end_losses
            self.rec_loss = rec_loss
            self.loss = rec_loss

            ''''''
            # computational graph
            if (self.args.dot_file is not None) and os.path.exists(self.args.dot_file) is False:
                g = CG.build_computational_graph((self.rec_loss,))
                with open(self.args.dot_file, 'w') as f:
                    f.write(g.dump())

            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss}, observer=self)
            return self.loss

        return loss_f

    def neg_loglikelihood_fun(self, target, distribution, mask=None):

        sum = 0

        batch_size, _ = distribution.shape

        for i in six.moves.range(batch_size):
            # if (len(distribution))
            if distribution[i][target[i]].data <= 0:
                if mask is not None:
                    m_sum = self.xp.sum(mask[i])
                    print("index:{}, target[i]:{}, len:{}".format(i, target[i], m_sum))
            sum += -F.log(distribution[i][target[i]])

        return sum

    '''
    def get_evaluation_fun(self):

        def evaluate_func(c, c_char, c_feature, c_mask, q, q_char, q_feature, q_mask, target, d_text):

            # eval_time = utils.Timer()
            # f1 = AverageMeter()
            # exact_match = AverageMeter()

            # Run through examples
            start_scores, end_scores = self.forward(c, c_char, c_feature, c_mask, q, q_char, q_feature, q_mask)

            size = len(start_scores)

            pbar = ProgressBar()
            for i, (s, e) in pbar(enumerate(zip(start_scores, end_scores))):
                pred_s = F.argmax(s)
                pred_e = F.argmax(e)

                start_p = pred_s.data
                end_p = pred_e.data

                if start_p > end_p:
                    self.exact_match.update(0)
                    self.f1.update(0)
                    continue

                # prediction = c[i][start_p:end_p]
                prediction = ' '.join(d_text[i][start_p:end_p])
                ground_truths = []
                for truth in target[i]:
                    # ground_truths.extend(c[i][truth[0]:truth[1]])
                    ground_truths.append(''.join(d_text[i][truth[0]:truth[1]]))

                self.exact_match.update(metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths))
                self.f1.update(metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths))

            chainer.report({"validation/main/f1": self.f1.avg, "validation/main/em": self.exact_match.avg},
                           observer=self)

        return evaluate_func
    '''
