import chainer
from chainer import Variable
from chainer import Parameter
import chainer.functions as F
import chainer.links as L
import six
from linkers_v6 import *
from evaluation import *
from progressbar import ProgressBar
import math
import os
import chainer.computational_graph as CG
from collections import Counter
from chainer.backends import cuda

from bilm import Elmo
import json


class MReader_V6(chainer.Chain):

    def __init__(self, args):
        super(MReader_V6, self).__init__()

        with self.init_scope():
            self.args = args

            # add dictionary
            """
            # self.dictionary = json.load(args.dictionary_file)
            # self.dictionary = args.dictionary
            self.dict_embedding_weight = self.xp.load(args.dict_embedding_file)
            self.dict_embedding = L.EmbedID(len(self.dict_embedding_weight), self.args.dict_embedding_dim,
                                            initialW=self.dict_embedding_weight, ignore_label=-1)

            self.dictionary_encoder = L.NStepBiLSTM(n_layers=1, in_size=args.dict_max_length,
                                                    out_size=args.encoder_hidden_size, dropout=args.dict_dropout)
            """

            self.dict_attn_module = AttnModule(args.embedding_dim + args.char_hidden_size * 2)

            if args.use_elmo:
                self.elmo = Elmo(
                    args.options_file,
                    args.weight_file,
                    # num_output_representations=2,
                    num_output_representations=1,
                    requires_grad=False,
                    do_layer_norm=False,
                    dropout=0.)

            # gamma
            self.gamma = Parameter(
                initializer=self.xp.array([3]).astype('f')
            )

            # gamma
            self.sigma_a = Parameter(
                initializer=self.xp.array([3]).astype('f')
            )

            # gamma
            self.sigma_b = Parameter(
                initializer=self.xp.array([3]).astype('f')
            )
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

            # encoder_input_size = args.embedding_dim + args.char_hidden_size * 2 + args.pos_size + args.ner_size + args.qtype_size + 1
            if args.use_elmo:
                encoder_input_size = args.embedding_dim + args.char_hidden_size * 2 + args.pos_size + args.ner_size + \
                                     1024 + 1
            else:
                encoder_input_size = args.embedding_dim + args.char_hidden_size * 2 + args.pos_size + args.ner_size + 1

            if args.use_dict:
                encoder_input_size += args.dict_embedding_dim

            self.encoder_bilstm = L.NStepBiLSTM(n_layers=1, in_size=encoder_input_size,
                                                out_size=args.encoder_hidden_size, dropout=args.encoder_dropout)

            # Interactive aligning, self aligning and aggregating
            self.interactive_aligners = chainer.ChainList()
            self.interactive_SFUs = chainer.ChainList()
            self.self_aligners = chainer.ChainList()
            self.self_SFUs = chainer.ChainList()
            self.aggregate_rnns = chainer.ChainList()

            context_hidden_size = 2 * args.encoder_hidden_size

            for i in six.moves.range(args.hops):
                # Iterative Aligner
                self.interactive_aligners.append(InteractiveAligner_V6(dim=self.args.nonlinear_dim))
                self.interactive_SFUs.append(SFU_V6(context_hidden_size, 3 * context_hidden_size))

                # Self Aligner
                self.self_aligners.append(SelfAttnAligner_V6(dim=self.args.nonlinear_dim))
                self.self_SFUs.append(SFU_V6(context_hidden_size, 3 * context_hidden_size))

                if i < args.hops - 1:
                    self.aggregate_rnns.append(L.NStepBiLSTM(n_layers=1, in_size=context_hidden_size,
                                                             out_size=args.encoder_hidden_size,
                                                             dropout=args.encoder_dropout))
                else:
                    self.aggregate_rnns.append(L.NStepBiLSTM(n_layers=1, in_size=context_hidden_size * args.hops,
                                                             out_size=args.encoder_hidden_size,
                                                             dropout=args.encoder_dropout))

            self.mem_ans_ptr = MemAnsPtr_V6_Variant(self.args, context_hidden_size, 3 * context_hidden_size)

            # self.f1 = AverageMeter()
            # self.exact_match = AverageMeter()

    def forward(self, c, c_char, c_feature, c_mask, q, q_char, q_feature, q_mask,
                c_has_gloss=None, c_gloss=None, q_has_gloss=None, q_gloss=None):
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

        # [batch_size, seq_len, dims] -> list of [seq_len, dims]
        split_c_char_bilstm_input = [F.squeeze(i) for i in split_c_char_bilstm_input]

        # [batch_size, seq_len, dims] -> list of [seq_len, dims]
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

        # add dict
        if self.args.use_dict:
            c_dict_input = F.concat(c_input, axis=2)
            q_dict_input = F.concat(q_input, axis=2)

            batch_size = c_dict_input.shape[0]

            # c_dict_output = []
            # q_dict_output = []
            c_dict_output = self.xp.zeros((batch_size, max_c_seq_len, self.args.dict_embedding_dim),
                                          dtype=self.xp.float32)
            q_dict_output = self.xp.zeros((batch_size, max_q_seq_len, self.args.dict_embedding_dim),
                                          dtype=self.xp.float32)
            # for each data
            for i in six.moves.range(batch_size):
                c_dict_o_seq = []
                # for j in six.moves.range(F.sum(self.xp.asarray(c_mask, dtype=self.xp.float32)[i]).data):
                j = 0
                """
                while j < len(c_mask[i]) and (c_mask[i][j] != 0):
                    if c_has_gloss[i][j] != 1:
                        c_dict_output_var = Variable(self.xp.zeros(self.args.dict_embedding_dim))
                    else:
                        dict_data = c_gloss[i, j, :, :]
                        definition_seq = self.dict_embedding(dict_data)
                        definition_list = [definition_seq[t] for t in range(len(definition_seq))]
                        _, _, definition_c_emb = self.dictionary_encoder(None, None, definition_list)
                        # _, _, definition_c_emb = self.dictionary_encoder(definition_seq)
                        definition_c_emb = F.stack(definition_c_emb, axis=0)

                        c_dict_output_var = self.dict_attn_module(F.expand_dims(c_dict_input[i][j], axis=0),
                                                                  definition_c_emb[:, -1, :])
                    c_dict_o_seq.append(c_dict_output_var)
                    j += 1
                c_dict_output.append(c_dict_o_seq)

                q_dict_o_seq = []
                # for j in six.moves.range(F.sum(self.xp.asarray(c_mask, dtype=self.xp.float32)[i]).data):
                j = 0
                while j < len(q_mask[i]) and (q_mask[i][j] != 0):
                    if q_has_gloss[i][j] != 1:
                        q_dict_output_var = Variable(self.xp.zeros(self.args.dict_embedding_dim))
                    else:
                        dict_data = q_gloss[i, j, :]
                        definition_seq = self.dict_embedding(dict_data)
                        definition_list = [definition_seq[t] for t in range(len(definition_seq))]
                        _, _, definition_q_emb = self.dictionary_encoder(None, None, definition_list)
                        # _, _, definition_q_emb = self.dictionary_encoder(definition_seq)

                        # definition_q_emb = F.stack(definition_q_emb, axis=0)

                        q_dict_output_var = self.dict_attn_module(F.expand_dims(q_dict_input[i][j], axis=0),
                                                                  definition_q_emb[:, -1, :])
                    q_dict_o_seq.append(q_dict_output_var)
                    j += 1
                q_dict_output.append(q_dict_o_seq)
                """
                while j < len(c_mask[i]) and (c_mask[i][j] != 0):
                    if c_has_gloss[i][j] != 1:
                        c_dict_output_var = Variable(self.xp.zeros(self.args.dict_embedding_dim, dtype=self.xp.float32))
                    else:
                        c_dict_output_var = self.dict_attn_module(F.expand_dims(c_dict_input[i][j], axis=0),
                                                                  c_gloss[i, j])
                    # c_dict_o_seq.append(c_dict_output_var)
                    c_dict_output[i][j] = F.squeeze(c_dict_output_var).data
                    j += 1
                # c_dict_output.append(c_dict_o_seq)

                q_dict_o_seq = []
                j = 0
                while j < len(q_mask[i]) and (q_mask[i][j] != 0):
                    if q_has_gloss[i][j] != 1:
                        q_dict_output_var = Variable(self.xp.zeros(self.args.dict_embedding_dim, dtype=self.xp.float32))
                    else:
                        q_dict_output_var = self.dict_attn_module(F.expand_dims(q_dict_input[i][j], axis=0),
                                                                  q_gloss[i, j])
                    # q_dict_o_seq.append(q_dict_output_var)

                    q_dict_output[i][j] = F.squeeze(q_dict_output_var).data
                    j += 1
                # q_dict_output.append(q_dict_o_seq)

            c_input.append(c_dict_output)
            q_input.append(q_dict_output)

        # Add additional features
        if self.args.num_features > 0:
            c_pos_feat, c_ner_feat, c_em_feat, c_qtype_feat = F.split_axis(c_feature, 4, axis=2)

            ## feature embedding
            c_pos_input = self.pos_embedding(c_pos_feat)
            c_ner_input = self.ner_embedding(c_ner_feat)

            # not mentioned in V6
            # c_qtype_input = self.q_type_embedding(c_qtype_feat)

            c_pos_input = F.squeeze(c_pos_input)
            c_ner_input = F.squeeze(c_ner_input)
            # c_em_feat_input = F.squeeze(c_em_feat).data.astype(self.xp.float32)
            c_em_feat_input = c_em_feat.data.astype(self.xp.float32)
            # not mentioned in V6
            # c_qtype_input = F.squeeze(c_qtype_input)

            c_input.append(c_pos_input)
            c_input.append(c_ner_input)
            c_input.append(c_em_feat_input)
            # not mentioned in V6
            # c_input.append(c_qtype_input)

            q_pos_feat, q_ner_feat, q_em_feat, q_qtype_feat = F.split_axis(q_feature, 4, axis=2)
            q_pos_input = self.pos_embedding(q_pos_feat)
            q_ner_input = self.ner_embedding(q_ner_feat)
            # not mentioned in V6
            # q_qtype_input = self.q_type_embedding(q_qtype_feat)

            q_pos_input = F.squeeze(q_pos_input)
            q_ner_input = F.squeeze(q_ner_input)
            # q_em_feat_input = F.squeeze(q_em_feat).data.astype(self.xp.float32)
            q_em_feat_input = q_em_feat.data.astype(self.xp.float32)
            # not mentioned in V6
            # q_qtype_input = F.squeeze(q_qtype_input)

            q_input.append(q_pos_input)
            q_input.append(q_ner_input)
            q_input.append(q_em_feat_input)
            # not mentioned in V6
            # q_input.append(q_qtype_input)

            # c_input.append(c_feature)
            # q_input.append(q_feature)

        """
        if self.args.use_elmo:
            ## add elmo
            
            batch_size = context_ids.shape[0]
            context_max_length = context_ids.shape[1]
            question_max_length = question_ids.shape[1]
            context_embeddings = []
            question_embeddings = []

            # comment: elmo batch require data to be ordered by their lengths -- longest sequences first
            for i in range(batch_size):
                context_elmo = self.elmo.forward(
                    self.xp.asarray([context_ids[i][:self.xp.sum(c_mask[i])]], dtype=self.xp.int32))
                context_embeddings.append(F.pad_sequence(context_elmo["elmo_representations"][0],
                                                         length=context_max_length))

                question_elmo = self.elmo.forward(
                    self.xp.asarray([question_ids[i][:self.xp.sum(q_mask[i])]], dtype=self.xp.int32))
                question_embeddings.append(F.pad_sequence(question_elmo["elmo_representations"][0],
                                                          length=question_max_length))

            c_input.append(F.vstack(context_embeddings))
            q_input.append(F.vstack(question_embeddings))
        """
        # Encode context with bi-lstm
        c_input = F.concat(c_input, axis=2)
        c_input_bilstm = [i for i in c_input]

        # _, _, context = self.encoder_bilstm(None, None, F.concat(c_input, axis=2))
        _, _, context = self.encoder_bilstm(None, None, c_input_bilstm)

        # q_input.append(F.vstack(question_embeddings))
        # Encode question with bi-lstm
        q_input = F.concat(q_input, axis=2)
        q_input_bilstm = [i for i in q_input]
        # _, _, question = self.encoder_bilstm(None, None, F.concat(q_input, axis=2))
        _, _, question = self.encoder_bilstm(None, None, q_input_bilstm)

        # Align and aggregate
        c_check = context

        c_check = F.stack(c_check, axis=0)
        question = F.stack(question, axis=0)

        z_mem = []

        e = None
        b = None

        """"""
        for i in six.moves.range(self.args.hops):
            q_tide, e = self.interactive_aligners[i](c_check, question, q_mask, e, b, self.gamma)
            # q_tide, _ = self.interactive_aligners[i](c_check, question, q_mask, e, b, self.args.gamma)

            h = self.interactive_SFUs[i](c_check,
                                         F.concat([q_tide, c_check * q_tide, c_check - q_tide], axis=2))

            h_tide, b = self.self_aligners[i](h, c_mask, b_param=b, gamma=self.gamma)
            # h_tide, _ = self.self_aligners[i](h, c_mask, b_param=b, gamma=self.args.gamma)

            z = self.self_SFUs[i](h, F.concat([h_tide, h * h_tide, h - h_tide], axis=2))

            z_mem.append(z)

            # batch to tuple
            c_hat_input = []
            if i < self.args.hops - 1:
                c_hat_input = [F.squeeze(item) for item in z]
            elif i == self.args.hops - 1:
                # res_z = F.concat(*z_mem, axis=z.shape[-1])
                res_z = z_mem[0]

                for j in range(1, len(z_mem)):
                    res_z = F.concat((res_z, z_mem[j]), axis=-1)
                c_hat_input = [F.squeeze(item) for item in res_z]

            _, _, c_check = self.aggregate_rnns[i](None, None, c_hat_input)
            c_check = F.stack(c_check, axis=0)

        # start_scores, end_scores = self.mem_ans_ptr.forward(c_check, question, c_mask, q_mask)
        start_scores, end_scores = self.mem_ans_ptr(c_check, question, c_mask, q_mask)

        return start_scores, end_scores

    def forward_elmo(self, c, c_char, c_feature, c_mask, q, q_char, q_feature, q_mask, context_ids=None,
                     question_ids=None,
                     c_has_gloss=None, c_gloss=None, q_has_gloss=None, q_gloss=None):
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

        # [batch_size, seq_len, dims] -> list of [seq_len, dims]
        split_c_char_bilstm_input = [F.squeeze(i) for i in split_c_char_bilstm_input]

        # [batch_size, seq_len, dims] -> list of [seq_len, dims]
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

        # add dict
        if self.args.use_dict:
            c_dict_input = F.concat(c_input, axis=2)
            q_dict_input = F.concat(q_input, axis=2)

            batch_size = c_dict_input.shape[0]

            # c_dict_output = []
            # q_dict_output = []
            c_dict_output = self.xp.zeros((batch_size, max_c_seq_len, self.args.dict_embedding_dim),
                                          dtype=self.xp.float32)
            q_dict_output = self.xp.zeros((batch_size, max_q_seq_len, self.args.dict_embedding_dim),
                                          dtype=self.xp.float32)
            # for each data
            for i in six.moves.range(batch_size):
                c_dict_o_seq = []
                # for j in six.moves.range(F.sum(self.xp.asarray(c_mask, dtype=self.xp.float32)[i]).data):
                j = 0
                """
                while j < len(c_mask[i]) and (c_mask[i][j] != 0):
                    if c_has_gloss[i][j] != 1:
                        c_dict_output_var = Variable(self.xp.zeros(self.args.dict_embedding_dim))
                    else:
                        dict_data = c_gloss[i, j, :, :]
                        definition_seq = self.dict_embedding(dict_data)
                        definition_list = [definition_seq[t] for t in range(len(definition_seq))]
                        _, _, definition_c_emb = self.dictionary_encoder(None, None, definition_list)
                        # _, _, definition_c_emb = self.dictionary_encoder(definition_seq)
                        definition_c_emb = F.stack(definition_c_emb, axis=0)

                        c_dict_output_var = self.dict_attn_module(F.expand_dims(c_dict_input[i][j], axis=0),
                                                                  definition_c_emb[:, -1, :])
                    c_dict_o_seq.append(c_dict_output_var)
                    j += 1
                c_dict_output.append(c_dict_o_seq)

                q_dict_o_seq = []
                # for j in six.moves.range(F.sum(self.xp.asarray(c_mask, dtype=self.xp.float32)[i]).data):
                j = 0
                while j < len(q_mask[i]) and (q_mask[i][j] != 0):
                    if q_has_gloss[i][j] != 1:
                        q_dict_output_var = Variable(self.xp.zeros(self.args.dict_embedding_dim))
                    else:
                        dict_data = q_gloss[i, j, :]
                        definition_seq = self.dict_embedding(dict_data)
                        definition_list = [definition_seq[t] for t in range(len(definition_seq))]
                        _, _, definition_q_emb = self.dictionary_encoder(None, None, definition_list)
                        # _, _, definition_q_emb = self.dictionary_encoder(definition_seq)

                        # definition_q_emb = F.stack(definition_q_emb, axis=0)

                        q_dict_output_var = self.dict_attn_module(F.expand_dims(q_dict_input[i][j], axis=0),
                                                                  definition_q_emb[:, -1, :])
                    q_dict_o_seq.append(q_dict_output_var)
                    j += 1
                q_dict_output.append(q_dict_o_seq)
                """
                while j < len(c_mask[i]) and (c_mask[i][j] != 0):
                    if c_has_gloss[i][j] != 1:
                        c_dict_output_var = Variable(self.xp.zeros(self.args.dict_embedding_dim, dtype=self.xp.float32))
                    else:
                        c_dict_output_var = self.dict_attn_module(F.expand_dims(c_dict_input[i][j], axis=0),
                                                                  c_gloss[i, j])
                    # c_dict_o_seq.append(c_dict_output_var)
                    c_dict_output[i][j] = F.squeeze(c_dict_output_var).data
                    j += 1
                # c_dict_output.append(c_dict_o_seq)

                q_dict_o_seq = []
                j = 0
                while j < len(q_mask[i]) and (q_mask[i][j] != 0):
                    if q_has_gloss[i][j] != 1:
                        q_dict_output_var = Variable(self.xp.zeros(self.args.dict_embedding_dim, dtype=self.xp.float32))
                    else:
                        q_dict_output_var = self.dict_attn_module(F.expand_dims(q_dict_input[i][j], axis=0),
                                                                  q_gloss[i, j])
                    # q_dict_o_seq.append(q_dict_output_var)

                    q_dict_output[i][j] = F.squeeze(q_dict_output_var).data
                    j += 1
                # q_dict_output.append(q_dict_o_seq)

            c_input.append(c_dict_output)
            q_input.append(q_dict_output)

        # Add additional features
        if self.args.num_features > 0:
            c_pos_feat, c_ner_feat, c_em_feat, c_qtype_feat = F.split_axis(c_feature, 4, axis=2)

            ## feature embedding
            c_pos_input = self.pos_embedding(c_pos_feat)
            c_ner_input = self.ner_embedding(c_ner_feat)

            # not mentioned in V6
            # c_qtype_input = self.q_type_embedding(c_qtype_feat)

            c_pos_input = F.squeeze(c_pos_input)
            c_ner_input = F.squeeze(c_ner_input)
            # c_em_feat_input = F.squeeze(c_em_feat).data.astype(self.xp.float32)
            c_em_feat_input = c_em_feat.data.astype(self.xp.float32)
            # not mentioned in V6
            # c_qtype_input = F.squeeze(c_qtype_input)

            c_input.append(c_pos_input)
            c_input.append(c_ner_input)
            c_input.append(c_em_feat_input)
            # not mentioned in V6
            # c_input.append(c_qtype_input)

            q_pos_feat, q_ner_feat, q_em_feat, q_qtype_feat = F.split_axis(q_feature, 4, axis=2)
            q_pos_input = self.pos_embedding(q_pos_feat)
            q_ner_input = self.ner_embedding(q_ner_feat)
            # not mentioned in V6
            # q_qtype_input = self.q_type_embedding(q_qtype_feat)

            q_pos_input = F.squeeze(q_pos_input)
            q_ner_input = F.squeeze(q_ner_input)
            # q_em_feat_input = F.squeeze(q_em_feat).data.astype(self.xp.float32)
            q_em_feat_input = q_em_feat.data.astype(self.xp.float32)
            # not mentioned in V6
            # q_qtype_input = F.squeeze(q_qtype_input)

            q_input.append(q_pos_input)
            q_input.append(q_ner_input)
            q_input.append(q_em_feat_input)
            # not mentioned in V6
            # q_input.append(q_qtype_input)

            # c_input.append(c_feature)
            # q_input.append(q_feature)

        if self.args.use_elmo:
            ## add elmo
            '''
            context_embeddings = self.elmo.forward(context_ids)
            question_embeddings = self.elmo.forward(question_ids)
            '''
            batch_size = context_ids.shape[0]
            context_max_length = context_ids.shape[1]
            question_max_length = question_ids.shape[1]
            context_embeddings = []
            question_embeddings = []

            # comment: elmo batch require data to be ordered by their lengths -- longest sequences first
            for i in range(batch_size):
                context_elmo = self.elmo.forward(
                    self.xp.asarray([context_ids[i][:self.xp.sum(c_mask[i])]], dtype=self.xp.int32))
                context_embeddings.append(F.pad_sequence(context_elmo["elmo_representations"][0],
                                                         length=context_max_length))

                question_elmo = self.elmo.forward(
                    self.xp.asarray([question_ids[i][:self.xp.sum(q_mask[i])]], dtype=self.xp.int32))
                question_embeddings.append(F.pad_sequence(question_elmo["elmo_representations"][0],
                                                          length=question_max_length))

            c_input.append(F.vstack(context_embeddings))
            q_input.append(F.vstack(question_embeddings))

        # Encode context with bi-lstm
        c_input = F.concat(c_input, axis=2)
        c_input_bilstm = [i for i in c_input]

        # _, _, context = self.encoder_bilstm(None, None, F.concat(c_input, axis=2))
        _, _, context = self.encoder_bilstm(None, None, c_input_bilstm)

        # q_input.append(F.vstack(question_embeddings))
        # Encode question with bi-lstm
        q_input = F.concat(q_input, axis=2)
        q_input_bilstm = [i for i in q_input]
        # _, _, question = self.encoder_bilstm(None, None, F.concat(q_input, axis=2))
        _, _, question = self.encoder_bilstm(None, None, q_input_bilstm)

        # Align and aggregate
        c_check = context

        c_check = F.stack(c_check, axis=0)
        question = F.stack(question, axis=0)

        z_mem = []

        e = None
        b = None

        """"""
        for i in six.moves.range(self.args.hops):
            q_tide, e = self.interactive_aligners[i](c_check, question, q_mask, e, b, self.gamma)
            # q_tide, _ = self.interactive_aligners[i](c_check, question, q_mask, e, b, self.args.gamma)

            h = self.interactive_SFUs[i](c_check,
                                         F.concat([q_tide, c_check * q_tide, c_check - q_tide], axis=2))

            h_tide, b = self.self_aligners[i](h, c_mask, b_param=b, gamma=self.gamma)
            # h_tide, _ = self.self_aligners[i](h, c_mask, b_param=b, gamma=self.args.gamma)

            z = self.self_SFUs[i](h, F.concat([h_tide, h * h_tide, h - h_tide], axis=2))

            z_mem.append(z)

            # batch to tuple
            c_hat_input = []
            if i < self.args.hops - 1:
                c_hat_input = [F.squeeze(item) for item in z]
            elif i == self.args.hops - 1:
                # res_z = F.concat(*z_mem, axis=z.shape[-1])
                res_z = z_mem[0]

                for j in range(1, len(z_mem)):
                    res_z = F.concat((res_z, z_mem[j]), axis=-1)
                c_hat_input = [F.squeeze(item) for item in res_z]

            _, _, c_check = self.aggregate_rnns[i](None, None, c_hat_input)
            c_check = F.stack(c_check, axis=0)

        # start_scores, end_scores = self.mem_ans_ptr.forward(c_check, question, c_mask, q_mask)
        start_scores, end_scores = self.mem_ans_ptr(c_check, question, c_mask, q_mask)

        return start_scores, end_scores

    def set_arg(self, args):
        self.args = args

    def mle_loss_func(self, c, c_char, c_feature, c_mask, q, q_char, q_feature, q_mask, target, start_scores,
                      end_scores):

        # start_scores, end_scores = self.forward(c, c_char, c_feature, c_mask, q, q_char, q_feature, q_mask)

        start_losses = self.neg_loglikelihood_fun(target[:, 0, 0], start_scores)

        end_losses = self.neg_loglikelihood_fun(target[:, 0, 1], end_scores, c_mask)

        rec_loss = start_losses + end_losses

        return rec_loss

    def sampling_func(self, probs):

        # sampled_i = 0
        try:
            histogram = self.xp.random.multinomial(1, probs)
            sampled_i = int(self.xp.nonzero(histogram)[0])

        except Exception as e:
            choices = range(len(probs))
            sampled_i = self.xp.random.choice(choices, p=probs)

        return sampled_i

    def f1_score(self, prediction, ground_truth):
        """Compute the geometric mean of precision and recall for answer tokens."""

        # prediction = self.xp.array(map(str, prediction))
        # ground_truth = self.xp.array(map(str, ground_truth))

        # common = Counter(prediction) & Counter(ground_truth)
        common = Counter(cuda.to_cpu(prediction)) & Counter(cuda.to_cpu(ground_truth))
        num_same = cuda.to_gpu(self.xp.array(sum(common.values()), dtype=self.xp.int32), self.args.gpu)

        """
        g_dict = {}
        for i in ground_truth:
            if i in g_dict:
                g_dict[i] += 1
            else:
                g_dict[i] = 1

        cnt = 0
        for i in prediction:
            if i in g_dict:
                if g_dict[i] > 0:
                    g_dict[i] -= 1
                    cnt += 1

        num_same = cnt
        """

        if num_same == 0:
            return 0

        precision = 1.0 * num_same / len(prediction)
        recall = 1.0 * num_same / len(ground_truth)
        """

        if num_same[0] == 0:
            return 0

        precision = 1.0 * num_same[0] / len(prediction)
        recall = 1.0 * num_same[0] / len(ground_truth)
        """
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def rl_loss_func(self, c, c_char, c_feature, c_mask, q, q_char, q_feature, q_mask, target, start_scores,
                     end_scores):

        rl_loss = 0

        # start_scores, end_scores = self.forward(c, c_char, c_feature, c_mask, q, q_char, q_feature, q_mask)

        for j, (context, start_dis, end_dis, gold) in enumerate(zip(c, start_scores, end_scores, target)):

            max_s = 0
            max_e = 0

            max_val = 0

            for k in six.moves.range(len(start_dis.data)):

                val1 = start_dis.data[max_s]
                if start_dis.data[k] > val1:
                    val1 = start_dis.data[k]
                    max_s = k

                val2 = end_dis.data[k]
                if val1 * val2 > max_val:
                    max_e = k
                    max_val = val1 * val2

            start_p = max_s
            end_p = max_e

            # cont_a = context[start_p:end_p + 1]
            # cont_b = context[gold[0]:gold[1] + 1]

            r_baseline = self.f1_score(context[start_p:end_p + 1], context[gold[0, 0]:gold[0, 1] + 1])

            sample_s = self.sampling_func(start_dis.data)

            # end distribution normalization
            end_sum = self.xp.sum(end_dis.data[sample_s:])
            end_dis_sample = end_dis.data[sample_s:] / end_sum
            sample_e = self.sampling_func(end_dis_sample)

            r_sample = self.f1_score(context[sample_s:sample_s + sample_e + 1], context[gold[0, 0]: gold[0, 1] + 1])

            reward = r_sample - r_baseline
            if reward > 0:
                # rl_loss += -(F.log(start_dis[sample_s]) + F.log(end_dis[sample_e])) * (r_sample - r_baseline)
                rl_loss += -(F.log(start_dis[sample_s]) + F.log(end_dis[sample_s + sample_e])) * reward
            else:
                rl_loss += (F.log(start_dis[start_p]) + F.log(end_dis[end_p])) * reward

        return rl_loss

    def get_loss_function(self):

        def loss_f_elmo(c, c_char, c_feature, c_mask, q, q_char, q_feature, q_mask, context_ids, question_ids,
                        c_has_gloss, c_gloss, q_has_gloss, q_gloss, target):
            """
            # rec_loss = 0
            start_scores, end_scores = self.forward(c, c_char, c_feature, c_mask, q, q_char, q_feature, q_mask)

            # start_losses = F.bernoulli_nll(start_scores, target[:, 0, 0].astype(self.xp.float32))
            # end_losses = F.bernoulli_nll(end_scores, target[:, 0, 1].astype(self.xp.float32))

            # start_losses = F.bernoulli_nll(target[:, 0, 0].astype(self.xp.float32), start_scores)
            # end_losses = F.bernoulli_nll(target[:, 0, 1].astype(self.xp.float32), end_scores)
            start_losses = self.neg_loglikelihood_fun(target[:, 0, 0], start_scores)

            end_losses = self.neg_loglikelihood_fun(target[:, 0, 1], end_scores, c_mask)

            rec_loss = start_losses + end_losses
            """
            rl_loss = 0

            start_scores, end_scores = self.forward_elmo(c, c_char, c_feature, c_mask, q, q_char, q_feature, q_mask,
                                                         context_ids, question_ids,
                                                         c_has_gloss, c_gloss, q_has_gloss, q_gloss)

            mle_loss = self.mle_loss_func(c, c_char, c_feature, c_mask, q, q_char, q_feature, q_mask, target,
                                          start_scores, end_scores)

            # if self.args.lambda_param != 1:
            if self.args.fine_tune:
                rl_loss = self.rl_loss_func(c, c_char, c_feature, c_mask, q, q_char, q_feature, q_mask, target,
                                            start_scores, end_scores)

            self.rec_loss = mle_loss
            # self.loss = self.args.lambda_param * mle_loss + (1 - self.args.lambda_param) * rl_loss
            if self.args.fine_tune:
                self.loss = mle_loss / (2 * F.square(self.sigma_a[0])) \
                            + rl_loss / (2 * F.square(self.sigma_b[0])) \
                            + F.log(F.square(self.sigma_a[0])) \
                            + F.log(F.square(self.sigma_b[0]))
            else:
                self.loss = mle_loss

            """
            # computational graph
            if (self.args.dot_file is not None) and os.path.exists(self.args.dot_file) is False:
                g = CG.build_computational_graph((self.rec_loss,))
                with open(self.args.dot_file, 'w') as f:
                    f.write(g.dump())
            """

            chainer.report(
                {'mle_loss': mle_loss, 'rl_loss': rl_loss, 'loss': self.loss}, observer=self)
            return self.loss

        def loss_f_dict(c, c_char, c_feature, c_mask, q, q_char, q_feature, q_mask,
                        c_has_gloss, c_gloss, q_has_gloss, q_gloss, target):
            """
            # rec_loss = 0
            start_scores, end_scores = self.forward(c, c_char, c_feature, c_mask, q, q_char, q_feature, q_mask)

            # start_losses = F.bernoulli_nll(start_scores, target[:, 0, 0].astype(self.xp.float32))
            # end_losses = F.bernoulli_nll(end_scores, target[:, 0, 1].astype(self.xp.float32))

            # start_losses = F.bernoulli_nll(target[:, 0, 0].astype(self.xp.float32), start_scores)
            # end_losses = F.bernoulli_nll(target[:, 0, 1].astype(self.xp.float32), end_scores)
            start_losses = self.neg_loglikelihood_fun(target[:, 0, 0], start_scores)

            end_losses = self.neg_loglikelihood_fun(target[:, 0, 1], end_scores, c_mask)

            rec_loss = start_losses + end_losses
            """
            rl_loss = 0

            start_scores, end_scores = self.forward(c, c_char, c_feature, c_mask, q, q_char, q_feature, q_mask,
                                                    c_has_gloss, c_gloss, q_has_gloss, q_gloss)

            mle_loss = self.mle_loss_func(c, c_char, c_feature, c_mask, q, q_char, q_feature, q_mask, target,
                                          start_scores, end_scores)

            # if self.args.lambda_param != 1:
            if self.args.fine_tune:
                rl_loss = self.rl_loss_func(c, c_char, c_feature, c_mask, q, q_char, q_feature, q_mask, target,
                                            start_scores, end_scores)

            self.rec_loss = mle_loss
            # self.loss = self.args.lambda_param * mle_loss + (1 - self.args.lambda_param) * rl_loss
            if self.args.fine_tune:
                self.loss = mle_loss / (2 * F.square(self.sigma_a[0])) \
                            + rl_loss / (2 * F.square(self.sigma_b[0])) \
                            + F.log(F.square(self.sigma_a[0])) \
                            + F.log(F.square(self.sigma_b[0]))
            else:
                self.loss = mle_loss

            """
            # computational graph
            if (self.args.dot_file is not None) and os.path.exists(self.args.dot_file) is False:
                g = CG.build_computational_graph((self.rec_loss,))
                with open(self.args.dot_file, 'w') as f:
                    f.write(g.dump())
            """

            chainer.report(
                {'mle_loss': mle_loss, 'rl_loss': rl_loss, 'loss': self.loss}, observer=self)
            return self.loss

        def loss_f(c, c_char, c_feature, c_mask, q, q_char, q_feature, q_mask, target):
            """
            # rec_loss = 0
            start_scores, end_scores = self.forward(c, c_char, c_feature, c_mask, q, q_char, q_feature, q_mask)

            # start_losses = F.bernoulli_nll(start_scores, target[:, 0, 0].astype(self.xp.float32))
            # end_losses = F.bernoulli_nll(end_scores, target[:, 0, 1].astype(self.xp.float32))

            # start_losses = F.bernoulli_nll(target[:, 0, 0].astype(self.xp.float32), start_scores)
            # end_losses = F.bernoulli_nll(target[:, 0, 1].astype(self.xp.float32), end_scores)
            start_losses = self.neg_loglikelihood_fun(target[:, 0, 0], start_scores)

            end_losses = self.neg_loglikelihood_fun(target[:, 0, 1], end_scores, c_mask)

            rec_loss = start_losses + end_losses
            """
            rl_loss = 0

            start_scores, end_scores = self.forward(c, c_char, c_feature, c_mask, q, q_char, q_feature, q_mask,
                                                    None, None, None, None)

            mle_loss = self.mle_loss_func(c, c_char, c_feature, c_mask, q, q_char, q_feature, q_mask, target,
                                          start_scores, end_scores)

            # if self.args.lambda_param != 1:
            if self.args.fine_tune:
                rl_loss = self.rl_loss_func(c, c_char, c_feature, c_mask, q, q_char, q_feature, q_mask, target,
                                            start_scores, end_scores)

            self.rec_loss = mle_loss
            # self.loss = self.args.lambda_param * mle_loss + (1 - self.args.lambda_param) * rl_loss
            if self.args.fine_tune:
                self.loss = mle_loss / (2 * F.square(self.sigma_a[0])) \
                            + rl_loss / (2 * F.square(self.sigma_b[0])) \
                            + F.log(F.square(self.sigma_a[0])) \
                            + F.log(F.square(self.sigma_b[0]))
            else:
                self.loss = mle_loss

            """
            # computational graph
            if (self.args.dot_file is not None) and os.path.exists(self.args.dot_file) is False:
                g = CG.build_computational_graph((self.rec_loss,))
                with open(self.args.dot_file, 'w') as f:
                    f.write(g.dump())
            """

            chainer.report(
                {'mle_loss': mle_loss, 'rl_loss': rl_loss, 'loss': self.loss}, observer=self)
            return self.loss

        # return loss_f
        if self.args.use_elmo:
            return loss_f_elmo
        elif self.args.use_dict:
            return loss_f_dict
        else:
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
