import chainer
import chainer.functions as F
import chainer.links as L
from linkers import *


class InteractiveModel(chainer.Chain):

    def __init__(self, args):
        super(InteractiveModel, self).__init__()
        self.args = args

        with self.init_scope():
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

            self.inference_model = InferenceModel(self.args.encoder_hidden_size, self.args.encoder_hidden_size * 3)
            self.intra_sentence_model = InferenceModel(self.args.encoder_hidden_size, self.args.encoder_hidden_size * 3,
                                                       intra=True)

            '''
            self.prediction_bilstm = L.NStepBiLSTM(n_layers=1, in_size=self.args.encoder_hidden_size,
                                                   out_size=2 * self.args.encoder_hidden_size,
                                                   dropout=args.encoder_dropout)
            '''
            self.prediction = PredNet(self.args.encoder_hidden_size * 2, self.args.encoder_hidden_size)

    def __call__(self, c, c_char, c_feature, c_mask, q, q_char, q_feature, q_mask):
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
            q_em_feat_input = q_em_feat.data.astype(self.xp.float32)
            q_qtype_input = F.squeeze(q_qtype_input)

            q_input.append(q_pos_input)
            q_input.append(q_ner_input)
            q_input.append(q_em_feat_input)
            q_input.append(q_qtype_input)

        # Encode context with bi-lstm
        c_input = F.concat(c_input, axis=2)
        c_input_bilstm = [i for i in c_input]

        _, _, context = self.encoder_bilstm(None, None, c_input_bilstm)

        # Encode question with bi-lstm
        q_input = F.concat(q_input, axis=2)
        q_input_bilstm = [i for i in q_input]
        _, _, question = self.encoder_bilstm(None, None, q_input_bilstm)

        context = F.stack(context, axis=0)
        question = F.stack(question, axis=0)

        s_slash, q_slash = self.inference_model(context, question)
        s_tri, q_tri = self.intra_sentence_model(context, question)

        s_bar = F.concat([s_slash, s_tri], axis=s_tri.data.ndim - 1)
        q_bar = F.concat([q_slash, q_tri], axis=q_tri.data.ndim - 1)

        y = self.prediction(s_bar, q_bar)

        return y


class InferenceModel(chainer.Chain):

    def __init__(self, input_size, fusion_size, intra=False):
        super(InferenceModel, self).__init__()

        self.intra = intra

        with self.init_scope():
            # self.infer_fusion = SFU(input_size, fusion_size)
            self.fusion_s = SFU(input_size, fusion_size, 1)
            self.fusion_q = SFU(input_size, fusion_size, 1)

    def __call__(self, s, q):
        A = F.matmul(s, q, transb=True)
        A_shape = A.data.shape
        A_weight = F.softmax(A, axis=len(A_shape) - 1)

        # intra-sentence model: a[i][j] = -inf
        # to ensure word is not aligned to itself
        if self.intra:
            max_i = len(A_shape) - 2
            max_j = len(A_shape) - 1

            max = max_i if max_i > max_j else max_j

            cond = self.xp.eye(max, dtype=self.xp.float32)
            mask = cond * -self.xp.inf

            A_weight = F.where(cond, mask, A_weight)

        B = F.matmul(A_weight, q)
        C = F.matmul(A_weight, s, transa=True)

        s_slash = self.fusion_s(s, B, s * B, s - B)
        q_slash = self.fusion_q(q, C, q * C, q - C)

        return s_slash, q_slash


class PredNet(chainer.Chain):

    def __init__(self, input_size, output_size, dropout=0):
        super(PredNet, self).__init__()

        with self.init_scope():
            self.pred_bilstm = L.NStepBiLSTM(n_layers=1, in_size=input_size,
                                             out_size=output_size,
                                             dropout=dropout)

            self.L = L.Linear(None, 2)

    def __call__(self, s, q):
        s_bar, _, _ = self.pred_bilstm(None, None, s)

        s_bar_new = F.concat(s_bar, axis=1)

        q_bar, _, _ = self.pred_bilstm(None, None, q)

        q_bar_new = F.concat(q_bar, axis=1)

        # mean-max pooling
        # stub
        # use addition as example
        summarized_vector = s_bar_new + q_bar_new

        s_linear_output = self.gelu(self.L(summarized_vector))

        y = F.softmax(s_linear_output)

        return y

    def gelu(self, x):
        return 0.5 * x * (1 + F.tanh(F.sqrt(2 / self.xp.pi) * (x + 0.044715 * F.pow(x, 3))))
