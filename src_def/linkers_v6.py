import chainer
from chainer import Variable
from chainer import Parameter
import chainer.functions as F
import chainer.links as L
import six


class SFU_V6(chainer.Chain):

    def __init__(self, input_size, fusion_size, fusion_type=0):
        super(SFU_V6, self).__init__()

        self.fusion_type = fusion_type

        with self.init_scope():
            self.linear_r = L.Linear(input_size + fusion_size, input_size)
            self.linear_g = L.Linear(input_size + fusion_size, input_size)

    def __call__(self, x, fusions):
        return self.forward(x, fusions)

    def forward(self, x, fusions):
        """
        Args:
        -
        Inputs:
            - x: (batch, seq_len, hidden_size)
            - fusions: (batch, seq_len, hidden_size * num of features)
        Outputs:
            - o: (batch, seq_len, hidden_size)
        """
        # r_f = F.concat((x, fusions), axis=2)
        r_f = F.concat((x, fusions), axis=x.data.ndim - 1)

        # for linear link, reshape (batch_size, seq_len, hidden_size) -> (batch_size*seq_len, hidden_size)
        batch_size, seq_len, hidden_size = r_f.shape
        r_f = F.reshape(r_f, (batch_size * seq_len, hidden_size))

        # add activation function selection 0:relu 1:gelu
        r = F.relu(self.linear_r(r_f)) if self.fusion_type == 0 else self.gelu(self.linear_r(r_f))

        g = F.sigmoid(self.linear_g(r_f))

        # reshape r for calculation
        _, r_output_size = r.shape
        r = F.reshape(r, (batch_size, seq_len, r_output_size))
        _, g_output_size = g.shape
        g = F.reshape(g, (batch_size, seq_len, g_output_size))

        o = g * r + (1 - g) * x

        # reshape output: (batch_size*seq_len, hidden_size) -> (batch_size, seq_len, hidden_size)
        # o = F.reshape(o, (batch_size, seq_len, hidden_size))

        return o

    def gelu(self, x):
        return 0.5 * x * (1 + F.tanh(F.sqrt(2 / self.xp.pi) * (x + 0.044715 * F.pow(x, 3))))


class InteractiveAligner_V6(chainer.Chain):

    def __init__(self, dim):
        super(InteractiveAligner_V6, self).__init__()

        with self.init_scope():
            self.dim = dim
            self.Lu = L.Linear(None, dim)
            self.Lv = L.Linear(None, dim)

    def __call__(self, context, query, q_mask, e_prev_param=None, b_prev_param=None, gamma=0):
        return self.forward(context, query, q_mask, e_prev_param, b_prev_param, gamma)

    def forward(self, context, query, q_mask, e_prev_param=None, b_prev_param=None, gamma=0):
        """
        Args:
        -
        Inputs:
            #- context: (seq1_len, batch, hidden_size)
            #- query: (seq2_len, batch, hidden_size)
            - context: (batch, seq1_len, hidden_size)
            - query: (batch, seq2_len, hidden_size)
            - q_mask: (batch, seq2_len)
        Outputs:
            #- output: (seq1_len, batch, hidden_size)
            - output: (batch, seq1_len, hidden_size)
            - alpha: (batch, seq1_len, seq2_len)
        """
        # c_trans = F.transpose(context, axes=(1, 0, 2))
        # q_trans = F.transpose(query, axes=(1, 0, 2))

        c_trans = context

        q_trans = query

        E = F.batch_matmul(

            F.relu(
                F.reshape(
                    self.Lv(
                        F.reshape(
                            q_trans,
                            (q_trans.shape[0] * q_trans.shape[1], q_trans.shape[2])
                        )
                    ),
                    (q_trans.shape[0], q_trans.shape[1], self.dim)
                )
            ),
            F.relu(
                F.reshape(
                    self.Lu(
                        F.reshape(
                            c_trans,
                            (c_trans.shape[0] * c_trans.shape[1], c_trans.shape[2])
                        )
                    ),
                    (c_trans.shape[0], c_trans.shape[1], self.dim)
                )
            ),
            transb=True
        )  # (batch, seq1_len, seq2_len)

        batch_size, q_len, c_len = E.shape

        # e = F.softmax(E, axis=2)  # (batch, seq1_len, seq2_len)

        if e_prev_param is not None and b_prev_param is not None:
            # e_slash = F.batch_matmul(e_prev_param, b_prev_param)
            e_slash = F.batch_matmul(
                F.softmax(e_prev_param, axis=2),
                F.softmax(b_prev_param, axis=1)
            )
            gamma = F.broadcast_to(
                F.expand_dims(
                    F.expand_dims(
                        gamma, 0
                    ),
                    0
                ),
                E.shape
            )
            E = E + gamma * e_slash

        if q_mask is not None:
            # q_mask = F.cast(F.broadcast_to(F.expand_dims(q_mask, axis=1),(batch_size, c_len, q_len)), 'float32')
            q_mask = F.broadcast_to(F.expand_dims(q_mask, axis=2), (batch_size, q_len, c_len))
            # B = B * q_mask
            infinit_matrix = self.xp.ones((batch_size, q_len, c_len), dtype=self.xp.float32) * -1 * self.xp.inf
            # B = F.where((q_mask == 1), B, -1 * self.xp.inf)
            cond = q_mask.data.astype(self.xp.bool)
            e = F.where(cond, E, infinit_matrix)

        e = F.softmax(e, axis=1)

        q_slash = F.batch_matmul(e, q_trans, transa=True)

        """
        if e_prev_param is not None and b_prev_param is not None:
            # e_slash = F.batch_matmul(e_prev_param, b_prev_param)
            e_slash = F.batch_matmul(
                F.softmax(e_prev_param, axis=2),
                F.softmax(b_prev_param, axis=1)
            )
            E = E + gamma * e_slash
        """
        return q_slash, E


class SelfAttnAligner_V6(chainer.Chain):

    def __init__(self, dim):
        super(SelfAttnAligner_V6, self).__init__()

        with self.init_scope():
            self.dim = dim
            self.Lu = L.Linear(None, dim)
            self.Lv = L.Linear(None, dim)

    def __call__(self, h, h_mask, b_param=None, gamma=0):
        return self.forward(h, h_mask, b_param, gamma)

    def forward(self, h, h_mask=None, b_param=None, gamma=0):
        """
        Args:
            -
        Inputs:
            #- h: (seq_len, batch, hidden_size)
            - h: (batch, seq_len, hidden_size)
            - h_mask: (batch, seq_len)
        Outputs:
            #- output: (seq_len, batch, hidden_size)
            - output: (batch, seq_len, hidden_size)
            - alpha: (batch, seq_len, seq_len)
        """
        batch_size, seq_len, _ = h.shape

        # B = F.batch_matmul(self.Lu(h), self.Lv(h), transb=True)  # (batch, seq_len, seq_len)
        B = F.batch_matmul(
            F.relu(
                F.reshape(
                    self.Lu(
                        F.reshape(
                            h,
                            (h.shape[0] * h.shape[1], h.shape[2])
                        )
                    ), (h.shape[0], h.shape[1], self.dim)
                )
            ),
            F.relu(
                F.reshape(
                    self.Lv(
                        F.reshape(
                            h,
                            (h.shape[0] * h.shape[1], h.shape[2])
                        )
                    ), (h.shape[0], h.shape[1], self.dim)
                )
            ),
            transb=True)  # (batch, seq_len, seq_len)

        mask = self.xp.eye(seq_len, dtype=self.xp.float32)
        mask = 1 - mask
        mask = F.broadcast_to(F.expand_dims(mask, axis=0), (batch_size, seq_len, seq_len))
        # B = B * mask

        if b_param is not None:
            B_slash = F.batch_matmul(
                F.softmax(b_param, axis=2),
                F.softmax(b_param, axis=1)
            )

            gamma = F.broadcast_to(
                F.expand_dims(
                    F.expand_dims(
                        gamma, 0
                    ),
                    0
                ),
                B_slash.shape
            )

            B = B + gamma * B_slash

        """
        infinit_matrix = self.xp.ones((batch_size, seq_len, seq_len), dtype=self.xp.float32) * -1 * self.xp.inf
        # B = F.where((q_mask == 1), B, -1 * self.xp.inf)
        cond = mask.data.astype(self.xp.bool)
        b = F.where(cond, B, infinit_matrix)
        """
        B = B * mask
        b = F.softmax(B, axis=2)  # (batch, seq_len, seq_len)

        c_slash = F.batch_matmul(b, h)
        """
        if b_param is not None:
            B_slash = F.batch_matmul(
                F.softmax(b_param, axis=2),
                F.softmax(b_param, axis=1)
            )
            B = B + gamma * B_slash
            B = B * mask
        """
        return c_slash, B


class FNnet_V6(chainer.Chain):

    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(FNnet_V6, self).__init__()

        with self.init_scope():
            self.linear_h = L.Linear(input_size, hidden_size)
            self.linear_o = L.Linear(hidden_size, output_size)
            self.dropout = dropout

    def __call__(self, input_data):
        return self.forward(input_data)

    def forward(self, input_data, mask=None):
        # reshape data
        batch_size, seq_len, hidden_size = input_data.shape

        input_data = F.reshape(input_data, (batch_size * seq_len, hidden_size))
        hidden_data = F.relu(self.linear_h(input_data))
        dropout_data = F.dropout(hidden_data)
        output_data = self.linear_o(dropout_data)

        _, output_size = output_data.shape
        output_data = F.reshape(output_data, (batch_size, seq_len, output_size))

        return output_data


class MemAnsPtr_V3(chainer.Chain):

    def __init__(self, args):
        super(MemAnsPtr_V6, self).__init__()

        self.args = args
        with self.init_scope():
            self.FFN_start = chainer.ChainList()
            self.SFU_start = chainer.ChainList()
            self.FFN_end = chainer.ChainList()
            self.SFU_end = chainer.ChainList()

            hidden_size = self.args.encoder_hidden_size

            for _ in six.moves.range(self.args.ptr_hops):
                self.FFN_start.append(FNnet(6 * hidden_size, hidden_size, 1, self.args.dropout))
                self.SFU_start.append(SFU(2 * hidden_size, 2 * hidden_size))
                self.FFN_end.append(FNnet(6 * hidden_size, hidden_size, 1, self.args.dropout))
                self.SFU_end.append(SFU(2 * hidden_size, 2 * hidden_size))

    def __call__(self, c, y, c_mask, y_mask):
        return self.forward(c, y, c_mask, y_mask)

    def forward(self, c, y, c_mask, y_mask):

        batch_size, seq_len, hidden_size = c.shape
        # z_s = F.squeeze(y[:, -1, :])
        z_s = F.expand_dims(y[:, -1, :], axis=1)
        z_e = None
        s = None
        e = None
        p_s = None
        p_e = None

        for i in six.moves.range(self.args.ptr_hops):
            '''
            # z_s_ = F.broadcast_to(F.expand_dims(z_s, axis=1), (batch_size, seq_len, hidden_size))
            z_s_ = F.broadcast_to(z_s, (batch_size, seq_len, hidden_size))
            s = F.squeeze(self.FFN_start[i].forward(F.concat((c, z_s_, c * z_s_), axis=2)))

            cond = c_mask.astype(self.xp.bool)
            # s = F.where(c_mask == 1, s, -self.xp.inf)

            infinit_matrix = self.xp.ones(s.shape, dtype=self.xp.float32) * -1 * self.xp.inf
            s = F.where(cond, s, infinit_matrix)

            p_s = F.softmax(F.squeeze(s), axis=1)
            u_s = F.batch_matmul(F.expand_dims(p_s, axis=1), c)
            z_e = self.SFU_start[i](z_s, u_s)
            # z_e_ = F.broadcast_to(F.expand_dims(z_e, axis=1), (batch_size, seq_len, hidden_size))
            z_e_ = F.broadcast_to(z_e, (batch_size, seq_len, hidden_size))
            e = F.squeeze(self.FFN_end[i].forward(F.concat((c, z_e_, c * z_e_), 2)))

            # e = F.where(c_mask == 1, e, -1 * self.xp.inf)
            e = F.where(cond, e, infinit_matrix)

            p_e = F.softmax(F.squeeze(e), axis=1)
            u_e = F.batch_matmul(F.expand_dims(p_e, axis=1), c)
            z_s = self.SFU_end[i](z_e, u_e)  # [B,1,H]
            '''

            # z_s_ = F.broadcast_to(F.expand_dims(z_s, axis=1), (batch_size, seq_len, hidden_size))
            z_s_ = F.broadcast_to(z_s, (batch_size, seq_len, hidden_size))
            s = F.squeeze(self.FFN_start[i](F.concat((c, z_s_, c * z_s_), axis=2)))

            cond = c_mask.astype(self.xp.bool)
            # s = F.where(c_mask == 1, s, -self.xp.inf)

            infinit_matrix = self.xp.ones(s.shape, dtype=self.xp.float32) * -1 * self.xp.inf
            s = F.where(cond, s, infinit_matrix)

            p_s = F.softmax(F.squeeze(s), axis=1)
            u_s = F.batch_matmul(F.expand_dims(p_s, axis=1), c)
            z_e = self.SFU_start[i](z_s, u_s)
            # z_e_ = F.broadcast_to(F.expand_dims(z_e, axis=1), (batch_size, seq_len, hidden_size))
            z_e_ = F.broadcast_to(z_e, (batch_size, seq_len, hidden_size))
            e = F.squeeze(self.FFN_end[i](F.concat((c, z_e_, c * z_e_), 2)))

            # e = F.where(c_mask == 1, e, -1 * self.xp.inf)
            e = F.where(cond, e, infinit_matrix)

            p_e = F.softmax(F.squeeze(e), axis=1)
            u_e = F.batch_matmul(F.expand_dims(p_e, axis=1), c)
            z_s = self.SFU_end[i](z_e, u_e)  # [B,1,H]

        return p_s, p_e


class MemAnsPtr_V6_Variant(chainer.Chain):

    def __init__(self, args, input_size, fusion_size):
        super(MemAnsPtr_V6_Variant, self).__init__()

        self.args = args
        with self.init_scope():
            self.linear_s = L.Linear(input_size + fusion_size, input_size)
            self.linear_e = L.Linear(input_size + fusion_size, input_size)
            """
            self.w_s = Variable(self.xp.array([input_size, 1], dtype=self.xp.float32))
            self.w_e = Variable(self.xp.array([input_size, 1], dtype=self.xp.float32))
            self.w_q = Variable(self.xp.array([input_size, 1], dtype=self.xp.float32))
            """

            """
            self.w_e = Variable(self.xp.zeros((input_size, 1), dtype=self.xp.float32))
            self.w_q = Variable(self.xp.zeros((input_size, 1), dtype=self.xp.float32))
            """
            self.w_s = Parameter(
                initializer=self.xp.random.randn(input_size, 1).astype('f')
            )
            self.w_e = Parameter(
                initializer=self.xp.random.randn(input_size, 1).astype('f')
            )
            self.w_q = Parameter(
                initializer=self.xp.random.randn(input_size, 1).astype('f')
            )

            self.e_fusion = SFU_V6(input_size, fusion_size)

    def __call__(self, c, y, c_mask, y_mask):
        return self.forward(c, y, c_mask, y_mask)

    def forward(self, c, y, c_mask, y_mask):
        batch_size, seq_len, dims = c.shape

        a_score = F.batch_matmul(y, F.broadcast_to(F.expand_dims(self.w_q, 0), (batch_size, self.w_q.shape[0], 1)))

        a_score = F.softmax(
            F.squeeze(a_score)  # [b,q]
        )  # [b,q]

        """
        s = F.batch_matmul(
            F.expand_dims(
                a_score,
                axis=1
            ),
            y,
            transa=True
        )
        """
        s = F.batch_matmul(
            F.expand_dims(
                a_score,
                axis=1
            ),
            y
        )

        s_broad = F.broadcast_to(s, (batch_size, seq_len, dims))

        s_value = F.batch_matmul(
            F.tanh(
                F.reshape(
                    self.linear_s(
                        F.reshape(
                            F.concat(
                                [c, s_broad, c * s_broad, c - s_broad],
                                axis=2
                            ),
                            (batch_size * seq_len, dims * 4)
                        )
                    ),
                    (batch_size, seq_len, dims)
                )
            ),
            F.broadcast_to(
                F.expand_dims(self.w_s, 0),
                (batch_size, self.w_s.shape[0], 1)
            )
        )

        cond = c_mask.astype(self.xp.bool)

        s_value = F.squeeze(s_value)

        infinit_matrix = self.xp.ones(s_value.shape, dtype=self.xp.float32) * -1 * self.xp.inf
        s_value = F.where(cond, s_value, infinit_matrix)

        p_s = F.softmax(s_value)

        """
        l = c * F.broadcast_to(
            F.reshape(
                p_s,
                (batch_size, seq_len, 1)
            ),
            (batch_size, seq_len, dims)
        )
        """
        l = F.batch_matmul(
            F.reshape(
                p_s,
                (batch_size, 1, seq_len)
            ),
            c
        )

        s_broad_one = F.broadcast_to(s, (batch_size, 1, dims))  # s is [b,1,d], so this may be not necessary

        s_slash = self.e_fusion(
            s_broad_one,
            F.concat(
                [l, s_broad_one * l, s_broad_one - l],
                axis=2
            )
        )  # (batch, 1, dim)

        s_slash_broad = F.broadcast_to(s_slash, (batch_size, seq_len, dims))

        e_value = F.batch_matmul(
            F.tanh(
                F.reshape(
                    self.linear_e(
                        F.reshape(
                            F.concat(
                                [c, s_slash_broad, c * s_slash_broad, c - s_slash_broad],
                                axis=2
                            ),
                            (batch_size * seq_len, dims * 4)
                        )
                    ),
                    (batch_size, seq_len, dims)
                )
            ),
            F.broadcast_to(
                F.expand_dims(self.w_e, 0),
                (batch_size, self.w_e.shape[0], 1)
            )
        )

        e_value = F.squeeze(e_value)

        e_value = F.where(cond, e_value, infinit_matrix)

        p_e = F.softmax(e_value)

        return p_s, p_e


class AttnModule(chainer.Chain):

    def __init__(self, output_size):
        super(AttnModule, self).__init__()

        with self.init_scope():
            self.AttnW = L.Linear(None, output_size)

    def __call__(self, u, dict_embedding):
        # dict_num, dict_embed_size = dict_embedding.shape

        attn_score = F.softmax(
            u @
            F.transpose(F.tanh(self.AttnW(dict_embedding)))
        )  # (1, dict_num)

        return attn_score @ dict_embedding
