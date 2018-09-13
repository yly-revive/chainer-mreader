import chainer
from chainer import Variable
from chainer import Parameter
import chainer.functions as F
import chainer.links as L
import six


class SFU(chainer.Chain):

    def __init__(self, input_size, fusion_size):
        super(SFU, self).__init__()

        with self.init_scope():
            self.linear_r = L.Linear(input_size + fusion_size, input_size)
            self.linear_g = L.Linear(input_size + fusion_size, input_size)

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
        r = F.tanh(self.linear_r(r_f))
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


class InteractiveAligner(chainer.Chain):

    def __init__(self):
        super(InteractiveAligner, self).__init__()

    def forward(self, context, query, q_mask):
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

        '''
        B = F.batch_matmul(c_trans, q_trans, transb=True)  # (batch, seq1_len, seq2_len)

        batch_size, c_len, q_len = B.shape
        if q_mask is not None:
            # q_mask = F.cast(F.broadcast_to(F.expand_dims(q_mask, axis=1),(batch_size, c_len, q_len)), 'float32')
            q_mask = F.broadcast_to(F.expand_dims(q_mask, axis=1), (batch_size, c_len, q_len))
            # B = B * q_mask
            infinit_matrix = self.xp.ones((batch_size, c_len, q_len), dtype=self.xp.float32) * -1 * self.xp.inf
            # B = F.where((q_mask == 1), B, -1 * self.xp.inf)
            cond = q_mask.data.astype(self.xp.bool)
            B = F.where(cond, B, infinit_matrix)
        b = F.softmax(B, axis=1)  # (batch, seq1_len, seq2_len)

        q_slash = F.batch_matmul(b, q_trans)
        '''

        B = F.batch_matmul(c_trans, q_trans, transb=True)  # (batch, seq1_len, seq2_len)

        batch_size, c_len, q_len = B.shape
        if q_mask is not None:
            # q_mask = F.cast(F.broadcast_to(F.expand_dims(q_mask, axis=1),(batch_size, c_len, q_len)), 'float32')
            q_mask = F.broadcast_to(F.expand_dims(q_mask, axis=1), (batch_size, c_len, q_len))
            # B = B * q_mask
            infinit_matrix = self.xp.ones((batch_size, c_len, q_len), dtype=self.xp.float32) * -1 * self.xp.inf
            # B = F.where((q_mask == 1), B, -1 * self.xp.inf)
            cond = q_mask.data.astype(self.xp.bool)
            B = F.where(cond, B, infinit_matrix)
        b = F.softmax(B, axis=2)  # (batch, seq1_len, seq2_len)

        q_slash = F.batch_matmul(b, q_trans)

        return q_slash, b


class SelfAttnAligner(chainer.Chain):

    def __init__(self):
        super(SelfAttnAligner, self).__init__()

    def forward(self, h, h_mask):
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

        B = F.batch_matmul(h, h, transb=True)  # (batch, seq_len, seq_len)

        mask = self.xp.eye(seq_len, dtype=self.xp.float32)
        mask = 1 - mask
        mask = F.broadcast_to(F.expand_dims(mask, axis=0), (batch_size, seq_len, seq_len))

        B = B * mask

        b = F.softmax(B, axis=2)  # (batch, seq_len, seq_len)

        c_slash = F.batch_matmul(b, h)

        return c_slash, b


class FNnet(chainer.Chain):

    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(FNnet, self).__init__()

        with self.init_scope():
            self.linear_h = L.Linear(input_size, hidden_size)
            self.linear_o = L.Linear(hidden_size, output_size)
            self.dropout = dropout

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


class MemAnsPtr(chainer.Chain):

    def __init__(self, args):
        super(MemAnsPtr, self).__init__()

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

        return p_s, p_e
