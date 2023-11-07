import random

import numpy as np
import torch
import sys
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
FLOAT_MIN = -sys.float_info.max


# 对比学习计算，需要的对象
class Contrast(torch.nn.Module):
    def __init__(self, pos_logits, neg_logits, pos_labels, neg_labels, log_feats, pos):
        super(Contrast, self).__init__()
        self.pos_logits = pos_logits
        self.neg_logits = neg_logits
        self.pos_labels = pos_labels
        self.neg_labels = neg_labels
        self.log_feats = log_feats
        self.pos = pos

# 前馈网络  包含着公式(9),(10),(11)
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):  # wried, why fusion X 2?

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


# 时间注意力层 包含着(6),(7),(8)
class TimeAwareMultiHeadAttention(torch.nn.Module):
    # required homebrewed mha layer for Ti/SASRec experiments
    def __init__(self, hidden_size, head_num, dropout_rate, dev):
        super(TimeAwareMultiHeadAttention, self).__init__()
        self.Q_w = torch.nn.Linear(hidden_size, hidden_size)
        self.K_w = torch.nn.Linear(hidden_size, hidden_size)
        self.V_w = torch.nn.Linear(hidden_size, hidden_size)

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = hidden_size // head_num
        self.dropout_rate = dropout_rate
        self.dev = dev

    def forward(self, queries, keys, time_mask, attn_mask, time_matrix_K, time_matrix_V, abs_pos_K, abs_pos_V):
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)

        # head dim * batch dim for parallelization (h*N, T, C/h)
        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

        time_matrix_K_ = torch.cat(torch.split(time_matrix_K, self.head_size, dim=3), dim=0)
        time_matrix_V_ = torch.cat(torch.split(time_matrix_V, self.head_size, dim=3), dim=0)
        abs_pos_K_ = torch.cat(torch.split(abs_pos_K, self.head_size, dim=2), dim=0)
        abs_pos_V_ = torch.cat(torch.split(abs_pos_V, self.head_size, dim=2), dim=0)

        # batched channel wise matmul to gen attention weights
        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))
        attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))
        attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1).to(self.dev)

        # seq length adaptive scaling
        # 公式(8)
        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)

        # key masking, -2^32 lead to leaking, inf lead to nan
        # 0 * inf = nan, then reduce_sum([nan,...]) = nan

        # fixed a bug pointed out in https://github.com/pmixer/TiSASRec.pytorch/issues/2
        # time_mask = time_mask.unsqueeze(-1).expand(attn_weights.shape[0], -1, attn_weights.shape[-1])
        time_mask = time_mask.unsqueeze(-1).repeat(self.head_num, 1, 1)
        time_mask = time_mask.expand(-1, -1, attn_weights.shape[-1])
        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)
        paddings = torch.ones(attn_weights.shape) *  (-2**32+1) # -1e23 # float('-inf')
        paddings = paddings.to(self.dev)
        attn_weights = torch.where(time_mask, paddings, attn_weights) # True:pick padding
        attn_weights = torch.where(attn_mask, paddings, attn_weights) # enforcing causality

        attn_weights = self.softmax(attn_weights) # code as below invalids pytorch backward rules
        # attn_weights = torch.where(time_mask, paddings, attn_weights) # weird query mask in tf impl
        # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/4
        # attn_weights[attn_weights != attn_weights] = 0 # rm nan for -inf into softmax case
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(V_)
        outputs += attn_weights.matmul(abs_pos_V_)
        outputs += attn_weights.unsqueeze(2).matmul(time_matrix_V_).reshape(outputs.shape).squeeze(2)

        # (num_head * N, T, C / num_head) -> (N, T, C)
        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2) # div batch_size

        return outputs

    # 将绝对时间加入自注意层
    def new_forward(self, queries, keys, time_mask, attn_mask, time_matrix_K, time_matrix_V, time_abs_matrix_K, time_abs_matrix_V, abs_pos_K, abs_pos_V):
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)

        # head dim * batch dim for parallelization (h*N, T, C/h)
        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

        time_matrix_K_ = torch.cat(torch.split(time_matrix_K, self.head_size, dim=3), dim=0)
        time_matrix_V_ = torch.cat(torch.split(time_matrix_V, self.head_size, dim=3), dim=0)
        time_abs_matrix_K_ = torch.cat(torch.split(time_abs_matrix_K, self.head_size, dim=2), dim=0)
        time_abs_matrix_V_ = torch.cat(torch.split(time_abs_matrix_V, self.head_size, dim=2), dim=0)
        abs_pos_K_ = torch.cat(torch.split(abs_pos_K, self.head_size, dim=2), dim=0)
        abs_pos_V_ = torch.cat(torch.split(abs_pos_V, self.head_size, dim=2), dim=0)


        # batched channel wise matmul to gen attention weights
        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))
        attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))
        attn_weights += Q_.matmul(torch.transpose(time_abs_matrix_K_, 1, 2))
        attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)

        # seq length adaptive scaling
        # 公式(8)
        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)

        # key masking, -2^32 lead to leaking, inf lead to nan
        # 0 * inf = nan, then reduce_sum([nan,...]) = nan

        # fixed a bug pointed out in https://github.com/pmixer/TiSASRec.pytorch/issues/2
        # time_mask = time_mask.unsqueeze(-1).expand(attn_weights.shape[0], -1, attn_weights.shape[-1])
        time_mask = time_mask.unsqueeze(-1).repeat(self.head_num, 1, 1)
        time_mask = time_mask.expand(-1, -1, attn_weights.shape[-1])
        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)
        paddings = torch.ones(attn_weights.shape) * (-2**32+1) # -1e23 # float('-inf')
        paddings = paddings.to(self.dev)
        attn_weights = torch.where(time_mask, paddings, attn_weights) # True:pick padding
        attn_weights = torch.where(attn_mask, paddings, attn_weights) # enforcing causality

        attn_weights = self.softmax(attn_weights) # code as below invalids pytorch backward rules
        # attn_weights = torch.where(time_mask, paddings, attn_weights) # weird query mask in tf impl
        # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/4
        # attn_weights[attn_weights != attn_weights] = 0 # rm nan for -inf into softmax case
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(V_)
        outputs += attn_weights.matmul(abs_pos_V_)
        outputs += attn_weights.matmul(time_abs_matrix_V_)
        outputs += attn_weights.unsqueeze(2).matmul(time_matrix_V_).reshape(outputs.shape).squeeze(2)

        # (num_head * N, T, C / num_head) -> (N, T, C)
        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2) # div batch_size

        return outputs

    def new_rate_forward(self, queries, keys, time_mask, attn_mask, time_matrix_K, time_matrix_V, abs_pos_K, abs_pos_V, ratings_K, ratings_V, time_abs_K, time_abs_V, weight, weight_pos, item_K, item_V):
        '''
        注意力层方法
        :param queries:
        :param keys:
        :param time_mask:
        :param attn_mask:
        :param time_matrix_K:
        :param time_matrix_V:
        :param abs_pos_K:
        :param abs_pos_V:
        :param ratings_K:
        :param ratings_V:
        :param time_abs_K:
        :param time_abs_V:
        :return:
        '''
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)

        # head dim * batch dim for parallelization (h*N, T, C/h)
        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

        # 下面除了相对时间戳是[128,50,50,50] 其余都是[128,50,50]
        kuan = time_abs_K.shape[1]
        yin = time_abs_K.shape[2]
        weight = weight.reshape([1, kuan, yin])
        new_weight = weight
        for i in range(Q_.shape[0] - 1):
            new_weight = torch.cat((new_weight, weight), dim=0).to(self.dev)
        # weight = weight.reshape([1, 50])
        # sum_weight = weight
        # for i in range(49):
        #     sum_weight = torch.cat((sum_weight, weight), dim=0).to(self.dev)
        # sum_weight = sum_weight.reshape([1, 50, 50])
        # new_weight = sum_weight
        # for i in range(Q_.shape[0] - 1):
        #     new_weight = torch.cat((new_weight, sum_weight), dim=0).to(self.dev)

        time_matrix_K_ = torch.cat(torch.split(time_matrix_K, self.head_size, dim=3), dim=0)
        time_matrix_V_ = torch.cat(torch.split(time_matrix_V, self.head_size, dim=3), dim=0)
        time_abs_K_ = torch.cat(torch.split(time_abs_K, self.head_size, dim=2), dim=0)
        time_abs_V_ = torch.cat(torch.split(time_abs_V, self.head_size, dim=2), dim=0)
        time_abs_K_ = new_weight * time_abs_K_
        time_abs_V_ = new_weight * time_abs_V_
        abs_pos_K_ = torch.cat(torch.split(abs_pos_K, self.head_size, dim=2), dim=0)
        abs_pos_V_ = torch.cat(torch.split(abs_pos_V, self.head_size, dim=2), dim=0)
        abs_pos_K_ = new_weight * abs_pos_K_
        abs_pos_V_ = new_weight * abs_pos_V_
        ratings_K_ = torch.cat(torch.split(ratings_K, self.head_size, dim=2), dim=0)
        ratings_V_ = torch.cat(torch.split(ratings_V, self.head_size, dim=2), dim=0)
        item_K_ = torch.cat(torch.split(item_K, self.head_size, dim=2), dim=0)
        item_V_ = torch.cat(torch.split(item_V, self.head_size, dim=2), dim=0)

        # batched channel wise matmul to gen attention weights
        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))
        attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))
        attn_weights += Q_.matmul(torch.transpose(ratings_K_, 1, 2))
        attn_weights += Q_.matmul(torch.transpose(item_K_, 1, 2))
        attn_weights += Q_.matmul(torch.transpose(time_abs_K_, 1, 2))
        attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1).to(self.dev)
        # attn_weights += time_abs_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1).to(self.dev)

        # seq length adaptive scaling
        # 公式(8)
        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)

        # key masking, -2^32 lead to leaking, inf lead to nan
        # 0 * inf = nan, then reduce_sum([nan,...]) = nan

        # fixed a bug pointed out in https://github.com/pmixer/TiSASRec.pytorch/issues/2
        # time_mask = time_mask.unsqueeze(-1).expand(attn_weights.shape[0], -1, attn_weights.shape[-1])
        time_mask = time_mask.unsqueeze(-1).repeat(self.head_num, 1, 1)
        time_mask = time_mask.expand(-1, -1, attn_weights.shape[-1])
        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)
        paddings = torch.ones(attn_weights.shape) * (-2**32+1)  # -1e23 # float('-inf')
        paddings = paddings.to(self.dev)
        attn_weights = torch.where(time_mask, paddings, attn_weights) # True:pick padding
        attn_weights = torch.where(attn_mask, paddings, attn_weights) # enforcing causality

        attn_weights = self.softmax(attn_weights) # code as below invalids pytorch backward rules
        # attn_weights = torch.where(time_mask, paddings, attn_weights) # weird query mask in tf impl
        # https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/4
        # attn_weights[attn_weights != attn_weights] = 0 # rm nan for -inf into softmax case
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(V_)
        outputs += attn_weights.matmul(abs_pos_V_)
        outputs += attn_weights.matmul(ratings_V_)
        outputs += attn_weights.matmul(item_V_)
        outputs += attn_weights.matmul(time_abs_V_)
        outputs += attn_weights.unsqueeze(2).matmul(time_matrix_V_).reshape(outputs.shape).squeeze(2)
        # outputs += attn_weights.unsqueeze(2).matmul(time_abs_V_).reshape(outputs.shape).squeeze(2)

        # (num_head * N, T, C / num_head) -> (N, T, C)
        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2)  # div batch_size

        return outputs


# 自注意力层
class TiSASRec(torch.nn.Module):  # similar to torch.nn.MultiheadAttention
    def __init__(self, user_num, item_num, time_num, args):
        '''
        :param user_num: 用户总数量
        :param item_num: 物品总数量
        :param time_num: 时间戳最大值
        :param args:  系统参数设定
        '''
        super(TiSASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        '''
        输入：
        torch.nn.Embedding(
        num_embeddings, – 词典的大小尺寸，比如总共出现5000个词，那就输入5000。此时index为（0-4999）
        embedding_dim,– 嵌入向量的维度，即用多少维来表示一个符号。
        padding_idx=None,– 填充id，比如，输入长度为100，但是每次的句子长度并不一样，后面就需要用统一的数字填充，而这里就是指定这个数字，这样，网络在遇到填充id时，就不会计算其与其它符号的相关性。（初始化为0）
        max_norm=None, – 最大范数，如果嵌入向量的范数超过了这个界限，就要进行再归一化。
        norm_type=2.0, – 指定利用什么范数计算，并用于对比max_norm，默认为2范数。
        scale_grad_by_freq=False, 根据单词在mini-batch中出现的频率，对梯度进行放缩。默认为False.
        sparse=False, – 若为True,则与权重矩阵相关的梯度转变为稀疏张量。
        _weight=None)
        输出：
        [规整后的句子长度，样本个数（batch_size）,词向量维度]
        '''
        # 公式(3)
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)  # 随机初始化词向量，词向量值在正态分布N(0,1)中随机取值。
        self.new_item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.old_item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.time_abs_matrix_emb = torch.nn.Embedding(args.max_time_span + 1, args.hidden_units, padding_idx=0)
        list_rand = []
        for i in range(args.hidden_units):
            list_rand.append(random.random())
        self.weight_item = torch.nn.Parameter(torch.FloatTensor(list_rand))

        list_rand = []
        for i in range(1):
            list_rand.append(random.uniform(6, 10 ))
        self.r_sim = torch.nn.Parameter(torch.FloatTensor(list_rand))

        list_rand = []
        for i in range(1):
            list_rand.append(random.uniform(0, 1))
        self.r_contrast = torch.nn.Parameter(torch.FloatTensor(list_rand))

        list_rand = []
        for i in range(args.hidden_units):
            list_rand.append(random.random())
        self.weight_pos = torch.nn.Parameter(torch.FloatTensor(list_rand))

        list_rand = []
        for i in range(args.hidden_units):
            list_rand.append(random.random())
        self.weight_new_pos = torch.nn.Parameter(torch.FloatTensor(list_rand))

        list_rand = []
        for i in range(args.hidden_units // 10):
            z_list = []
            for j in range(args.hidden_units):
                z_list.append(random.random())
            list_rand.append(z_list)
        self.weight_neight = torch.nn.Parameter(torch.FloatTensor(list_rand))



        # list_rand = []
        # for i in range(args.hidden_units):
        #     list_rand.append(random.random())
        # self.weight_3 = torch.nn.Parameter(torch.FloatTensor(list_rand))
        #
        # list_rand = []
        # for i in range(args.hidden_units):
        #     list_rand.append(random.random())
        # self.bias = torch.nn.Parameter(torch.FloatTensor(list_rand))

        list_rand = []
        for i in range(args.maxlen):
            z_list = []
            for j in range(args.hidden_units):
                z_list.append(random.random())
            list_rand.append(z_list)

        self.weight_high = torch.nn.Parameter(torch.FloatTensor(list_rand))

        list_rand = []
        for i in range(args.hidden_units):
            z_list = []
            for j in range(args.hidden_units):
                z_list.append(random.random())
            list_rand.append(z_list)
        self.weight_high_pos = torch.nn.Parameter(torch.FloatTensor(list_rand))
        self.linear_transform = torch.nn.Linear(args.hidden_units * 2, args.hidden_units, bias=True)
        # 公式(4)
        self.abs_pos_K_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.abs_pos_V_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        # self.abs_pos_K_emb = torch.nn.Embedding(args.maxlen * 2, args.hidden_units)
        # self.abs_pos_V_emb = torch.nn.Embedding(args.maxlen * 2, args.hidden_units)
        # ******
        # 公式(5)
        self.time_matrix_K_emb = torch.nn.Embedding(args.time_span + 1, args.hidden_units)
        self.time_matrix_V_emb = torch.nn.Embedding(args.time_span + 1, args.hidden_units)
        # 新增属性 评分
        self.rate_emb = torch.nn.Embedding(args.maxfen + 1, args.hidden_units)
        self.rate_K_emb = torch.nn.Embedding(args.maxfen + 1, args.hidden_units)
        self.rate_V_emb = torch.nn.Embedding(args.maxfen + 1, args.hidden_units)
        self.rate_K_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.rate_V_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.item_K_emb = torch.nn.Embedding(user_num + 1, args.hidden_units)
        self.item_V_emb = torch.nn.Embedding(user_num + 1, args.hidden_units)
        self.item_K_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.item_V_dropout = torch.nn.Dropout(p=args.dropout_rate)
        # ******
        # ******
        self.time_abs_matrix_K_emb = torch.nn.Embedding(args.max_time_span + 1, args.hidden_units)
        self.time_abs_matrix_V_emb = torch.nn.Embedding(args.max_time_span + 1, args.hidden_units)
        # ******

        self.item_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.abs_pos_K_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.abs_pos_V_emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.time_matrix_K_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.time_matrix_V_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.time_abs_matrix_K_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.time_abs_matrix_V_dropout = torch.nn.Dropout(p=args.dropout_rate)

        # 2层layerNorm前向传播(args.hidden_units, eps=1e-8)
        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        # 2层TimeAwareMultiHeadAttention(args.hidden_units(隐藏维度), args.num_heads,args.dropout_rate(dropout率),args.device(运行环境))
        self.attention_layers = torch.nn.ModuleList()
        # 2层layerNorm前向传播(args.hidden_units, eps=1e-8)
        self.forward_layernorms = torch.nn.ModuleList()
        # 2层PointWiseFeedForward(args.hidden_units, args.dropout_rate)
        self.forward_layers = torch.nn.ModuleList()
        # 公式(11)
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        # 两层注意力层相关
        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = TimeAwareMultiHeadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate,
                                                            args.device)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def seq2feats(self, user_ids, log_seqs, time_matrices, time_abs_matrices):
        '''

        :param user_ids: batch_size个不同的随机用户id
        :param log_seqs: batch_size个用户排除掉最后一个，倒数maxlen个物品集合 (-2, -2-maxlen) 集合大小(batch_size, maxlen) 如果用户没有那么多物品 进行左置补零
        :param time_matrices: batch_size个用户的r矩阵
        :return:
        '''
        seqs_tensor = torch.LongTensor(log_seqs).to(self.dev)  # 象征论文中哪个变量
        seqs = self.item_emb(seqs_tensor)
        seqs *= self.item_emb.embedding_dim ** 0.5  # sqrt(d)
        seqs = self.item_emb_dropout(seqs)

        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])  # tile(集合，(y轴复制扩大倍数,x轴复制扩大倍数))
        positions = torch.LongTensor(positions).to(self.dev)
        abs_pos_K = self.abs_pos_K_emb(positions)  # 可学习位置嵌入矩阵 自注意机制中的键
        abs_pos_V = self.abs_pos_V_emb(positions)  # 可学习位置嵌入矩阵 自注意机制中的值
        abs_pos_K = self.abs_pos_K_emb_dropout(abs_pos_K)
        abs_pos_V = self.abs_pos_V_emb_dropout(abs_pos_V)

        time_matrices = torch.LongTensor(time_matrices).to(self.dev)  # 时间戳相关的
        time_matrix_K = self.time_matrix_K_emb(time_matrices)
        time_matrix_V = self.time_matrix_V_emb(time_matrices)
        time_abs_matrices = torch.LongTensor(time_abs_matrices).to(self.dev)
        time_abs_matrix_K = self.time_abs_matrix_K_emb(time_abs_matrices)
        time_abs_matrix_V = self.time_abs_matrix_V_emb(time_abs_matrices)
        time_matrix_K = self.time_matrix_K_dropout(time_matrix_K)
        time_matrix_V = self.time_matrix_V_dropout(time_matrix_V)
        time_abs_matrix_K = self.time_abs_matrix_K_dropout(time_abs_matrix_K)
        time_abs_matrix_V = self.time_abs_matrix_V_dropout(time_abs_matrix_V)
        # mask 0th items(placeholder for dry-run) in log_seqs
        # would be easier if 0th item could be an exception for training
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        # i是一层前向传播
        for i in range(len(self.attention_layers)):
            # Self-attention, Q=layernorm(seqs), K=V=seqs
            # seqs = torch.transpose(seqs, 0, 1) # (N, T, C) -> (T, N, C)
            Q = self.attention_layernorms[i](seqs)  # PyTorch mha requires time first fmt
            # mha_outputs = self.attention_layers[i](Q, seqs,
            #                                 timeline_mask, attention_mask,
            #                                 time_matrix_K, time_matrix_V,
            #                                 abs_pos_K, abs_pos_V)
            # 下面这个效果不好
            new_mha_outputs = self.attention_layers[i].new_forward(Q, seqs,
                                                   timeline_mask, attention_mask,
                                                   time_matrix_K, time_matrix_V,
                                                   time_abs_matrix_K, time_abs_matrix_V,
                                                   abs_pos_K, abs_pos_V)
            seqs = Q + new_mha_outputs
            # seqs = Q + new_mha_outputs
            # seqs = torch.transpose(seqs, 0, 1) # (T, N, C) -> (N, T, C)

            # Point-wise Feed-forward, actually 2 Conv1D for channel wise fusion
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)  # z[i]

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def new_seq2feats(self, user_ids, log_seqs, time_matrices, time_abs_matrices, ratings_seq, ratings_pos, weight_pos, item_train, item_index, item_num):
        '''

        :param user_ids: batch_size个不同的随机用户id
        :param log_seqs: batch_size个用户排除掉最后一个，倒数maxlen个物品集合 (-2, -2-maxlen) 集合大小(batch_size, maxlen) 如果用户没有那么多物品 进行左置补零
        :param time_matrices: batch_size个用户的r矩阵
        :param ratings 评分相关参数
        :return:
        '''
        weight = self.weight_high
        weight_position = self.weight_pos
        weight_new_pos = self.weight_new_pos
        seqs_tensor = torch.LongTensor(log_seqs).to(self.dev)  # 象征论文中哪个变量
        seqs = self.item_emb(seqs_tensor)
        seqs *= self.item_emb.embedding_dim ** 0.5  # sqrt(d)
        seqs = self.item_emb_dropout(seqs)
        new_positions = np.zeros([log_seqs.shape[0], log_seqs.shape[1]], dtype=np.int32)
        # for i in range(log_seqs.shape[0]):
        #     for j in range(log_seqs.shape[1]):
        #         if log_seqs[i][j] in item_num and item_num[log_seqs[i][j]] != 0:
        #             # new_positions[i][j] = int((j + int(item_index[log_seqs[i][j]] // item_num[log_seqs[i][j]])) // 2)
        #             new_positions[i][j] = int(weight_new_pos[j] * j + weight_position[j] * (item_index[log_seqs[i][j]] // item_num[log_seqs[i][j]]))
        #         else:
        #             new_positions[i][j] = j
        # positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])  # tile(集合，(y轴复制扩大倍数,x轴复制扩大倍数))
        positions = torch.LongTensor(new_positions).to(self.dev)
        abs_pos_K = self.abs_pos_K_emb(positions)  # 可学习位置嵌入矩阵 自注意机制中的键
        abs_pos_V = self.abs_pos_V_emb(positions)  # 可学习位置嵌入矩阵 自注意机制中的值
        abs_pos_K = self.abs_pos_K_emb_dropout(abs_pos_K)
        abs_pos_V = self.abs_pos_V_emb_dropout(abs_pos_V)
        ratings_seq = torch.LongTensor(ratings_seq).to(self.dev)
        rate_K = self.rate_K_emb(ratings_seq)
        rate_V = self.rate_V_emb(ratings_seq)
        rate_K = self.rate_K_dropout(rate_K)
        rate_V = self.rate_V_dropout(rate_V)

        item_train = torch.LongTensor(item_train).to(self.dev)
        item_K = self.item_K_emb(item_train)
        item_V = self.item_V_emb(item_train)
        item_K = self.item_K_dropout(item_K)
        item_V = self.item_V_dropout(item_V)

        time_matrices = torch.LongTensor(time_matrices).to(self.dev)  # 时间戳相关的
        time_matrix_K = self.time_matrix_K_emb(time_matrices)
        time_matrix_V = self.time_matrix_V_emb(time_matrices)
        time_abs_matrices = torch.LongTensor(time_abs_matrices).to(self.dev)
        time_abs_matrix_K = self.time_abs_matrix_K_emb(time_abs_matrices)
        time_abs_matrix_V = self.time_abs_matrix_V_emb(time_abs_matrices)
        time_matrix_K = self.time_matrix_K_dropout(time_matrix_K)
        time_matrix_V = self.time_matrix_V_dropout(time_matrix_V)
        time_abs_matrix_K = self.time_abs_matrix_K_dropout(time_abs_matrix_K)
        time_abs_matrix_V = self.time_abs_matrix_V_dropout(time_abs_matrix_V)
        # mask 0th items(placeholder for dry-run) in log_seqs
        # would be easier if 0th item could be an exception for training
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        # i是一层前向传播
        for i in range(len(self.attention_layers)):
            # Self-attention, Q=layernorm(seqs), K=V=seqs
            # seqs = torch.transpose(seqs, 0, 1) # (N, T, C) -> (T, N, C)
            Q = self.attention_layernorms[i](seqs)  # PyTorch mha requires time first fmt
            mha_outputs = self.attention_layers[i].new_rate_forward(Q, seqs,
                                            timeline_mask, attention_mask,
                                            time_matrix_K, time_matrix_V,
                                            rate_K, rate_V,
                                            abs_pos_K, abs_pos_V,
                                            time_abs_matrix_K, time_abs_matrix_V,
                                            weight, weight_pos,
                                            item_K, item_V)
            # 下面这个效果不好
            # new_mha_outputs = self.attention_layers[i].new_forward(Q, seqs,
            #                                        timeline_mask, attention_mask,
            #                                        time_matrix_K, time_matrix_V,
            #                                        time_abs_matrix_K, time_abs_matrix_V,
            #                                        abs_pos_K, abs_pos_V)
            seqs = Q + mha_outputs
            # seqs = Q + new_mha_outputs
            # seqs = torch.transpose(seqs, 0, 1) # (T, N, C) -> (N, T, C)

            # Point-wise Feed-forward, actually 2 Conv1D for channel wise fusion
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)  # z[i]

        log_feats = self.last_layernorm(seqs)


        return log_feats

    def forward(self, user_ids, log_seqs, time_matrices, time_abs_marices, pos_seqs, neg_seqs):  # for training
        '''
        训练方法
        :param user_ids: batch_size个不同的随机用户id
        :param log_seqs: batch_size个用户排除掉最后一个，倒数maxlen个物品集合 (-2, -2-maxlen) 集合大小(batch_size, maxlen) 如果用户没有那么多物品 进行左置补零
        :param time_matrices: batch_size个用户的r矩阵
        :param pos_seqs: batch_size个用户倒数maxlen个物品的id集合 (-1, -1-maxlen) 集合大小(batch_size, maxlen)
        :param neg_seqs: 存放的是不存在于batch_size个用户的物品id集合 集合大小(batch_size, maxlen)
        :return:
        pos_logits 用户拥有物品得出的预测值
        neg_logits 用户未拥有物品得出的预测值
        '''
        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices, time_abs_marices)
        weight = self.weight_item
        new_log_feats = []
        for item in log_feats:
            item_s = item.tolist() * weight
            new_log_feats.append(item_s)
        log_feats = torch.tensor(new_log_feats).to(self.div)
        # log_final_feats = self.seq2feats(user_ids, pos_seqs, time_matrices, time_abs_marices)
        # feat_valid = log_final_feats[:, -1, :]
        # total_feat = log_feats[:, :, :]
        # weight = self.weight_item
        # total_sum = total_feat.matmul(weight)
        # final_feat = torch.sum(total_sum, dim=1).to(self.dev)
        # list_final = []
        # for i in final_feat:
        #     i_max = max(i.tolist())
        #     i_min = min(i.tolist())
        #     i_feat = list(
        #         map(lambda x: (2 * ((x - i_min) / (i_max - i_min)) - 1),
        #             i.tolist()))
        #     list_final.append(i_feat)
        # final_feat = torch.tensor(list_final)
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits  # pos_pred, neg_pred


    def new_rate_forward(self, user_ids, log_seqs, time_matrices, time_abs_marices, pos_seqs, neg_seqs, ratings_seq, ratings_pos, item_train, item_index, item_num):  # for training
        '''
        训练方法
        :param user_ids: batch_size个不同的随机用户id
        :param log_seqs: batch_size个用户排除掉最后一个，倒数maxlen个物品集合 (-2, -2-maxlen) 集合大小(batch_size, maxlen) 如果用户没有那么多物品 进行左置补零
        :param time_matrices: batch_size个用户的r矩阵
        :param pos_seqs: batch_size个用户倒数maxlen个物品的id集合 (-1, -1-maxlen) 集合大小(batch_size, maxlen)
        :param neg_seqs: 存放的是不存在于batch_size个用户的物品id集合 集合大小(batch_size, maxlen)
        :return:
        pos_logits 用户拥有物品得出的预测值
        neg_logits 用户未拥有物品得出的预测值
        '''
        # log_feats = self.seq2feats(user_ids, log_seqs, time_matrices, time_abs_marices)
        weight_pos = self.weight_high_pos
        kuan = weight_pos.shape[0]
        yin = weight_pos.shape[1]
        weight_pos = weight_pos.reshape([1, kuan, yin])
        log_feats = self.new_seq2feats(user_ids, log_seqs, time_matrices, time_abs_marices, ratings_seq, ratings_pos, weight_pos, item_train, item_index, item_num)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        '''
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        '''
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        # pos_logits = (log_feats * pos_embs).sum(dim=-1)
        # neg_logits = (log_feats * neg_embs).sum(dim=-1)
        # new_pos_logits = []
        # new_neg_logits = []
        # for i in pos_logits:
        #     z_i = i * weight
        #     new_pos_logits.append(z_i.tolist())
        # for j in neg_logits:
        #     z_j = j * weight
        #     new_neg_logits.append(z_j.tolist())
        # pos_logits = torch.tensor(new_pos_logits, dtype=torch.float32).to(self.dev)
        # neg_logits = torch.tensor(new_neg_logits, dtype=torch.float32).to(self.dev)


        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits, log_feats  # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, time_matrices, time_abs_matrices, item_indices): # for inference
        '''
        预测偏好分数
        :param user_ids:
        :param log_seqs:
        :param time_matrices: 时间戳相对值的r矩阵
        :param time_abs_matrices: 时间戳绝对值的r矩阵
        :param item_indices: 101个物品位置信息 1个测试集的 100个该用户未拥有的物品
        :return: item_indices通过模型得出的偏好分数
        '''
        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices, time_abs_matrices)
        # total_feat = log_feats[:, :, :]
        # weight = self.weight_item
        # total_sum = total_feat.matmul(weight)
        # final_feat = torch.sum(total_sum, dim=1).to(self.dev)
        # final_feat = torch.div(final_feat, final_feat.shape[1])
        # final_feat_mean = torch.div(final_feat, final_feat.shape[1])
        # final_feat_max = max(final_feat.tolist())
        # final_feat_min = min(final_feat.tolist())
        # final_feat = torch.tensor(list(map(lambda x: (2 * ((x - final_feat_min)/ (final_feat_max-final_feat_min))-1), final_feat.tolist()))).to(self.dev)
        final_feat_only = log_feats[:, -1, :] # only use last QKV classifier, a waste
        # final_feat_un = final_feat.unsqueeze(-1)
        # 初步判断公式(12)  可更改的地方
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)
        # logits = item_embs.matmul(final_feat_un).squeeze(-1)
        old_logits = item_embs.matmul(final_feat_only.unsqueeze(-1)).squeeze(-1)
        # preds = self.pos_sigmoid(logits) # rank same item list for different users
        # logits = self.pos_sigmoid(logits)
        return old_logits #, logits  # preds # (U, I)

    def rate_predict(self, user_ids, log_seqs, time_matrices, time_abs_matrices, rate_seqs, rate_pos, item_indices, item_train, item_index, item_num): # for inference
        '''
        预测偏好分数
        :param user_ids:
        :param log_seqs:
        :param time_matrices: 时间戳相对值的r矩阵
        :param time_abs_matrices: 时间戳绝对值的r矩阵
        :param item_indices: 101个物品位置信息 1个测试集的 100个该用户未拥有的物品
        :return: item_indices通过模型得出的偏好分数
        '''
        # log_feats = self.seq2feats(user_ids, log_seqs, time_matrices, time_abs_matrices)
        weight_pos = self.weight_high_pos
        new_log_feats = self.new_seq2feats(user_ids, log_seqs, time_matrices, time_abs_matrices, rate_seqs, rate_pos, weight_pos, item_train, item_index, item_num)
        # final_feat_only = log_feats[:, -1, :] # only use last QKV classifier, a waste
        final_feat_only = new_log_feats[:, -1, :]
        total_feat = new_log_feats[0]

        # final_feat_only = torch.add(final_feat_only[0], total_feat)
        # 初步判断公式(12)  可更改的地方
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)
        # final_feat_only = final_feat
        logits = item_embs.matmul(final_feat_only.unsqueeze(-1)).squeeze(-1)
        old_logits = logits[0]

        # total_feat = total_feat[:-1]
        # total_feat_sum = torch.div(torch.sum(total_feat, dim=0).to(self.dev), total_feat.shape[0]).to(self.dev)
        # total_logits = item_embs.matmul(total_feat_sum.unsqueeze(-1)).squeeze(-1)
        # old_logits_mean = torch.mean(old_logits)
        new_logits = old_logits
        # old_logits = torch.add(old_logits, total_logits).to(self.dev)
        # old_logits = logits
        # preds = self.pos_sigmoid(logits) # rank same item list for different users
        # logits = self.pos_sigmoid(logits)
        return new_logits #, logits  # preds # (U, I)

    # def new_predict(self, user_ids, log_seqs, time_matrices, time_abs_matrices, item_indices, new_item_matrices, item_idxs): # for inference
    #     '''
    #     :param user_ids:
    #     :param log_seqs:
    #     :param time_matrices:
    #     :param time_abs_matrices:
    #     :param item_indices:
    #     :param new_item_matrices:
    #     :return:
    #     '''
    #     log_feats = self.seq2feats(user_ids, log_seqs, time_matrices, time_abs_matrices)
    #
    #     final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste
    #     # 初步判断公式(12)  可更改的地方
    #     new_item_embs = self.item_emb(torch.LongTensor(new_item_matrices).to(self.dev))
    #     w = self.weight_item
    #     sim_items = new_item_embs[:-100]
    #     # new_item = torch.sum(sim_items[:-1], dim=0).matmul(w).tolist()
    #     # old_item = sim_items[-1].tolist()
    #     sum_items = torch.sum(sim_items, dim=0)
    #     new_item_embs_minmax = []
    #     result_item = sum_items.matmul(w)
    #     new_li = result_item.tolist()
    #     new_item_embs_minmax.append(new_li)
    #     for i in new_item_embs.tolist()[-100:]:
    #         new_item_embs_minmax.append(i)
    #     sum_new_item_embs = torch.tensor(new_item_embs_minmax).to(self.dev)
    #     new_logits = sum_new_item_embs.matmul(final_feat.unsqueeze(-1).to(self.dev)).squeeze(-1).to(self.dev)
    #     return new_logits  # , new_item, old_item  # preds # (U, I)

