import functools
import sys
import copy
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Process, Queue
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from numpy.linalg import norm


# 最大-最小值归一化
def min_max_gui(sum_list):
    ma_list = max(sum_list)
    mi_list = min(sum_list)
    chu = ma_list - mi_list
    if chu == 0:
        chu = chu + 1e-6
    for i in range(len(sum_list)):
        sum_list[i] = (sum_list[i] - mi_list) / chu
    ma_list = max(sum_list)
    if ma_list > 1:
        sum_list = min_max_gui(sum_list)
    return sum_list


# 最大绝对值归一化
def abs_max_gui(sum_list):
    chu = max(abs(sum_list))
    if chu == 0:
        chu = chu + 1e-6
    for i in range(len(sum_list)):
        sum_list[i] = sum_list[i] / chu
    ma_list = max(sum_list)
    mi_list = min(sum_list)
    if ma_list > 1 or mi_list < -1:
        sum_list = abs_max_gui(sum_list)
    return sum_list


# 最大绝对值归一化
def base_max_gui(sum_list):
    chu = max(sum_list)
    if chu == 0:
        chu = chu + 1e-6
    for i in range(len(sum_list)):
        sum_list[i] = sum_list[i] / chu
    ma_list = max(sum_list)
    mi_list = min(sum_list)
    if ma_list > 1 or mi_list < -1:
        sum_list = abs_max_gui(sum_list)
    return sum_list


# 归一化方法的选择
def selct_gui(sum_list, flag):
    if flag == 0:
        sum_list = min_max_gui(sum_list)
    elif flag == 1:
        sum_list = abs_max_gui(sum_list)
    elif flag == 2:
        sum_list = base_max_gui(sum_list)
    return sum_list


# 对比学习的计算
def low_contrast(bce_criterion, contrast_dict, r_sim, contrast_user, u):
    fanzhi = 5  # 反对对数
    contrast_loss = 0
    loss = 0
    # r_sim = r_sim.tolist()[0]
    # 对比学习损失计算
    for u_ind in range(len(u)):
        user_id = u[u_ind]
        log_feats_1 = contrast_dict[u[u_ind]][0].log_feats
        log_feats_2 = contrast_dict[u[u_ind]][1].log_feats
        log_feats_1 = log_feats_1.cpu().detach().numpy()
        log_feats_2 = log_feats_2.cpu().detach().numpy()
        cos_zheng = cosine_similarity(log_feats_1, log_feats_2)
        cos_zheng = np.sum(cos_zheng, axis=0)
        cos_zheng_gui = selct_gui(cos_zheng, 2)  # max归一化
        chu = 0
        user_other = contrast_user[user_id]
        zhui = 1  # 为防止exp()得出值过大，设的固定参数
        r_sim = r_sim * zhui
        # 防止分母为0
        if r_sim == 0:
            r_sim = 1e-08
        cos_l = len(cos_zheng_gui)
        e_zheng = np.sum(cos_zheng_gui) / (r_sim * cos_l)
        random_list = []
        for fu_ind in range(fanzhi):
            fu_random = random.randint(0, len(user_other)-1)
            while fu_random in random_list:
                fu_random = random.randint(0, len(user_other) - 1)
            random_list.append(fu_random)
        for fu_ind in random_list:
            user_other_id = user_other[fu_ind][0]
            # 下面是两个负对
            log_feats_fu_1 = contrast_dict[user_other_id][0].log_feats
            log_feats_fu_1 = log_feats_fu_1.cpu().detach().numpy()
            log_feats_fu_2 = contrast_dict[user_other_id][1].log_feats
            log_feats_fu_2 = log_feats_fu_2.cpu().detach().numpy()
            cos_fu_1 = np.sum(cosine_similarity(log_feats_1, log_feats_fu_1), axis=0)
            cos_fu_gui_1 = selct_gui(cos_fu_1, 2)
            cos_fu_2 = np.sum(cosine_similarity(log_feats_1, log_feats_fu_2), axis=0)
            cos_fu_gui_2 = selct_gui(cos_fu_2, 2)
            cos_fu_3 = np.sum(cosine_similarity(log_feats_2, log_feats_fu_1), axis=0)
            cos_fu_gui_3 = selct_gui(cos_fu_3, 2)
            cos_fu_4 = np.sum(cosine_similarity(log_feats_2, log_feats_fu_2), axis=0)
            cos_fu_gui_4 = selct_gui(cos_fu_4, 2)
            if fu_ind == u_ind:
                continue
            # consin = np.dot(log_feats_1,log_feats_2) / (norm(log_feats_1)*norm(log_feats_2))
            fu_l = len(cos_fu_gui_1)
            fu_l2 = len(cos_fu_gui_2)
            consin = e_zheng + np.sum(cos_fu_gui_1) / (r_sim)
            consin_2 = e_zheng + np.sum(cos_fu_gui_2) / (r_sim)
            consin_3 = e_zheng + np.sum(cos_fu_gui_3) / (r_sim)
            consin_4 = e_zheng + np.sum(cos_fu_gui_4) / (r_sim)
            chu = chu + math.exp(consin) + math.exp(consin_2) + math.exp(consin_3) + math.exp(consin_4)
            # chu = chu + math.exp(consin) + math.exp(consin_2)
            # chu = chu + math.exp(consin)
        chu = chu + 1e-08  # 防止分母为0
        con_loss = -math.log(math.exp(e_zheng) / chu)
        contrast_loss += con_loss
        for contrast_step in range(2):
            contrast = contrast_dict[u[u_ind]][contrast_step]
            pos = contrast.pos
            pos_logits = contrast.pos_logits
            pos_labels = contrast.pos_labels
            neg_logits = contrast.neg_logits
            neg_labels = contrast.neg_labels
            indices = np.where(pos != 0)
            loss += bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
    return loss, contrast_loss


# 对比学习挑选数据排序方法
def contrast_cmp(a, b):
    if a[1] < b[1]:
        return 1
    else:
        return -1


# 为对比学习挑选数据
def contrast_select(users, items):
    u_lenght = len(users)
    contrast_user = defaultdict(list)
    for i in range(u_lenght):
        my = users[i]
        my_items = items[i]
        if my in contrast_user:
            continue
        for j in range(0, u_lenght):
            if j == i:
                continue
            jishu = 0
            other_items = items[j]
            for item in other_items:
                if item == 0:
                    continue
                if item in my_items:
                    jishu += 1
            contrast_user[my].append([users[j], jishu])
    for user in users:
        user_list = contrast_user[user]
        user_list = sorted(user_list, key=functools.cmp_to_key(contrast_cmp))
        contrast_user[user] = user_list
    return contrast_user


def random_neq(l, r, s):
    '''
    返回一个不在s集合中的随机数,区间(l, r)
    :param l: 随机数左区间
    :param r: 右区间
    :param s: s集合
    :return: t 这个数 不存在s中
    '''
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def concat_abs_time(time_seq, item_idx, time_span, maxlen):
    size = time_seq.shape[0]
    time_abs_matrix = np.zeros([size], dtype=np.int32)
    min_time = min(time_seq)
    max_time = max(time_seq)
    for i in range(size):
        # time_abs_matrix[i] = min(max_time - (time_seq[i] - min_time), time_span * maxlen)
        time_abs_matrix[i] = min(time_seq[i] - min_time, time_span * maxlen)
    return time_abs_matrix

def computeRePos(time_seq, time_span):
    '''
    计算r[i][j]
    :param time_seq: 该用户时间戳数组
    :param time_span: 论文中的k
    :return: 返回求得的时间戳矩阵
    '''
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    # mean_time = mean_matrix(time_seq, time_span)
    for i in range(size):
        for j in range(size):
            # if abs(time_seq[i]-time_seq[j]) >= mean_time:
            #     span = abs(time_seq[i]-time_seq[j]) - mean_time
            # else:
            #     span = abs(time_seq[i]-time_seq[j])
            span = abs(time_seq[i] - time_seq[j])
            if span > time_span:
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = span
    return time_matrix


def mean_matrix(time_seq, time_span):
    size = time_seq.shape[0]
    ji_matrix = np.zeros([size, size], dtype=np.int32)
    mean_time = 0
    sum = 0
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i]-time_seq[j])
            ji_matrix[i][j] = span
            if i == j or span > (time_span * size):
                continue
            else:
                sum = sum + 1
                mean_time += span
    mean_time = int(mean_time / sum)
    return mean_time

def Relation(user_train, usernum, maxlen, time_span):
    '''
    计算二维数组r[i][j]
    :param user_train: {"用户id":[[物品id,时间戳],...],...}
    :param usernum: 用户数量
    :param maxlen:开辟数组的大小
    :param time_span: 论文中计算r所用的阈值k
    :return: r[i][j]二维时间戳矩阵 处理好的时间戳
    '''
    data_train = dict()

    for user in tqdm(range(1, usernum+1), desc='Preparing relation matrix'):
        time_seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break
        data_train[user] = computeRePos(time_seq, time_span)
    return data_train


# 创新方法
def Relation_abs(user_train, usernum, maxlen, time_span):
    '''
    计算二维数组r_abs[i]
    :param user_train: {"用户id":[[物品id,时间戳],...],...}
    :param usernum: 用户数量
    :param maxlen:开辟数组的大小
    :param time_span: 论文中计算r所用的阈值k
    :return: r_abs[i]绝对时间戳矩阵 处理好的时间戳
    '''
    data_train = dict()
    for user in tqdm(range(1, usernum+1), desc='Preparing relation abs_matrix'):
        time_seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        time_list = list(map(lambda x:x[1], user_train[user]))
        min_time = min(time_list)
        max_time = max(time_list)
        max_time = max_time - min_time
        for i in reversed(user_train[user][:-1]):
            # for j in range(maxlen):
            #     time_seq[idx][j] = min(i[1] - min_time, time_span * (maxlen))
            #     # time_seq[idx][j] = min(i[1], time_span * (maxlen))
            # time_seq[idx] = min(max_time - (i[1] - min_time), time_span * (maxlen))
            time_seq[idx] = min(i[1] - min_time, time_span * (maxlen))
            idx -= 1
            if idx == -1: break
        data_train[user] = time_seq
    return data_train


# 多进程运算方法
def sample_function(user_train, usernum, itemnum, batch_size, maxlen, relation_matrix, relation_abs_matrix, result_queue, SEED):
    '''
    :param user_train: {"用户id":[[物品id,时间戳],...],...}
    :param usernum: 用户数量
    :param itemnum: 物品数量
    :param batch_size: 每轮选取数据的量
    :param maxlen:
    :param relation_matrix: 关系矩阵
    :param result_queue: 结果队列
    :param SEED:
    :return:
    user 用户id
    seq 该用户排除掉最后一个，倒数maxlen个物品集合 (-2, -2-maxlen)
    time_seq 用户相应的时间戳集合
    time_matrix 该用户的r矩阵
    pos 该用户倒数maxlen个物品的id集合 (-1, -1-maxlen)
    neg 存放的是不存在于该用户的物品id集合 干什么用??
    '''
    def sample(user):
        seq = np.zeros([maxlen], dtype=np.int32)
        time_seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        ratings_seq = np.zeros([maxlen], dtype=np.int32)
        ratings_pos = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1][0]  # 用户最后一个物品的物品id
        nxrate = user_train[user][-1][2]
        idx = maxlen - 1
        ts = set(map(lambda x: x[0], user_train[user]))  # 该用户物品id集合
        for i in reversed(user_train[user][:-1]):  # reversed()将集合反转再进行遍历
            seq[idx] = i[0]
            ratings_seq[idx] = i[2]
            time_seq[idx] = i[1]
            pos[idx] = nxt
            ratings_pos[idx] = nxrate
            if nxt != 0:
                neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i[0]
            nxrate = i[2]
            idx -= 1
            if idx == -1: break
        time_matrix = relation_matrix[user]  # r矩阵
        time_abs_matrix = relation_abs_matrix[user]

        return (user, seq, time_seq, time_matrix, time_abs_matrix, pos, neg, ratings_seq, ratings_pos)

    np.random.seed(SEED)  # 随机数种子
    while True:  # 结束条件??
        one_batch = []
        for i in range(batch_size):
            user = np.random.randint(1, usernum + 1)
            # 如果用户购买记录小于等于1 则再从新随机一个用户id
            while len(user_train[user]) <= 1:
                user = np.random.randint(1, usernum + 1)
            one_batch.append(sample(user))

        result_queue.put(zip(*one_batch))

class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, relation_matrix, relation_abs_matrix, batch_size=64, maxlen=10, n_workers=1):
        '''
        得出运行时所需要的数据
        :param User: 用户训练数据集 {"用户id":[[物品id,时间戳计算结果],...],....}
        :param usernum: 用户数量
        :param itemnum: 物品数量
        :param relation_matrix: r矩阵
        :param batch_size: 单轮处理数量
        :param maxlen: 开辟空间最大值
        :param n_workers: 线程数
        '''
        self.result_queue = Queue(maxsize=n_workers * 10)  # 结果队列，后面取数据方便一些
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                # target为线程调用方法 args 传入参数
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      relation_matrix,
                                                      relation_abs_matrix,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

def timeSlice(time_set):
    '''
    将所有数据的时间戳与最小时间戳进行减法运算
    :param time_set: 所有用户与物品的时间戳
    :return: 所有时间戳与最小时间的差值集合
    '''
    time_min = min(time_set)
    time_map = dict()
    for time in time_set:  # float as map key?
        time_map[time] = int(round(float(time-time_min)))
    return time_map


def conlatitem(datasets):
    Item_train = defaultdict(list)
    item_lens = {}
    len_user = datasets[3]
    for i in range(3):
        dataset = datasets[i]
        for user_id in dataset:
            for item in dataset[user_id]:
                Item_train[item[0]].append(user_id)
    for item in Item_train:
        item_lens[item] = len(Item_train[item])
    return Item_train, item_lens


def con_item(seq, item_lens):
    item_train = np.zeros([seq.shape[0], seq.shape[1]], dtype=np.int32)
    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            if seq[i][j] == 0:
                item_train[i][j] = 0
            else:
                item_train[i][j] = item_lens[seq[i][j]]
    return item_train


def global_item(seq, item_train):
    item_index = defaultdict(int)
    item_num = defaultdict(int)
    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            if seq[i][j] == 0:
                continue
            else:
                item_index[seq[i][j]] = item_index[seq[i][j]] + j+1
                item_num[seq[i][j]] = item_num[seq[i][j]] + 1
    return item_index, item_num


def cleanAndsort(User, time_map):
    '''
    数据处理
    :param User:里面放的是 {用户：[[物品，时间戳],...]},...
    :param time_map: 所有时间戳与最小时间的差值集合
    :return:
    User-res:{"用户id":[[物品id,时间戳计算结果],...],....}, 所有用户数量，所有物品数量， 所有时间戳计算结果中的最大值
    '''
    User_filted = dict()  # 与User一样
    user_set = set()  # 用户的集合
    item_set = set()  # 物品的集合
    for user, items in User.items():
        user_set.add(user)
        User_filted[user] = items
        for item in items:
            item_set.add(item[0])

    # 下面是为用户与物品重新附上新的下标*****
    user_map = dict()
    item_map = dict()
    for u, user in enumerate(user_set):
        user_map[user] = u+1
    for i, item in enumerate(item_set):
        item_map[item] = i+1
    # ******结束******

    for user, items in User_filted.items():
        User_filted[user] = sorted(items, key=lambda x: x[1])  # 快排 条件：是物品的时间戳

    User_res = dict()
    for user, items in User_filted.items():
        # User_res存储数据是:{用户下标：[[物品下标，时间戳差值],...]}
        User_res[user_map[user]] = list(map(lambda x: [item_map[x[0]], time_map[x[1]], x[2]], items))

    time_max = set()
    for user, items in User_res.items():  # User_res存储数据是:{用户下标：[[物品下标，时间戳],...]}
        time_list = list(map(lambda x: x[1], items))  # 时间差值list
        time_diff = set()
        for i in range(len(time_list)-1):
            if time_list[i+1]-time_list[i] != 0:
                time_diff.add(time_list[i+1]-time_list[i])  # 相邻两个时间戳的差值
        if len(time_diff)==0:
            time_scale = 1
        else:
            time_scale = min(time_diff)  # 两个时间戳差值的最小值
        time_min = min(time_list)  # 该用户所有时间戳的最小值
        # [((所有的时间戳-该用户物品时间戳最小值)/物品时间戳差值最小值)+1]向下取整
        # User_res[user] = list(map(lambda x: [x[0], int(round((x[1]-time_min)/time_scale)+1)], items))
        User_res[user] = list(map(lambda x: [x[0], int(round((x[1] - time_min) / time_scale) + 1), x[2]], items))
        time_max.add(max(set(map(lambda x: x[1], User_res[user]))))

    return User_res, len(user_set), len(item_set), max(time_max)


def data_partition(fname):
    '''
    读取数据与数据处理主方法
    :param fname: 文件名字
    :return:
    [训练数据集， 验证数据集， 测试数据集， 用户数量， 物品数量， 时间戳最大值]
    '''
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}  # 训练集
    user_valid = {}  # 验证集
    user_test = {}   # 测试集
    print('Preparing data...')
    f = open('data/%s.txt' % fname, 'r')
    time_set = set()

    user_count = defaultdict(int)  # 用户拥有物品的数量
    item_count = defaultdict(int)  # 物品有多少用户拥有的数量
    
    # 下面是读数据与数据的统计
    for line in f:
        try:
            u, i, rating, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        user_count[u]+=1
        item_count[i]+=1
    f.close()
    # ******读取结束******
    
    # 下面是读数据与数据的筛选
    f = open('data/%s.txt' % fname, 'r')  # try?...ugly data pre-processing code
    for line in f:
        try:
            u, i, rating, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
            rating = None
        u = int(u)
        i = int(i)
        timestamp = float(timestamp)
        if user_count[u] < 5 or item_count[i] < 5:  # hard-coded
            continue
        time_set.add(timestamp)
        if rating != None:
            User[u].append([i, timestamp, int(float(rating))])
        else:
            User[u].append([i, timestamp])
    f.close()
    # ******读取结束******
    
    time_map = timeSlice(time_set)   # 时间方法计算
    User, usernum, itemnum, timenum = cleanAndsort(User, time_map)  # 时间戳的处理
    for user in User:   # User:{"用户id":[[物品id,时间戳计算结果],...],....}
        nfeedback = len(User[user])
        # 当用户时间戳数量小于3的情况 只有训练集
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]  # 将所有数据直到倒数第二位都是训练数据
            user_valid[user] = []
            user_valid[user].append(User[user][-2])  # 倒数第二个作为有效数据
            user_test[user] = []
            user_test[user].append(User[user][-1])  # 该用户最后一个当做测试集
    print('Preparing done...')
    return [user_train, user_valid, user_test, usernum, itemnum, timenum]


def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        
        seq[idx] = valid[u][0][0]
        time_seq[idx] = valid[u][0][1]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break
        rated = set(map(lambda x: x[0],train[u]))
        rated.add(valid[u][0][0])
        rated.add(test[u][0][0])
        new_item_idx = list(rated)
        rated.add(0)
        item_idx = [test[u][0][0]]  # 该用户没有物品
        only_idx = []
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            new_item_idx.append(t)
            only_idx.append(t)

        time_matrix = computeRePos(time_seq, args.time_span)
        time_abs_matrix = concat_abs_time(time_seq, item_idx, args.max_time_span, args.maxlen)

        # predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix], [time_abs_matrix], item_idx]])
        new_predictions = -model.new_predict(*[np.array(l) for l in [[u], [seq], [time_matrix], [time_abs_matrix], only_idx, new_item_idx, item_idx]])
        # predictions = predictions[0]  # 当new_predict()使用时 注掉
        predictions = new_predictions[0]
        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.',end='')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def new_evaluate(model, dataset, args, item_lens):
    [train, valid, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    new_items = []
    old_items = []
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        rate_seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1

        seq[idx] = valid[u][0][0]
        time_seq[idx] = valid[u][0][1]
        rate_seq[idx] = valid[u][0][2]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            rate_seq[idx] = i[2]
            idx -= 1
            if idx == -1: break
        rate_pre = [test[u][0][2]]
        rated = set(map(lambda x: x[0], train[u]))
        rated.add(valid[u][0][0])
        rated.add(test[u][0][0])
        new_item_idx = list(rated)
        only_idx = list(rated)
        rated.add(0)
        item_idx = [test[u][0][0]]  # 该用户没有物品
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            new_item_idx.append(t)

        time_matrix = computeRePos(time_seq, args.time_span)
        time_abs_matrix = concat_abs_time(time_seq, item_idx, args.max_time_span, args.maxlen)
        item_train = con_item(seq.reshape([1, seq.shape[0]]), item_lens)[0]
        item_index, item_num = global_item(seq.reshape([1, seq.shape[0]]), item_train)
        # predictions = model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix], [time_abs_matrix], item_idx]])
        # old_predictions= model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix], [time_abs_matrix], item_idx]])
        old_predictions = model.rate_predict(*[np.array(l) for l in [[u], [seq], [time_matrix], [time_abs_matrix], [rate_seq], [rate_pre], item_idx, [item_train], [item_index], [item_num]]])
        # predictions = -model.new_predict(*[np.array(l) for l in [[u], [seq], [time_matrix], [time_abs_matrix], only_idx, new_item_idx, item_idx]])
        # new_items.append(new_predictions[1])
        # old_items.append(new_predictions[2])
        old_predictions = -old_predictions  # 当new_predict()使用时 注掉
        # new_predictions = -new_predictions[0]
        # predictions = -new_predictions[0][0]
        old_rank = old_predictions.argsort().argsort()[0].item()
        # new_rank = new_predictions.argsort().argsort()[0].item()
        # rank = new_rank
        # rank = min(old_rank, new_rank)
        rank = old_rank
        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end='')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user # , new_items, old_items

def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break

        rated = set(map(lambda x: x[0], train[u]))
        rated.add(valid[u][0][0])
        new_item_idx = list(rated)
        rated.add(0)
        only_idx = []
        item_idx = [valid[u][0][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            new_item_idx.append(t)
            only_idx.append(t)

        time_matrix = computeRePos(time_seq, args.time_span)
        time_abs_matrix = concat_abs_time(time_seq, item_idx, args.max_time_span, args.maxlen)
        # predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix], [time_abs_matrix], item_idx]])
        predictions = -model.new_predict(*[np.array(l) for l in [[u], [seq], [time_matrix], [time_abs_matrix], only_idx, new_item_idx, item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()
        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.',end='')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

def new_evaluate_valid(model, dataset, args, item_lens):
    [train, valid, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    new_items = []
    old_items = []
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        rate_seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            rate_seq[idx] = i[2]
            idx -= 1
            if idx == -1: break

        rated = set(map(lambda x: x[0], train[u]))
        rated.add(valid[u][0][0])
        new_item_idx = list(rated)
        rated.add(0)
        only_idx = []
        item_idx = [valid[u][0][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            new_item_idx.append(t)
            only_idx.append(t)
        rate_pre = [valid[u][0][2]]
        time_matrix = computeRePos(time_seq, args.time_span)
        time_abs_matrix = concat_abs_time(time_seq, item_idx, args.max_time_span, args.maxlen)
        item_train = con_item(seq.reshape([1, seq.shape[0]]), item_lens)[0]
        item_index, item_num = global_item(seq.reshape([1, seq.shape[0]]), item_train)
        # predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix], [time_abs_matrix], item_idx]])
        # old_predictions = model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix], [time_abs_matrix], item_idx]])
        old_predictions = model.rate_predict(*[np.array(l) for l in [[u], [seq], [time_matrix], [time_abs_matrix], [rate_seq], [rate_pre], item_idx, [item_train], [item_index], [item_num]]])
        # new_items.append(predictions[1])
        # old_items.append(predictions[2])
        # predictions = predictions[0]
        old_predictions = -old_predictions # 当new_predict()使用时 注掉
        # new_predictions = -new_predictions[0]
        # predictions = -new_predictions[0][0]
        old_rank = old_predictions.argsort().argsort()[0].item()

        # new_rank = new_predictions.argsort().argsort()[0].item()
        # rank = min(old_rank, new_rank)
        rank = old_rank
        # rank = predictions.argsort().argsort()[0].item()
        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.',end='')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user # , new_items, old_items
