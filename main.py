# coding=utf-8
import os
import time
import torch
import pickle
import argparse

from model import TiSASRec
from tqdm import tqdm
from utils import *

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', required=True)
# parser.add_argument('--train_dir', required=True)
parser.add_argument('--dataset', default="ml-1m")  # 数据集
parser.add_argument('--train_dir', default="default")  # 训练路径
parser.add_argument('--batch_size', default=128, type=int)  # 每轮所用个数
parser.add_argument('--lr', default=0.001, type=float)  # 学习率
# parser.add_argument('--lr', default=0.0012, type=float)
parser.add_argument('--maxlen', default=50, type=int)  # 开辟空间大小
parser.add_argument('--hidden_units', default=50, type=int)  # 潜在维数?
parser.add_argument('--num_blocks', default=2, type=int)  # FFN 2层的意思
parser.add_argument('--num_epochs', default=201, type=int)  # 学习轮数
# parser.add_argument('--num_epochs', default=101, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)  # dropout率
parser.add_argument('--l2_emb', default=0.00005, type=float)  # 正则化系数
# parser.add_argument('--device', default='cpu', type=str)  # 运行环境
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)  # 如果是True 不进行训练直接预测 前提是 有已经存在的模型
parser.add_argument('--state_dict_path', default=None, type=str)  # 已经存在的模型的路径
# parser.add_argument('--time_span', default=256, type=int)  # k值 对r时间戳剪切时用的
parser.add_argument('--time_span', default=6000, type=int)
parser.add_argument('--time_slice', default=600, type=int)  # 时间片 根据时间片切割（新增的）

args = parser.parse_args()
if __name__ == '__main__':
    if not os.path.isdir(args.dataset + '_' + args.train_dir):
        os.makedirs(args.dataset + '_' + args.train_dir)
    with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    f.close()

    # 1.数据处理
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum, timenum] = dataset
    num_batch = len(user_train) // args.batch_size  # 将数据按照设定的个数分成相应的分数
    cc = 0.0  # 所有用户评分记录数量
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')

    # 下面是判断 该模型是否已经存在
    try:
        relation_matrix = pickle.load(open('data/relation_matrix1_%s_%d_%d.pickle'%(args.dataset, args.maxlen, args.time_span),'rb'))
    except:
        relation_matrix = Relation(user_train, usernum, args.maxlen, args.time_span)  # 二维相对位置时间戳矩阵
        relation_abs_matrix = Relation_abs(user_train, usernum, args.maxlen, args.time_span)  # 一维绝对位置的时间戳矩阵
    #    pickle.dump(relation_matrix, open('data/relation_matrix_%s_%d_%d.pickle'%(args.dataset, args.maxlen, args.time_span),'wb'))
    # ******判断结束******
    # 2.准备训练数据
    sampler = WarpSampler(user_train, usernum, itemnum, relation_matrix, relation_abs_matrix, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    # 3.创建模型
    model = TiSASRec(usernum, itemnum, itemnum, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_uniform_(param.data)  # 是一个服从均匀分布的Glorot初始化器
        except:
            pass  # just ignore those failed init layers

    model.train()  # enable model training

    epoch_start_idx = 1
    # 判断有没有已经训练的模型 自己训练时暂时用不到
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
    # ******结束******
    # 测试
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
    # ******结束******
    # 损失函数
    bce_criterion = torch.nn.BCEWithLogitsLoss()  # 这个损失函数是sigmoid + BCELoss
    # 优化器
    weight_paras = []
    paras = []
    for name, p in model.named_parameters():
        if name == "weight_item":
            p.requires_grad = True
            weight_paras.append(p)
        else:
            p.requires_grad = True
            paras.append(p)
    # adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    adam_optimizer = torch.optim.Adam(paras, lr=args.lr, betas=(0.9, 0.98))
    adam_optimiz = torch.optim.Adam(weight_paras, lr=0.0002, betas=(0.9, 0.98))
    bce_loss = torch.nn.SmoothL1Loss(reduction='mean')
    # 优化器参数
    '''
    params(iterable)：可用于迭代优化的参数或者定义参数组的dicts。
    lr(float, optional) ：学习率(默认: 1e-3)
    betas(Tuple[float, float], optional)：用于计算梯度的平均和平方的系数(默认: (0.9, 0.999))
    eps(float, optional)：为了提高数值稳定性而添加到分母的一个项(默认: 1e-8)
    weight_decay(float, optional)：权重衰减(如L2惩罚)(默认: 0)
    step(closure=None)函数：执行单一的优化步骤
    closure (callable, optional)：用于重新评估模型并返回损失的一个闭包
    '''

    T = 0.0
    # 记录每轮运行时间
    t0 = time.time()

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only:
            break  # just to decrease identition
        # 总数/batch_size 的份数 来当成循环次数
        for step in range(num_batch):
            # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            # 从result_queue中得到数据
            u, seq, time_seq, time_matrix, time_abs_matrix, pos, neg = sampler.next_batch()  # tuples to ndarray
            '''
            u batch_size个不同的随机用户id
            seq batch_size个用户排除掉最后一个，倒数maxlen个物品集合 (-2, -2-maxlen) 集合大小(batch_size, maxlen) 如果用户没有那么多物品 进行左置补零
            time_seq batch_size个用户相应的时间戳集合 集合大小(batch_size, maxlen)
            time_matrix batch_size个用户的r矩阵
            pos batch_size个用户倒数maxlen个物品的id集合 (-1, -1-maxlen) 集合大小(batch_size, maxlen)
            neg 存放的是不存在于该用户的物品id集合 干什么用?? 集合大小(batch_size, maxlen)
            '''
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            time_seq, time_matrix, time_abs_matrix = np.array(time_seq), np.array(time_matrix), np.array(time_abs_matrix)
            # 预测值
            pos_logits, neg_logits = model(u, seq, time_matrix, time_abs_matrix, pos, neg)
            # 真实值
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()

            indices = np.where(pos != 0)
            '''损失计算 公式(14)'''
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            # loss += bce_loss(pos_items[0], pos_items_labels[0])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            # 下面是正则项的损失
            for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.abs_pos_K_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
                # new_loss += args.l2_emb * torch.norm(param)
            for param in model.abs_pos_V_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
                # new_loss += args.l2_emb * torch.norm(param)
            for param in model.time_matrix_K_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.time_matrix_V_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.time_abs_matrix_K_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            for param in model.time_abs_matrix_V_emb.parameters(): loss += args.l2_emb * torch.norm(param)
            '''***结束***'''

            # # new_loss.requires_grad_(True)
            # # new_loss.backward(retain_graph=True)

            loss.backward()
            adam_optimizer.step()
            # adam_optimiz.zero_grad()
            # pos_items, pos_items_labels = model.new_forward(u, seq, time_matrix, time_abs_matrix, pos, neg)
            # torch.autograd.set_detect_anomaly(True)
            # new_loss = bce_loss(pos_items[0], pos_items_labels[0])  # 针对位置的改造
            # new_loss.backward(retain_graph=True)
            # adam_optimiz.step()

            # print("weight:{}  pos_item:{} real_item:{}".format(model.weight_item[0][0], pos_items[0][0], pos_items_labels[0][0]))
            # print("loss in epoch {} iteration {}: {} and item_loss {}".format(epoch, step, loss.item(), new_loss.item()))  # expected 0.4~0.6 after init few epochs
            # print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))

        # 每20轮 进行一次输出 输出验证集与测试集的成果
        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            # 测试集的结果

            t_test = new_evaluate(model, dataset, args)
            # new_loss = bce_criterion(torch.tensor(t_test[2]), torch.tensor(t_test[3]))
            # 验证集的结果
            t_valid = new_evaluate_valid(model, dataset, args)
            # new_loss += bce_criterion(torch.tensor(t_test[2]), torch.tensor(t_test[3]))
            # new_loss.requires_grad_(True)
            # adam_optimiz.zero_grad()
            # new_loss.backward()
            # adam_optimiz.step()
            print()
            # print("w的权值:", paras[0][0][0:2])
            # print("item_emb的权值:", model.item_emb.weight[0][0:2])
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
            model.train()
        # 当最后一轮的时候 将模型存下来
        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'TiSASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))

    f.close()
    sampler.close()
    print("Done")
