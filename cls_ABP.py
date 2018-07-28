from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
import util 
import classifier
import classifier2
import sys
import model
import glob
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='FLO', help='FLO')
parser.add_argument('--dataroot', default='/media/jianwen/ExtraDrive1/yizhe/Xian_Yongqin/cvpr17GBU/data/', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discriminator')
parser.add_argument('--nepoch', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=1, help='weight of the classification loss')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netG_name', default='')
parser.add_argument('--netD_name', default='')
parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
parser.add_argument('--outname', help='folder to output data and model checkpoints')
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--val_every', type=int, default=10)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
parser.add_argument('--latent_dim', type=int, default=20, help='dimention of latent z')
parser.add_argument('--hidden_dim', type=int, default=4096, help='dimention of hidden layer in generator')
parser.add_argument('--latent_var', type=float, default=1, help='variance of prior distribution z')

parser.add_argument('--sigma',   type=float, default=0.3, help='variance of random noise')
parser.add_argument('--sigma_U', type=float, default=1, help='variance of U_tau')
parser.add_argument('--langevin_s', type=float, default=0.1, help='s in langevin sampling')  # 0.1 or 0.3

parser.add_argument('--langevin_step', type=int, default=5, help='langevin step in each iteration')  # 10
# Hyper-parameter KNN
parser.add_argument('--N_KNN', type=int, default=20, help='K value')
opt = parser.parse_args()
print(opt)


try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


exp_info = 'GBU_{}'.format(opt.dataset)
exp_params = 'Eu{}_Rls{}'.format(1, 1)  # opt.CENT_LAMBDA, opt.REG_W_LAMBDA)

out_dir  = 'out/{:s}'.format(exp_info)
out_subdir = 'out/{:s}/{:s}'.format(exp_info, exp_params)
if not os.path.exists('out'):
    os.mkdir('out')
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
if not os.path.exists(out_subdir):
    os.mkdir(out_subdir)

print("The output dictionary is {}".format(out_subdir), 'red')

# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)
# Construct the latent variable z for each Image
train_z = torch.FloatTensor(len(data.train_feature), opt.latent_dim).cuda()
train_z.normal_(0, opt.latent_var)

# initialize generator and discriminator
# netG = model.MLP_G(opt)
# if opt.netG != '':
#     netG.load_state_dict(torch.load(opt.netG))
# print(netG)
#
# netD = model.MLP_CRITIC(opt)
# if opt.netD != '':
#     netD.load_state_dict(torch.load(opt.netD))
# print(netD)

# classification loss, Equation (4) of the paper
cls_criterion = nn.NLLLoss()

opt.nclass_all = data.nclass_all
opt.resSize = data.feature_dim
opt.attSize = data.att_dim

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
one = torch.FloatTensor([1])
mone = one * -1
input_label = torch.LongTensor(opt.batch_size)
input_idx = torch.LongTensor(opt.batch_size)
if opt.cuda:
    # netD.cuda()
    # netG.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    one = one.cuda()
    mone = mone.cuda()
    cls_criterion.cuda()
    input_label = input_label.cuda()

def sample():
    batch_feature, batch_label, batch_att, batch_idx = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_idx.copy_(batch_idx)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))

def generate_syn_feature(netG, classes, attribute, num, opt):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.latent_dim)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
        
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, opt.latent_var)
        output = netG(torch.cat((Variable(syn_att, volatile=True), Variable(syn_noise, volatile=True)), 1))
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label

# setup optimizer
# optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


# def calc_gradient_penalty(netD, real_data, fake_data, input_att):
#     #print real_data.size()
#     alpha = torch.rand(opt.batch_size, 1)
#     alpha = alpha.expand(real_data.size())
#     if opt.cuda:
#         alpha = alpha.cuda()
#
#     interpolates = alpha * real_data + ((1 - alpha) * fake_data)
#
#     if opt.cuda:
#         interpolates = interpolates.cuda()
#
#     interpolates = Variable(interpolates, requires_grad=True)
#
#     disc_interpolates = netD(interpolates, Variable(input_att))
#
#     ones = torch.ones(disc_interpolates.size())
#     if opt.cuda:
#         ones = ones.cuda()
#
#     gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
#                               grad_outputs=ones,
#                               create_graph=True, retain_graph=True, only_inputs=True)[0]
#
#     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
#     return gradient_penalty


class Conditional_Generator(nn.Module):
    def __init__(self, opt):
        super(Conditional_Generator, self).__init__()
        #   312 + 20  || 4096
        self.w1 = Variable(torch.FloatTensor(opt.attSize + opt.latent_dim, opt.hidden_dim).cuda(), requires_grad=True)
        self.b1 = Variable(torch.FloatTensor(opt.hidden_dim).cuda(), requires_grad=True)
        self.w2 = Variable(torch.FloatTensor(opt.hidden_dim, opt.resSize).cuda(), requires_grad=True)
        self.b2 = Variable(torch.FloatTensor(opt.resSize).cuda(), requires_grad=True)

        self.lrelu = nn.LeakyReLU(0.2, True).cuda()
        self.relu = nn.ReLU(True).cuda()

        # must initialize!
        self.w1.data.normal_(0, 0.02)
        self.w2.data.normal_(0, 0.02)
        self.b1.data.fill_(0)
        self.b2.data.fill_(0)

    def forward(self, att):
        a1 = self.lrelu(torch.mm(att, self.w1) + self.b1)
        a2 = self.relu(torch.mm(a1, self.w2) + self.b2)
        return a2


# Construct Log likelihood loss
def getloss(pred, x, z, opt):
    loss = 1/(2*opt.sigma**2) * torch.pow(x - pred, 2).sum() + 1/2 * torch.pow(z, 2).sum()
    loss /= x.size(0)
    return loss

class Result(object):
    def __init__(self):
        self.best_acc = 0.0
        self.best_iter = 0.0
        self.best_acc_S_T = 0.0
        self.best_acc_U_T = 0.0
        self.acc_list = []
        self.iter_list = []


import numpy as np
def KNN_CLASSIFIER(syn_feature, syn_label, test_feature, test_label, opt):

    # labels = [np.where(test_id == i) for i in test_label]
    # labels = np.asarray(labels).squeeze()
    test_label = test_label.numpy()
    idx = np.argsort(syn_label)
    syn_label = syn_label[idx]
    syn_feature = syn_feature[idx]

    gen_feat_mean = list()
    gen_feat = list()

    from sklearn.metrics.pairwise import cosine_similarity

    # Euclidean predict K-nearest Neighbor
    sim = cosine_similarity(test_feature, syn_feature)
    idx_mat = np.argsort(-1 * sim, axis=1)
    label_mat = (idx_mat[:, 0:opt.N_KNN] / opt.syn_num).astype(int)
    preds = np.zeros(label_mat.shape[0])
    for i in range(label_mat.shape[0]):
        (values, counts) = np.unique(label_mat[i], return_counts=True)
        preds[i] = values[np.argmax(counts)]

    # MCA
    acc_list = []
    for i in np.unique(test_label):
        acc_list.append((test_label[test_label == i] == preds[test_label == i]).mean())
    acc_cos = np.asarray(acc_list).mean()
    return acc_cos


def KNN_CLASSIFIER_GZSL(it, syn_feature, syn_label, data, opt, result):

    # labels = [np.where(test_id == i) for i in test_label]
    # labels = np.asarray(labels).squeeze()
    # data.test_seen_feature
    # data.test_seen_label
    # test_label = test_label.numpy()
    # idx = np.argsort(syn_label)
    # syn_label = syn_label[idx]
    # syn_feature = syn_feature[idx]
    test_seen_label = data.test_seen_label.numpy()
    test_seen_feature = data.test_seen_feature.numpy()
    test_unseen_label = data.test_unseen_label.numpy()
    test_unseen_feature = data.test_unseen_feature.numpy()

    """ S->T
    """
    # Euclidean predict K-nearest Neighbor
    sim = cosine_similarity(test_seen_feature, syn_feature)
    idx_mat = np.argsort(-1 * sim, axis=1)

    label_mat = syn_label.numpy()[idx_mat[:, 0:opt.N_KNN]]
    #label_mat = (idx_mat[:, 0:opt.N_KNN] / opt.syn_num).astype(int)
    preds = np.zeros(label_mat.shape[0])
    for i in range(label_mat.shape[0]):
        (values, counts) = np.unique(label_mat[i], return_counts=True)
        preds[i] = values[np.argmax(counts)]

    # MCA
    acc_list = []
    for i in np.unique(test_seen_label):
        acc_list.append((test_seen_label[test_seen_label == i] == preds[test_seen_label == i]).mean())
    acc_S_T = np.asarray(acc_list).mean()

    """ U->T
    """
    # Euclidean predict K-nearest Neighbor
    sim = cosine_similarity(test_unseen_feature, syn_feature)
    idx_mat = np.argsort(-1 * sim, axis=1)

    label_mat = syn_label.numpy()[idx_mat[:, 0:opt.N_KNN]]
    # label_mat = (idx_mat[:, 0:opt.N_KNN] / opt.syn_num).astype(int)
    preds = np.zeros(label_mat.shape[0])
    for i in range(label_mat.shape[0]):
        (values, counts) = np.unique(label_mat[i], return_counts=True)
        preds[i] = values[np.argmax(counts)]

    # MCA
    acc_list = []
    for i in np.unique(test_unseen_label):
        acc_list.append((test_unseen_label[test_unseen_label == i] == preds[test_unseen_label == i]).mean())
    acc_U_T = np.asarray(acc_list).mean()

    acc = (2 * acc_S_T * acc_U_T) / (acc_S_T + acc_U_T)

    result.acc_list += [acc]
    result.iter_list += [it]
    result.save_model = False
    if acc > result.best_acc:
        result.best_acc = acc
        result.best_iter = it
        result.best_acc_S_T = acc_S_T
        result.best_acc_U_T = acc_U_T
        result.save_model = True

    print("H {:.2f}%  S->T {:.2f}%  U->T {:.2f}%  Best_H [{:.2f}% {:.2f}% {:.2f}%] KNN".format(acc*100, acc_S_T*100,
                                                                                          acc_U_T*100,
                                                                                          result.best_acc*100,
                                                                                          result.best_acc_S_T*100,
                                                                                          result.best_acc_U_T*100,
                                                                                          ))

def save_model(it, netG, train_z, random_seed, log, fout):
    torch.save({
        'it': it + 1,
        'state_dict_G': netG.state_dict(),
        'latent_z': train_z,
        'random_seed': random_seed,
        'log': log,
    }, fout)


# train a classifier on seen classes, obtain \theta of Equation (4)
pretrain_cls = classifier.CLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses), data.seenclasses.size(0), opt.resSize, opt.cuda, 0.001, 0.5, 50, 100, opt.pretrain_classifier)

# freeze the classifier during the optimization
for p in pretrain_cls.model.parameters(): # set requires_grad to False
    p.requires_grad = False


net_Generator = Conditional_Generator(opt)
net_Generator.cuda()
print(net_Generator)
optimizerG = optim.Adam([net_Generator.w1, net_Generator.b1, net_Generator.w2, net_Generator.b2],
                        lr=opt.lr, weight_decay=1e-3)  # betas=(opt.beta1, 0.999))

best_acc_cls = 0.0
best_acc_cos = 0.0
best_epoch = 0
best_epoch_gzsl = 0
result = Result()
result_gzsl_knn = Result()
result_gzsl_cls = Result()

for epoch in range(opt.nepoch):
    for i in range(0, data.ntrain, opt.batch_size):
        sample()
        input_zv = Variable(train_z[input_idx.numpy(),], requires_grad=True)
        input_resv = Variable(input_res)
        input_attv = Variable(input_att)
        optimizer_z = torch.optim.Adam([input_zv], lr=opt.lr, weight_decay=1e-3) # betas=(opt.beta1, 0.999))  # 0.5

        # Alternative update Weight and latent_batch z
        for em_step in range(2):  # EM_STEP
            # update w
            for _ in range(1):  # 1
                pred = net_Generator.forward(torch.cat((input_attv, input_zv), 1))
                loss = getloss(pred, input_resv, input_zv, opt)

                # classification loss
                # c_errG = cls_criterion(pretrain_cls.model(pred), Variable(input_label))
                loss_T = loss  # + opt.cls_weight * c_errG

                optimizerG.zero_grad()
                loss_T.backward()
                # gradient clip makes it converge much faster!
                torch.nn.utils.clip_grad_norm([net_Generator.w1, net_Generator.b1, net_Generator.w2, net_Generator.b2], 1)
                optimizerG.step()
            # print("E step loss {}".format(loss.data[0]))
            # update z

            for _ in range(opt.langevin_step):  # 5 or 10
                U_tau = torch.FloatTensor(input_zv.shape).normal_(0, 0.1).cuda()
                pred = net_Generator.forward(torch.cat((input_attv, input_zv), 1))
                loss = getloss(pred, input_resv, input_zv, opt)

                # classification loss
                # c_errG = cls_criterion(pretrain_cls.model(pred), Variable(input_label))
                loss_T = opt.langevin_s*2/2 * loss  # + opt.cls_weight * c_errG

                optimizer_z.zero_grad()
                loss_T.backward()
                # gradient clip makes it converge much faster!
                torch.nn.utils.clip_grad_norm([input_zv], 1)
                optimizer_z.step()
                if epoch < opt.nepoch/2:
                    input_zv.data += opt.langevin_s * U_tau
            # print("M step loss {}".format(loss.data[0]))
        train_z[input_idx.numpy(),] = input_zv.data

        # Evaluated twice per Epoch
        if i==int(data.ntrain / opt.batch_size /2)*opt.batch_size or i==int(data.ntrain/opt.batch_size)*opt.batch_size:
            if epoch > 5:
                # evaluate the model, set G to evaluation mode
                net_Generator.eval()

                syn_feature, syn_label = generate_syn_feature(net_Generator, data.unseenclasses, data.attribute,
                                                              opt.syn_num, opt)

                cls = classifier2.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data,
                                             data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25,
                                             opt.syn_num, False)
                acc = cls.acc
                if acc > best_acc_cls:
                    best_acc_cls = acc

                acc_cos = KNN_CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses),
                                         data.test_unseen_feature,
                                         util.map_label(data.test_unseen_label, data.unseenclasses), opt)
                if acc_cos > best_acc_cos:
                    best_acc_cos = acc_cos
                    best_epoch = epoch

                print('unseen class accuracy= {:.2f}%, KNN_cos {:.2f}%  Best: {:.2f}%||{:.2f}%'.format(acc * 100,
                                                                                                       acc_cos * 100,
                                                                                                       best_acc_cls * 100,
                                                                                                       best_acc_cos * 100))

                # Generalized zero-shot learning
                syn_feature, syn_label = generate_syn_feature(net_Generator, data.unseenclasses, data.attribute,
                                                              opt.syn_num, opt)
                train_X = torch.cat((data.train_feature, syn_feature), 0)
                train_Y = torch.cat((data.train_label, syn_label), 0)
                nclass = opt.nclass_all
                cls = classifier2.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 25,
                                             opt.syn_num, True)


                result_gzsl_cls.acc_list += [cls.H]
                result_gzsl_cls.save_model = False
                if cls.H > result_gzsl_cls.best_acc:
                    result_gzsl_cls.best_acc = cls.H
                    result_gzsl_cls.best_acc_S_T = cls.acc_seen
                    result_gzsl_cls.best_acc_U_T = cls.acc_unseen
                    result_gzsl_cls.save_model = True
                    best_epoch_gzsl = epoch
                print(
                    "H {:.2f}%  S->T {:.2f}%  U->T {:.2f}%  Best_H [{:.2f}% {:.2f}% {:.2f}%] CLS".format(cls.H * 100,
                                                                                                     cls.acc_seen*100,
                                                                                                     cls.acc_unseen*100,
                                                                                                     result_gzsl_cls.best_acc*100,
                                                                                                     result_gzsl_cls.best_acc_S_T*100,
                                                                                                     result_gzsl_cls.best_acc_U_T*100,
                                                                                                     ))

                syn_feature, syn_label = generate_syn_feature(net_Generator, data.allclasses, data.attribute,
                                                              60, opt)
                KNN_CLASSIFIER_GZSL(0, syn_feature, syn_label, data, opt, result_gzsl_knn)


                # reset G to training mode
                net_Generator.train()
    # if i % 200 == 0 and i:
    print("[{}/{}] loss {}".format(epoch, opt.nepoch, loss_T.data[0]))
with open('{}_Result_sum_ZSL_GZSL.txt'.format(opt.dataset), 'a') as f:
    f.write(('Z_dim: {}, Lgv: Sigma {}, Step {}, Batchsize {}, Best_acc: {:.2f}% || {:.2f}%, GZSL [{:.2f}% {:.2f}% {:.2f}%] ||'
            ' [{:.2f}% {:.2f}% {:.2f}%] best_epoch: [ZSL {}, GZSL {}] Random_seed: {}\n').format(opt.latent_dim, opt.langevin_s, opt.langevin_step,
                                                                                                 opt.batch_size,
                                                                    best_acc_cls*100, best_acc_cos*100,
                                                                   result_gzsl_cls.best_acc * 100,
                                                                   result_gzsl_cls.best_acc_S_T * 100,
                                                                   result_gzsl_cls.best_acc_U_T * 100,
                                                                   result_gzsl_knn.best_acc * 100,
                                                                   result_gzsl_knn.best_acc_S_T * 100,
                                                                   result_gzsl_knn.best_acc_U_T * 100,
                                                                   best_epoch,
                                                                   best_epoch_gzsl,
                                                                   opt.manualSeed))




