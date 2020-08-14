import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import Dataset, DataLoader
import scipy.integrate as integrate
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CLASSIFIER:
    def __init__(self, opt, model, dataset, _train_X, _train_Y, netGs):

        self.train_X = _train_X
        self.train_Y = _train_Y

        self.test_seen_feature = torch.from_numpy(dataset.pfc_feat_data_train)
        self.test_seen_label = torch.from_numpy(dataset.labels_train)

        self.test_novel_feature = torch.from_numpy(dataset.pfc_feat_data_test)
        self.test_novel_label = torch.tensor(dataset.labels_test) + dataset.train_cls_num

        self.nclass = dataset.test_cls_num + dataset.train_cls_num
        self.seenclasses = torch.unique(self.test_seen_label)
        self.novelclasses = torch.unique(self.test_novel_label)

        self.opt = opt
        self.dataset = dataset
        self.weights = dataset.weights

        self.seenclasses_num = self.seenclasses.shape[0]
        self.novelclasses_num = self.novelclasses.shape[0]

        self.batch_size = self.opt.cls_batch_size
        self.nepoch = self.opt.cls_epoch
        # self.nclass = _nclass
        self.input_dim = _train_X.size(1)
        print('self.input_dim')
        print(self.input_dim)

        self.average_loss = 0

        self.model = []
        #
        for cls in model:
            self.model.append(cls.cuda())

        self.criterion = nn.NLLLoss()

        self.input = torch.FloatTensor(self.batch_size, self.input_dim).cuda()
        self.label = torch.LongTensor(self.batch_size).cuda()
        self.lr = self.opt.cls_lr

        self.optimizers = []
        for num in range(len(self.model)):
            f = list(filter(lambda x:  x.requires_grad, self.model[num].parameters()))
            self.optimizers.append(optim.Adam(f, lr=self.lr,  betas=(0.9, 0.999)))#

        self.criterion.cuda()
        self.input = self.input.cuda()
        self.label = self.label.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]

        self.loss = 0

        self.used_indices = torch.LongTensor([]).cuda()
        self.all_indices  = torch.linspace(0,self.ntrain-1,self.ntrain).long().cuda()

        self.current_epoch = 0

        self.acc_novel, self.acc_seen, self.H, self.acc = 0, 0, 0, 0

        self.intra_epoch_accuracies = [()]*10
        self.retrieval()
        # '''
        self.acc = self.fit_zsl()

        if opt.dataset == 'CUB2011':
            if self.acc >= 0.46:
                torch.save({
                    'state_dict_G0': netGs[0].state_dict(),
                    'state_dict_G1': netGs[1].state_dict(),
                    'state_dict_G2': netGs[2].state_dict(),
                    'state_dict_G3': netGs[3].state_dict(),
                    'state_dict_G4': netGs[4].state_dict(),
                    'state_dict_G5': netGs[5].state_dict(),
                    'state_dict_G6': netGs[6].state_dict(),
                }, "CUB_"+opt.splitmode+str(float(self.acc))+".tar")
        else:
            if self.acc >= 0.365:
                torch.save({
                    'state_dict_G0': netGs[0].state_dict(),
                    'state_dict_G1': netGs[1].state_dict(),
                    'state_dict_G2': netGs[2].state_dict(),
                    'state_dict_G3': netGs[3].state_dict(),
                    'state_dict_G4': netGs[4].state_dict(),
                    'state_dict_G5': netGs[5].state_dict(),
                }, "NAB_"+opt.splitmode+str(float(self.acc))+".tar")

    def fit_zsl(self):
        best_acc = 0
        mean_loss = 0
        a = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):

                for cls in self.model:
                    cls.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                inputv = self.input
                labelv = self.label
                for num in range(len(self.model)):
                    output = self.model[num](inputv[:, num*512:(num+1)*512])
                    loss = self.criterion(output, labelv)
                    mean_loss += loss.item()  # data[0]
                    loss.backward()
                    self.optimizers[num].step()

            self.current_epoch += 1

            with torch.no_grad():
                acc, weighted_acc = self.val(self.test_novel_feature, self.test_novel_label, self.novelclasses)
            print("acc: %.3f, weighted_acc: %.3f"%(acc.item(), weighted_acc.item()))

            if weighted_acc > best_acc:
                best_acc = weighted_acc

        return best_acc


    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            #print(start, end)
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            #print(start, end)
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]


    def val(self, test_X, test_label, target_classes):
        # weights = [0.999325, 0.62739927, 0.5503919, 0.72866416, 0.41019028, 0.64912724, 0.35950154]
        # weights = [0.7746, 0.5016, 0.4214, 0.0233, 0.4582, 0.0685, 0.5237]
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        weighted_predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            outputs = torch.zeros([end-start, target_classes.size(0)]).cuda()
            weighted_outputs = torch.zeros([end-start, target_classes.size(0)]).cuda()
            # for num in range(len(self.model)):
            for num in [1, 2, 3, 4, 5]:
                output = self.model[num](test_X[start:end, num*512:(num+1)*512].cuda())
                outputs += output
                weighted_outputs += self.weights[num] * output
            # output = self.model(test_X[start:end].cuda())
            _, predicted_label[start:end] = torch.max(outputs.data, 1)
            _, weighted_predicted_label[start:end] = torch.max(weighted_outputs.data, 1)

            start = end

        acc = self.compute_per_class_acc(map_label(test_label, target_classes), predicted_label, target_classes.size(0))
        weighted_acc = self.compute_per_class_acc(map_label(test_label, target_classes), weighted_predicted_label, target_classes.size(0))

        return acc, weighted_acc

    def compute_per_class_acc(self, test_label, predicted_label, nclass):

        per_class_accuracies = torch.zeros(nclass).float().cuda().detach()

        target_classes = torch.arange(0, nclass, out=torch.LongTensor()).cuda() #changed from 200 to nclass on 24.06.
        predicted_label = predicted_label.cuda()
        test_label = test_label.cuda()

        for i in range(nclass):

            is_class = test_label==target_classes[i]

            per_class_accuracies[i] = torch.div((predicted_label[is_class]==test_label[is_class]).sum().float(), is_class.sum().float())

        return per_class_accuracies.mean()

    def retrieval(self):

        self.part_cls_centrild = np.zeros((self.dataset.test_cls_num, self.dataset.part_num, 512))
        for i in range(self.dataset.test_cls_num):
            for p in range(self.dataset.part_num):
                self.part_cls_centrild[i][p] = np.mean(self.train_X[self.train_Y == i, p * 512:(p + 1) * 512].numpy(), axis=0)

        test_num = self.test_novel_feature.shape[0]
        dist = np.zeros((self.dataset.test_cls_num, test_num))
        for p in range(self.dataset.part_num):
            dist_p = cosine_similarity(self.part_cls_centrild[:, p, :], self.test_novel_feature[:, p * 512:(p + 1) * 512])
            dist = dist + dist_p * self.dataset.weights[p]

        precision_100 = torch.zeros(self.dataset.test_cls_num)
        precision_50 = torch.zeros(self.dataset.test_cls_num)
        precision_25 = torch.zeros(self.dataset.test_cls_num)

        dist = torch.from_numpy(-dist)
        for i in range(self.dataset.test_cls_num):
            is_class = (self.test_novel_label-self.dataset.train_cls_num) == i
            # print(is_class.sum())
            cls_num = int(is_class.sum())

            #100%
            _, idx = torch.topk(dist[i, :], cls_num, largest=False)
            precision_100[i] = (is_class[idx]).sum().float()/cls_num

            #50%
            cls_num_50 = int(cls_num/2)
            _, idx = torch.topk(dist[i, :], cls_num_50, largest=False)
            precision_50[i] = (is_class[idx]).sum().float()/cls_num_50

            #25%
            cls_num_25 = int(cls_num/4)
            _, idx = torch.topk(dist[i, :], cls_num_25, largest=False)
            precision_25[i] = (is_class[idx]).sum().float()/cls_num_25
        print("retrieval results 100%%: %.3f 50%%: %.3f 25%%: %.3f"%(precision_100.mean().item(),
                        precision_50.mean().item(), precision_25.mean().item()))




# retrieval:  tensor(0.4171) tensor(0.4793) tensor(0.5061)

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i

    return mapped_label

