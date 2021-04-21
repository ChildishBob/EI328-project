import pickle
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
import random
import torch.backends.cudnn as cudnn

sys.path.append('./')

from UDAB.models import UDAB
from datatools.dataloader import SEEDDataset
from torch.utils.data import DataLoader


# random.seed(0)
# torch.manual_seed(0)

cudnn.benchmark = True


LEARNING_RATE = 1e-3
NUM_EPOCH = 20
CUDA = True
BATCH_SIZE = 256
LOSS = 'NLL'
MOMENTUM = 0.5
LAMBDA = 0.5
TRANSFER_BASE = 0
TEST_IDX = 4


acc_list = []
for TEST_IDX in range(15):
    print(f'========== test index {TEST_IDX} ==========')
    model = UDAB(track_running_stats=True, momentum=MOMENTUM)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # MSE loss
    # class_loss = nn.MSELoss()
    # domain_loss = nn.MSELoss()

    # NLL loss
    class_loss = nn.NLLLoss()
    domain_loss = nn.NLLLoss()

    if CUDA:
        model = model.cuda()

    for p in model.parameters():
        p.requires_grad = True

    dataset_source = SEEDDataset(train=True, one_hot=False, test_idx=TEST_IDX)
    dataloader_source = DataLoader(dataset=dataset_source, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)

    dataset_target = SEEDDataset(train=False, one_hot=False, test_idx=TEST_IDX)
    dataloader_target = DataLoader(dataset=dataset_target, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)

    
    for epoch in range(NUM_EPOCH):

        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)

        model.train()
        for i in range(len_dataloader):
            p = float(i + (epoch-TRANSFER_BASE) * len_dataloader) / (NUM_EPOCH-TRANSFER_BASE) / len_dataloader
            alpha = LAMBDA * (2. / (1. + np.exp(-10 * p)) - 1)

            if epoch < TRANSFER_BASE:
                alpha = 0
            
            # train model using source data
            data_source = data_source_iter.next()
            s_input, s_label = data_source

            
            optimizer.zero_grad()
            
            batch_size = len(s_label)

            # MSE loss
            # source_domain_label = torch.stack(batch_size * [torch.tensor([1, 0], dtype=torch.float32)])

            # NLL loss
            source_domain_label = torch.zeros(batch_size).long()

            if CUDA:
                s_input = s_input.cuda()
                s_label = s_label.cuda()
                source_domain_label = source_domain_label.cuda()
            
            source_class_pred, source_domain_pred = model(input_data=s_input, alpha=alpha)

            source_c_loss = class_loss(source_class_pred, s_label)
            source_d_loss = domain_loss(source_domain_pred, source_domain_label)

            # train model using target data
            data_target = data_target_iter.next()
            t_input, t_label = data_target

            batch_size = len(t_input)

            # MSE loss
            # target_domain_label = torch.stack(batch_size * [torch.tensor([0, 1], dtype=torch.float32)])

            # NLL loss
            target_domain_label = torch.ones(batch_size).long()

            if CUDA:
                t_input = t_input.cuda()
                target_domain_label = target_domain_label.cuda()
            
            target_class_pred, target_domain_pred = model(input_data=t_input, alpha=alpha)

            target_d_loss = domain_loss(target_domain_pred, target_domain_label)

            loss = source_c_loss + source_d_loss + target_d_loss

            # loss = source_c_loss

            loss.backward()
            optimizer.step()

        # classification accuracy of current model on test data (target domain)
        model.eval()
        num_correct = 0
        test_input, test_label = dataset_target[:]
        # print(test_input.shape)
        if CUDA:
            test_input = test_input.cuda()
        with torch.no_grad():
            target_class_pred, _ = model(input_data=test_input, alpha=alpha)
        # print(target_class_pred[0])
        target_class_pred = target_class_pred.cpu()
        for j in range(len(dataset_target)):
            if (LOSS == 'MSE' and np.argmax(target_class_pred[j].numpy()) == np.argmax(test_label[j].numpy())) or (LOSS == 'NLL' and np.argmax(target_class_pred[j].numpy()) == test_label[j]):
                num_correct += 1
        
        acc = round(num_correct/len(dataset_target), 4)
        print(f'Epoch {epoch} Accuracy {acc:.4f}')
    acc_list.append(acc)

    # plt.plot(range(1, NUM_EPOCH+1), accuracies)
    # plt.savefig(f'./UDAB/results/testOn{TEST_IDX}_{LEARNING_RATE:.1e}_mom{MOMENTUM}_lam{LAMBDA}.png')

print(f'Accuracy list:\n{acc_list}')
print(f'Mean accuracy: {sum(acc_list)/len(acc_list):.4f}')
    

