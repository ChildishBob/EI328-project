import sys
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn

sys.path.append('./')

from ADDA.models import FeatureExtractor, Classifier, Discriminator
from datatools.dataloader import SEEDDataset
from torch.utils.data import DataLoader


cudnn.benchmark = True

LEARNING_RATE = 1e-4
NUM_EPOCH_PRE = 10
NUM_EPOCH = 10
CUDA = True
BATCH_SIZE = 256
BETA1 = 0.5
BETA2 = 0.9
BATCHNORM_TRACK = False
MOMENTUM = 0.5
# TEST_IDX = 14

pre_acc_list = []
da_acc_list = []
for TEST_IDX in range(15):
    print(f'###### test idx {TEST_IDX} ######')

    # train source feature extractor and classifier on source domain data
    source_feature_extractor = FeatureExtractor(track_running_stats=BATCHNORM_TRACK, momentum=MOMENTUM)
    classifier = Classifier(track_running_stats=BATCHNORM_TRACK, momentum=MOMENTUM)
    criterion = nn.CrossEntropyLoss()

    if CUDA:
        source_feature_extractor = source_feature_extractor.cuda()
        classifier = classifier.cuda()

    optimizer = optim.Adam(list(source_feature_extractor.parameters())+list(classifier.parameters()), lr=LEARNING_RATE, betas=(BETA1, BETA2))

    dataset_source = SEEDDataset(train=True, one_hot=False, test_idx=TEST_IDX)
    dataloader_source = DataLoader(dataset=dataset_source, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
    dataset_target = SEEDDataset(train=False, one_hot=False, test_idx=TEST_IDX)
    dataloader_target = DataLoader(dataset=dataset_target, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)

    # pretrain
    print('========== Pretrain Stage ==========')
    for epoch in range(NUM_EPOCH_PRE):
        data_source_iter = iter(dataloader_source)
        for step in range(len(dataloader_source)):
            data_source = data_source_iter.next()
            s_input, s_label = data_source

            optimizer.zero_grad()

            batch_size = len(data_source)

            if CUDA:
                s_input = s_input.cuda()
                s_label = s_label.cuda()
            
            pred = classifier(source_feature_extractor(s_input))
            loss = criterion(pred, s_label)

            loss.backward()
            optimizer.step()

        t_input, t_label = dataset_target[:]
        if CUDA:
            t_input = t_input.cuda()
            t_label = t_label.cuda()
        
        with torch.no_grad():
            pred_class_score = classifier(source_feature_extractor(t_input))

        pred_class = pred_class_score.max(1)[1]

        acc = round((pred_class == t_label).float().mean().cpu().numpy().tolist(), 4)

        
        print(f'Pretrain Epoch: {epoch}, Accuracy: {acc:.4f}')

    pre_acc_list.append(acc)

    # train target feature extractor and discriminator
    target_feature_extractor = FeatureExtractor(track_running_stats=BATCHNORM_TRACK, momentum=MOMENTUM)
    target_feature_extractor.load_state_dict(source_feature_extractor.state_dict())

    discriminator = Discriminator(track_running_stats=BATCHNORM_TRACK, momentum=MOMENTUM)

    if CUDA:
        target_feature_extractor.cuda()
        discriminator.cuda()

    optimizer_tfe = optim.Adam(target_feature_extractor.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

    dataset_source = SEEDDataset(train=True, one_hot=False, test_idx=TEST_IDX)
    dataloader_source = DataLoader(dataset=dataset_source, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)

    dataset_target = SEEDDataset(train=False, one_hot=False, test_idx=TEST_IDX)
    dataloader_target = DataLoader(dataset=dataset_target, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)

    # accuracies = []
    print('========== Train Stage ==========')
    for epoch in range(NUM_EPOCH):
        discriminator.train()
        target_feature_extractor.train()
        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)

        for step in range(len_dataloader):
            data_source = data_source_iter.next()
            s_input, s_label = data_source
            data_target = data_target_iter.next()
            t_input, t_label = data_target

            if CUDA:
                s_input = s_input.cuda()
                t_input = t_input.cuda()

            optimizer_disc.zero_grad()
            feat_src = source_feature_extractor(s_input)
            feat_tgt = target_feature_extractor(t_input)
            feat_concat = torch.cat((feat_src, feat_tgt), dim=0)

            pred_concat = discriminator(feat_concat.detach())

            label_src = torch.zeros(feat_src.shape[0]).long()
            label_tgt = torch.ones(feat_tgt.shape[0]).long()
            label_concat = torch.cat((label_src, label_tgt), dim=0)     
            
            if CUDA:
                label_concat = label_concat.cuda()
            loss_disc = criterion(pred_concat, label_concat)
            loss_disc.backward()

            optimizer_disc.step()

            # train target feature extractor
            optimizer_disc.zero_grad()
            optimizer_tfe.zero_grad()

            feat_tgt = target_feature_extractor(t_input)
            pred_tgt = discriminator(feat_tgt)
            label_tgt = torch.zeros(feat_tgt.shape[0]).long()

            if CUDA:
                label_tgt = label_tgt.cuda()

            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()
            optimizer_tfe.step()
        
        discriminator.eval()
        target_feature_extractor.eval()
        t_input, t_label = dataset_target[:]
        if CUDA:
            t_input = t_input.cuda()
            t_label = t_label.cuda()
        
        with torch.no_grad():
            pred_class_score = classifier(target_feature_extractor(t_input))

        pred_class = pred_class_score.max(1)[1]

        acc = round((pred_class == t_label).float().mean().cpu().numpy().tolist(), 4)
        # accuracies.append(acc)
        print(f'Train Epoch: {epoch}, Accuracy: {acc:.4f}')
    
    da_acc_list.append(acc)

print(f'Pre Accuracy List:\n{pre_acc_list}')
print(f'Pre Mean Accuracy: {sum(pre_acc_list)/len(pre_acc_list):.4f}')

print(f'DA Accuracy list:\n{da_acc_list}')
print(f'DA Mean accuracy: {sum(da_acc_list)/len(da_acc_list):.4f}')

# plt.plot(range(1, NUM_EPOCH+1), accuracies)
# plt.savefig(f'ADDA/results/{BETA1}_{BETA2}_test0_{pretrain_acc:.3f}.png')







