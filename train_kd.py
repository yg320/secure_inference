import os
import pickle
import argparse

import torch
from tqdm import tqdm
from torch.optim import SGD
from collections import OrderedDict
from torch.utils.data import DataLoader
from mmcls.models import build_classifier


from kd_training_utils.utils import train_kd, test
from kd_training_utils.train_utils import log
from kd_training_utils.datasets import get_dataset
from kd_training_utils.architectures_unstructured import get_architecture

from research.distortion.arch_utils.factory import arch_utils_factory
from research.distortion.arch_utils.classification.resnet.resnet18_cifar import ResNet18_CIFAR_Utils
from research.mmlab_extension.classification.resnet_cifar_v2 import ResNet_CIFAR_V2  # TODO: why is this needed?
from research.mmlab_extension.classification.resnet import MyResNet  # TODO: why is this needed?

parser = argparse.ArgumentParser(description='train with kd')
parser.add_argument('--outdir', type=str, help='target dir for log and network',
                    default='/code/{WORK_DIR}/classification_regular/resnet18_cifar100/experiments/0.06/kd_finetune')
parser.add_argument('--student_path', type=str, help='path to the student model checkpoint',
                    default='/code/{WORK_DIR}/classification_regular/resnet18_cifar100/experiments/epoch_1.pth')
parser.add_argument('--teacher_path', type=str, help='path to the teacher model checkpoint',
                    default='/code/{WORK_DIR}/classification_from_snl/resnet18_cifar100.pth')
parser.add_argument('--relu_spec_file', type=str, help='path to the relu spec file',
                    default='/code/{WORK_DIR}/classification_regular/resnet18_cifar100/distortion/'
                            'block_spec/0.06.pickle')

args = parser.parse_args()

LOG_FILE_NAME = 'kd_finetune.log'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

outdir = args.outdir
student_path = args.student_path
teacher_path = args.teacher_path
relu_spec_file = args.relu_spec_file


def convert_mmcls_state_dict_to_snl_state_dict(mmcls_state_dict):
    new_mmcls_state_dict = OrderedDict({})
    for k in mmcls_state_dict:
        if k.startswith('backbone.') and 'downsample' not in k:
            new_mmcls_state_dict[k[len('backbone.'):]] = mmcls_state_dict[k]
        if k.startswith('head.fc'):
            new_mmcls_state_dict[k.replace('head.fc', 'linear')] = mmcls_state_dict[k]
        if 'downsample' in k:
            new_key = k.replace('downsample', 'shortcut')
            new_mmcls_state_dict[new_key[len('backbone.'):]] = mmcls_state_dict[k]
    return new_mmcls_state_dict


def finetune_student_model(student_model, teacher_model, train_loader, test_loader, outdir, epochs=20, lr=1e-3,
                           momentum=0.9, weight_decay=0.0005, arch='resnet18_in',
                           dataset='cifar100'):
    finetune_epoch = epochs

    optimizer = SGD(student_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, finetune_epoch)

    # don't optimize for the alphas, we assume that their location is fixed.
    log_file_path = os.path.join(outdir, LOG_FILE_NAME)

    log(log_file_path, "Finetuning the model")

    best_top1 = 0
    for epoch in range(1, 1 + finetune_epoch):
        train_loss, train_top1, train_top5 = train_kd(train_loader, student_model, teacher_model, optimizer, criterion, epoch,
                                                      device)
        test_loss, test_top1, test_top5 = test(test_loader, student_model, criterion, device, 100, display=True)
        log(log_file_path, f"[{epoch:03d}|{finetune_epoch:03d}] test accuracy: {test_top1} [%]")
        scheduler.step()

        if best_top1 < test_top1:
            best_top1 = test_top1
            is_best = True
        else:
            is_best = False

        if is_best:
            torch.save({
                'arch': arch,
                'state_dict': student_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(outdir,
                            f'kd_finetune_best_checkpoint_{arch}_{dataset}.pth.tar'))

    log(log_file_path, "Final best Prec@1 = {}%".format(best_top1))
    return student_model



class MyNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


my_args = MyNamespace(alpha=1e-05, arch='resnet18_in', batch=128, block_type='LearnableAlpha',
                      budegt_type='absolute', dataset='cifar100', epochs=2000, finetune_epochs=100,
                      gamma=0.1, gpu=0, logname='resnet18_in_unstructured_.txt', lr=0.001, lr_step_size=30,
                      momentum=0.9, num_of_neighbors=4,
                      print_freq=100, relu_budget=15000, stride=1, threshold=0.01,
                      weight_decay=0.0005, workers=4)


teacher_mmcls_state_dict = torch.load(teacher_path, map_location=torch.device(device))['state_dict']
student_mmcls_state_dict = torch.load(student_path, map_location=torch.device(device))['state_dict']


teacher_model = get_architecture('resnet18_in', 'cifar100', device, my_args)
teacher_mismatch = teacher_model.load_state_dict(convert_mmcls_state_dict_to_snl_state_dict(teacher_mmcls_state_dict),
                                                 strict=False)
teacher_mismatch = teacher_model.load_state_dict(teacher_mmcls_state_dict,
                                                 strict=False)
print(teacher_mismatch)
student_model = get_architecture('resnet18_in', 'cifar100', device, my_args)
student_model.load_state_dict(convert_mmcls_state_dict_to_snl_state_dict(student_mmcls_state_dict), strict=False)


train_dataset = get_dataset('cifar100', 'train')
test_dataset = get_dataset('cifar100', 'test')
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=256,
                          num_workers=1, pin_memory=False)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=256,
                         num_workers=1, pin_memory=False)
criterion = torch.nn.CrossEntropyLoss().to(device)
test_loss, test_top1, test_top5 = test(test_loader, teacher_model, criterion, device, 100, display=True)
print(f"[TEACHER] (test_loss, test_top1, test_top5) = ({test_loss}, {test_top1}, {test_top5})")
test_loss, test_top1, test_top5 = test(test_loader, student_model, criterion, device, 100, display=True)
print(f"[STUDENT BEFORE REPLACING TO BRELUS] (test_loss, test_top1, test_top5) = ({test_loss}, {test_top1}, {test_top5})")

# load the model as if it was mmcls version, and apply the utility which switches relus with brelus.
model_config = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR_V2',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))

model = build_classifier(model_config)
layer_name_to_block_sizes = pickle.load(open(relu_spec_file, 'rb'))
arch_utils = ResNet18_CIFAR_Utils()
arch_utils.set_bReLU_layers(model, layer_name_to_block_sizes)
model.load_state_dict(torch.load(student_path, map_location=device)['state_dict'])
model = model.to(device)
model.eval()
# sanity check - the model's accuracy:
correct = 0
total = 0
for batch in tqdm(test_loader):
    images, labels = batch[0].to(device), batch[1].to(device)
    with torch.no_grad():
        pred_scores = model.forward_test(images)
    preds = torch.cat([torch.from_numpy(p).unsqueeze(0) for p in pred_scores]).argmax(1).to(device)
    correct += (preds == labels).sum().item()
    total += preds.shape[0]
print(correct / total * 100.0)

# take the pytorch version and switch the alpha blocks with BReLU blocks.
for layer in [f'layer{i}' for i in range(1, 1+4)]:
    for block in [0, 1]:
        for idx in [1, 2]:
            exec(f'student_model.{layer}[{block}].alpha{idx} = model.backbone.{layer}[{block}].relu_{idx}')

test_loss, test_top1, test_top5 = test(test_loader, student_model, criterion, device, 100, display=True)
print(f"[STUDENT] (test_loss, test_top1, test_top5) = ({test_loss}, {test_top1}, {test_top5})")


os.makedirs(outdir, exist_ok=True)
finetune_student_model(student_model, teacher_model, train_loader, test_loader, outdir, epochs=160)
