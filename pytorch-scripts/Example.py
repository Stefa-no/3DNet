#Demo for CS7-GV1

#general modules
from __future__ import print_function, division
import os
import sys
import argparse
import time
import datetime
import copy
import numpy as np

#pytorch modules
import torch
import torch.nn as nn
from torchvision import datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from torch.autograd import Variable
import pdb

#user defined modules
import Augmentation as ag
import ResNeXt as rx
import Models
import utils
from Test import Test

from functools import partial
import pickle
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
parser = argparse.ArgumentParser(description='CS7-GV1 Final Project');

today = datetime.datetime.now().strftime("%Y_%B_%d_%I:%M%p");

#add/remove arguments as required. It is useful when tuning hyperparameters from bash scripts
parser.add_argument('--aug', type=str, default = '', help='data augmentation strategy')
parser.add_argument('--datapath', type=str, default='', 
               help='root folder for data.It contains two sub-directories train and val')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')               
parser.add_argument('--pretrained', action='store_true',
                    help='use pretrained model')
parser.add_argument('--batch_size', type=int, default = 64,
                    help='batch size')
parser.add_argument('--model', type=str, default = None, help='Specify model to use for training.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=25,
                    help='upper epoch limit')
parser.add_argument('--tag', type=str, default=None,
                    help='unique_identifier used to save results')
parser.add_argument('--base_width', type=int, default=4,
                    help='base width for ResNeXt.')
parser.add_argument('--cardinality', type=int, default=32,
                    help='number of convolution groups in resnext')
parser.add_argument('--lr_decay', type=float, default=0.1,
                    help='Learning rate decay multiplier')
parser.add_argument('--num_classes', type=int, default=200,
                    help='Learning rate decay multiplier')
parser.add_argument('--optimizer', type=str, default='SGD',
                    help='gradient descent optimization algorithm ')
parser.add_argument('--no_preactivation', action='store_false',
                    help='if set, runs ResNexT without preactivation')
parser.add_argument('--no_stochastic', action='store_false',
                    help='if set, runs ResNexT without stochastic depth')
parser.add_argument('--no_personalized', action='store_false',
                    help='if set, runs ResNexT without personalized inception block')
parser.add_argument('--loss_fun', type=str, default='Cross',
                    help='if set, runs ResNexT without personalized inception block')
parser.add_argument('--activ_fun', type=str, default='relu',
                    help='if set, runs ResNexT without personalized inception block')
parser.add_argument('--cm', type=str, default='no',
                    help='if set, runs ResNexT without personalized inception block')
                    
parser.add_argument('--filename', type=str,
                    help='file name of the file with name')
args = parser.parse_args();
default_tag = args.model +'_'+ today;
#if not args.tag:
#    print('Please specify tag...')
#    exit()
print (args)

#Define augmentation strategy
augmentation_strategy = ag.Augmentation(args.aug);
data_transforms = augmentation_strategy.applyTransforms();
##

#Root directory
data_dir = args.datapath;
##

######### Data Loader ###########
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
             for x in ['train', 'val']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=args.batch_size,
                                               shuffle=True, num_workers=16) # set num_workers higher for more cores and faster data loading
             for x in ['train', 'val']}
                 
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = dsets['train'].classes
#################################

#set GPU flag
use_gpu = args.cuda;
##

#Load model . Once you define your own model in Models.py, you can call it from here. 
if args.model == 'ResNet18':
    current_model = Models.resnet18(args.pretrained)
    num_ftrs = current_model.fc.in_features
    current_model.fc = nn.Linear(num_ftrs, len(dset_classes));
    
    
elif args.model == 'DenseNet161':
    current_model = Models.densenet161(args.pretrained)
    
    
elif args.model == 'AlexNet':
    current_model = Models.alexnet(args.pretrained)
    
    
elif args.model == 'VGG16':
    current_model = Models.vgg16(args.pretrained)
    
    
elif args.model == 'ResNeXt':
    current_model = rx.resnext18(args.base_width, args.cardinality, args.num_classes, args.no_preactivation, args.no_stochastic, args.no_personalized, args.activ_fun)

elif args.model == 'ResNeXt_test20':
    path = 'ResNeXt_test20_2018_February_25_02:14PM.model' 
    if args.cm != 'yes' and args.cm != 'no':
        path = args.cm   
    current_model = torch.load(path)
    num_ftrs = current_model.fc.in_features
    current_model.fc = nn.Linear(num_ftrs, 20);
        
    
elif args.model == 'Demo':
    current_model = Models.demo_model();
    
    
else :
    print ("Model %s not found"%(args.model))
    exit();    


if use_gpu:
    current_model = current_model.cuda();
    
# uses a cross entropy loss as the loss function
# http://pytorch.org/docs/master/nn.html#
if args.loss_fun == 'Cross':
        criterion = nn.CrossEntropyLoss()
elif args.loss_fun == 'Hinge':
        criterion = nn.MultiMarginLoss(p=1, margin=0.5)

#uses stochastic gradient descent for learning
# http://pytorch.org/docs/master/optim.html
if args.optimizer == 'SGD':
        optimizer_ft = optim.SGD(current_model.parameters(), lr=args.lr, momentum=0.9)
elif args.optimizer == 'Adam':
        optimizer_ft = optim.Adam(current_model.parameters(), lr=args.lr)
elif args.optimizer == 'Adamax':
        optimizer_ft = optim.Adamax(current_model.parameters(), lr=args.lr)
        
scheduler_ft = ReduceLROnPlateau(optimizer_ft, 'min', factor = args.lr_decay, patience = 2, verbose = True)







def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=25):
    since = time.time()
    if not args.filename:
        log =open('./logs/'+ default_tag +'.log', 'w+')
    else:
        log =open('./logs/'+ args.filename +'.log', 'w+')
    print( 'model=' +args.model+' aug=' + args.aug +' batch_size=' + str(args.batch_size) +' lr=' + str(args.lr) +' epochs=' + str(args.epochs) +' pretrained=' + str(args.pretrained) + ' base_width=' + str(args.base_width) + ' cardinality=' + str(args.cardinality)+ ' lr_decay=' + str(args.lr_decay) + ' optimizer=' + str(args.optimizer), file=log);
    
    best_model = model
    best_acc = 0.0
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            top1.reset()
            top5.reset()
            # Iterate over data.
            for count, data in enumerate(dset_loaders[phase]):
                # get the inputs
                inputs, labels1 = data
                labels1 = labels1.cuda(async=True)
                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels1.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels1)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                prec1, prec5 = utils.accuracy(outputs.data, labels1, topk=(1, 5))
                top1.update(prec1[0], inputs.size(0))
                top5.update(prec5[0], inputs.size(0))
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                #if count%10 == 0:
                #    print('Batch %d || Running Loss = %0.6f || Running Accuracy = %0.6f'%(count+1,running_loss/(args.batch_size*(count+1)),running_corrects/(args.batch_size*(count+1))))
                #print('Running Loss = %0.6f'%(running_loss/(args.batch_size*(count+1))))

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]
            

            print('Epoch %d || %s Loss: %.4f || Acc: %.4f || Top1: %.4f || Top5: %.4f'%(epoch,
                phase, epoch_loss, epoch_acc, top1.avg, top5.avg),end = ' || ')
            print('Epoch %d || %s Loss: %.4f || Acc: %.4f || Top1: %.4f || Top5: %.4f'%(epoch,
                phase, epoch_loss, epoch_acc, top1.avg, top5.avg),end = ' || ', file=log)
            #pdb.set_trace();
            if phase == 'val':
                print ('\n', end='');
                print ('\n', end='', file=log);
                lr_scheduler.step(epoch_loss);
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60), file=log)
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val Acc: {:4f}'.format(best_acc), file=log)
    log.close()
    return best_model


#comment the block below if you are not training 
######################
if args.cm=='no':
    trained_model = train_model(current_model, criterion, optimizer_ft, scheduler_ft,
                          num_epochs=args.epochs);
    with open(default_tag+'.model', 'wb') as f:
        torch.save(trained_model, f);
######################    
## uncomment the lines blow while testing.
else:
    path = 'ResNeXt_test20_2018_February_25_02:14PM.model' 
    if args.cm != 'yes':
        path = args.tag
    trained_model = torch.load(path);
    testDataPath = './data20/'
    t = Test(args.aug,trained_model);
    scores = t.testfromdir(testDataPath);
    #pdb.set_trace();
    np.savetxt(args.tag+'.txt', scores, fmt='%0.5f',delimiter=',')
