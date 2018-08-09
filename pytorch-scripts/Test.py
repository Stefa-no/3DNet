import os

#pytorch modules
import torch
from torchvision import datasets
from torch.autograd import Variable
import pdb
import numpy as np
import Augmentation as ag
import utils

class Test():
    def __init__(self, aug, model, use_gpu = True):
        #Define augmentation strategy
        self.augmentation_strategy = ag.Augmentation(aug);
        self.data_transforms = self.augmentation_strategy.applyTransforms();
        self.model = model;
        self.model.train(False)
        self.use_gpu = use_gpu
        
    def testfromdir(self,datapath,batch_size = 32):
        #Root directory
        data_dir = datapath;
        
        ######### Data Loader ###########
        dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), self.data_transforms[x])
                        for x in ['val']}
        #Set shuffle to False for testing
        dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size,
                                                        shuffle=False, num_workers=16)# set num_workers higher for more cores and faster data loading
                        for x in ['val']}

        scores = torch.cuda.FloatTensor();
        all_labels = torch.LongTensor(); #to compute top 5 accuracy
        labelsAll = []
        for count, data in enumerate(dset_loaders['val']):
            # get the inputs
            inputs, labels = data
            # wrap them in Variable
            if self.use_gpu:
                inputs, labels = Variable(inputs.cuda()), \
                    Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

                # forward
            outputs = self.model(inputs)
            all_labels = torch.cat((all_labels, data[1]), 0) #add curr_labels to labels
            labelsAll.extend(labels.data)
            scores = torch.cat((scores,torch.nn.functional.softmax(outputs).data),0);

        #compute the confusion matrix
        confusionMat = utils.confusionMatrix(all_labels, scores, 20)
        print(confusionMat)

        #compute top 5 accuracy
        isTop5 = utils.testAccuracy(5, all_labels, scores)
        print(isTop5)
        
        np.savetxt('labels.txt', labelsAll, fmt='%0.5f',delimiter=',') 
        return(scores.cpu().numpy());

