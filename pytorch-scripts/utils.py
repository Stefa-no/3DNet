import torch
from torchvision import datasets
from torch.autograd import Variable
import pdb
import numpy as np



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


def testAccuracy(factorK, labels, scores):
    accuracy = 0;
    nb_images = labels.size()[0];
    for idx in range(nb_images):
        sortedWeights = sorted(range(len(scores[idx])), key=lambda i: scores[idx][i])[-factorK:]
        for k in range(factorK):
            if (labels[idx] == sortedWeights[k]):
                accuracy += 1;
    percentage = float(accuracy) / float(nb_images); 
    return percentage;




def confusionMatrix(labels, scores, nb_classes):
    nb_images = labels.size()[0];

    #Create confusion matrix
    confusion = torch.cuda.FloatTensor(nb_classes, nb_classes).zero_()

    probs, preds = torch.max(scores,1);
    for idx in range(nb_images):
        pred_idx = preds[idx]
        label_idx = labels[idx]
        confusion[label_idx][pred_idx] = confusion[label_idx][pred_idx] + 1;
    return confusion;

    
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

