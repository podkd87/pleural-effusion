import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import math
import os
import sys
from util import AverageMeter

sys.argv=['']

class small_encoder(nn.Module):
    def __init__(self, input_len, class_param, dimension):
        super(small_encoder, self).__init__()
        self.classifier_layers = nn.ModuleList()
        self.input_len = input_len
        self.dimension = dimension
        self._make_layer(class_param)
        
    def _make_layer(self, class_param):
        for i, param in enumerate(class_param):
            if i==0:
                self.classifier_layers.append(nn.Linear(self.input_len, param))
                self.classifier_layers.append(nn.BatchNorm1d(param))
                self.classifier_layers.append(nn.Dropout(0.3))
                pre_param = param

            elif i==len(class_param)-1:
                self.classifier_layers.append(nn.Linear(pre_param, param))
                self.classifier_layers.append(nn.BatchNorm1d(param))
                self.classifier_layers.append(nn.Dropout(0.3))
                self.classifier_layers.append(nn.Linear(param, self.dimension))
            else:
                self.classifier_layers.append(nn.Linear(pre_param, param))
                self.classifier_layers.append(nn.BatchNorm1d(param))
                self.classifier_layers.append(nn.Dropout(0.3))
                pre_param = param


    def classifier(self, train_x):
        for i, layer in enumerate(self.classifier_layers):
            if i!=len(self.classifier_layers)-1:
                if isinstance(layer, nn.Linear):
                    train_x = F.sigmoid(layer(train_x))

                else:
                    train_x = layer(train_x)
            else:
                train_x = layer(train_x)

        return train_x
    
    def forward(self, train_x):
        # Encoder 를 통과

        encoded = self.classifier(train_x)

        return encoded
    
class added_on_model(nn.Module):
    def __init__(self, model_, parameter, dimension):
        super(added_on_model, self).__init__()
        self.classifier_layers = nn.ModuleList()
        self.loaded_model = model_.eval()
        self.parameter = parameter
        self.hidden_dimension = dimension
        self.new_model()
        
    def new_model(self):
        for i, param in enumerate(self.parameter):
            if i==0:
                self.classifier_layers.append(nn.Linear(self.hidden_dimension, param))
                pre_param = param
            elif i==len(self.parameter)-1:
                self.classifier_layers.append(nn.Linear(pre_param, param))
                pre_param = param                
            else:
                self.classifier_layers.append(nn.Linear(pre_param, param))
                pre_param = param
                    
        return
    
    def classifier(self, train_x):
        train_x = self.loaded_model(train_x)
        
        for i, layer in enumerate(self.classifier_layers):
            if i!=len(self.classifier_layers)-1:
                if isinstance(layer, nn.Linear):
                    train_x = nn.ReLU()(layer(train_x))

                else:
                    train_x = layer(train_x)
            else:
                train_x = layer(train_x)

        return train_x
    
    def forward(self, train_x):
        # Encoder 를 통과

        encoded = self.classifier(train_x)

        return encoded
    

class small_model(nn.Module):
    def __init__(self, x_len, class_param, class_len):
        super(small_model, self).__init__()
              
        self.classifier_layers = nn.ModuleList()

        for i, param in enumerate(class_param):
            if i==0:
                self.classifier_layers.append(nn.Linear(x_len, param))
                self.classifier_layers.append(nn.Dropout(0.3))
                pre_param = param
            elif i==len(class_param)-1:
                self.classifier_layers.append(nn.Linear(pre_param, param))
                self.classifier_layers.append(nn.Dropout(0.3))
                self.classifier_layers.append(nn.Linear(param, class_len))        
            else:
                self.classifier_layers.append(nn.Linear(pre_param, param))
                self.classifier_layers.append(nn.Dropout(0.3))
                pre_param = param
                

    def classifier(self, train_x):
        
        for i, layer in enumerate(self.classifier_layers):
            if i!=len(self.classifier_layers)-1:
                if isinstance(layer, nn.Linear):
                    train_x = nn.ReLU()(layer(train_x))

                else:
                    train_x = layer(train_x)
            else:
                last = layer(train_x)
                
        pre_last = train_x
        last = last
        return pre_last, last
    
    def forward(self, z):
        # Encoder 를 통과

        pre_last, last = self.classifier(z)
        
        return pre_last, last

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.2,
                        help='learning rate')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format("effusion")
    opt.tb_path = './save/SupCon/{}_tensorboard'.format("effusion")

    return opt

def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()
    losses = AverageMeter()

    end = time.time()
    for idx, (train_x, labels) in enumerate(train_loader):
        train_x = train_x.cuda()
        labels = labels.cuda()
        bsz = labels.shape[0]

        #compute loss
        output = model(train_x)
        output = output.reshape(output.shape[0],1,-1)
        output_2x = torch.cat((output, output), dim=1)
        loss = criterion(output_2x, labels)
                
        #update metric
        losses.update(loss.item(), bsz)
        
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print info
        if (epoch + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'training loss {loss.val:.3f} (average: {loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), loss=losses))
            sys.stdout.flush()
        
    return losses.avg

def validate(val_loader, model, criterion, epoch, opt):
    """validation"""
    model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for idx, (train_x, labels) in enumerate(val_loader):
            train_x = train_x.cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(train_x)
            output = output.reshape(output.shape[0],1,-1)
            output_2x = torch.cat((output, output), dim=1)
            loss = criterion(output_2x, labels)

            # update metric
            losses.update(loss.item(), bsz)

            if (epoch+1) % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'val Loss: {loss.val:.4f} (val loss average: {loss.avg:.4f})'.format(
                       idx, len(val_loader), loss=losses))

    return losses.avg