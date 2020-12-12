import warnings
warnings.filterwarnings("ignore")
import argparse
import json
# import matplotlib
# import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import cuda
import sys, os
import random
import numpy as np
from sklearn import metrics
import models as Model
# from SiameseLoss import ContrastiveLoss
import evaluate
import data
import gc
import csv
from pdb import set_trace as stop

import kipoi
from copy import deepcopy
from scipy.stats import pearsonr
from tqdm import tqdm

# python train.py --cell_type=Cell1 --model_name=attchrome --epochs=120 --lr=0.0001 --data_root=data/ --save_root=Results/

parser = argparse.ArgumentParser(description='DeepDiff')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--model_type', type=str, default='attchrome', help='DeepDiff variation')
parser.add_argument('--clip', type=float, default=1,help='gradient clipping')
parser.add_argument('--epochs', type=int, default=30, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=16, help='')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout applied to layers (0 = no dropout) if n_layers LSTM > 1')
parser.add_argument('--cell_type', type=str, default='E003', help='cell type 1')
parser.add_argument('--save_root', type=str, default='./Results/', help='where to save')
parser.add_argument('--model_root', type=str, default=None, help='where to load model')
parser.add_argument('--model_all_cells', action='store_true', help='model_all_cells')
parser.add_argument('--data_root', type=str, default='./data/', help='data location')
parser.add_argument('--gpuid', type=int, default=0, help='CUDA gpu')
parser.add_argument('--n_hms', type=int, default=5, help='number of histone modifications')
parser.add_argument('--n_bins', type=int, default=100, help='number of bins')
parser.add_argument('--bin_rnn_size', type=int, default=32, help='bin rnn size')
parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
parser.add_argument('--unidirectional', action='store_true', help='bidirectional/undirectional LSTM')
parser.add_argument('--save_attention_maps',action='store_true', help='set to save validation beta attention maps')
parser.add_argument('--attentionfilename', type=str, default='beta_attention.txt', help='where to save attnetion maps')
parser.add_argument('--test_on_saved_model',action='store_true', help='only test on saved model')
parser.add_argument('--kipoi_model',action='store_true', help='only test on saved model')
parser.add_argument('--pgd', type=int, default=None, help='pgd')
parser.add_argument('--pgd_steps',type=int, default=2, help='pgd')
parser.add_argument('--pgd_mask',type=str, default=None, help='pgd mask, fg or bg')
parser.add_argument('--pgd_mask_threshold',type=int, default=1, help='pgd mask threshold')
parser.add_argument('--input_threshold',type=int, default=None, help='input threshold')
parser.add_argument('--random_attack',type=int, default=None, help='random_attack')
parser.add_argument('--test_on_train',action='store_true', help='test_on_train')
parser.add_argument('--all_cell_types',action='store_true', help='all_cell_types')
parser.add_argument('--patient',type=int, default=10, help='patient')


args = parser.parse_args()



def main(args):
    torch.manual_seed(1)


    model_name = ''
    if args.model_all_cells:
      model_name += ('*')+('_')
    else:
      model_name += (args.cell_type)+('_')

    model_name+=args.model_type



    args.bidirectional=not args.unidirectional

    print('the model name: ',model_name)
    args.data_root+=''
    args.save_root+=''
    args.dataset=args.cell_type
    args.data_root = os.path.join(args.data_root)
    print('loading data from:  ',args.data_root)
    args.save_root = os.path.join(args.save_root,args.dataset)
    if not os.path.exists(args.save_root):
      os.makedirs(args.save_root)
    print('saving results in from: ',args.save_root)
    if args.model_root is None:
      model_dir = os.path.join(args.save_root,model_name)
    else:
      if args.model_all_cells:
        args.model_root = os.path.join(args.model_root,'*')
      else:
        args.model_root = os.path.join(args.model_root,args.dataset)
      model_dir = os.path.join(args.model_root,model_name)
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)



    attentionmapfile=model_dir+'/'+args.attentionfilename
    print('==>processing data')
    Train,Valid,Test = data.load_data(args)






    print('==>building model')
    if args.model_type == 'deepchrome':
      model = Model.DeepChromeModel(args)
    elif args.model_type == 'linearchrome':
      model = Model.LinearChromeModel(args)
    elif args.model_type == 'meanlinearchrome':
      model = Model.SimpleLinearChromeModel(args)
    else:
      model = Model.att_chrome(args)



    if torch.cuda.device_count() > 0:
      torch.cuda.manual_seed_all(1)
      dtype = torch.cuda.FloatTensor
      cuda.set_device(args.gpuid)
      model.type(dtype)
      print('Using GPU '+str(args.gpuid))
    else:
      dtype = torch.FloatTensor

    #print(model)
    if(args.test_on_saved_model==False):
      print("==>initializing a new model")
      for p in model.parameters():
        p.data.uniform_(-0.1,0.1)


    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    #optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum)
    def train(TrainData):
      model.train()
      # initialize attention
      diff_targets = torch.zeros(TrainData.dataset.__len__(),1)
      predictions = torch.zeros(diff_targets.size(0),1)

      all_attention_bin=torch.zeros(TrainData.dataset.__len__(),(args.n_hms*args.n_bins))
      all_attention_hm=torch.zeros(TrainData.dataset.__len__(),args.n_hms)

      num_batches = int(math.ceil(TrainData.dataset.__len__()/float(args.batch_size)))
      all_gene_ids=[None]*TrainData.dataset.__len__()
      per_epoch_loss = 0
      print('Training')
      t = tqdm(total=len(TrainData))
      for idx, Sample in (enumerate(TrainData)):

        start,end = (idx*args.batch_size), min((idx*args.batch_size)+args.batch_size, TrainData.dataset.__len__())
      

        inputs_1 = Sample['input']
        batch_diff_targets = Sample['label'].unsqueeze(1).float()

        
        optimizer.zero_grad()

        if args.pgd:
          if args.pgd_steps == 2:
            alpha = args.pgd
          else:
            alpha = 1
          batch_predictions, corr = pgd_attack(model, inputs_1.type(dtype), batch_diff_targets, eps=args.pgd, alpha=alpha, iters=args.pgd_steps)
        else:
          # batch_predictions,batch_beta,batch_alpha = model(inputs_1.type(dtype))
          batch_predictions = model(inputs_1.type(dtype))

        #batch_predictions= model(inputs_1.type(dtype))

        loss = F.binary_cross_entropy_with_logits(batch_predictions.cpu(), batch_diff_targets,reduction='mean')

        per_epoch_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        # all_attention_bin[start:end]=batch_alpha.data
        # all_attention_hm[start:end]=batch_beta.data

        diff_targets[start:end,0] = batch_diff_targets[:,0]
        all_gene_ids[start:end]=Sample['geneID']
        batch_predictions = torch.sigmoid(batch_predictions)
        predictions[start:end] = batch_predictions.data.cpu()

        t.update(1)
        
      per_epoch_loss=per_epoch_loss/num_batches
      return predictions,diff_targets,all_attention_bin,all_attention_hm,per_epoch_loss,all_gene_ids


    def pgd_attack(model, inputs_1, batch_diff_targets, eps=1, alpha=1.0, iters=2):

        ori_inputs_1 = inputs_1.data

        if args.pgd_mask:
            if args.pgd_mask == 'fg':
                pgd_mask = (ori_inputs_1 > args.pgd_mask_threshold).to(torch.float32)  
            elif args.pgd_mask == 'bg':  
                pgd_mask = (ori_inputs_1 <= args.pgd_mask_threshold).to(torch.float32)

        #print(ori_images.min(), ori_images.max())
        for i in range(iters) :    
            inputs_1.requires_grad = True

            # batch_predictions,batch_beta,batch_alpha = model(inputs_1.type(dtype))
            if args.input_threshold:
              inputs = inputs_1 * (inputs_1 > args.input_threshold).to(torch.float32)
            else:
              inputs = inputs_1

            batch_predictions = model(inputs)

            if i < iters - 1 :

              model.zero_grad()

              loss = F.binary_cross_entropy_with_logits(batch_predictions, batch_diff_targets.cuda(),reduction='mean')

              loss.backward()

              adv_inputs_1 = inputs_1 + alpha*inputs_1.grad.sign()
              eta = torch.clamp(adv_inputs_1 - ori_inputs_1, min=-eps, max=eps)
              if args.pgd_mask:
                  eta = eta * pgd_mask

              inputs_1 = torch.clamp(ori_inputs_1 + eta, min=ori_inputs_1.min(), max=ori_inputs_1.max()).detach_()

        corr = pearsonr(inputs_1.detach().cpu().numpy().flatten(), ori_inputs_1.cpu().numpy().flatten())[0]

        return batch_predictions, corr

    def random_attack(model, inputs_1, batch_diff_targets, eps=1.0, iters=1):

        ori_inputs_1 = inputs_1.data

        if args.pgd_mask:
            if args.pgd_mask == 'fg':
                pgd_mask = (ori_inputs_1 > args.pgd_mask_threshold).to(torch.float32)  
            elif args.pgd_mask == 'bg':  
                pgd_mask = (ori_inputs_1 <= args.pgd_mask_threshold).to(torch.float32)

        #print(ori_images.min(), ori_images.max())
        for i in range(iters):

            eta = torch.randint(low=-eps, high=eps+1, size=inputs_1.shape).type(dtype)

            inputs_1 = torch.clamp(ori_inputs_1 + eta, min=ori_inputs_1.min(), max=ori_inputs_1.max()).detach_()

            if args.pgd_mask:
                  eta = eta * pgd_mask

            # batch_predictions,batch_beta,batch_alpha = model(inputs_1.type(dtype))
            if args.input_threshold:
              inputs = inputs_1 * (inputs_1 > args.input_threshold).to(torch.float32)
            else:
              inputs = inputs_1

            batch_predictions = model(inputs)

            #loss = F.binary_cross_entropy_with_logits(batch_predictions, batch_diff_targets.cuda(),reduction='mean')


        corr = pearsonr(inputs_1.detach().cpu().numpy().flatten(), ori_inputs_1.cpu().numpy().flatten())[0]

        return batch_predictions, corr

    def test(ValidData):
      if args.pgd:
        model.train()
      else:
        model.eval()

      diff_targets = torch.zeros(ValidData.dataset.__len__(),1)
      predictions = torch.zeros(diff_targets.size(0),1)

      all_attention_bin=torch.zeros(ValidData.dataset.__len__(),(args.n_hms*args.n_bins))
      all_attention_hm=torch.zeros(ValidData.dataset.__len__(),args.n_hms)

      num_batches = int(math.ceil(ValidData.dataset.__len__()/float(args.batch_size)))
      all_gene_ids=[None]*ValidData.dataset.__len__()
      per_epoch_loss = 0
      per_epoch_corr = 0

      t = tqdm(total=len(ValidData))
      for idx, Sample in (enumerate(ValidData)):

        #print(Sample['input'].min(), Sample['input'].max(), Sample['input'].mean(), Sample['input'].std(), Sample['input'].median())

        start,end = (idx*args.batch_size), min((idx*args.batch_size)+args.batch_size, ValidData.dataset.__len__())
        optimizer.zero_grad()

        inputs_1 = Sample['input']
        batch_diff_targets= Sample['label'].unsqueeze(1).float()

        

        if args.pgd:
          if args.pgd_steps == 2:
            alpha = args.pgd
          else:
            alpha = 1
          batch_predictions, corr = pgd_attack(model, inputs_1.type(dtype), batch_diff_targets, eps=args.pgd, alpha=alpha, iters=args.pgd_steps)
          per_epoch_corr += corr
        elif args.random_attack:
          batch_predictions, corr = random_attack(model, inputs_1.type(dtype), batch_diff_targets, eps=args.random_attack, iters=1)
          per_epoch_corr += corr
        else:
          # batch_predictions,batch_beta,batch_alpha = model(inputs_1.type(dtype))
          if args.input_threshold:
            inputs_1 = inputs_1 * (inputs_1 > args.input_threshold).to(torch.float32)
          batch_predictions = model(inputs_1.type(dtype))

        loss = F.binary_cross_entropy_with_logits(batch_predictions.cpu(), batch_diff_targets,reduction='mean')
        # all_attention_bin[start:end]=batch_alpha.data
        # all_attention_hm[start:end]=batch_beta.data


        diff_targets[start:end,0] = batch_diff_targets[:,0]
        all_gene_ids[start:end]=Sample['geneID']
        batch_predictions = torch.sigmoid(batch_predictions)
        predictions[start:end] = batch_predictions.data.cpu()

        per_epoch_loss += loss.item()

        t.update(1)
        
      per_epoch_loss=per_epoch_loss/num_batches
      per_epoch_corr=per_epoch_corr/num_batches
      return predictions,diff_targets,all_attention_bin,all_attention_hm,per_epoch_loss,all_gene_ids, per_epoch_corr




    best_valid_loss = 10000000000
    best_valid_avgAUPR=-1
    best_valid_avgAUC=-1
    best_test_avgAUC=-1
    if(args.test_on_saved_model==False):
      patients = args.patient
      for epoch in range(0, args.epochs):
        print('---------------------------------------- Training '+str(epoch+1)+' -----------------------------------')
        predictions,diff_targets,alpha_train,beta_train,train_loss,_ = train(Train)
        train_avgAUPR, train_avgAUC = evaluate.compute_metrics(predictions,diff_targets)

        predictions,diff_targets,alpha_valid,beta_valid,valid_loss,gene_ids_valid,test_corr = test(Valid)
        valid_avgAUPR, valid_avgAUC = evaluate.compute_metrics(predictions,diff_targets)

        predictions,diff_targets,alpha_test,beta_test,test_loss,gene_ids_test,test_corr = test(Test)
        test_avgAUPR, test_avgAUC = evaluate.compute_metrics(predictions,diff_targets)

        if(valid_avgAUC >= best_valid_avgAUC):
            # save best epoch -- models converge early
          best_valid_avgAUC = valid_avgAUC
          best_test_avgAUC = test_avgAUC
          torch.save(model.cpu().state_dict(),model_dir+"/"+model_name+'_avgAUC_model.pt')
          model.type(dtype)
          patients = args.patient
        else:
          patients -= 1
        if patients <= 0:
          break

        print("Epoch:",epoch)
        print("train avgAUC:",train_avgAUC)
        print("valid avgAUC:",valid_avgAUC)
        print("test avgAUC:",test_avgAUC)
        print("best valid avgAUC:", best_valid_avgAUC)
        print("best test avgAUC:", best_test_avgAUC)

    
      print("\nFinished training")
      print("Best validation avgAUC:",best_valid_avgAUC)
      print("Best test avgAUC:",best_test_avgAUC)



      if(args.save_attention_maps):
        attentionfile=open(attentionmapfile,'w')
        attentionfilewriter=csv.writer(attentionfile)
        beta_test=beta_test.numpy()
        for i in range(len(gene_ids_test)):
          gene_attention=[]
          gene_attention.append(gene_ids_test[i])
          for e in beta_test[i,:]:
            gene_attention.append(str(e))
          attentionfilewriter.writerow(gene_attention)
        attentionfile.close()
        
      return best_test_avgAUC, test_corr


    else:
      if args.kipoi_model:
          model.load_state_dict(kipoi.get_model("AttentiveChrome/{}".format(args.cell_type)).model.state_dict())
      else:
          model.load_state_dict(torch.load(model_dir+"/"+model_name+'_avgAUC_model.pt'))

      predictions,diff_targets,alpha_test,beta_test,test_loss,gene_ids_test,test_corr = test(Test)
      test_avgAUPR, test_avgAUC = evaluate.compute_metrics(predictions,diff_targets)
      print("test avgAUC:",test_avgAUC)
      print("test corr:",test_corr)

      if(args.save_attention_maps):
        attentionfile=open(attentionmapfile,'w')
        attentionfilewriter=csv.writer(attentionfile)
        beta_test=beta_test.numpy()
        for i in range(len(gene_ids_test)):
          gene_attention=[]
          gene_attention.append(gene_ids_test[i])
          for e in beta_test[i,:]:
            gene_attention.append(str(e))
          attentionfilewriter.writerow(gene_attention)
        attentionfile.close()
      return test_avgAUC, test_corr

auc_list = []
corr_list = []
og_args = deepcopy(args)
skip = False
for cell_type_name in np.sort(list(os.listdir(args.data_root))):
    if cell_type_name == 'E003':
      skip = False
    if skip:
      continue
    args.cell_type = cell_type_name
    if args.all_cell_types:
      args.cell_type = '*'
    #try:
        
    auc, corr = main(args)
     
    auc_list.append(auc)
    np.savetxt(args.save_root+'total_auc.txt', auc_list)
    np.save(args.save_root+'total_auc.npy', auc_list)

    if args.pgd or args.random_attack:
      corr_list.append(corr)
      np.savetxt(args.save_root+'total_corr.txt', corr_list)
      np.save(args.save_root+'total_corr.npy', corr_list)
      
    # except Exception as e:
    #    print(e)
    args = deepcopy(og_args)

    if args.all_cell_types:
      break

print(auc_list)
print('total auc: {}'.format(np.mean(auc_list)))


