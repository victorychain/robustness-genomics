import torch
import collections
import pdb
import torch.utils.data
import csv
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import math
from pdb import set_trace as stop
import numpy as np
import glob
import logging
import multiprocessing




def loadData(filename,windows):
	with open(filename) as fi:
		csv_reader=csv.reader(fi)
		data=list(csv_reader)

		ncols=(len(data[0]))
	fi.close()
	nrows=len(data)
	ngenes=nrows/windows
	nfeatures=ncols-1
	print("Number of genes: %d" % ngenes)
	print("Number of entries: %d" % nrows)
	print("Number of HMs: %d" % nfeatures)

	count=0
	attr=collections.OrderedDict()

	for i in range(0,nrows,windows):
		hm1=torch.zeros(windows,1)
		hm2=torch.zeros(windows,1)
		hm3=torch.zeros(windows,1)
		hm4=torch.zeros(windows,1)
		hm5=torch.zeros(windows,1)
		for w in range(0,windows):
			hm1[w][0]=int(data[i+w][2])
			hm2[w][0]=int(data[i+w][3])
			hm3[w][0]=int(data[i+w][4])
			hm4[w][0]=int(data[i+w][5])
			hm5[w][0]=int(data[i+w][6])
		geneID=str(data[i][0].split("_")[0])

		thresholded_expr = int(data[i+w][7])

		attr[count]={
			'geneID':geneID,
			'expr':thresholded_expr,
			'hm1':hm1,
			'hm2':hm2,
			'hm3':hm3,
			'hm4':hm4,
			'hm5':hm5
		}
		count+=1

	return attr


class HMData(Dataset):
	# Dataset class for loading data
	def __init__(self,data_cell1,transform=None):
		self.c1=data_cell1
	def __len__(self):
		return len(self.c1)
	def __getitem__(self,i):
		final_data_c1=torch.cat((self.c1[i]['hm1'],self.c1[i]['hm2'],self.c1[i]['hm3'],self.c1[i]['hm4'],self.c1[i]['hm5']),1)
		label=self.c1[i]['expr']
		geneID=self.c1[i]['geneID']
		sample={'geneID':geneID,
			   'input':final_data_c1,
			   'label':label,
			   }
		return sample

# def load_data(args):
#     '''
#     Loads data into a 3D tensor for each of the 3 splits.

#     '''
#     if not args.test_on_train:
#       if not args.test_on_saved_model:
#           print("==>loading train data")
#           cell_train_dict1=loadData(args.data_root+"/"+args.cell_type+"/classification/train.csv",args.n_bins)
#           train_inputs = HMData(cell_train_dict1)

#           print("==>loading valid data")
#           cell_valid_dict1=loadData(args.data_root+"/"+args.cell_type+"/classification/valid.csv",args.n_bins)
#           valid_inputs = HMData(cell_valid_dict1)

#       print("==>loading test data")
#       cell_test_dict1=loadData(args.data_root+"/"+args.cell_type+"/classification/test.csv",args.n_bins)
#       test_inputs = HMData(cell_test_dict1)

#       if args.test_on_saved_model:
#           Train = None
#           Valid = None
#       else:
#           Train = torch.utils.data.DataLoader(train_inputs, batch_size=args.batch_size, shuffle=True)
#           Valid = torch.utils.data.DataLoader(valid_inputs, batch_size=args.batch_size, shuffle=False)
#       Test = torch.utils.data.DataLoader(test_inputs, batch_size=args.batch_size, shuffle=False)
#     else:
#       print("==>loading train data")
#       cell_train_dict1=loadData(args.data_root+"/"+args.cell_type+"/classification/train.csv",args.n_bins)
#       train_inputs = HMData(cell_train_dict1)
#       Train = torch.utils.data.DataLoader(train_inputs, batch_size=args.batch_size, shuffle=True)
#       Test = None
#       Valid = None

#     return Train,Valid,Test

def load_data(args):
    '''
    Loads data into a 3D tensor for each of the 3 splits.

    '''
    if not args.test_on_train:
      if not args.test_on_saved_model:
          print("==>loading train data")
          train_inputs = DeepChromeDataset([args.data_root+"/"+args.cell_type+"/classification/train.csv"])

          print("==>loading valid data")
          valid_inputs = DeepChromeDataset([args.data_root+"/"+args.cell_type+"/classification/valid.csv"])

      print("==>loading test data")
      test_inputs = DeepChromeDataset([args.data_root+"/"+args.cell_type+"/classification/test.csv"])

      if args.test_on_saved_model:
          Train = None
          Valid = None
      else:
          Train = torch.utils.data.DataLoader(train_inputs, batch_size=args.batch_size, shuffle=True)
          Valid = torch.utils.data.DataLoader(valid_inputs, batch_size=args.batch_size, shuffle=False)
      Test = torch.utils.data.DataLoader(test_inputs, batch_size=args.batch_size, shuffle=False)
    else:
      print("==>loading train data")
      train_inputs = DeepChromeDataset([args.data_root+"/"+args.cell_type+"/classification/train.csv"])
      Train = torch.utils.data.DataLoader(train_inputs, batch_size=args.batch_size, shuffle=True)
      Test = None
      Valid = None

    return Train,Valid,Test


class DeepChromeDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, num_procs=24):
        self.dataroot = dataroot # Should be a list of glob strings.
        self.num_procs = num_procs
        self.samples = [] # List of tuples (torch.Tensor[100x5], torch.Tensor[1])

        self._load_from_dataroot()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return {
            'input' : torch.from_numpy(sample['X']).float(),
            'label' : torch.from_numpy(sample['y']).squeeze(-1),
            'geneID' : sample['gene_id'],
        }

    def _load_from_dataroot(self):
        files = []
        for glob_str in self.dataroot:
            files.extend(glob.glob(glob_str))
        assert len(files) != 0
        
        # Code is inefficient and I'm lazy.
        with multiprocessing.Pool(self.num_procs) as pool:
            proc_results = pool.map(self._load_file_faster, files)

        for result in proc_results:
            # self.samples is going to contain numpy arrays.
            # Fix this in __getitem__()
            self.samples.extend(result)

    def _load_file(self, fname):
        """
        Pls excuse the shit code.
        """

        # file_contents = {
        #     "gene_id" : {
        #         "bin_id" : [HM1_count, ..., HM5_count]
        #         ...
        #         "expression" : 0/1
        #     }
        # }
        file_contents = dict()

        with open(fname, 'r') as f:
            reader = csv.DictReader(
                f,
                fieldnames=[
                    "gene_id", 
                    "bin_id", 
                    "H3K27me3_count", 
                    "H3K36me3_count", 
                    "H3K4me1_count", 
                    "H3K4me3_count", 
                    "H3K9me3_count", 
                    "gene_expression"
                ]
            )
            
            for row in reader:
                if row['gene_id'] not in file_contents:
                    file_contents[row['gene_id']] = dict()
                
                file_contents[row['gene_id']][row['bin_id']] = [
                    row['H3K27me3_count'],
                    row['H3K36me3_count'],
                    row['H3K4me1_count'],
                    row['H3K4me3_count'],
                    row['H3K9me3_count'],
                ]

                # Sanity check.
                assert file_contents[row['gene_id']].get('expression', None) == None \
                    or file_contents[row['gene_id']].get('expression', None) == row['gene_expression']
                
                file_contents[row['gene_id']]['expression'] = row['gene_expression']
        
        # Now that we have file contents loaded, create X and Y.
        samples = []
        for gene_id, bins in file_contents.items():
            # Sanity check we have 100 bins for each gene_id
            assert len(bins) == 100 + 1 # Add 1 for the expression.

            Y = torch.zeros((1))
            X = torch.zeros((100, 5))
            for key in bins:
                if key == 'expression':
                    Y[0] = int(bins[key])
                else:
                    bin_id = int(key) - 1 # Indices go from 0-99, but the CSV has it in 1-100
                    X[bin_id][0] = float(bins[key][0]) # H3K27me3_count
                    X[bin_id][1] = float(bins[key][1]) # H3K36me3_count
                    X[bin_id][2] = float(bins[key][2]) # H3K4me1_count
                    X[bin_id][3] = float(bins[key][3]) # H3K4me3_count
                    X[bin_id][4] = float(bins[key][4]) # H3K9me3_count
            
            # Convert to numpy here, and caller is responsible for converting back to PyTorch
            # This is because of multiprocessing being weird with PyTorch.
            samples.append({
                "X" : X.numpy(),
                "Y" : Y.numpy(),
                "gene_id" : gene_id
            })

        return samples

    def _load_file_faster(self, fname):
        """
        Pls excuse the shit code.
        """

        samples = dict()

        with open(fname, 'r') as f:
            reader = csv.DictReader(
                f,
                fieldnames=[
                    "gene_id", 
                    "bin_id", 
                    "H3K27me3_count", 
                    "H3K36me3_count", 
                    "H3K4me1_count", 
                    "H3K4me3_count", 
                    "H3K9me3_count", 
                    "gene_expression"
                ]
            )
            
            for row in reader:
                gene_id = row['gene_id']
                bin_id = int(row['bin_id']) - 1
                hm1 = int(row['H3K27me3_count'])
                hm2 = int(row['H3K36me3_count'])
                hm3 = int(row['H3K4me1_count'])
                hm4 = int(row['H3K4me3_count'])
                hm5 = int(row['H3K9me3_count'])
                gene_expression = int(row['gene_expression'])

                if gene_id not in samples:
                    samples[gene_id] = {
                        "X" : np.zeros((100, 5)),
                        "y" : None, 
                        "gene_id" : gene_id
                    }
                
                samples[gene_id]["X"][bin_id][0] = hm1
                samples[gene_id]["X"][bin_id][1] = hm2
                samples[gene_id]["X"][bin_id][2] = hm3
                samples[gene_id]["X"][bin_id][3] = hm4
                samples[gene_id]["X"][bin_id][4] = hm5

                # Sanity check.
                assert samples[gene_id]['y'] == None \
                    or samples[gene_id]['y'][0] == gene_expression
                
                samples[gene_id]['y'] = np.array([gene_expression])
        
        return list(samples.values())