import numpy as np
import torch
import os
from torch.autograd import Variable
#from .base_model import BaseModel
import sys
from . import pytorch_DIW_scratch
import os
#from .utils import load_state_dict_from_url
from torch.utils.model_zoo import load_url as load_state_dict_from_url


#import torch

class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '_%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device_id=gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self): #network, network_label, epoch_label):
        #save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        #save_path = os.path.join(self.save_dir, save_filename)
        #print(save_path)

        # hard code for now
        # load best generalization model
        url = 'http://www.cs.cornell.edu/projects/megadepth/dataset/models/best_generalization_net_G.pth'
        state_dict = load_state_dict_from_url(url)
        #model.load_state_dict(state_dict)

        """
        is this right ???
        """
        model = torch.load(state_dict) 
        return model
        # network.load_state_dict(torch.load(save_path))
    """
    do i need self here !!??
    """
    def update_learning_rate(self): 
        pass


class HGModel(BaseModel):
    def name(self):
        return 'HGModel'

    def __init__(self, opt):
        BaseModel.initialize(self, opt)

        print("===========================================LOADING Hourglass NETWORK====================================================")
        model = pytorch_DIW_scratch.pytorch_DIW_scratch
        model= torch.nn.parallel.DataParallel(model, device_ids = [0,1])
        #model_parameters = self.load_network(model, 'G', 'best_vanila')
        #model_parameters = self.load_network(model, 'G', 'best_generalization') # load with best generalization
        model_parameters = self.load_network()
        model.load_state_dict(model_parameters)
        self.netG = model.cuda()


    def batch_classify(self, z_A_arr, z_B_arr, ground_truth ):
        threashold = 1.1
        depth_ratio = torch.div(z_A_arr, z_B_arr)

        depth_ratio = depth_ratio.cpu()

        estimated_labels = torch.zeros(depth_ratio.size(0))

        estimated_labels[depth_ratio > (threashold)] = 1
        estimated_labels[depth_ratio < (1/threashold)] = -1

        diff = estimated_labels - ground_truth
        diff[diff != 0] = 1

        # error 
        inequal_error_count = diff[ground_truth != 0]
        inequal_error_count =  torch.sum(inequal_error_count)

        error_count = torch.sum(diff) #diff[diff !=0]
        # error_count = error_count.size(0)

        equal_error_count = error_count - inequal_error_count
        

        # total 
        total_count = depth_ratio.size(0)
        ground_truth[ground_truth !=0 ] = 1

        inequal_count_total = torch.sum(ground_truth)
        equal_total_count = total_count - inequal_count_total


        error_list = [equal_error_count, inequal_error_count, error_count]
        count_list = [equal_total_count, inequal_count_total, total_count]

        return error_list, count_list 


    def computeSDR(self, prediction_d, targets):
        #  for each image 
        total_error = [0,0,0]
        total_samples = [0,0,0]

        for i in range(0, prediction_d.size(0)):

            if targets['has_SfM_feature'][i] == False:
                continue
            
            x_A_arr = targets["sdr_xA"][i].squeeze(0)
            x_B_arr = targets["sdr_xB"][i].squeeze(0)
            y_A_arr = targets["sdr_yA"][i].squeeze(0)
            y_B_arr = targets["sdr_yB"][i].squeeze(0)

            predict_depth = torch.exp(prediction_d[i,:,:])
            predict_depth = predict_depth.squeeze(0)
            ground_truth = targets["sdr_gt"][i]

            # print(x_A_arr.size())
            # print(y_A_arr.size())

            z_A_arr = torch.gather( torch.index_select(predict_depth, 1 ,x_A_arr.cuda()) , 0, y_A_arr.view(1, -1).cuda())# predict_depth:index(2, x_A_arr):gather(1, y_A_arr:view(1, -1))
            z_B_arr = torch.gather( torch.index_select(predict_depth, 1 ,x_B_arr.cuda()) , 0, y_B_arr.view(1, -1).cuda())

            z_A_arr = z_A_arr.squeeze(0)
            z_B_arr = z_B_arr.squeeze(0)

            error_list, count_list  = self.batch_classify(z_A_arr, z_B_arr,ground_truth)

            for j in range(0,3):
                total_error[j] += error_list[j]
                total_samples[j] += count_list[j]

        return  total_error, total_samples


    def evaluate_SDR(self, input_, targets):
        input_images = Variable(input_.cuda() )
        prediction_d = self.netG.forward(input_images) 

        total_error, total_samples = self.computeSDR(prediction_d.data, targets)

        return total_error, total_samples

    def rmse_Loss(self, log_prediction_d, mask, log_gt):
        N = torch.sum(mask)
        log_d_diff = log_prediction_d - log_gt
        log_d_diff = torch.mul(log_d_diff, mask)
        s1 = torch.sum( torch.pow(log_d_diff,2) )/N 

        s2 = torch.pow(torch.sum(log_d_diff),2)/(N*N)  
        data_loss = s1 - s2

        data_loss = torch.sqrt(data_loss)

        return data_loss

    def evaluate_RMSE(self, input_images, prediction_d, targets):
        count = 0            
        total_loss = Variable(torch.cuda.FloatTensor(1))
        total_loss[0] = 0
        mask_0 = Variable(targets['mask_0'].cuda(), requires_grad = False)
        d_gt_0 = torch.log(Variable(targets['gt_0'].cuda(), requires_grad = False))

        for i in range(0, mask_0.size(0)):
 
            total_loss +=  self.rmse_Loss(prediction_d[i,:,:], mask_0[i,:,:], d_gt_0[i,:,:])
            count += 1

        return total_loss.data[0], count


    def evaluate_sc_inv(self, input_, targets):
        input_images = Variable(input_.cuda() )
        prediction_d = self.netG.forward(input_images) 
        rmse_loss , count= self.evaluate_RMSE(input_images, prediction_d, targets)

        return rmse_loss, count


    def switch_to_train(self):
        self.netG.train()

    def switch_to_eval(self):
        self.netG.eval()


def create_model(opt):
    model = None
    #from .HG_model import HGModel
    model = HGModel(opt)
    print("model [%s] was created" % (model.name()))
    return model


"""
might not need this option, maybe hardcode somewhere
"""

import argparse
import os
#from util import util

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # self.parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        # self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='unet_256', help='selects model to use for netG')
        # self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2')
        self.parser.add_argument('--name', type=str, default='test_local', help='name of the experiment. It decides where to store samples and models')
        # self.parser.add_argument('--align_data', action='store_true',
                                # help='if True, the datasets are loaded from "test" and "train" directories and the data pairs are aligned')
        self.parser.add_argument('--model', type=str, default='pix2pix',
                                 help='chooses which model to use. cycle_gan, one_direction_test, pix2pix, ...')
        # self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--identity', type=float, default=0.0, help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        """
        need to take care of this 
        """
        #self.opt.isTrain = self.isTrain   # train or test
        self.opt.isTrain = False  # not going to train it

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir =  os.path.join(self.opt.checkpoints_dir, self.opt.name)
        #util.mkdirs(expr_dir)
        mkdirs(expr_dir) # not importing util
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--no_flip'  , action='store_true', help='if specified, do not flip the images for data argumentation')

        # NOT-IMPLEMENTED self.parser.add_argument('--preprocessing', type=str, default='resize_and_crop', help='resizing/cropping strategy')
        self.isTrain = True
