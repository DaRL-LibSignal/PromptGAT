import numpy as np
from copy import deepcopy
import torch
from torch import nn, no_grad
import torch.nn.functional as F
from  torch import optim
from torch.utils.data import DataLoader, Dataset
import os
from agent.utils import idx2onehot
import pickle as pkl
import random
from functools import lru_cache
from common.utils import get_weather
from prompt_data.ChatGPT.prompt_api import Conversation, split_data

def quantile_loss(y_pred, y_true):
    """
    not used, only for previous explore of quantile
    """
    quantiles = [0.025, 0.5, 0.975]
    losses = []
    sigmoid_fun = nn.Sigmoid()
    cross_entropy = nn.CrossEntropyLoss()
    for i, q in enumerate(quantiles):

        ###########v1:
        # errors =  y_true - y_pred
        # losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(0))
        ###########v2:
        # map the logit into the (0, 1) as probablity
        sigmoid_y_pred = sigmoid_fun(y_pred)
        errors =  cross_entropy(sigmoid_y_pred, y_true)
        losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(0))

    loss = torch.mean(torch.sum(torch.cat(losses, dim=0), dim=0))
    return loss

def log_passing_lane_actinon(traj, lanes, fix_time=30):
    # TODO: only for one intersection and fixedtime agent now
    # save in {time: {lanes: num}} format
    lanes_dict = {l:0 for l in lanes}
    record = {i: deepcopy(lanes_dict) for i in range(120)}
    # preprocess time
    for k in traj.keys():
        route = traj[k]
        tmp_road = route[0][0]
        tmp_interval =  (route[0][1]+route[0][2]-1) // fix_time # -1 for integer result, 5 sec yellow ensures this 
        record[tmp_interval][tmp_road] += 1 
    return record

def write_action_record(path, record, struc, fix_time=30):
    result = []
    result.append(str(struc))
    for t in range(int(3600/ fix_time)):
        tmp = []
        for r in struc:
            tmp.append(str(record[t][r]).ljust(5, ' '))
        result.append(str(tmp).replace(",", "").replace("'", ""))
    # temp = str([str(int(i)).ljust(5, ' ') for i in obs[idx][0]])
    with open(file=path, mode='w+', encoding='utf-8') as wf:
        for line in result:
            wf.writelines(line + "\n")

class NN_dataset(Dataset):
    def __init__(self, feature, target):
        self.len = len(feature)
        self.features = torch.from_numpy(feature).float()
        self.target = torch.from_numpy(target).float()

    def __getitem__(self, idx):
        return self.features[idx, :], self.target[idx]

    def __len__(self):
        return self.len

class UNCERTAINTY_predictor(object):
    def __init__(self, logger, in_dim, out_dim, DEVICE, model_dir, data_dir, backward=False, history=1):
        super(UNCERTAINTY_predictor, self).__init__()
        self.epo = 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.model =None
        self.backward = backward
        self.make_model()
        self.DEVICE = DEVICE
        self.model.to(DEVICE).float()
        if not backward:
            self.criterion = nn.MSELoss()
            self.learning_rate = 0.0001
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.learning_rate = 0.00001
        
        self.history = history
        self.batch_size = 64
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        # self.online_optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate * 0.1, momentum=0.9)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.99))
        self.online_optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.99))
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.logger = logger

        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None

    def load_dataset(self):
        train_data = generate_forward_dataset(self.data_dir, backward=self.backward, history=self.history)
        # train val split
        split_point = int(train_data['x_train'].shape[0] * 0.8)
        # shuffled when create
        if self.x_train is not None:
            self.x_train = np.concatenate((self.x_train, train_data['x_train'][: split_point]))
            self.y_train = np.concatenate((self.y_train, train_data['y_train'][: split_point]))
            self.x_val = np.concatenate((self.x_val, train_data['x_train'][split_point :]))
            self.y_val = np.concatenate((self.y_val, train_data['y_train'][split_point :]))     
        else:       
            self.x_train = train_data['x_train'][: split_point]
            self.y_train = train_data['y_train'][: split_point]
            self.x_val = train_data['x_train'][split_point :]
            self.y_val = train_data['y_train'][split_point :]
        
        # shuffle in dataloader
        print('dataset batch size: ', self.y_train.shape[0])


    def predict(self, x):
        x = x.to(self.DEVICE)
        with no_grad():
            output = self.model.forward(x)
            result, uncertainty = output[0], output[1]
        return result, uncertainty

    def train(self, epochs, writer, sign):
        train_loss = 0.0
        train_dataset = NN_dataset(self.x_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        if self.backward:
            txt = 'inverse'
        else:
            txt = 'forward'
        print(f"Epoch {self.epo - 1} Training")
        # epoch_quantiles = []
        for e in range(epochs):
            for i, data in enumerate(train_loader):
                x, y_true = data
                self.optimizer.zero_grad()
                x = x.to(self.DEVICE)
                y_true = y_true.to(self.DEVICE)
                result  = self.model(x)
                y_pred, uncertainty = result[0], result[1]

                # standard loss
                standard_loss = self.criterion(y_pred, y_true)

                loss = standard_loss 
               
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            # if self.backward:
            #     epoch_quantiles.append(np.mean(record_quantile))
 
            if e == 0:
                ave_loss = train_loss/ len(train_dataset)
                self.logger.info(f'epoch {e}: {txt} train average loss {ave_loss}.')
                test_loss = self.test(e, txt)

                if sign == 'inverse':
                    writer.add_scalar("ave_train_Loss/start_inverse_train", ave_loss, self.epo)
                    writer.add_scalar("ave_test_Loss/start_inverse_test", test_loss, self.epo)
                    # writer.add_scalar("quantile/inverse_quantile", epoch_quantiles[-1], self.epo)
                    # writer.add_scalar("quantile/inverse_quantile_details", np.array(epoch_quantiles), np.arange(start=0, stop=(self.epo+1) * len(epoch_quantiles)))
                elif sign == 'forward':
                    writer.add_scalar("ave_train_Loss/start_forward_train", ave_loss, self.epo)
                    writer.add_scalar("ave_test_Loss/start_forward_test", test_loss, self.epo)

            elif e == epochs - 1:
                ave_loss = train_loss / len(train_dataset)
                self.logger.info(f'epoch {e}: {txt} train average loss {ave_loss}.')
                test_loss = self.test_inverse(e, txt)

                if sign == 'inverse':
                    # writer.add_scalar("quantile/inverse_quantile", epoch_quantiles[-1], self.epo)
                    # writer.add_scalar("quantile/inverse_quantile_details", np.array(epoch_quantiles), np.arange(start=(self.epo) * len(epoch_quantiles), stop=(self.epo +1)* len(epoch_quantiles)))
                    writer.add_scalar("ave_train_Loss/end_inverse_train", ave_loss, self.epo)
                    writer.add_scalar("ave_test_Loss/end_inverse_test", test_loss, self.epo)
                elif sign == 'forward':
                    writer.add_scalar("ave_train_Loss/end_forward_train", ave_loss, self.epo)
                    writer.add_scalar("ave_test_Loss/end_forward_test", test_loss, self.epo)

            train_loss = 0.0
        self.epo += 1
        
        if sign == 'inverse':
            # print("uncertainty now is:"+str(uncertainty))
            # return the uncertainty if inverse:
            return uncertainty
        # import matplotlib.pyplot as plt
        # plt.scatter(np.arange(len(epoch_quantiles)), epoch_quantiles)
        # plt.show()
    
    def test(self, e, txt):
        test_loss = 0.0
        test_dataset = NN_dataset(self.x_val, self.y_val)
        test_loader = DataLoader(test_dataset, batch_size=64,shuffle=True)
        with no_grad():
            for i, data in enumerate(test_loader):
                x, y_true = data
                x = x.to(self.DEVICE)
                y_true = y_true.to(self.DEVICE)
                result = self.model(x)
                y_pred, uncertainty = result[0], result[1]
                # print(y_pred)
                # print(y_true)
                # normal loss, different from the training part
                loss = self.criterion(y_pred, y_true)
                test_loss += loss.item()
        test_loss = test_loss / len(test_dataset)
        self.logger.info(f'epoch {e}: {txt} test average loss {test_loss}.')
        return test_loss
    
    def test_inverse(self, e, txt):
        test_loss = 0.0
        test_dataset = NN_dataset(self.x_val, self.y_val)
        test_loader = DataLoader(test_dataset, batch_size=64,shuffle=True)
        with no_grad():
            for i, data in enumerate(test_loader):
                x, y_true = data
                x = x.to(self.DEVICE)
                y_true = y_true.to(self.DEVICE)
                result = self.model(x)
                y_pred, uncertainty = result[0], result[1]

                # standard loss
                standard_loss = self.criterion(y_pred, y_true)

                # chosen version:
                loss = standard_loss
             
                test_loss += loss.item()

        test_loss = test_loss / len(test_dataset)
        self.logger.info(f'epoch {e}: {txt} test average loss {test_loss}.')
        return test_loss
    
    def make_model(self):
        # v1:
        # self.model = N_net(self.in_dim, self.out_dim, self.backward).float()
        # v2: add loss to optimze
        self.model = N_net_back(self.in_dim, self.out_dim, self.backward).float()

    def load_model(self, type_model):
        if self.backward:
            txt = 'inverse'
        else:
            txt = 'forward'
        name = f"NN_inference_{txt}.pt"
        model_name = os.path.join(self.model_dir, name)
        if type_model == 'forward':
            self.model = N_net(self.in_dim, self.out_dim, self.backward)
        elif type_model == 'inverse':
            self.model = N_net_back(self.in_dim, self.out_dim, self.backward)

        self.model.load_state_dict(torch.load(model_name))
        self.model = self.model.float().to(self.DEVICE)

    def save_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if self.backward:
            txt = 'inverse'
        else:
            txt = 'forward'
        name = f"NN_inference_{txt}.pt"
        model_name = os.path.join(self.model_dir, name)
        torch.save(self.m34del.state_dict(), model_name)

class NN_predictor(object):
    def __init__(self, logger, in_dim, out_dim, DEVICE, model_dir, data_dir, backward=False, history=1):
        super(NN_predictor, self).__init__()
        self.epo = 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.model =None
        self.backward = backward
        self.make_model(type_model='forward')
        self.DEVICE = DEVICE
        self.model.to(DEVICE).float()
        if not backward:
            self.criterion = nn.MSELoss()
            self.learning_rate = 0.0001
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.learning_rate = 0.00001
        
        self.history = history
        self.batch_size = 64
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        # self.online_optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate * 0.1, momentum=0.9)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.99))
        self.online_optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.99))
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.logger = logger

        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None

        # train_data = generate_forward_dataset('collected/train.pkl', backward=self.backward)
        # self.x_train = train_data['x_train']
        # self.x_val = train_data['x_val']
        # self.y_train = train_data['y_train']
        # self.y_val = train_data['y_val']

        # test_data = generate_forward_dataset('collected/test.pkl',backward=self.backward)
        # self.x_test = test_data['x_val']
        # self.y_test = test_data['y_val']

    def load_dataset(self, conv=None):
        train_data = generate_forward_dataset(self.data_dir, backward=self.backward, history=self.history, conv=conv)
        # train val split
        split_point = int(train_data['x_train'].shape[0] * 0.8)
        # shuffled when create
        if self.x_train is not None:
            self.x_train = np.concatenate((self.x_train, train_data['x_train'][: split_point]))
            self.y_train = np.concatenate((self.y_train, train_data['y_train'][: split_point]))
            self.x_val = np.concatenate((self.x_val, train_data['x_train'][split_point :]))
            self.y_val = np.concatenate((self.y_val, train_data['y_train'][split_point :]))     
        else:       
            self.x_train = train_data['x_train'][: split_point]
            self.y_train = train_data['y_train'][: split_point]
            self.x_val = train_data['x_train'][split_point :]
            self.y_val = train_data['y_train'][split_point :]
        
        # shuffle in dataloader
        print('dataset batch size: ', self.y_train.shape[0])


    def predict(self, x):
        x = x.to(self.DEVICE)
        with no_grad():
            result = self.model.forward(x)
        return result

    def train(self, epochs, writer, sign):
        train_loss = 0.0
        train_dataset = NN_dataset(self.x_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        if self.backward:
            txt = 'inverse'
        else:
            txt = 'forward'
        print(f"Epoch {self.epo - 1} Training")
        # epoch_quantiles = []
        for e in range(epochs):
            record_quantile = []
            for i, data in enumerate(train_loader):
                x, y_true = data
                self.optimizer.zero_grad()
                x = x.to(self.DEVICE)
                y_true = y_true.to(self.DEVICE)
                result = self.model(x)
                y_pred, u = result[0], result[1]
                loss = self.criterion(y_pred, y_true)
                # add the quantile loss
                # if self.backward:
                #     quantile_value = quantile_loss(y_pred, y_true)
                    
                #     record_quantile.append(quantile_value.detach().numpy())
                # else: 
                #     loss = self.criterion(y_pred, y_true)
                # loss = QuantileLoss()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            # if self.backward:
            #     epoch_quantiles.append(np.mean(record_quantile))
 
            if e == 0:
                ave_loss = train_loss/ len(train_dataset)
                self.logger.info(f'epoch {e}: {txt} train average loss {ave_loss}.')
                if self.backward:
                    test_loss = self.testest_inverset(e, txt)
                else:
                    test_loss = self.test(e, txt)

                if sign == 'inverse':
                    writer.add_scalar("ave_train_Loss/start_inverse_train", ave_loss, self.epo)
                    writer.add_scalar("ave_test_Loss/start_inverse_test", test_loss, self.epo)
                    # writer.add_scalar("quantile/inverse_quantile", epoch_quantiles[-1], self.epo)
                    # writer.add_scalar("quantile/inverse_quantile_details", np.array(epoch_quantiles), np.arange(start=0, stop=(self.epo+1) * len(epoch_quantiles)))
                elif sign == 'forward':
                    writer.add_scalar("ave_train_Loss/start_forward_train", ave_loss, self.epo)
                    writer.add_scalar("ave_test_Loss/start_forward_test", test_loss, self.epo)

            elif e == epochs - 1:
                ave_loss = train_loss / len(train_dataset)
                self.logger.info(f'epoch {e}: {txt} train average loss {ave_loss}.')
                test_loss = self.test(e, txt)

                if sign == 'inverse':
                    # writer.add_scalar("quantile/inverse_quantile", epoch_quantiles[-1], self.epo)
                    # writer.add_scalar("quantile/inverse_quantile_details", np.array(epoch_quantiles), np.arange(start=(self.epo) * len(epoch_quantiles), stop=(self.epo +1)* len(epoch_quantiles)))
                    writer.add_scalar("ave_train_Loss/end_inverse_train", ave_loss, self.epo)
                    writer.add_scalar("ave_test_Loss/end_inverse_test", test_loss, self.epo)
                elif sign == 'forward':
                    writer.add_scalar("ave_train_Loss/end_forward_train", ave_loss, self.epo)
                    writer.add_scalar("ave_test_Loss/end_forward_test", test_loss, self.epo)

            train_loss = 0.0
        self.epo += 1
        
        if sign == 'inverse':
            # return the uncertainty if inverse:
            return 0
        # import matplotlib.pyplot as plt
        # plt.scatter(np.arange(len(epoch_quantiles)), epoch_quantiles)
        # plt.show()

    def test(self, e, txt):
        test_loss = 0.0
        test_dataset = NN_dataset(self.x_val, self.y_val)
        test_loader = DataLoader(test_dataset, batch_size=64,shuffle=True)
        with no_grad():
            for i, data in enumerate(test_loader):
                x, y_true = data
                x = x.to(self.DEVICE)
                y_true = y_true.to(self.DEVICE)
                y_pred, uncertainty = self.model(x)
                y_pred = y_pred.to(self.DEVICE)

                loss = self.criterion(y_pred, y_true)
                test_loss += loss.item()
        test_loss = test_loss / len(test_dataset)
        self.logger.info(f'epoch {e}: {txt} test average loss {test_loss}.')
        return test_loss
    


    def make_model(self, type_model):
        if type_model == 'forward':
            self.model = N_net(self.in_dim, self.out_dim, self.backward).float()
        elif type_model == 'inverse':
            self.model = N_net_back(self.in_dim, self.out_dim, self.backward)

        # self.model = LSTMPredictor(self.in_dim, self.out_dim, self.backward).float()

    def load_model(self, type_model):
        if self.backward:
            txt = 'inverse'
        else:
            txt = 'forward'
        name = f"NN_inference_{txt}.pt"
        model_name = os.path.join(self.model_dir, name)
        if type_model == 'forward':
            self.model = N_net(self.in_dim, self.out_dim, self.backward)
        elif type_model == 'inverse':
            self.model = N_net_back(self.in_dim, self.out_dim, self.backward)
            
        self.model.load_state_dict(torch.load(model_name))
        self.model = self.model.float().to(self.DEVICE)

    def save_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if self.backward:
            txt = 'inverse'
        else:
            txt = 'forward'
        name = f"NN_inference_{txt}.pt"
        model_name = os.path.join(self.model_dir, name)
        torch.save(self.m34del.state_dict(), model_name)

#################################
"""
origin N_net based model:
"""

# This function to generate evidence is used for the first example
def relu_evidence(logits):
    relu_net = torch.nn.ReLU()
    return relu_net(logits)


# This one usually works better and used for the second and third examples
# For general settings and different datasets, you may try this one first
def exp_evidence(logits):
    return torch.exp(torch.clip(logits, -10, 10))


# This one is another alternative and
# usually behaves better than the relu_evidence
def softplus_evidence(logits):
    softplus_net = torch.nn.Softplus()
    return softplus_net(logits)

def var_torch(shape, init=None):
    if init is None:
        data_ini = torch.empty(size=shape)
        std = (2 / shape[0]) ** 0.5
        init = torch.nn.init.trunc_normal_(tensor=data_ini, std=std)

    return init

def KL(alpha):
    # beta = tf.constant(np.ones((1, K)), dtype=tf.float32)
    K = 8
    beta = torch.ones((1, K))
    # S_alpha = tf.reduce_sum(alpha, axis=1, keep_dims=True)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    # S_beta = tf.reduce_sum(beta, axis=1, keep_dims=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    # lnB = tf.lgamma(S_alpha) - tf.reduce_sum(tf.lgamma(alpha), axis=1, keep_dims=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def loss_EDL(p, alpha, global_step, annealing_step):
    # annealing_step3:10*n_batches
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1

    A = torch.sum(p * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = torch.Tensor(np.array(np.min([1.0, global_step / annealing_step])))

    alp = E * (1 - p) + 1
    B = annealing_coef * KL(alp)
    result = A + B
    print(result)
    return (result)

def loss_v2(logits, labels):
    soft_max_value = F.log_softmax(logits, dim=1) # map to (normalized version) on dim1, taken an row, normalize its columns
    loss_value = soft_max_value * labels
    loss_sum = -torch.sum(loss_value, dim=1)
    loss_k = torch.mean(loss_sum)
    return loss_k

def l2_penalty(w):
    return (w**2).sum() / 2


# for inverse model
class N_net_back(nn.Module):
    def __init__(self, size_in, size_out, backward):
        super(N_net_back, self).__init__()
        self.backward = backward

        self.dense_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(size_in, 64) # current verrsion: 16 * 4 = 64
        )
        self.dense_2 = nn.Linear(64, 128)
        self.dense_3 = nn.Linear(128, 128)
        self.dense_4 = nn.Linear(128, 20)
        self.dense_5 = nn.Linear(20, size_out)


    def var_torch(shape, init=None):
        if init is None:
            data_ini = torch.empty(size=shape)
            std = (2 / shape[0]) ** 0.5
            init = torch.nn.init.trunc_normal_(tensor=data_ini, std=std)

        return init
    

    def forward(self, x):
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = F.relu(self.dense_3(x))
        x = F.relu(self.dense_4(x)) #（64， 500）
        u = torch.Tensor([0.5])# set a fake value, just for value passing
        x = self.dense_5(x)

        return x, u


# for forward model
class N_net(nn.Module):
    def __init__(self, size_in, size_out, backward):
        super(N_net, self).__init__()
        self.backward = backward

        self.embedding_main = nn.Linear(64, 64) # make main expressive
        self.embedding_extra = nn.Linear(128, 16) # make extra condense
        self.embedding_weather = nn.Linear(1, 8) # impress 8 lanes

        self.dense_1 = nn.Linear(80, 256)
        self.dense_2 = nn.Linear(256, 128)
        self.dense_3 = nn.Linear(128, 128)
        self.dense_4 = nn.Linear(128, 20)
        self.dense_5 = nn.Linear(20, size_out)
    def var_torch(shape, init=None):
        if init is None:
            data_ini = torch.empty(size=shape)
            std = (2 / shape[0]) ** 0.5
            init = torch.nn.init.trunc_normal_(tensor=data_ini, std=std)

        return init
    

    def forward(self, x):
        batch = x.shape[0]
        # x (64, 224), 224 = 56 * 4 (memory) ==> 56: 16, 40,,, * 4
        resize_part = x.view(batch, 4, 8, -1) # (64, 4, 8, 7): 1 1 4 1
        
        original_main_part = resize_part[..., :2] # (64, 4, 8, 2) # important part
        original_main_part = original_main_part.reshape(batch, -1)  # (64, 16*4) 

        extra_part = resize_part[..., 2:6] # (64, 4, 8, 4) # real features, exclude weather
        extra_part = extra_part.reshape(batch, -1) # (64, 128) 

        main_embed = self.embedding_main(original_main_part) # (64, 64) 
        extra_embed = self.embedding_extra(extra_part) # (64, 16) 

        # fusion = torch.cat((main_embed, extra_embed, weat_embed), dim=1) # (64, 88)
        fusion = torch.cat((main_embed, extra_embed), dim=1) # (64, 80)

        fusion_embed = F.relu(self.dense_1(fusion)) # (64, 88)==> (64, 256)
        fusion_embed = F.relu(self.dense_2(fusion_embed)) # (256, 128)
        fusion_embed = F.relu(self.dense_3(fusion_embed)) # (128, 128)
        fusion_embed = F.relu(self.dense_4(fusion_embed)) # (128, 20)
        x = self.dense_5(fusion_embed)

        u = torch.Tensor([0.5]) # manually defined, minize the computation 

        return x, u


def prompt_for_info(prompt_element, conv):
    # vehicle_num, phase_value, weather_name
    capacity_num = str(50)
    vehicle_num, phase_value, weather_name = prompt_element

    vehicle_num = str(vehicle_num)
    weather_name = str(weather_name)
    
    prompt_info = "Given the knowledge: during snowy weather, the average acceleration of the vehicle is 0.5 meters per second squared, the average deceleration of the vehicle is 1.5 meters per second squared, and the average emergency deceleration of the vehicle is about 2 meters per second squared and the average delay is 0.5 second. Knowing that these indicatiors might vary based on the vehicle number and snowy level, they will have bias based on the standard value. If for a lane with " +vehicle_num+ " vehicles while capacity is " + capacity_num +" , and under "+ weather_name +"weather, each lane has the capacity as" + capacity_num + ",  please assume the four indicators (average acceleration, average deceleration, average emergency deceleration and average delay) in the format of (replace {value } as assumed value, only one value allowed): [average acceleration: {value}], [average deceleration: {value}], [average emergency deceleration: {value}], [average delay: {value}]"

    res = conv.get_response(prompt_info=prompt_info, max_token=2000, model="text-davinci-003")
    resu = conv.get_response_text(res)
    spl = split_data(resu[10:])
    print("obtained prompt result, keep training...")
    return np.array(spl)





def generate_forward_dataset(file, action=8, backward=False, history=1, conv=None):
    prompt_forward = True

    if prompt_forward:
        pass
    
    with open(file, 'rb') as f:
        contents = pkl.load(f)

    feature = list()
    target = list()
    if backward: # inverse dataset
        assert history == 1
        for e in range(contents[0].__len__()):
            for s in range(360):
                # -------------------------------------- v0 -----------------------------------------------
                x = np.concatenate((contents[0][e][s], contents[0][e][s+1]), axis=1)
                # -------------------------------------- v1 -----------------------------------------------
                feature.append(x)
                y = idx2onehot(contents[1][e][s], action)
                target.append(y)

    else: # forward dataset
        unit_size = 16 + 8 * (4 + 1) # 4: path info, 1: weather info = 56
        input_size = history * unit_size # (4 * 56)
        for e in range(contents[0].__len__()):
            seq = np.zeros((1, input_size)) # input_size = 64 ： 16 * 4， now we need : 56 * 4
            weather_name, weather_value = get_weather() # random sample weather condition, for 360 time step, are the same
            for s in range(360):
                phase_info = idx2onehot(contents[1][e][s][0], action)
                path_info = []
                for i in range(len(contents[0][e][s][0])):
                    veichle_num = contents[0][e][s][0][i]
                    phase_value = phase_info[0][i]
                    input_info = [veichle_num, phase_value, weather_value]

                    prompt_element = [veichle_num, phase_value, weather_name]
                    path_i = prompt_for_info(prompt_element, conv=conv)
                    path_info.append(path_i)

                # path_info = np.array(path_info).reshape(1, 8, -1) # (1, 8, 7)
                path_info = np.array(path_info).reshape(1, 8, -1)
                weather_re = np.expand_dims(np.repeat([weather_value], len(contents[0][e][s][0])), axis=(0, 2)) # (1, 8, 1)
                ob_re = np.expand_dims(contents[0][e][s], axis = 2) # (1, 8) => (1, 8, 1)
                one_hot_re = np.expand_dims(phase_info, axis=2) # (1, 8) => (1, 8, 1)
                
                new_feature = np.concatenate((ob_re, one_hot_re, path_info, weather_re), axis=2) # (1, 8, 7) [1, 1, 4, 1]
                # save_prompt_data(str(new_feature))
                x = np.reshape(new_feature, (1, -1))

                # x = np.concatenate((contents[0][e][s], phase_info, path_info), axis=1) # (1, 40) == (1, 56)
                seq = np.concatenate((seq, x), axis=1)
                feature.append(seq[:, -input_size :])

                # the target y is simple: array([[0., 0., 0., 0., 0., 0., 3., 0.]], dtype=float32) (1, 8)
                y = contents[0][e][s+1]
                target.append(y)

    feature= np.concatenate(feature, axis=0) # now : (360, 56 * 4), orginal (360, 64): = 16 * 4
    target = np.concatenate(target, axis=0)


    total_idx = len(target)
    sample_idx = range(total_idx)
    sample_idx = random.sample(sample_idx, len(sample_idx))
    x_train = feature[sample_idx]
    y_train = target[sample_idx]
    dataset = {'x_train': x_train, 'y_train': y_train}
    return dataset

# def generate_forward_dataset(file, action=8, backward = False):
#     with open(file, 'rb') as f:
#         contents = pkl.load(f)

#     feature = list()
#     target = list()
#     if backward:
#         for e in range(contents[0].__len__()):
#             for s in range(360, 0, -1):
#                 x = np.concatenate((contents[0][e][s], idx2onehot(contents[1][e][s-1][0], action)), axis=1)
#                 feature.append(x)
#                 y = contents[0][e][s-1]
#                 target.append(y)
#     else:
#         for e in range(contents[0].__len__()):
#             for s in range(360):
#                 x = np.concatenate((contents[0][e][s], idx2onehot(contents[1][e][s][0], action)), axis=1)
#                 feature.append(x)
#                 y = contents[0][e][s+1]
#                 target.append(y)

#     feature= np.concatenate(feature, axis=0)
#     target = np.concatenate(target, axis=0)
#     total_idx = len(target)
#     sample_idx = range(total_idx)
#     sample_idx = random.sample(sample_idx, len(sample_idx))
#     x_train = feature[sample_idx[: int(0.8 * total_idx)]]
#     y_train = target[sample_idx[: int(0.8 * total_idx)]]
#     x_test = feature[sample_idx[int(0.8 * total_idx) :]]
#     y_test = target[sample_idx[int(0.8 * total_idx) :]]
#     dataset = {'x_train': x_train, 'y_train': y_train, 'x_val': x_test, 'y_val': y_test}
#     return dataset