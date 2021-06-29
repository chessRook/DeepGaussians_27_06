import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import math
from tqdm import tqdm
import random
import torch.nn as nn
import neptune.new as neptune

#######################################################
run = neptune.init(project='dan.nav/DeepGaussians',
                   api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vY'
                             'XBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsI'
                             'joiaHR0cHM6Ly9hcHAubmVwdHVuZS5haS'
                             'IsImFwaV9rZXkiOiI0MTNiNzMxYS03Mjg2'
                             'LTQ2ODMtYjQ0Yi0zNDJjMmE2YTYxZDcifQ==')

########################################################




class Trainable(nn.Module):
    def __init__(self, gammas, ixy, hx):
        super(Trainable, self).__init__()
        self.gammas = gammas
        self.ixy = ixy
        self.hx = hx

    def forward(self, betas):
        rhs = self.rhs(betas)
        lhs = self.lhs(betas)
        return rhs, lhs


class Calculator(Trainable):
    def __init__(self, lambdas, gammas, ixy, hx, alpha, c, dim):
        super(Calculator, self).__init__(gammas, ixy, hx)
        self.lambdas = lambdas
        self.gammas = gammas
        self.ixy = ixy
        self.hx = hx
        self.alpha = alpha
        self.c = c
        self.dim = dim

    def rhs(self, beta):
        rhs = 1 - self.iy(beta)
        return rhs

    def lhs(self, beta):
        lhs_1 = 1 - self.ix(beta)
        if lhs_1 < 0:
            lhs_1 = torch.abs(lhs_1) * 20.
        lhs = self.c * torch.pow(lhs_1, self.alpha)
        return lhs

    def ix(self, beta):
        ixt = self.ixt(beta)
        hx = self.hx
        ix = ixt / hx
        return ix

    def iy(self, beta):
        ity = self.ity(beta)
        ixy = self.ixy
        iy = ity / ixy
        return iy

    def ity(self, beta):
        ixt = self.ixt(beta)
        second_chechik_term = self.chechik_term(beta, ixt)
        ity = ixt + second_chechik_term
        return ity

    def ixt(self, beta):
        k_beta = self.beta_placer(beta)
        lambda_max = 1. - torch.pow(k_beta.float() + 1, -1)
        multipliers = self.multipliers(k_beta, lambda_max)
        terms = self.gammas * multipliers
        ixt = torch.sum(terms)
        return ixt

    def chechik_term(self, beta, ixt):
        k_betas = self.beta_placer(beta).squeeze().tolist()
        masker = torch.tensor([[1 for k in range(k_beta)]
                               + [0.] * (self.dim - k_beta) for k_beta in k_betas]).to('cuda')
        masked_lambdas = self.lambdas * masker
        masked_gammas_un_normalized = self.gammas * masker
        n_I = torch.sum(masked_gammas_un_normalized)
        masked_gammas = masked_gammas_un_normalized / n_I
        pow_term = torch.pow(masked_lambdas, masked_gammas)
        gm_term = torch.prod(pow_term)
        ############################################
        masked_comp_lambdas = (1. - self.lambdas) * masker
        comp_prod_term = torch.pow(masked_comp_lambdas, masked_gammas)
        gm_comp_term = torch.prod(comp_prod_term)
        ##############################################
        exp_term = torch.exp(2 * ixt / n_I)
        #################################################
        inner_log_term = gm_comp_term + exp_term * gm_term
        log_term = -n_I / 2 * torch.log(inner_log_term)
        return log_term

    def beta_placer(self, beta):
        lambda_ = 1 - torch.pow(beta, -1)
        k_frac = torch.pow(1 - lambda_, -1) - 1
        k = torch.floor(k_frac).int()
        return k

    def multipliers(self, k_betas, lambda_max):
        k_betas_lst = k_betas.squeeze().clone().detach().tolist()
        lambda_max_lst = lambda_max.squeeze().clone().detach().tolist()
        multipliers_lst = [[self.k_multiplier_ixt(k, lambda_max).squeeze().item() for k in range(1, k_beta + 1)]
                           + [0., ] * (self.dim - k_beta) for k_beta, lambda_max in zip(k_betas_lst, lambda_max_lst)]
        multipliers = torch.tensor(multipliers_lst, device='cuda')
        return multipliers

    def k_multiplier_ixt(self, k, lambda_max):
        lambda_k = torch.tensor(1 - 1 / (k + 1)).to('cuda')
        log_term = torch.log((lambda_max * (1 - lambda_k)) / (lambda_k * (1 - lambda_max)))
        return log_term


class Trainer(Calculator):
    def __init__(self):
        self.device = 'cuda'
        self.dim = 1_00
        self.lambdas = torch.tensor([1 - 1 / (k + 1) for k in range(1, self.dim + 1)]).to('cuda')
        self.gammas = torch.tensor([2., ] * self.dim).to('cuda')
        self.hx = torch.tensor([5_000., ]).to('cuda')
        self.ixy = torch.tensor([4_50.]).to('cuda')
        self.alpha = 1.92
        self.c = np.exp(-.866)
        self.optim = None
        self.lr = .005
        Calculator.__init__(self, self.lambdas, self.gammas, self.ixy, self.hx, self.alpha, self.c, self.dim)
        ##############################
        run["JIRA"] = "NPT-952"
        run["parameters"] = {"learning_rate": self.lr,
                             "optimizer": "Adam"}
        ##############################

    def plotter(self):
        lhs, rhs = [], []
        for beta in range(1, self.dim):
            lhs_ = self.lhs(beta)
            rhs_ = self.rhs(beta)
            #####################
            lhs.append(lhs_)
            rhs.append(rhs_)
            #####################

    def optimizer(self):
        self.train_mode()
        for i in tqdm(range(10_000)):
            betas = self.beta_randomizer(batch_size=10)
            rhs, lhs = self.forward(betas)
            loss_ten = self.losser(rhs, lhs)
            loss = loss_ten
            loss.backward()
            self.optim.step()
            #####################
            self.note_me(i, loss, rhs, lhs, betas)
            #####################
            self.optim.zero_grad()

    def losser(self, rhs, lhs):
        diff = torch.abs(lhs - rhs).mean()
        return diff

    def beta_randomizer(self, batch_size=1):
        beta_1 = torch.tensor([(self.dim - 1) * random.random() + 1 for _ in range(batch_size)], device='cuda')
        beta_2 = beta_1.unsqueeze(-1)
        return beta_2

    def note_me(self, i, loss, rhs, lhs, betas):
        if i % 3 == 0:
            print(f'loss: {loss}')

        run["loss"].log(loss)
        run["H(X)"].log(self.hx)
        run["I(X, Y)"].log(self.ixy)
        run["RHS"].log(rhs)
        run["LHS"].log(lhs)
        run["I(X, Y)"].log(lhs)

    def train_mode(self):
        ####################################
        self.gammas.requires_grad = True
        self.hx.requires_grad = True
        self.ixy.requires_grad = True
        ####################################
        self.optim = torch.optim.Adam((self.gammas, self.hx, self.ixy), lr=self.lr)
        ####################################


def runner():
    trainer = Trainer()
    trainer.optimizer()
    trainer.plotter()


if __name__ == '__main__':
    runner()
