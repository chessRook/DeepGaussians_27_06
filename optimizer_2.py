import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import math
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import random

writer = SummaryWriter()


# TODO current BUG RHS isn't diff while LHS is diff.

class Calculator:
    def __init__(self, lambdas, gammas, ixy, hx, alpha, c, dim):
        self.lambdas = lambdas
        self.gammas = gammas
        self.ixy = ixy
        self.hx = hx
        self.alpha = alpha
        self.c = c
        self.dim = dim

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

    def chechik_term(self, beta, ixt):
        k_beta = self.beta_placer(beta)
        masker = torch.tensor([1 for k in range(k_beta)]
                              + [0.] * (self.dim - k_beta)).to('cuda')
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

    def ixt(self, beta):
        k_beta = self.beta_placer(beta)
        lambda_max = 1 - 1 / (k_beta + 1)
        multipliers = torch.tensor([self.k_multiplier_ixt(k, lambda_max) for k in range(1, k_beta + 1)]
                                   + [0.] * (self.dim - k_beta)).to('cuda')
        terms = self.gammas * multipliers
        ixt = torch.sum(terms)
        return ixt

    def k_multiplier_ixt(self, k, lambda_max):
        lambda_k = torch.tensor(1 - 1 / (k + 1)).to('cuda')
        log_term = torch.log((lambda_max * (1 - lambda_k)) / (lambda_k * (1 - lambda_max)))
        return log_term

    def beta_placer(self, beta):
        lambda_ = 1 - beta ** -1
        k_frac = 1 / (1 - lambda_) - 1
        k = math.floor(k_frac)
        return k


class Trainer(Calculator):
    def __init__(self):
        self.device = 'cuda'
        self.dim = 1_00
        self.lambdas = torch.tensor([1 - 1 / (k + 1) for k in range(1, self.dim + 1)]).to('cuda')
        self.gammas = torch.tensor([2., ] * self.dim).to('cuda')
        self.hx = torch.tensor([5_00., ]).to('cuda')
        self.ixy = torch.tensor([4_50.]).to('cuda')
        self.alpha = 1.92
        self.c = np.exp(-.866)
        self.optim = None
        self.lr = .005
        Calculator.__init__(self, self.lambdas, self.gammas, self.ixy, self.hx, self.alpha, self.c, self.dim)

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
            beta = self.beta_randomizer(batch_size=10)
            loss_ten = \
                (self.losser(beta[0]) + self.losser(beta[1]) + self.losser(beta[2]) + self.losser(beta[3])
                 + self.losser(beta[4]) + self.losser(beta[5]) + self.losser(beta[6]) + self.losser(beta[7])
                 + self.losser(beta[8]) + self.losser(beta[9])) / 10
            loss = loss_ten
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            #####################
            self.note_me(i, loss)

    def beta_randomizer(self, batch_size=1):
        beta_1 = [(self.dim - 1) * random.random() + 1 for _ in range(batch_size)]
        return beta_1

    def rhs(self, beta):
        rhs = 1 - self.iy(beta)
        return rhs

    def lhs(self, beta):
        lhs_1 = 1 - self.ix(beta)
        if lhs_1 < 0:
            lhs_1 = torch.abs(lhs_1) * 1_000.
        lhs = self.c * torch.pow(lhs_1, self.alpha)
        return lhs

    def losser(self, beta):
        rhs = self.rhs(beta)
        lhs = self.lhs(beta)
        diff = torch.square(lhs - rhs)
        return diff

    def note_me(self, i, loss):
        if i % 1 == 0:
            # TODO change me
            print(f'loss: {loss}')

        writer.add_scalar('loss', loss, i)
        writer.add_scalar('H(X)', self.hx, i)
        writer.add_scalar('I(X, Y)', self.ixy, i)

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
