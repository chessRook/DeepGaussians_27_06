import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision


class Dists:
    def __init__(self):
        self.max_beta = 1_000
        self.margin = 10
        self.ixy = 1.
        self.hx = 1_000.
        self.lambdas = self.lambdas_builder()
        self.gammas = self.gammas_builder()

    def gammas_builder(self):
        gammas = []
        for i in range(2, self.max_beta + self.margin):
            next_gamma = self.optimizer(self.lambdas, gammas)
            gammas.append(next_gamma)
            self.ixy_updator(self.lambdas, gammas)

    def optimizer(self, lambdas, gammas):
        """Looks after the next gamma"""
        max_lambda = self.max_lambda(lambdas, gammas)

    def max_lambda(self, lambdas, gammas):
        num_known = len(gammas)
        max_lambda = lambdas[num_known]
        return max_lambda

    def lambdas_builder(self):
        lambdas = []
        for i in range(2, self.max_beta + self.margin):
            lambdas.append(1 - 1 / (i + 1))
        return lambdas

    def itx(self, lambdas, gammas, max_lambda):
        sum_ = 0
        for lambda_, gamma in zip(lambdas, gammas):
            if lambda_ >= max_lambda:
                break
            sum_ += gamma * np.log((max_lambda * (1 - lambda_)) / ((1 - max_lambda) * lambda_))
        return sum_

    def ity(self, itx, max_lambda, lambdas, gammas):
        term_1 = term_2 = term_3 = 0
        for lambda_, gamma in zip(lambdas, gammas):
            if lambda_ >= max_lambda:
                break
            term_1 += gamma * np.log(lambda_)
            term_2 += gamma * np.log(1 - lambda_)
            term_3 += gamma
        avg_bar = term_2 / term_3
        avg = term_1 / term_3
        prod_bar = np.exp(avg_bar)
        prod = np.exp(avg)
        exp_term = np.exp(2 * itx / term_3)
        log_term = -(term_3 / 2) * np.log(prod_bar + exp_term * prod)
        ity = itx - log_term
        return ity

    def ixy_updator(self, lambdas, gammas):
        self.ixy = sum(gamma * np.log(1 / lambda_) for lambda_, gamma in zip(lambdas, gammas))
