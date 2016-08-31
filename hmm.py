#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import numpy as np
from numpy.linalg import norm


# ------------------------------------------------------------------------------

class HMM:
	"""
	Class to represent HHMs.
	"""

	def __init__(self, hidden_states):
		"""
		Construct the HMM object.

		:param hidden_states: number of hidden units of the model
		:type hidden_states: int
		:param observ_symbols: symbols to be observed
		:type observ_symbols: array-like = [#observ]
		:return: self instance
		"""
		self.N = hidden_states

	# --------------------------------------------------------------------------

	def _init_model(self, seed=1234):
		"""
		Initialize the model with uniformly distributed random values.

		:param seed: seed for random generator
		:return: self instance
		"""
		np.random.seed(seed)

		A = np.random.rand(self.N, self.N)
		A /= A.sum(axis=1, keepdims=True)

		B = np.random.rand(self.N, self.M)
		B /= B.sum(axis=1, keepdims=True)

		pi = np.random.rand(self.N)
		pi /= pi.sum()

		self.A, self.B, self.pi = A, B, pi

	# --------------------------------------------------------------------------

	def fit(self, y, seed=1234, tol=1e-6):
		"""
		Fit the model to given obversation sequence y using the Baum-Welch
		algorithm.

		:param y: observation sequence
		:type y: array-like, shape = [n_features, n_datapoints]
		:param seed: seed to be used for random
		:type seed: int
		:return: fitted model
		:rtype: HMM
		"""
		self.T = y.shape[0]
		self.observations, self.obs_idx = np.unique(y, return_inverse=True)
		self.M = self.observations.shape[0]
		self.obs_idx_map = dict(zip(self.observations, np.arange(self.M)))

		self._init_model(seed)

		diff = np.inf
		while diff > tol:
			self._calc_alpha()
			self._calc_beta()

			self._calc_xi()
			self._calc_gamma2()

			# update vars
			pi = self.gamma[:, 0]
			A = np.sum(self.xi, axis=0) / np.sum(self.gamma[:, :-1], axis=1, keepdims=True)

			idxs = self.obs_idx == np.arange(self.M)[:, np.newaxis]
			B = np.sum(self.gamma[:, np.newaxis] * idxs, axis=2)
			B /= np.sum(self.gamma, axis=1, keepdims=True)

			diff = norm(np.array([norm(self.pi - pi), norm(self.A - A), norm(self.B - B)]))

			self.A, self.B, self.pi = A, B, pi

		self.fitted = True

	# --------------------------------------------------------------------------

	def _calc_alpha(self):
		"""
		Calculate forward variables, i.e. alpha-matrix with a_ij = a_j(i) (j - time) and array-like = [M, T]
		"""
		alpha = np.empty((self.N, self.T))
		alpha[:, 0] = np.multiply(self.pi, self.B[:, self.obs_idx[0]])

		for i in range(1, self.T):
			alpha[:, i] = np.multiply(alpha[:, i - 1].dot(self.A), self.B[:, self.obs_idx[i]])

		self.alpha = alpha

	# --------------------------------------------------------------------------

	def _calc_beta(self):
		"""
		Calculate backward variables, i.e. beta-matrix with b_ij = b_j(i) (j - time) and array-like = [M, T]
		"""
		beta = np.empty((self.N, self.T))
		beta[:, -1] = 1

		for i in range(self.T - 2, -1, -1):
			beta[:, i] = np.multiply(beta[:, i + 1], self.B[:, self.obs_idx[i + 1]]).dot(self.A.T)

		self.beta = beta

	# --------------------------------------------------------------------------

	def _calc_xi(self):
		"""
		Calculate the xi (tensor) with
			axis0 - time
			axis1 - i
			axis2 - j
		"""
		# b_j(O_t+1) * beta_t+1(j)
		b = self.B[:, self.obs_idx[1:]] * self.beta[:, 1:]

		xi = np.expand_dims(self.alpha[:, :-1].T, axis=2) * np.expand_dims(b.T, axis=1)
		xi *= self.A[np.newaxis]
		# normalize for each t
		xi /= np.sum(xi, axis=(1, 2), keepdims=True)

		self.xi = xi

	# --------------------------------------------------------------------------

	def _calc_gamma(self):
		"""
		Calculate the gammas; dimension of self.gamma = [N, T]
		"""
		# for t=1,..,T-1 we use the xis
		gamma = np.sum(self.xi, axis=2)

		g_t = self.alpha[:, -1] * self.beta[:, -1]
		g_t /= np.sum(g_t)

		gamma = np.hstack((gamma.T, g_t[:, np.newaxis]))

		self.gamma = gamma
		return gamma

	# --------------------------------------------------------------------------

	def _calc_gamma2(self):
		"""
		Calculate the gammas; dimension of self.gamma = [N, T]
		Use just alpha and beta for calculation
		"""
		gamma = self.alpha * self.beta
		# normalize
		gamma /= np.sum(gamma, axis=0)

		self.gamma = gamma
		return gamma

	# --------------------------------------------------------------------------
	# Just for testing purpose
	def _numerator_xi(self, i, j, t):
		return (self.alpha[i, t] * self.beta[j, t + 1] * self.A[i, j] * self.B[j, self.obs_idx[t + 1]])

	def _gen_xi(self):
		xi = np.empty_like(self.xi)
		for t, i, j in np.ndindex(*xi.shape):
			xi[t, i, j] = self._numerator_xi(i, j, t)
		return xi / xi.sum(axis=(1, 2), keepdims=True)

	# --------------------------------------------------------------------------

	def obs_seq_prob(self, observations):
		"""
		Calculate the probability of the given sequence wrt. the model.

		:param observations: observation sequence
		:type: array-like = [T]
		:return: probability
		:rtype: float
		"""
		assert self.fitted, "First fit the model"

		self.T = observations.shape[0]
		self.obs_idx = np.array([self.obs_idx_map[o] for o in observations])
		self._calc_alpha()
		return self.alpha[:, -1].sum()


# ------------------------------------------------------------------------------

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Execute task a) and b) for the given observation sequence \"v1, v4, v3, v1, v2\".")
	parser.add_argument("--seed", "-s", help="set a seed", type=int)
	args = parser.parse_args()

	obs = np.array([1, 4, 3, 1, 2])
	t = HMM(3)

	if args.seed:
		t.fit(obs, seed=args.seed)
	else:
		t.fit(obs)

	print "A:\n{}".format(np.array_str(t.A, precision=4, suppress_small=True))
	print "B:\n{}".format(np.array_str(t.B, precision=4, suppress_small=True))
	print "pi:\n{}".format(np.array_str(t.pi, precision=4, suppress_small=True))

	print "-" * 60

	print "Pr({}) = {}".format(obs, t.obs_seq_prob(obs))
