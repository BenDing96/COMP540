import numpy as np

class ProbabilityModel:

    # Returns a single sample (independent of values returned on previous calls).
    # The returned value is an element of the model's sample space.
    def sample(self):
        return np.random.uniform(1, -1)


# The sample space of this probability model is the set of real numbers, and
# the probability measure is defined by the density function 
# p(x) = 1/(sigma * (2*pi)^(1/2)) * exp(-(x-mu)^2/2*sigma^2)
class UnivariateNormal(ProbabilityModel):
    
    # Initializes a univariate normal probability model object
    # parameterized by mu and (a positive) sigma
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def sample(self):
        # Central limit theorem
        N = 100
        sampleSum = 0
        for _ in range(N):
            sampleSum += np.random.uniform(-1, 1)
        sample = sampleSum * self.sigma/np.sqrt(N) + self.mu

        return sample

    
# The sample space of this probability model is the set of D dimensional real
# column vectors (modeled as numpy.array of size D x 1), and the probability 
# measure is defined by the density function 
# p(x) = 1/(det(Sigma)^(1/2) * (2*pi)^(D/2)) * exp( -(1/2) * (x-mu)^T * Sigma^-1 * (x-mu) )
class MultiVariateNormal(ProbabilityModel):
    
    # Initializes a multivariate normal probability model object 
    # parameterized by Mu (numpy.array of size D x 1) expectation vector
    # and symmetric positive definite covariance Sigma (numpy.array of size D x D)
    def __init__(self, Mu, Sigma):
        self.Mu = Mu
        self.Sigma = Sigma

    def sample(self):
        N = len(self.Mu)
        sampleList = []
        for _ in range(N):
            sampleList.append(UnivariateNormal(0, 1).sample())
        sample = self.Mu + np.dot(self.Sigma, np.array(sampleList))
        return sample
    

# The sample space of this probability model is the finite discrete set {0..k-1}, and 
# the probability measure is defined by the atomic probabilities 
# P(i) = ap[i]
class Categorical(ProbabilityModel):
    
    # Initializes a categorical (a.k.a. multinom, multinoulli, finite discrete) 
    # probability model object with distribution parameterized by the atomic probabilities vector
    # ap (numpy.array of size k).
    def __init__(self, ap):
        self.ap = ap

    def sample(self):
        k = len(self.ap)
        cdf = []
        # Cumulative Distribution function
        for i in range(1, k+1):
            sum = 0
            for j in range(len(self.ap[:i])):
                sum += self.ap[j]
            cdf.append(sum)
        num = np.random.uniform(0, 1)
        for i in range(k):
            if num < cdf[i]:
                return i


# The sample space of this probability model is the union of the sample spaces of 
# the underlying probability models, and the probability measure is defined by 
# the atomic probability vector and the densities of the supplied probability models
# p(x) = sum ad[i] p_i(x)
class MixtureModel(ProbabilityModel):

    def __init__(self, ap, pm):
        self.ap = ap
        self.pm = pm

    def sample(self):
        i = Categorical(self.ap).sample()
        res = self.pm[i]
        return res.sample()
