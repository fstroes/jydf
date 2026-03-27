import numpy as np
from scipy.stats import qmc, norm


class JYDF:
    """
    Jianming Ye's Degrees of Freedom (JYDF) Estimator.
    """

    def __init__(self, tau=0.1):
        self.tau = tau

    def estimate_df(self, model_wrapper, y=None, iterations=100, epsilons=None):
        n = len(y)
        total_sensitivity = 0

        for i in range(iterations):

            epsilon = np.random.normal(0, self.tau, size=n)

            # 1. Perturb y
            y_perturbed = y + epsilon
            y_perturbed_neg = y - epsilon

            # 2. Get new y_hat
            y_hat_perturbed = model_wrapper.predict_y_hat(y_perturbed)
            y_hat_perturbed_neg = model_wrapper.predict_y_hat(y_perturbed_neg)

            # 3. Covariance penalty: sum( (dy_hat) * (dy) )
            total_sensitivity += np.dot((y_hat_perturbed - y_hat_perturbed_neg)/2, epsilon)

        return total_sensitivity / (iterations * (self.tau ** 2)) # average covariance divided by the
        # variance of the perturbation

# simply do this for local, show how averaging over many give global estimates will give you the global complexity