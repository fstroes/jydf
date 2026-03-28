# Python implementation of Jianming Ye's Degrees of Freedom

This repository provides a Python implementation of **Jianming Ye's (1998)** generalized framework for measuring the "Generalized Degrees of Freedom" (GDF). This method is capable of estimating the complexity of **any** modeling procedure, including non-linear, non-parametric, regularized or non-Gaussian methods even when the model is discontinuous in the data and SURE (Stein's Unbiased Risk Estimate) is invalid.
The estimate consistently estimates the self influence and the resulting degrees of freedom estimate $\hat{\text{df}}$ can therefore be plugged into metrics like Akaike's information criterion (AIC) as the model degrees of freedom.

The estimate is local (dependent on the sample) but by sampling over many simulated datasets a global estimate of the model complexity can also be obtained (see code example).

## Key Features
- **Model Agnostic:** Works with any model wrapper providing a `predict_y_hat` method.
- **Negative Covariate Sampling:** Uses antithetic perturbations ($y + \epsilon$ and $y - \epsilon$) to significantly reduce simulation variance.

## Core Logic
The estimator calculates the sensitivity of the fitted values $\hat{y}$ to perturbations in the observed data $y$. By averaging the covariance between the error and the change in prediction, we arrive at the effective degrees of freedom:

$$GDF = \frac{\sum_{i=1}^{n} \text{cov}(\hat{y}_i, y_i)}{\tau^2}$$

## Implementation Example
The following logic is contained in `code_example.py`. It compares a standard 3rd-degree polynomial against various Nadaraya-Watson (Kernel) regressions, including a "one-step-ahead" predictive model with an optimized bandwidth $h$.

```python
# Refer to code_example.py for the full implementation of wrappers and simulation logic.
# The JYDF class (in jydf.py) handles the perturbation and sensitivity math.

## Code Example Overview

The `code_example.py` script demonstrates how to use the **JYDF** estimator across various modeling techniques—ranging from simple parametric models to adaptive, non-parametric ones. It also showcases how to scale the estimation process using Python's `ThreadPoolExecutor` for parallel processing.

### The Wrapper Pattern
To use any custom model with `JYDF`, you must implement a simple **Wrapper Class**. This architecture ensures that the estimator can treat any model as a "black box" by decoupling the data structure from the estimation logic.

**Requirements for the Wrapper:**
* **Exogenous Data Storage:** All feature data ($X$) should be stored in the class during initialization (`__init__`).
* **Sample-Driven Prediction:** The class must provide a `predict_y_hat(y)` method. This method takes a **new sample** $y$ (the perturbed version), fits or applies the model using the stored $X$ data, and returns the resulting predictions ($\hat{y}$).

---

### Implementation Snippets

#### 1. Simple Parametric Wrapper (Polynomial)
For standard linear models, you can pre-calculate the Moore-Penrose pseudoinverse during initialization to make the thousands of perturbations extremely fast.
```python
class SimplePolyWrapper:
    def __init__(self, x, degree=3):
        # Pre-calculate the 'exogenous' part for speed
        self.X = np.vander(x, degree + 1)
        self.X_pinv = np.linalg.pinv(self.X)

    def predict_y_hat(self, y):
        # Standard OLS: Just a matrix multiplication for each new y
        beta = self.X_pinv @ y
        return self.X @ beta