/*
Hierarchical Wrapped Cauchy Mixture Model

This Stan model implements a hierarchical mixture of wrapped Cauchy distributions 
for analyzing angular data (turning angles). The model includes group-level 
variations in mixture proportions with fixed component means.

Key Features:
- Multiple mixture components with wrapped Cauchy distributions
- Hierarchical structure for group-level mixing proportions
- Fixed component means (0 or π)
- Ordered concentration parameters for identifiability
- Custom wrapped Cauchy implementation

Data Structure:
- n_obs: number of observations
- n_grp: number of groups
- k: number of mixture components
- y: observed angles in radians [-π, π]
*/

functions {
  /*
  Wrapped Cauchy log likelihood
  
  Args:
    y: observed angle (-π to π)
    mu: mean/location parameter (-π to π)
    rho: concentration parameter (0 to 1)
  */
  real wcauchy_log_lik(real y, real mu, real rho) {
    real lprob = log(1/(2*pi()) * (1-rho^2) / (1+rho^2-2*rho*cos(y-mu))); 
    return(lprob);
  }
    
  /*
  Generate random draws from wrapped Cauchy distribution
  
  Args:
    mu: mean/location parameter
    rho: concentration parameter
  */
  real wcauchy_rng(real mu, real rho) {
    real value; 
    if (rho == 0) {
      value = uniform_rng(-1, 1 * pi());
    } else {
      value = cauchy_rng(mu, -log(rho));      
    }
    return(value);
  }
}

data {
  // Dimensions
  int<lower=1> n_obs;                    // number of observations
  int<lower=1> n_grp;                    // number of groups
  int<lower=2> k;                        // number of mixture components
  
  // Data
  real y[n_obs];                         // observed angles
  int<lower=1, upper=n_grp> grp[n_obs];  // group indicators
}

parameters {
  // Mixture Proportions
  simplex[k] theta[n_grp];               // group-level mixing proportions
  simplex[k] theta0;                     // population-level mixing proportions
  
  // Distribution Parameters
  ordered[k] rho_raw;                    // unconstrained concentration parameters
}

transformed parameters {
  vector[k] mu;                          // component means
  vector[k] rho;                         // concentration parameters
  
  // Fix means to 0 for all components
  for (c in 1:k) {
    mu[c] = 0;
  }
  
  // Transform concentration parameters to (0,1)
  rho = inv_logit(rho_raw);
}

model {
  vector[k] log_theta[n_grp];
  
  // Priors
  rho_raw ~ normal(0, 1);
  
  // Group-level Priors
  for (g in 1:n_grp) {
    theta[g] ~ dirichlet(theta0);
    log_theta[g] = log(theta[g]);
  }
  
  // Likelihood
  for (i in 1:n_obs) {
    vector[k] lps = log_theta[grp[i]];
    for (c in 1:k) {
      lps[c] += wcauchy_log_lik(y[i], mu[c], rho[c]);
    }
    target += log_sum_exp(lps);
  } 
}

generated quantities {
  vector[n_obs] ytilde;    // posterior predictive samples
  vector[n_obs] log_lik;   // pointwise log-likelihood
  
  for (i in 1:n_obs) {
    vector[k] lps;
    // Generate posterior predictive samples
    int comp = categorical_rng(theta[grp[i]]);
    ytilde[i] = wcauchy_rng(mu[comp], rho[comp]);
    
    // Calculate pointwise log-likelihood
    for (c in 1:k) {
      lps[c] = log(theta[grp[i]][c]) + wcauchy_log_lik(y[i], mu[c], rho[c]);
    }
    log_lik[i] = log_sum_exp(lps);
  }
}
