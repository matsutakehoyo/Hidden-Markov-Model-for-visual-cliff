/*
Hierarchical von Mises Mixture Model

This Stan model implements a hierarchical mixture of von Mises distributions 
for analyzing angular data (turning angles). The model includes group-level 
variations in mixture proportions with fixed component means.

Key Features:
- Multiple mixture components with von Mises distributions
- Hierarchical structure for group-level mixing proportions
- Fixed component means
- Ordered concentration parameters for identifiability
- Dirichlet prior for mixing proportions

Data Structure:
- n_obs: number of observations
- n_grp: number of groups
- k: number of mixture components
- y: observed angles in radians [-π, π]
- mu: fixed component means
*/

data {
  // Dimensions
  int<lower=1> n_obs;                    // number of observations
  int<lower=1> n_grp;                    // number of groups
  int<lower=2> k;                        // number of mixture components
  
  // Data
  real y[n_obs];                         // observed angles
  int<lower=1, upper=n_grp> grp[n_obs];  // group indicators
  
  // Fixed Parameters
  real<lower=-pi(), upper=pi()> mu[k];   // component means
}

parameters {
  // Mixture Proportions
  simplex[k] theta[n_grp];               // group-level mixing proportions
  
  // Distribution Parameters
  positive_ordered[k] kappa_angle;        // concentration parameters (ordered for identifiability)
  
  // Hierarchical Parameters for Mixing Proportions
  simplex[k] phi;                        // population-level proportions
  real<lower=0> kappa;                   // concentration parameter
}

transformed parameters {
  vector[k] theta0;                      // population-level mixing proportions
  vector[k] alpha = kappa * phi;         // Dirichlet parameters
  
  for (c in 1:k) {
    theta0[c] = alpha[c]/sum(alpha);
  }
}

model {
  vector[k] log_theta[n_grp];
  
  // Hyperpriors for Mixing Proportions
  phi ~ beta(1, 1);
  kappa ~ normal(0, 5);  // allows substantial variation in Dirichlet parameters
  
  // Group-level Priors
  for (g in 1:n_grp) {
    theta[g] ~ dirichlet(alpha);
    log_theta[g] = log(theta[g]);
  }
  
  // Likelihood
  for (i in 1:n_obs) {
    vector[k] lps = log_theta[grp[i]];
    for (c in 1:k) {
      lps[c] += von_mises_lpdf(y[i] | mu[c], kappa_angle[c]);
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
    ytilde[i] = von_mises_rng(mu[comp], kappa_angle[comp]);
    
    // Calculate pointwise log-likelihood
    for (c in 1:k) {
      lps[c] = log(theta[grp[i]][c]) + 
               von_mises_lpdf(y[i] | mu[c], kappa_angle[c]);
    }
    log_lik[i] = log_sum_exp(lps);
  }
}
