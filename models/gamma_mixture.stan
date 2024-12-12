/*
Hierarchical Gamma Mixture Model

This Stan model implements a hierarchical mixture of gamma distributions 
for analyzing step length distributions. The model includes group-level 
variations in mixture components and proportions.

Key Features:
- Multiple mixture components with gamma distributions
- Hierarchical structure for group-level parameters
- Ordered means to ensure identifiability
- Dirichlet prior for mixing proportions

Data Structure:
- n_obs: number of observations
- n_grp: number of groups
- k: number of mixture components
- y: observed values
*/

data {
  // Dimensions
  int<lower=1> n_obs;                    // number of observations
  int<lower=1> n_grp;                    // number of groups
  int<lower=2> k;                        // number of mixture components
  
  // Data
  real y[n_obs];                         // observations
  int<lower=1, upper=n_grp> grp[n_obs];  // group indicators
}

parameters {
  // Mixture Proportions
  simplex[k] theta[n_grp];               // group-level mixing proportions
  
  // Distribution Parameters
  real<lower=0> mu0[k];                  // population-level means
  real<lower=0> mu0_sd;                  // variation in means
  positive_ordered[k] mu[n_grp];         // group-level means
  
  real<lower=0> va0[k];                  // population-level variances
  real<lower=0> va0_sd;                  // variation in variances
  vector<lower=0>[k] va[n_grp];          // group-level variances
  
  // Hierarchical Parameters for Mixing Proportions
  simplex[k] phi;                        // population-level proportions
  real<lower=0> conc;                    // concentration parameter
}

transformed parameters {
  // Population-level mixing proportions
  simplex[k] theta0;
  vector[k] alpha = conc * phi;
  
  for (c in 1:k) {
    theta0[c] = alpha[c]/sum(alpha);
  }
}

model {
  vector[k] log_theta[n_grp];
  
  // Group-level Priors
  for (g in 1:n_grp) {
    theta[g] ~ dirichlet(alpha);
    for (c in 1:k) {
      mu[g][c] ~ normal(mu0[c], mu0_sd);
      va[g][c] ~ normal(va0[c], va0_sd);
    }
    log_theta[g] = log(theta[g]);
  }
  
  // Population-level Priors
  mu0 ~ normal(0, 2);
  va0 ~ normal(0, 1);
  mu0_sd ~ normal(0, .5);
  va0_sd ~ normal(0, .5);
  
  // Hyperpriors
  phi ~ beta(2, 2);
  conc ~ normal(0, 5);
  
  // Likelihood
  for (i in 1:n_obs) {
    vector[k] lps = log_theta[grp[i]];
    for (c in 1:k) {
      lps[c] += gamma_lpdf(y[i] | 
                          mu[grp[i]][c] * mu[grp[i]][c] / va[grp[i]][c],  // shape
                          mu[grp[i]][c] / va[grp[i]][c]);                 // rate
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
    ytilde[i] = gamma_rng(mu[grp[i]][comp] * mu[grp[i]][comp] / va[grp[i]][comp],
                         mu[grp[i]][comp] / va[grp[i]][comp]);
    
    // Calculate pointwise log-likelihood
    for (c in 1:k) {
      lps[c] = log(theta[grp[i]][c]) + 
               gamma_lpdf(y[i] | mu[grp[i]][c] * mu[grp[i]][c] / va[grp[i]][c],
                                 mu[grp[i]][c] / va[grp[i]][c]);
    }
    log_lik[i] = log_sum_exp(lps);
  }
}
