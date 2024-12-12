/*
Hierarchical Lognormal Mixture Model

This Stan model implements a hierarchical mixture of lognormal distributions 
for analyzing step length distributions. The model includes group-level 
variations in mixture components and proportions.

Key Features:
- Multiple mixture components with lognormal distributions
- Hierarchical structure for group-level parameters
- Ordered means to ensure identifiability
- Dirichlet prior for mixing proportions

Data Structure:
- n_obs: number of observations
- n_grp: number of groups
- k: number of mixture components
- y: observed values (step lengths)
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
  ordered[k] log_mu[n_grp];              // group-level log means (ordered for identifiability)
  vector<lower=0>[k] log_sd[n_grp];      // group-level log standard deviations
  
  // Population-level Parameters
  real mu0[k];                           // population-level means
  real<lower=0> mu0_sd;                  // variation in means
  real<lower=0> sd0[k];                  // population-level standard deviations
  real<lower=0> sd0_sd;                  // variation in standard deviations
  
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
  
  // Group-level Priors
  for (g in 1:n_grp) {
    theta[g] ~ dirichlet(alpha);
    log_theta[g] = log(theta[g]);
    
    for (c in 1:k) {
      log_mu[g][c] ~ normal(mu0[c], mu0_sd);
      log_sd[g][c] ~ normal(sd0[c], mu0_sd);          
    }
  }
  
  // Population-level Priors
  for (c in 1:k) {
    sd0[c] ~ normal(0, 1);
  }
  mu0_sd ~ normal(0, 1);
  sd0_sd ~ normal(0, 1);
  
  // Hyperpriors for Mixing Proportions
  phi ~ beta(1, 1);
  kappa ~ normal(0, 5);  // allows substantial variation in Dirichlet parameters
  
  // Likelihood
  for (i in 1:n_obs) {
    vector[k] lps = log_theta[grp[i]];
    for (c in 1:k) {
      lps[c] += lognormal_lpdf(y[i] | log_mu[grp[i]][c], log_sd[grp[i]][c]);
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
    ytilde[i] = lognormal_rng(log_mu[grp[i]][comp], log_sd[grp[i]][comp]);
    
    // Calculate pointwise log-likelihood
    for (c in 1:k) {
      lps[c] = log(theta[grp[i]][c]) + 
               lognormal_lpdf(y[i] | log_mu[grp[i]][c], log_sd[grp[i]][c]);
    }
    log_lik[i] = log_sum_exp(lps);
  }
}
