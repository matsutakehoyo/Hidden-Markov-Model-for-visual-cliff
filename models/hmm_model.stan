/*
Hierarchical Hidden Markov Model for Visual Cliff Behavior Analysis

This Stan model implements a 3-state HMM to analyze mouse movement data from visual cliff experiments.
The model identifies and characterizes discrete behavioral states and their transitions in response 
to visual stimuli and environmental features.

Key Features:
- Three behavioral states (Resting, Exploring, Navigating)
- Structured transition matrix allowing only neighboring state transitions
- Hierarchical structure for individual and group-level parameters
- Regularization via horseshoe priors for transition coefficients
- Spatial influences modeled through sigmoid-transformed covariates

Movement Modeling:
- Step lengths: Gamma distribution with state-specific parameters
- Angular changes: Wrapped Cauchy distribution with dynamically adjusted concentration
- Environmental effects: Edge, cliff, and center influences modeled via sigmoid functions

Hierarchical Structure:
- Group level (experimental conditions)
- Sample level (individual trials)
- Observation level (movement data)

Data Structure:
- n_states: number of behavioral states (3)
- n_obs: total number of observations
- n_trk: number of tracks (continuous movement sequences)
- n_smp: number of samples/trials
- n_grp: number of experimental groups
- l: step lengths
- a: angles
- edge_l, edge_a: distances and angles to edge
- cliff_l: distance to cliff
- cov: environmental covariates matrix [intercept, cliff, edge, center]

Author: Take Matsuyama
Date: December 2024
*/

functions {
  /*
  Wrapped Cauchy log likelihood function
  
  Computes the log likelihood of an angle under the wrapped Cauchy distribution.
  Uses a numerically stable formulation to avoid potential underflow/overflow.
  
  Args:
    y: observed angle (-π to π)
    mu: mean/location parameter (-π to π)
    rho: concentration parameter (0 to 1)
  */
  real wcauchy_log_lik(real y, real mu, real rho) {
      real denom = 1 + rho^2 - 2 * rho * cos(y - mu);
      real lprob = log1m(rho^2) - log(2 * pi() * denom);
      return lprob;
  }

  /*
  Compute log transition probability matrix
  
  Calculates the structured TPM allowing only neighboring state transitions.
  Uses softmax to ensure proper probability normalization for each row.
  
  Args:
    n_states: number of states (3)
    beta_row: matrix of regression coefficients
    cov_vec: vector of covariates
  Returns:
    log_gamma: matrix of log transition probabilities
  */
  matrix compute_log_gamma(int n_states, matrix beta_row, vector cov_vec) {
    matrix[n_states, n_states] log_gamma;
    int index = 1;

    // Compute unnormalized log transition probabilities
    for(i in 1:n_states) { 
      for(j in 1:n_states) {
        if(abs(i-j) <= 1) { //self and adjacent transitions
          log_gamma[i,j] = beta_row[index,] * cov_vec;
          index += 1;
        } else {
          log_gamma[i,j] = negative_infinity();  // Non-adjacent transitions
        }
      }
    }

    // Normalize each row using log_softmax
    for (i in 1:n_states) {
      log_gamma[i, ] = (log_softmax(to_vector(log_gamma[i, ]')))';
    }

    return log_gamma;
  }

  /*
  Compute angle log likelihood with environmental influences
  
  Calculates the log likelihood of an angle considering edge, cliff, and center effects.
  Modifies both the expected angle (lambda) and concentration (rho) based on spatial features.
  
  Args:
    a_t: current angle
    a_t_minus_1: previous angle
    angle_type: movement direction (1: clockwise, 2: counterclockwise)
    edge_a: angle to nearest edge
    cliff_l, edge_l: distances to cliff and edge
    cliff_x, edge_x, center_x: influence thresholds
    cliff_beta, edge_beta, center_beta: influence slopes
    a_rho: base concentration
    cliff_rho, edge_rho, center_rho: feature-specific concentrations
  */
  real angle_log_lik(
      real a_t,
      real a_t_minus_1,
      int angle_type,
      real edge_a,
      real cliff_l,
      real cliff_x,
      real cliff_beta,
      real edge_l,
      real edge_x,
      real edge_beta,
      real center_l,
      real center_x,
      real center_beta,
      real a_rho,
      real cliff_rho, 
      real edge_rho,
      real center_rho) {
    // Calculate feature influences using sigmoid transformations
    real cliff_effect = inv_logit((cliff_l - cliff_x) * cliff_beta);
    real edge_effect = inv_logit((edge_l - edge_x) * edge_beta);
    real center_effect = inv_logit(((30-edge_l) - center_x) * center_beta);

    // Calculate expected angle (lambda) considering edge effect
    real lambda;
    if (angle_type == 1) {
        lambda = (1 - edge_effect) * a_t_minus_1 + edge_effect * (edge_a + pi() / 2);
    } else {
        lambda = (1 - edge_effect) * a_t_minus_1 + edge_effect * (edge_a - pi() / 2);
    }
    // Wrap lambda to [-π, π]
    lambda = atan2(sin(lambda), cos(lambda));

    // Calculate weighted concentration parameter
    real rho = (a_rho + cliff_effect * cliff_rho + edge_effect * edge_rho + center_effect * center_rho) / 
               (1 + edge_effect + cliff_effect + center_effect);

    // Return log likelihood under wrapped Cauchy distribution
    return wcauchy_log_lik(a_t, lambda, rho);
  }
}


*/
data {
  // Dimensions
  int<lower=1> n_states;          // number of states (3)
  int<lower=1> n_obs;            // total number of observations
  int<lower=1> n_trk;            // number of tracks
  int<lower=1> n_smp;            // number of samples/trials
  int<lower=1> n_grp;            // number of groups
  int<lower=1> n_cov;            // number of covariates
  
  // Indexing variables
  int<lower=1, upper=n_trk> trk[n_obs];  // track index for each observation
  int<lower=1, upper=n_smp> smp[n_obs];  // sample index for each observation
  int<lower=1, upper=n_grp> smp2grp[n_smp];  // group index for each sample
  
  // Movement data
  vector[n_obs] l;               // step lengths
  vector[n_obs] a;               // angles
  vector[n_obs] edge_l;          // distance to edge
  vector[n_obs] edge_a;          // angle to edge
  vector[n_obs] cliff_l;         // distance to cliff
  int<lower=1, upper=2> angle_type[n_obs];  // movement direction type
  
  // Environmental covariates
  matrix[n_obs, 1+n_cov] cov;    // covariate matrix [intercept, cliff, edge, center]
  
  // Fixed parameters
  real cliff_x;                  // cliff influence threshold
  real cliff_beta;               // cliff influence slope
}

parameters {
  // ===== TPM Regression Coefficients =====
  // Horseshoe prior structure for regularization
  matrix[3 * n_states - 2, n_cov+1] beta_raw[n_smp];    // individual-level coefficients
  matrix[3 * n_states - 2, n_cov+1] beta0[n_grp];       // group-level coefficients
  vector<lower = 0>[n_cov+1] beta_tau;                  // global shrinkage
  matrix<lower =0>[3 * n_states - 2, n_cov+1] beta_lambda;  // local shrinkage
  vector<lower = 0>[n_cov+1] beta0_sigma;               // group-level variation

  // ===== Step Length Parameters =====
  // Gamma distribution parameters
  positive_ordered[n_states] l_mu[n_smp];    // individual-level means
  vector<lower=0>[n_states] l_va[n_smp];     // individual-level variances
  positive_ordered[n_states] l_mu0[n_grp];   // group-level means
  vector<lower=0>[n_states] l_va0[n_grp];    // group-level variances
  real<lower = 0> l_mu0_sigma[n_states];     // mean variation across groups
  real<lower = 0> l_va0_sigma[n_states];     // variance variation across groups

  // ===== Angular Parameters =====
  // Base movement parameters (wrapped Cauchy)
  vector<lower = 0, upper =1>[n_states] a_rho0[n_grp];    // group-level concentration
  vector<lower = 0, upper =1>[n_states] a_rho[n_smp];     // individual-level concentration
  real<lower = 0> a_rho0_phi;                             // concentration hyperparameter

  // Edge influence parameters
  vector<lower=0, upper=1>[n_states] edge_rho0[n_grp];    // group-level edge effect
  vector<lower=0, upper=1>[n_states] edge_rho[n_smp];     // individual-level edge effect
  real<lower = 0> edge_rho0_phi;                          // edge effect hyperparameter
  
  // Cliff influence parameters
  vector<lower=0, upper=1>[n_states] cliff_rho0[n_grp];   // group-level cliff effect
  vector<lower=0, upper=1>[n_states] cliff_rho[n_smp];    // individual-level cliff effect
  real<lower = 0> cliff_rho0_phi;                         // cliff effect hyperparameter

  // Center influence parameters
  vector<lower=0, upper=1>[n_states] center_rho0[n_grp];  // group-level center effect
  vector<lower=0, upper=1>[n_states] center_rho[n_smp];   // individual-level center effect
  real<lower = 0> center_rho0_phi;                        // center effect hyperparameter

  // ===== Spatial Influence Parameters =====
  // Edge effects
  vector<upper=0>[n_grp] edge_beta0;                      // group-level edge slope
  vector<upper=0>[n_smp] edge_beta;                       // individual-level edge slope
  real<lower=0> edge_beta0_sigma;                         // edge slope variation
  vector<lower=0, upper=30>[n_grp] edge_x0;               // group-level edge threshold
  vector<lower=0, upper=30>[n_smp] edge_x;                // individual-level edge threshold
  real<lower=0> edge_x0_sigma;                            // edge threshold variation

  // Center effects
  vector<upper=0>[n_grp] center_beta0;                    // group-level center slope
  vector<upper=0>[n_smp] center_beta;                     // individual-level center slope
  real<lower=0> center_beta0_sigma;                       // center slope variation
  vector<lower=0, upper=30>[n_grp] center_x0;             // group-level center threshold
  vector<lower=0, upper=30>[n_smp] center_x;              // individual-level center threshold
  real<lower=0> center_x0_sigma;                          // center threshold variation
}

transformed parameters {
  // ===== State Transition Parameters =====
  matrix[n_states, n_states] log_gamma[n_obs];    // log transition probabilities for each observation
  
  // ===== Step Length Distribution Parameters =====
  vector[n_states] l_shape[n_smp];    // shape parameter for gamma distribution
  vector[n_states] l_rate[n_smp];     // rate parameter for gamma distribution
  
  // ===== Regression Coefficients =====
  matrix[3 * n_states - 2, n_cov+1] beta[n_smp];  // individual-level TPM coefficients
  
  // ===== Transformed Covariates =====
  matrix[n_obs, 1+n_cov] cov_trans;   // sigmoid-transformed environmental covariates

  // Calculate individual-level regression coefficients from hierarchical structure
  for (s in 1:n_smp) {
    // Transform TPM coefficients
    for (c in 1:(n_cov+1)) {
      beta[s,,c] = beta0[smp2grp[s],,c] + beta0_sigma[c] * beta_raw[s,,c];
    }
    
    // Calculate gamma distribution parameters for step lengths
    for (n in 1:n_states) {
      l_shape[s,n] = l_mu[s,n] * l_mu[s,n] / (l_va[s,n]);
      l_rate[s,n] = l_mu[s,n] / (l_va[s,n]);
    }
  }

  // Calculate transition probability matrices
  for (t in 1:n_obs) {
    // Transform environmental covariates using sigmoid functions
    for (c in 1:(n_cov+1)) {
      if (c == 1) {
        cov_trans[t,c] = 1;  // intercept term
      } else if (c==2) {
        // Cliff influence
        cov_trans[t,c] = inv_logit((cov[t,c] - cliff_x) * cliff_beta);
      } else if (c==3) {
        // Edge influence
        cov_trans[t,c] = inv_logit((cov[t,c] - edge_x[smp[t]]) * edge_beta[smp[t]]);
      } else if (c==4) {
        // Center influence
        cov_trans[t,c] = inv_logit((cov[t,c] - center_x[smp[t]]) * center_beta[smp[t]]);
      }
    }

    // Compute TPM or set to uniform for missing data
    if (l[t] != -99) {
      log_gamma[t] = compute_log_gamma(n_states, beta[smp[t]], cov_trans[t]');
    } else {
      for(i in 1:n_states) {
        log_gamma[t][i, ] = rep_row_vector(-log(n_states), n_states);
      }
    }
  }
}



model {
  // ===== Prior Distributions =====
  
  // --- TPM Coefficient Priors ---
  // Horseshoe prior structure
  beta_tau ~ cauchy(0,1);  // global shrinkage
  for (c in 1:(n_cov+1)) {
    for (i in 1:(3 * n_states - 2)) {
      beta_lambda[i,c] ~ cauchy(0,1);  // local shrinkage
      beta0[,i,c] ~ normal(0, beta_lambda[i,c] * beta_tau[c]);
    }
  }

  // Individual-level coefficient variations
  beta0_sigma ~ normal(0,1);
  for (i in 1:(3 * n_states - 2)) {
    for (c in 1:(n_cov+1))
      beta_raw[,i,c] ~ std_normal();
  }

  // --- Movement Parameter Priors ---
  // Group level
  for(g in 1:n_grp) {
    // Step length parameters
    l_mu0[g, ] ~ normal(0,2) T[0, ];
    l_va0[g, ] ~ normal(0,.2) T[0, ];
    
    // Angular concentration parameters
    a_rho0[g, ] ~ beta(2,2);      // base angle distribution
    cliff_rho0[g, ] ~ beta(2,2);  // near cliff
    edge_rho0[g, ] ~ beta(2,2);   // near wall  
    center_rho0[g, ] ~ beta(2,2); // near center  
  
    // Spatial influence parameters
    edge_beta0[g] ~ normal(0,10) T[, 0];
    center_beta0[g] ~ normal(0,10) T[, 0];
    target += beta_lpdf(edge_x0[g] / 30 | 1, 9) - log(30);
    target += beta_lpdf(center_x0[g] / 30 | 1, 9) - log(30);
  } 

  // Hyperparameters
  l_mu0_sigma ~ normal(0,.5) T[0, ];
  l_va0_sigma ~ normal(0,.5) T[0, ];
  
  // Concentration hyperparameters
  a_rho0_phi ~ gamma(2, 0.5);     // base angle
  cliff_rho0_phi ~ gamma(2, 0.5);  // cliff effect
  edge_rho0_phi ~ gamma(2, 0.5);   // edge effect  
  center_rho0_phi ~ gamma(2, 0.5); // center effect  
  
  // Spatial influence variations
  edge_x0_sigma ~ normal(0,1) T[0, ];    
  edge_beta0_sigma ~ normal(0,1) T[0, ];  
  center_x0_sigma ~ normal(0,1) T[0, ];    
  center_beta0_sigma ~ normal(0,1) T[0, ];  

  // --- Individual Level Parameters ---
  for (s in 1:n_smp) {
    edge_beta[s] ~ normal(edge_beta0[smp2grp[s]], edge_beta0_sigma);   
    edge_x[s] ~ normal(edge_x0[smp2grp[s]], edge_x0_sigma);   
    center_beta[s] ~ normal(center_beta0[smp2grp[s]], center_beta0_sigma);   
    center_x[s] ~ normal(center_x0[smp2grp[s]], center_x0_sigma);       
  }

  for (state in 1:n_states) {
    for (s in 1:n_smp) {
      // Step length parameters
      l_mu[s,state] ~ normal(l_mu0[smp2grp[s], state], l_mu0_sigma[state]);
      l_va[s,state] ~ normal(l_va0[smp2grp[s], state], l_va0_sigma[state]);

      // Angular concentration parameters
      a_rho[s,state] ~ beta(a_rho0[smp2grp[s], state] * a_rho0_phi, 
                           (1 - a_rho0[smp2grp[s], state]) * a_rho0_phi);
      edge_rho[s,state] ~ beta(edge_rho0[smp2grp[s], state] * edge_rho0_phi, 
                              (1 - edge_rho0[smp2grp[s], state]) * edge_rho0_phi);
      cliff_rho[s,state] ~ beta(cliff_rho0[smp2grp[s], state] * cliff_rho0_phi, 
                               (1 - cliff_rho0[smp2grp[s], state]) * cliff_rho0_phi);
      center_rho[s,state] ~ beta(center_rho0[smp2grp[s], state] * center_rho0_phi,
                                (1 - center_rho0[smp2grp[s], state]) * center_rho0_phi);
    }
  }

  // ===== Likelihood Calculation =====
  vector[n_states] logp;      // forward variables
  vector[n_states] logptemp;  // temporary storage for forward algorithm

  // Forward algorithm implementation for HMM likelihood
  for (t in 1:n_obs) {
    // Reset forward variables at the start of each track
    if(t == 1 || trk[t] != trk[t - 1]) {
      logp = rep_vector(-log(n_states), n_states);  // initialize with uniform state distribution
    }

    // Calculate forward probabilities for each state
    for (state in 1:n_states) {
      // Transition probability
      logptemp[state] = log_sum_exp(logp + log_gamma[t][ , state]);
      
      // Emission probability: step length
      if (l[t]!=-99) {
        logptemp[state] = logptemp[state] + 
          gamma_lpdf(l[t] | l_shape[smp[t],state], l_rate[smp[t],state]);
      }
      
      // Emission probability: angle
      if (t >= 2 && trk[t] == trk[t-1] && a[t]!=-99 && a[t-1]!=-99) {
        logptemp[state] = logptemp[state] +
          angle_log_lik(
            a[t], a[t-1], angle_type[t], edge_a[t], 
            cliff_l[t],  6, -1, 
            edge_l[t],  edge_x[smp[t]], edge_beta[smp[t]], 
            (30-edge_l[t]),  center_x[smp[t]], center_beta[smp[t]], 
            a_rho[smp[t],state], cliff_rho[smp[t],state], 
            edge_rho[smp[t],state], center_rho[smp[t],state]);
      }
    }
    
    // Update forward variables
    logp = logptemp;

    // Add log probability at the end of each track
    if(t == n_obs || trk[t+1] != trk[t]) {
      target += log_sum_exp(logp);  // marginalize over final states
    }
  }
}

generated quantities {
  // Output variables
  int<lower=1, upper=n_states> z_star[n_obs];  // most likely state sequence
  real logp_z_star;                            // log probability of most likely sequence
  vector[n_obs] log_lik;                       // observation-wise log likelihoods

  {
    // Viterbi algorithm workspace
    int back_ptr[n_obs, n_states];             // backpointers for state reconstruction
    real best_logp[n_obs, n_states];           // log probabilities for best paths

    // ===== Forward Pass =====
    for (t in 1:n_obs) {
      // Compute best_logp for each state at current timestep
      for (state in 1:n_states) {
        // Handle missing data case
        if (l[t] == -99) {
          best_logp[t, state] = -log(n_states);  // uniform distribution
          back_ptr[t, state] = 0;                // no meaningful backpointer
        } else {
          // Start of new track or first observation
          if (t == 1 || trk[t] != trk[t - 1]) {
            best_logp[t, state] = -log(n_states);  // uniform initial distribution
            back_ptr[t, state] = 0;
            // Add emission probability for step length
            best_logp[t, state] += gamma_lpdf(l[t] | l_shape[smp[t],state], l_rate[smp[t],state]);
          } else {
            // Initialize for finding best previous state
            best_logp[t, state] = negative_infinity();
            back_ptr[t, state] = 0;

            // Consider only allowed transitions (neighboring states)
            int min_prev_state = max(1, state - 1);
            int max_prev_state = min(n_states, state + 1);

            // Find best previous state
            for (j in min_prev_state:max_prev_state) {
              if (l[t - 1] != -99 && trk[t] == trk[t - 1]) {
                real logp = best_logp[t - 1, j];
                // Add transition probability
                logp += log_gamma[t][j, state];
                // Add emission probability for step length
                logp += gamma_lpdf(l[t] | l_shape[smp[t],state], l_rate[smp[t],state]);
                // Add emission probability for angle if available
                if (a[t]!=-99 && a[t-1]!=-99) {
                  logp += angle_log_lik(
                    a[t], a[t-1], angle_type[t], edge_a[t], 
                    cliff_l[t],  6, -1, 
                    edge_l[t], edge_x[smp[t]], edge_beta[smp[t]], 
                    (30-edge_l[t]),  center_x[smp[t]], center_beta[smp[t]], 
                    a_rho[smp[t],state], cliff_rho[smp[t],state], 
                    edge_rho[smp[t],state], center_rho[smp[t],state]);
                }
                // Update if this path is better
                if (logp > best_logp[t, state]) {
                  back_ptr[t, state] = j;
                  best_logp[t, state] = logp;
                }
              }
            }
          }
        }
      }
    }

    // ===== Backward Pass =====
    {
      // Find best final state
      real max_logp = negative_infinity();
      int best_last_state = 1;
      for (state in 1:n_states) {
        if (best_logp[n_obs, state] > max_logp) {
          max_logp = best_logp[n_obs, state];
          best_last_state = state;
        }
      }
      z_star[n_obs] = best_last_state;
      logp_z_star = max_logp;

      // Backtrack to reconstruct state sequence
      int t = n_obs;
      while (t >= 2) {
        int current_state = z_star[t];
        int prev_state = back_ptr[t, current_state];

        if (prev_state == 0 || trk[t] != trk[t - 1]) {
          // Start of track or no valid previous state
          real max_prev_logp = negative_infinity();
          int best_prev_state = 1;
          for (state in 1:n_states) {
            if (best_logp[t - 1, state] > max_prev_logp) {
              max_prev_logp = best_logp[t - 1, state];
              best_prev_state = state;
            }
          }
          z_star[t - 1] = best_prev_state;
        } else {
          z_star[t - 1] = prev_state;
        }
        t -= 1;
      }
    }

    // ===== Calculate Observation-wise Log-likelihood =====
    for (t in 1:n_obs) {
      int state = z_star[t];
      log_lik[t] = 0;
      
      // Step length log-likelihood
      if (l[t] != -99)
        log_lik[t] += gamma_lpdf(l[t] | l_shape[smp[t],state], l_rate[smp[t],state]);
      
      // Angle log-likelihood
      if (t >= 2 && trk[t] == trk[t - 1] && a[t] != -99 && a[t - 1] != -99) {
        log_lik[t] += angle_log_lik(
          a[t], a[t-1], angle_type[t], edge_a[t], 
          cliff_l[t],  6, -1, 
          edge_l[t], edge_x[smp[t]], edge_beta[smp[t]], 
          (30-edge_l[t]),  center_x[smp[t]], center_beta[smp[t]], 
          a_rho[smp[t],state], cliff_rho[smp[t],state], 
          edge_rho[smp[t],state], center_rho[smp[t],state]);
      }

      // Transition log-likelihood
      if (t >= 2 && trk[t] == trk[t - 1] && l[t] != -99) {
        int prev_state = z_star[t - 1];
        if (abs(prev_state - state) <= 1) {
          log_lik[t] += log_gamma[t][prev_state, state];
        } else {
          log_lik[t] += negative_infinity();
        }
      }
    }
  }
}
