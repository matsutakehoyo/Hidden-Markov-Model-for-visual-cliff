/*
Hierarchical Hidden Markov Model for Visual Cliff Behavior Analysis
===============================================================

This Stan model implements a hierarchical n-state Hidden Markov Model (HMM) to analyze 
animal movement patterns in visual cliff experiments. The model characterizes discrete 
behavioral states and their transitions in response to environmental features (edge, cliff, and center).

Model Structure
--------------
- Hierarchical organization with three levels:
  * Group level (experimental conditions)
  * Individual level (trials/samples)
  * Observation level (movement data)

Key Features
-----------
1. State Transitions:
   - Structured transition matrix allowing only neighboring state transitions
   - Environmental effects modeled through sigmoid-transformed covariates
   - Horseshoe prior for transition coefficients to handle sparsity

2. Movement Parameters:
   - Step lengths: Gamma distribution with state-specific parameters
   - Turning angles: Wrapped Cauchy distribution with dynamic concentration
   - Minimum separation between state means to avoid state collapse

3. Environmental Influences:
   - Edge effects: wall-following behavior
   - Cliff effects: avoidance/approach patterns
   - Center effects: open field behavior
   - Each effect modeled via sigmoid functions with learned thresholds

Input Data Structure
------------------
n_states : int
    Number of behavioral states (fixed at 3)
n_obs : int
    Total number of observations across all tracks
n_trk : int
    Number of continuous movement tracks
n_smp : int
    Number of individual trials/samples
n_grp : int
    Number of experimental groups
l : vector[n_obs]
    Step lengths (-99 for missing data)
a : vector[n_obs]
    Turning angles in radians (-99 for missing data)
edge_l : vector[n_obs]
    Distance to nearest edge
edge_a : vector[n_obs]
    Angle to nearest edge
cliff_l : vector[n_obs]
    Distance to cliff edge
cov : matrix[n_obs, 1+n_cov]
    Environmental covariates [intercept, cliff, edge, center]

Author: Take Matsuyama
Date: December 2024
License: MIT
*/

functions {
  /*
  Compute log likelihood for wrapped Cauchy distribution
  
  The wrapped Cauchy distribution is used for modeling angular data with varying
  concentration around a mean direction. This implementation uses a numerically
  stable formulation to prevent underflow/overflow.
  
  Parameters
  ----------
  y : real
      Observed angle in radians (-π to π)
  mu : real
      Mean/location parameter (-π to π)
  rho : real
      Concentration parameter (0 to 1), higher values = more concentrated
  
  Returns
  -------
  real
      Log likelihood value
  */
  real wcauchy_log_lik(real y, real mu, real rho) {
      real denom = 1 + rho^2 - 2 * rho * cos(y - mu);
      real lprob = log1m(rho^2) - log(2 * pi() * denom);
      return lprob;
  }

  /*
  Compute log transition probability matrix
  
  Calculates structured transition probabilities between states, allowing only
  transitions between neighboring states. Uses softmax for proper probability
  normalization.
  
  Parameters
  ----------
  n_states : int
      Number of states (3)
  beta_row : matrix
      Matrix of regression coefficients for transitions
  cov_vec : vector
      Vector of environmental covariates
  
  Returns
  -------
  matrix
      Log transition probability matrix
  */
  matrix compute_log_gamma(int n_states, matrix beta_row, vector cov_vec) {
    matrix[n_states, n_states] log_gamma;
    int index = 1;

    // Compute unnormalized log probabilities
    for(i in 1:n_states) { 
      for(j in 1:n_states) {
        if(abs(i-j) <= 1) {  // Allow only self and adjacent transitions
          log_gamma[i,j] = beta_row[index,] * cov_vec;
          index += 1;
        } else {
          log_gamma[i,j] = negative_infinity();
        }
      }
    }

    // Row-wise normalization
    for (i in 1:n_states) {
      log_gamma[i, ] = (log_softmax(to_vector(log_gamma[i, ]')))';
    }

    return log_gamma;
  }

  /*
  Compute angle log likelihood with environmental influences
  
  Calculates the log likelihood of an observed angle considering multiple
  environmental effects (edge, cliff, center). Both the expected angle and
  concentration parameter are modified based on spatial features.
  
  Parameters
  ----------
  a_t : real
      Current angle
  a_t_minus_1 : real
      Previous angle
  angle_type : int
      Movement direction (1: clockwise, 2: counterclockwise)
  edge_a : real
      Angle to nearest edge
  cliff_l, edge_l : real
      Distances to cliff and edge
  cliff_x, edge_x, center_x : real
      Influence thresholds
  cliff_beta, edge_beta, center_beta : real
      Influence slopes
  a_rho : real
      Base angular concentration
  cliff_rho, edge_rho, center_rho : real
      Feature-specific concentrations
  
  Returns
  -------
  real
      Log likelihood value
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
    // Calculate environmental effects using sigmoid functions
    real cliff_effect = inv_logit((cliff_l - cliff_x) * cliff_beta);
    real edge_effect = inv_logit((edge_l - edge_x) * edge_beta);
    real center_effect = inv_logit(((30-edge_l) - center_x) * center_beta);

    // Calculate expected angle based on edge following behavior
    real lambda;
    if (angle_type == 1) {
        lambda = (1 - edge_effect) * a_t_minus_1 + edge_effect * (edge_a + pi() / 2);
    } else {
        lambda = (1 - edge_effect) * a_t_minus_1 + edge_effect * (edge_a - pi() / 2);
    }
    // Ensure angle stays in [-π, π]
    lambda = atan2(sin(lambda), cos(lambda));

    // Calculate weighted concentration parameter
    real rho = (a_rho + cliff_effect * cliff_rho + edge_effect * edge_rho + 
                center_effect * center_rho) / 
               (1 + edge_effect + cliff_effect + center_effect);

    return wcauchy_log_lik(a_t, lambda, rho);
  }
}


data {
  //=============================================================================
  // Dimensions and Indexing Variables
  //=============================================================================
  
  // Core dimensions
  int<lower=1> n_states;     // Number of behavioral states (fixed at 3)
  int<lower=1> n_obs;        // Total number of observations
  int<lower=1> n_trk;        // Number of continuous movement tracks
  int<lower=1> n_smp;        // Number of samples/trials
  int<lower=1> n_grp;        // Number of experimental groups
  int<lower=1> n_cov;        // Number of environmental covariates
  
  // Indexing arrays for hierarchical structure
  int<lower=1, upper=n_trk> trk[n_obs];    // Track index for each observation
  int<lower=1, upper=n_smp> smp[n_obs];    // Sample/trial index for each observation
  int<lower=1, upper=n_grp> smp2grp[n_smp]; // Group index for each sample
  
  //=============================================================================
  // Movement Data
  //=============================================================================
  
  vector[n_obs] l;           // Step lengths (-99 for missing values)
  vector[n_obs] a;           // Turning angles in radians (-99 for missing)
  vector[n_obs] edge_l;      // Distance to nearest arena edge
  vector[n_obs] edge_a;      // Angle to nearest edge
  vector[n_obs] cliff_l;     // Distance to visual cliff edge
  
  // Movement direction type for angle calculations
  // 1: clockwise movement relative to edge
  // 2: counterclockwise movement relative to edge
  int<lower=1, upper=2> angle_type[n_obs];
  
  // Minimum separation between state means to ensure identifiability
  real<lower=0> min_mu_sep;
  
  //=============================================================================
  // Environmental Data
  //=============================================================================
  
  // Environmental covariate matrix [intercept, cliff, edge, center]
  // Used to modulate transition probabilities based on spatial features
  matrix[n_obs, 1+n_cov] cov;
  
  // Fixed parameters for cliff influence
  real cliff_x;              // Distance threshold for cliff effect
  real cliff_beta;           // Slope parameter for cliff sigmoid
}

/*
Model Implementation Notes
------------------------
1. Missing Data Handling:
   - Missing values are coded as -99
   - TPM calculation handles missing data by using uniform probabilities
   - Forward algorithm appropriately skips likelihood calculation for missing values

2. State Identifiability:
   - States are ordered by mean step length (positive_ordered constraint)
   - Minimum separation between state means is enforced
   - Horseshoe prior helps with parameter identification

3. Numerical Stability:
   - Log space calculations used throughout
   - Stable implementation of wrapped Cauchy distribution
   - Careful handling of edge cases in forward algorithm
*/

parameters {
  //=============================================================================
  // State Transition Parameters
  //=============================================================================
  
  // === Transition Probability Matrix (TPM) Regression Coefficients ===
  // Horseshoe prior structure for sparse regression
  matrix[3 * n_states - 2, n_cov+1] beta_raw[n_smp];    // Individual-level
  matrix[3 * n_states - 2, n_cov+1] beta0[n_grp];       // Group-level
  vector<lower=0>[n_cov+1] beta_tau;                    // Global shrinkage
  matrix<lower=0>[3 * n_states - 2, n_cov+1] beta_lambda; // Local shrinkage
  vector<lower=0>[n_cov+1] beta0_sigma;                 // Group variation
  
  //=============================================================================
  // Movement Parameters
  //=============================================================================
  
  // === Step Length Parameters ===
  // Gamma distribution parameters for each state
  positive_ordered[n_states] l_mu[n_smp];     // Individual means
  vector<lower=0>[n_states] l_va[n_smp];      // Individual variances
  positive_ordered[n_states] l_mu0[n_grp];    // Group means
  vector<lower=0>[n_states] l_va0[n_grp];     // Group variances
  real<lower=0> l_mu0_sigma[n_states];        // Mean variation
  real<lower=0> l_va0_sigma[n_states];        // Variance variation
  
  // === Angular Parameters ===
  // Base movement parameters (wrapped Cauchy distribution)
  vector<lower=0, upper=1>[n_states] a_rho0[n_grp];     // Group concentration
  vector<lower=0, upper=1>[n_states] a_rho[n_smp];      // Individual concentration
  real<lower=0> a_rho0_phi;                             // Concentration hyper-param
  
  //=============================================================================
  // Environmental Influence Parameters
  //=============================================================================
  
  // === Edge Effects ===
  vector<lower=0, upper=1>[n_states] edge_rho0[n_grp];  // Group effect
  vector<lower=0, upper=1>[n_states] edge_rho[n_smp];   // Individual effect
  real<lower=0> edge_rho0_phi;                          // Effect hyper-param
  vector<upper=0>[n_grp] edge_beta0;                    // Group slope
  vector<upper=0>[n_smp] edge_beta;                     // Individual slope
  real<lower=0> edge_beta0_sigma;                       // Slope variation
  vector<lower=0, upper=30>[n_grp] edge_x0;            // Group threshold
  vector<lower=0, upper=30>[n_smp] edge_x;             // Individual threshold
  real<lower=0> edge_x0_sigma;                         // Threshold variation
  
  // === Cliff Effects ===
  vector<lower=0, upper=1>[n_states] cliff_rho0[n_grp]; // Group effect
  vector<lower=0, upper=1>[n_states] cliff_rho[n_smp];  // Individual effect
  real<lower=0> cliff_rho0_phi;                         // Effect hyper-param
  
  // === Center Effects ===
  vector<lower=0, upper=1>[n_states] center_rho0[n_grp]; // Group effect
  vector<lower=0, upper=1>[n_states] center_rho[n_smp];  // Individual effect
  real<lower=0> center_rho0_phi;                         // Effect hyper-param
  vector<upper=0>[n_grp] center_beta0;                   // Group slope
  vector<upper=0>[n_smp] center_beta;                    // Individual slope
  real<lower=0> center_beta0_sigma;                      // Slope variation
  vector<lower=0, upper=30>[n_grp] center_x0;           // Group threshold
  vector<lower=0, upper=30>[n_smp] center_x;            // Individual threshold
  real<lower=0> center_x0_sigma;                        // Threshold variation
}


transformed parameters {
  //=============================================================================
  // Transition Probability Parameters
  //=============================================================================
  
  // Log transition probability matrices for each observation
  // Dimensions: [n_obs, n_states, n_states]
  // Each matrix contains log probabilities of transitioning between states
  matrix[n_states, n_states] log_gamma[n_obs];    
  
  //=============================================================================
  // Movement Distribution Parameters
  //=============================================================================
  
  // Gamma distribution parameters for step lengths, derived from mean/variance
  vector[n_states] l_shape[n_smp];    // Shape parameter for each state and sample
  vector[n_states] l_rate[n_smp];     // Rate parameter for each state and sample
  
  // Individual-level transition matrix coefficients
  // Dimensions: [n_smp, 3*n_states-2, n_cov+1]
  // Only adjacent state transitions are allowed
  matrix[3 * n_states - 2, n_cov+1] beta[n_smp];
  
  // Transformed environmental covariates incorporating spatial effects
  // Each covariate is transformed through a sigmoid function
  matrix[n_obs, 1+n_cov] cov_trans;
  
  //=============================================================================
  // Parameter Transformations
  //=============================================================================
  
  // Calculate individual-level parameters from hierarchical structure
  for (s in 1:n_smp) {
    // Transform TPM coefficients with regularization
    // Combines group-level effects with individual variation
    for (c in 1:(n_cov+1)) {
      beta[s,,c] = beta0[smp2grp[s],,c] + beta0_sigma[c] * beta_raw[s,,c];
    }
    
    // Calculate gamma distribution parameters for each state
    // Converts from mean/variance parameterization to shape/rate
    for (n in 1:n_states) {
      // Shape = μ²/σ², Rate = μ/σ²
      l_shape[s,n] = l_mu[s,n] * l_mu[s,n] / (l_va[s,n]);
      l_rate[s,n] = l_mu[s,n] / (l_va[s,n]);
    }
  }

  // Calculate transition probability matrices for each observation
  for (t in 1:n_obs) {
    // Transform environmental covariates using sigmoid functions
    for (c in 1:(n_cov+1)) {
      if (c == 1) {
        cov_trans[t,c] = 1;  // Intercept term
      } else if (c==2) {
        // Cliff influence: fixed threshold and slope parameters
        cov_trans[t,c] = inv_logit((cov[t,c] - cliff_x) * cliff_beta);
      } else if (c==3) {
        // Edge influence: learned individual-specific parameters
        cov_trans[t,c] = inv_logit((cov[t,c] - edge_x[smp[t]]) * 
                                  edge_beta[smp[t]]);
      } else if (c==4) {
        // Center influence: learned individual-specific parameters
        // Distance from edge used as proxy for center proximity
        cov_trans[t,c] = inv_logit((cov[t,c] - center_x[smp[t]]) * 
                                  center_beta[smp[t]]);
      }
    }

    // Compute transition probability matrix or set to uniform for missing data
    if (l[t] != -99) {
      // Calculate structured TPM allowing only neighboring transitions
      log_gamma[t] = compute_log_gamma(n_states, beta[smp[t]], cov_trans[t]');
    } else {
      // Use uniform distribution for missing data points
      // This ensures proper probability flow in forward algorithm
      for(i in 1:n_states) {
        log_gamma[t][i, ] = rep_row_vector(-log(n_states), n_states);
      }
    }
  }
}



model {
  //=============================================================================
  // Prior Distributions
  //=============================================================================
  
  // === TPM Coefficient Priors ===
  // Horseshoe prior structure for sparse regression
  // Global shrinkage parameter
  beta_tau ~ cauchy(0, 1);  
  
  // Local shrinkage parameters and coefficient priors
  for (c in 1:(n_cov+1)) {
    for (i in 1:(3 * n_states - 2)) {
      // Local shrinkage with half-Cauchy prior
      beta_lambda[i,c] ~ cauchy(0, 1);  
      // Group-level coefficients with adaptive shrinkage
      beta0[,i,c] ~ normal(0, beta_lambda[i,c] * beta_tau[c]);
    }
  }

  // Individual-level coefficient variations
  beta0_sigma ~ normal(0, 1);  // Prior for group-level standard deviation
  for (i in 1:(3 * n_states - 2)) {
    for (c in 1:(n_cov+1)) {
      // Standard normal prior for individual variations
      beta_raw[,i,c] ~ std_normal();
    }
  }

  // === Movement Parameter Priors ===
  // Group-level parameters
  for(g in 1:n_grp) {
    // Step length parameters - weakly informative priors
    l_mu0[g, ] ~ normal(0, 2) T[0, ];     // Positive mean
    l_va0[g, ] ~ normal(0, 0.2) T[0, ];   // Positive variance, tighter prior
    
    // Angular concentration parameters - Beta(2,2) for moderate concentration
    a_rho0[g, ] ~ beta(2, 2);             // Base angle distribution
    cliff_rho0[g, ] ~ beta(2, 2);         // Near cliff concentration
    edge_rho0[g, ] ~ beta(2, 2);          // Near edge concentration
    center_rho0[g, ] ~ beta(2, 2);        // Near center concentration
  
    // Spatial influence parameters
    // Negative slopes encourage avoidance behavior
    edge_beta0[g] ~ normal(0, 10) T[, 0];    // Edge avoidance
    center_beta0[g] ~ normal(0, 10) T[, 0];  // Center avoidance
    
    // Distance thresholds scaled to arena size (30 units)
    // Edge threshold: Beta(2,10) favors detection near walls
    target += beta_lpdf(edge_x0[g] / 30 | 2, 10) - log(30);
    // Center threshold: Beta(3,3) allows more variation
    target += beta_lpdf(center_x0[g] / 30 | 3, 3) - log(30);
  } 

  // === Hyperpriors ===
  // Half-normal priors for variance components
  l_mu0_sigma ~ normal(0, 0.5) T[0, ];    // Step length mean variation
  l_va0_sigma ~ normal(0, 0.5) T[0, ];    // Step length variance variation
  
  // Gamma priors for concentration parameters
  // Shape=2, rate=0.5 gives moderate prior concentration
  a_rho0_phi ~ gamma(2, 0.5);             // Base angle concentration
  cliff_rho0_phi ~ gamma(2, 0.5);         // Cliff effect concentration
  edge_rho0_phi ~ gamma(2, 0.5);          // Edge effect concentration
  center_rho0_phi ~ gamma(2, 0.5);        // Center effect concentration
  
  // Spatial influence variations
  // Half-normal priors for standard deviations
  edge_x0_sigma ~ normal(0, 1) T[0, ];    // Edge threshold variation    
  edge_beta0_sigma ~ normal(0, 1) T[0, ]; // Edge slope variation
  center_x0_sigma ~ normal(0, 1) T[0, ];  // Center threshold variation
  center_beta0_sigma ~ normal(0, 1) T[0, ]; // Center slope variation

  // === Individual Level Parameters ===
  for (s in 1:n_smp) {
    // Spatial influence parameters
    edge_beta[s] ~ normal(edge_beta0[smp2grp[s]], edge_beta0_sigma);   
    edge_x[s] ~ normal(edge_x0[smp2grp[s]], edge_x0_sigma);   
    center_beta[s] ~ normal(center_beta0[smp2grp[s]], center_beta0_sigma);   
    center_x[s] ~ normal(center_x0[smp2grp[s]], center_x0_sigma);       
  }

  // State-specific parameters
  for (state in 1:n_states) {
    for (s in 1:n_smp) {
      // Step length parameters
      l_mu[s,state] ~ normal(l_mu0[smp2grp[s], state], l_mu0_sigma[state]);
      
      // Enforce minimum separation between states
      if(state > 1) {  
        if(l_mu[s,state] - l_mu[s,state-1] < min_mu_sep) {
          // Soft constraint using normal penalty
          target += normal_lpdf(l_mu[s,state] - l_mu[s,state-1] | min_mu_sep, 0.05);
        }
      }

      // Step length variance with penalty for large values
      l_va[s,state] ~ normal(l_va0[smp2grp[s], state], l_va0_sigma[state]);
      target += -2 * log1p(l_va[s,state]);  // Penalize large variances

      // Angular concentration parameters with beta priors
      // Each uses hierarchical structure with concentration hyperparameter
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

  //=============================================================================
  // Likelihood Calculation using Forward Algorithm
  //=============================================================================
  
  vector[n_states] logp;      // Forward variables
  vector[n_states] logptemp;  // Temporary storage
  
  // Implement forward algorithm for HMM likelihood
  for (t in 1:n_obs) {
    // Initialize forward variables at start of each track
    if(t == 1 || trk[t] != trk[t - 1]) {
      // Uniform initial state distribution
      logp = rep_vector(-log(n_states), n_states);
    }

    // Calculate forward probabilities for each state
    for (state in 1:n_states) {
      // 1. Transition probability from previous states
      logptemp[state] = log_sum_exp(logp + log_gamma[t][ , state]);
      
      // 2. Emission probability: step length (if not missing)
      if (l[t] != -99) {
        logptemp[state] += gamma_lpdf(l[t] | l_shape[smp[t],state], 
                                            l_rate[smp[t],state]);
      }
      
      // 3. Emission probability: turning angle 
      // Only calculated if we have two consecutive observations in same track
      if (t >= 2 && trk[t] == trk[t-1] && a[t] != -99 && a[t-1] != -99) {
        logptemp[state] += angle_log_lik(
          a[t], a[t-1], angle_type[t], edge_a[t], 
          cliff_l[t], cliff_x, cliff_beta, 
          edge_l[t], edge_x[smp[t]], edge_beta[smp[t]], 
          (30-edge_l[t]), center_x[smp[t]], center_beta[smp[t]], 
          a_rho[smp[t],state], cliff_rho[smp[t],state], 
          edge_rho[smp[t],state], center_rho[smp[t],state]);
      }
    }
    
    // Update forward variables
    logp = logptemp;

    // Add log probability at end of each track
    if(t == n_obs || trk[t+1] != trk[t]) {
      target += log_sum_exp(logp);  // Marginalize over final states
    }
  }
}

generated quantities {
  //=============================================================================
  // Output Variables
  //=============================================================================
  
  // Most likely state sequence from Viterbi algorithm
  int<lower=1, upper=n_states> z_star[n_obs];
  
  // Log probability of the most likely state sequence
  real logp_z_star;
  
  // Observation-wise log likelihoods for model evaluation
  vector[n_obs] log_lik;

  {
    //===========================================================================
    // Viterbi Algorithm Implementation
    //===========================================================================
    
    // Workspace variables for Viterbi algorithm
    int back_ptr[n_obs, n_states];     // Backpointers for state reconstruction
    real best_logp[n_obs, n_states];   // Log probabilities for best paths

    //-------------------------------------------------------------------------
    // Forward Pass: Calculate best paths and store backpointers
    //-------------------------------------------------------------------------
    for (t in 1:n_obs) {
      // Calculate best_logp for each state at current timestep
      for (state in 1:n_states) {
        // Handle missing data case
        if (l[t] == -99) {
          // Use uniform distribution for missing data
          best_logp[t, state] = -log(n_states);
          back_ptr[t, state] = 0;  // No meaningful backpointer
        } else {
          // Start of new track or first observation
          if (t == 1 || trk[t] != trk[t - 1]) {
            // Initialize with uniform state distribution
            best_logp[t, state] = -log(n_states);
            back_ptr[t, state] = 0;
            // Add emission probability for step length
            best_logp[t, state] += gamma_lpdf(l[t] | l_shape[smp[t],state], 
                                                    l_rate[smp[t],state]);
          } else {
            // Initialize variables for finding best previous state
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
                logp += gamma_lpdf(l[t] | l_shape[smp[t],state], 
                                        l_rate[smp[t],state]);
                
                // Add emission probability for angle if available
                if (a[t] != -99 && a[t-1] != -99) {
                  logp += angle_log_lik(
                    a[t], a[t-1], angle_type[t], edge_a[t], 
                    cliff_l[t], cliff_x, cliff_beta, 
                    edge_l[t], edge_x[smp[t]], edge_beta[smp[t]], 
                    (30-edge_l[t]), center_x[smp[t]], center_beta[smp[t]], 
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

    //-------------------------------------------------------------------------
    // Backward Pass: Reconstruct most likely state sequence
    //-------------------------------------------------------------------------
    {
      // Find best final state
      real max_logp = negative_infinity();
      int best_last_state = 1;
      
      // Search over final states
      for (state in 1:n_states) {
        if (best_logp[n_obs, state] > max_logp) {
          max_logp = best_logp[n_obs, state];
          best_last_state = state;
        }
      }
      
      // Set final state and log probability
      z_star[n_obs] = best_last_state;
      logp_z_star = max_logp;

      // Backtrack to reconstruct state sequence
      int t = n_obs;
      while (t >= 2) {
        int current_state = z_star[t];
        int prev_state = back_ptr[t, current_state];

        if (prev_state == 0 || trk[t] != trk[t - 1]) {
          // Handle track boundaries or missing data
          // Find best state at previous timestep
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
          // Normal case: use backpointer
          z_star[t - 1] = prev_state;
        }
        t -= 1;
      }
    }

    //-------------------------------------------------------------------------
    // Calculate Observation-wise Log Likelihoods
    //-------------------------------------------------------------------------
    for (t in 1:n_obs) {
      int state = z_star[t];
      log_lik[t] = 0;
      
      // Step length log-likelihood
      if (l[t] != -99)
        log_lik[t] += gamma_lpdf(l[t] | l_shape[smp[t],state], 
                                      l_rate[smp[t],state]);
      
      // Angle log-likelihood for consecutive observations
      if (t >= 2 && trk[t] == trk[t - 1] && a[t] != -99 && a[t - 1] != -99) {
        log_lik[t] += angle_log_lik(
          a[t], a[t-1], angle_type[t], edge_a[t], 
          cliff_l[t], cliff_x, cliff_beta, 
          edge_l[t], edge_x[smp[t]], edge_beta[smp[t]], 
          (30-edge_l[t]), center_x[smp[t]], center_beta[smp[t]], 
          a_rho[smp[t],state], cliff_rho[smp[t],state], 
          edge_rho[smp[t],state], center_rho[smp[t],state]);
      }

      // Transition log-likelihood
      if (t >= 2 && trk[t] == trk[t - 1] && l[t] != -99) {
        int prev_state = z_star[t - 1];
        // Only add transition probability if states are neighbors
        if (abs(prev_state - state) <= 1) {
          log_lik[t] += log_gamma[t][prev_state, state];
        } else {
          log_lik[t] += negative_infinity();
        }
      }
    }
  }
}
