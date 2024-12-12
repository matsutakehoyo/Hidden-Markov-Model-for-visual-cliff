# Hidden Markov Models for Visual Cliff Analysis

This repository contains the implementation of Hidden Markov Models and mixture models for analyzing mouse behavior in the visual cliff test, as described in our paper "Hidden Markov Models Reveal Complex Visual Processing Behavior in the Mouse Visual Cliff Test".

## Overview

The repository provides Stan implementations of:
- A hierarchical Hidden Markov Model (HMM) for analyzing mouse movement patterns
- Mixture models for preliminary analysis of step lengths and turning angles
- Tools for analyzing behavioral states and their transitions in response to visual stimuli

## Repository Structure


## Model Details

### Hidden Markov Model
The main HMM implementation (`hmm_model.stan`) features:
- Three behavioral states (Resting, Exploring, Navigating)
- Structured transition matrices allowing only neighboring state transitions
- Integration of spatial covariates (cliff, edge, center)
- Hierarchical structure for individual and group-level parameters

### Mixture Models
Preliminary mixture models for analyzing movement components:
- Gamma and lognormal mixtures for step length distributions
- von Mises and wrapped Cauchy mixtures for angular distributions


## Data Processing Pipeline

1. `process_csv.Rmd`: Initial processing of DeepLabCut tracking data
   - Data cleaning and filtering
   - Movement path visualization
   - Preliminary boundary estimation

2. `estimate_field_circle.Rmd`: Apparatus boundary estimation
   - Estimates circular apparatus center and radius
   - Uses iterative grid search algorithm
   - Generates diagnostic plots

3. `correct_coordinates.Rmd`: Coordinate system standardization
   - Aligns cliff boundary to vertical axis
   - Centers coordinate system
   - Validates coordinate transformations


## Requirements
- RStan (>= 2.26.0)
- R (>= 4.0.0)

