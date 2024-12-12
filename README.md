# Hidden Markov Models for Visual Cliff Analysis

This repository contains the implementation of Hidden Markov Models and mixture models for analyzing mouse behavior in the visual cliff test, as described in our paper "Hidden Markov Models Reveal Complex Visual Processing Behavior in the Mouse Visual Cliff Test".

## Overview

The repository provides Stan implementations of:
- A hierarchical Hidden Markov Model (HMM) for analyzing mouse movement patterns
- Mixture models for preliminary analysis of step lengths and turning angles
- Tools for analyzing behavioral states and their transitions in response to visual stimuli

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

## Requirements
- RStan (>= 2.26.0)
- R (>= 4.0.0)

