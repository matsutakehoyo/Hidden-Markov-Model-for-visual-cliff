# Data Processing Scripts

This directory contains scripts for processing and analyzing visual cliff experiment data.

## Data Preprocessing

### `process_csv.R`
Processes raw coordinate data from DeepLabCut tracking:
- Combines multiple CSV files with experimental metadata
- Filters data based on tracking confidence
- Aligns bodypart coordinates
- Estimates apparatus boundaries
- Generates movement visualizations

Requirements:
- R (>= 4.0.0)
- tidyverse
- readxl
- hexbin
- zoo
- patchwork
- wesanderson
- MetBrewer

Input:
- CSV files from DeepLabCut
- `experiment.xlsx` with experimental metadata

Output:
- Processed RDS file containing cleaned and aligned coordinate data
- Diagnostic plots for data quality assessment
