# Classification Prediction Problem ----
# Reading in/collection of datasets

## load packages ----

library(tidyverse)
library(here)
library(knitr)

## load data ----
load(here("attempt_5/results/results_table_5.rda"))
load(here("attempt_3/results/results_table_final.rda"))

combined_results <- bind_rows(results_table_5, results_table_final) |> 
  kable()

combined_results

# save combined results
save(combined_results, file = here("figures/combined_results.rda"))