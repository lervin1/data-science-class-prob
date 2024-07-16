# Classification Prediction Problem ----
# Analysis of tuned and trained models (comparisons)

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(knitr)

# handle common conflicts
tidymodels_prefer()

# load fit/tuned models
load(here("attempt_4/results/btree_tuned_3.rda"))

# load testing data
load(here("data/test_airbnb.rda"))

# creating table to see best models ----
# boosted tree
tbl_btree_3 <- btree_tuned_3 |> 
  show_best(metric = "roc_auc") |> 
  slice_max(mean) |>
  slice_head(n = 1) |> 
  select(mean, n, std_err) |> 
  mutate(model = "Boosted Tree")


# creating results tables
results_table_4 <- bind_rows(tbl_btree_3) |> 
  select(model, mean, std_err, n) |> 
  arrange(desc(mean)) |> 
  rename("Mean ROC AUC" = mean,
         "Model" = model,
         "Standard Error" = std_err,
         "N" = n) |> 
  kable()

results_table_4

# save results tables
save(results_table_4, file = here("attempt_4/results/results_table_4.rda"))
