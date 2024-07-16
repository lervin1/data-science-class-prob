# Classification Prediction Problem ----
# Assess Top Model w/ ks rec

# load packages
library(tidyverse)
library(tidymodels)
library(here)

# load data
load(here("attempt_2/results/final_fit_2.rda"))
load(here("data/test_airbnb.rda"))

predicted_prob_svm_1 <-  predict(final_fit_2, test_airbnb, type = "prob")

prob_result_svm_1 <- test_airbnb |>
  select(id) |>
  bind_cols(predicted_prob_svm_1) |>
  select(id, .pred_1) |>
  rename(predicted = .pred_1)

prob_result_svm_1

# making boosted tree submission
write_csv(prob_result_svm_1, file = here("submissions/svm_submission_attempt_1.csv"))
