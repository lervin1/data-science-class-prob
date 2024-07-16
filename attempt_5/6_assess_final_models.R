# Classification Prediction Problem ----
# Assess Top Model w/ ks rec

# load packages
library(tidyverse)
library(tidymodels)
library(here)

# load data
load(here("attempt_5/results/final_fit_5.rda"))
load(here("data/test_airbnb.rda"))

predicted_prob_bt_5 <-  predict(final_fit_5, test_airbnb, type = "prob")

prob_result_bt_5 <- test_airbnb |>
  select(id) |>
  bind_cols(predicted_prob_bt_5) |>
  select(id, .pred_1) |>
  rename(predicted = .pred_1)

prob_result_bt_5

# making boosted tree submission
write_csv(prob_result_bt_5, file = here("submissions/bt_submission_attempt_5.csv"))