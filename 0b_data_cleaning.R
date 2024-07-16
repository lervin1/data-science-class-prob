# Classification Prediction Problem ----
# Data cleaning/wrangling for classification prediction problem

# load packages
library(tidyverse)
library(tidymodels)
library(here)
library(knitr)
library(naniar)

# cleaning for training dataset
train_airbnb <- train_classification |>
  mutate(host_response_rate = str_remove(host_response_rate, "%") |>
           as.numeric()/100,
         host_acceptance_rate = str_remove(host_acceptance_rate, "%") |>
           as.numeric()/100, 
         host_is_superhost = factor(host_is_superhost, labels = c(0, 1)),
         host_has_profile_pic = factor(host_has_profile_pic, labels = c(0, 1)),
         host_identity_verified = factor(host_identity_verified, labels = c(0, 1)),
         has_availability = factor(has_availability, labels = c(0, 1)),
         instant_bookable = factor(instant_bookable, labels = c(0, 1)),
         time_hosting = as.numeric(today() - host_since),
         across(where(is.character), as.factor)) |>
  select(-host_since)

# checking missingness for training dataset
missing_count <- colSums(is.na(train_airbnb))
knitr::kable(missing_count, col.names = c("Variable", "Count Missing"))

miss_sum <- miss_var_summary(train_airbnb) |> 
  slice_head(n = 17) |> 
  kable()

gg_miss_var(train_airbnb)

# cleaning for testing dataset
test_airbnb <- test_classification |> 
  mutate(host_response_rate = str_remove(host_response_rate, "%") |>
           as.numeric()/100,
         host_acceptance_rate = str_remove(host_acceptance_rate, "%") |>
           as.numeric()/100, 
         host_has_profile_pic = factor(host_has_profile_pic, labels = c(0, 1)),
         host_identity_verified = factor(host_identity_verified, labels = c(0, 1)),
         has_availability = factor(has_availability, labels = c(0, 1)),
         instant_bookable = factor(instant_bookable, labels = c(0, 1)),
         time_hosting = as.numeric(today() - host_since),
         across(where(is.character), as.factor)) |>
  select(-host_since)

# checking missingness for testing dataset
missing_count <- colSums(is.na(test_airbnb))
knitr::kable(missing_count, col.names = c("Variable", "Count Missing"))

miss_sum <- miss_var_summary(test_airbnb) |> 
  slice_head(n = 17) |> 
  kable()

gg_miss_var(test_airbnb)

# save out data
save(train_airbnb, file = here("data/train_airbnb.rda"))
save(test_airbnb, file = here("data/test_airbnb.rda"))