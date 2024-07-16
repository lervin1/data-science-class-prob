# Classification Prediction Problem ----
# Initial data checks, data splitting, & data folding

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)

# handle common conflicts
tidymodels_prefer()

# reading in data ----
load(here("data/train_airbnb.rda"))
load(here("data/test_airbnb.rda"))

# setting a seed ----

set.seed(1234)

# cross validation
airbnb_folds <- vfold_cv(train_airbnb, v = 5, repeats = 3,
                        strata = host_is_superhost)

# save out files ----
save(airbnb_folds, file = here("attempt_4/data-splits/airbnb_folds.rda"))
