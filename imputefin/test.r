# install.packages(c("imputeFin", 'data.table', "pbapply", "dplyr", "tidyr"))

library(imputeFin)
library(data.table)
library(pbapply)
library(dplyr)
library(tidyr)


# Import the data and look at the first six rows
get_data <- function(transform = FALSE){
  dataset <- read.csv(file = '../data/data_set_challenge.csv')
  rownames(dataset) <- dataset$Date
  dataset$Date <- NULL
  indexes_fx_rates <- NULL
  if(transform){
    types <- read.csv(file = '../data/final_mapping_candidat.csv') %>% select(Type, mapping_id)
    indexes_fx_rates <- types %>% filter(Type %in% c("BOND", "STOCK")) %>% pull(mapping_id)
    dataset[indexes_fx_rates, ] <- log(dataset[indexes_fx_rates, ])
  }
  lst <- list()
  lst$data <- dataset
  lst$index <- indexes_fx_rates
  return(lst)
}

lst <- get_data()
dataset <- lst$data
index <- lst$index
print(dim(dataset))
head(dataset)

impute_column <- function(ts){
  tryCatch(
    {
      return(impute_AR1_t(ts, remove_outliers = FALSE, verbose = FALSE))
    },
    error=function(err){
      return(ts)
    }
  )
}

standardize_colname <- function(cn){
  substr(cn, 2, 100)
}

impute_data <- function(){
  data <- dataset#[, 0:5]
  df <- cbind(row.names(data), data.frame(pbapply(data, 2, impute_column)))
  colnames(df) <- c("Date", unlist(lapply(list(colnames(data)), standardize_colname)))
  return(df)
}

df <- impute_data()

write.csv(df, "../data/r_submission.csv")
# 
# dataset_test <- read.csv(file = '../data/r_submission_test.csv') %>% select(-X)
# x2 <- sum(is.na(dataset_test))
# print(x2)
# types <- read.csv(file = '../data/final_mapping_candidat.csv') %>% select(Type, mapping_id)
# indexes_fx_rates <- types %>% filter(Type %in% c("BOND", "STOCK")) %>% pull(mapping_id)
# date <- dataset_test$Date
# dataset_test <- dataset_test %>% select(-Date)
# dataset_test[-indexes_fx_rates, ] <- log(dataset_test[-indexes_fx_rates, ])
# dataset_test <- cbind(Date=date, dataset_test)
# 
# subset <- dataset_test %>% filter(Date %in% c("28/10/2020", "29/10/2020", "30/10/2020", "01/11/2020", "02/11/2020"))
# subset$X32
# print(isna)
# x0 <- sum(is.na(dataset_test))
# dataset_test <- dataset_test %>% fill()
# x1 <- sum(is.na(dataset_test))
# 
# 
# print(x0)
# print(x1)
# print(x0 - x1)
# 
# write.csv(dataset_test, file = "../data/r_submission_group1.csv")
