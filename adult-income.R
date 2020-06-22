# README First
# This script creats a "rdas" directory to save training results for rmarkdown, 
# it requires
# 1. linux file system
# 2. the directory where the script runs writable 

# -------- adult income machine learning script --------
# Functionality of this script:
# 1. Download adult income file
# 2. Clean the data set
# 3. Create the training dataset, dev, and validation data set
# 4. Train five models with dev_train and dev_testand compute accuracy
# 5. Run final model on the validation set
# 6. Analyze prediction results

ptm <- proc.time()
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(ggrepel)) install.packages("ggrepel", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(reshape2)) install.packages("reshape2", repos = "http://cran.us.r-project.org")
if(!require(tinytex)) install.packages("tinytex", repos = "http://cran.us.r-project.org")
if(!require(factoextra)) install.packages("factoextra", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
if(!require(grid)) install.packages("grid", repos = "http://cran.us.r-project.org")

# ----Function Definitions ------

# -- build models
create_fits <- function (models, train_dat = devr_train, ks, mtry, save_results = FALSE){
  # Description:
  #  Build training models 
  # Input Params:
  #  models: models to be built
  #  train_dat: training data set
  #  ks: knn tuning param
  #  mtry: rf tuning param
  #  save_results: save training models
  # Return
  #  Training models
  
  print("------- Building Models ------")
  
  lapply(models, function(model){
    
    print(model)
    train_model <- switch(model,
                          "knn" = {train(income ~ ., method = model, 
                                         data = train_dat, tuneGrid=data.frame(k=ks))},
                          "rf" = {train(income ~ ., method = model, 
                                        data = train_dat, tuneGrid=data.frame(mtry = mtry))},
                          train(income ~ ., method = model, data = train_dat)
                          )
    if (save_results) {
      saveRDS(train_model, paste("rdas/", model, ".rda", sep = ''))
    }
    
    train_model
  })
}

# ---Compute Model Accuracy ---
compute_accuracy <- function(train_model, test_dat, save_results = FALSE) {
  # Discription: 
  #  Use train_model to generate prediction for test_dat 
  # input params: 
  #  train_model
  #  test_dat
  #  save_results: save confusionMatrix
  # Return:
  #  confusionMatrix's accuracy

  pred <- predict(train_model, test_dat, type="raw")
  cm <- confusionMatrix(pred, reference=test_dat$income)
  pred_test_dat <- test_dat %>% mutate(pred = pred)
  
  cmname <- paste(train_model$method, "cm", sep = '')
  datname <- paste(train_model$method, "pred_test_dat", sep = '') 
  if (save_results) {
    saveRDS(cm, paste("rdas/", cmname, ".rda", sep = ''))
    saveRDS(pred_test_dat, paste("rdas/", datname, ".rda", sep = ''))
    }
  
  cm$overall[["Accuracy"]]
}

# ---run_models function calls create_fits and run_accuracy to 
#    build models and then compute accuracy ---
run_models <- function(models, train_dat = devr_train, test_dat = devr_test, 
                       ks = 41, mtry = 3, test_note = "run models", save_results = FALSE){
  
  # Description: run training models
  # Arguments
  #  models: models to run
  #  train_dat: training data set
  #  test_dat: test data set
  #  ks: knn tuning param
  #  mtry: rf tuning param
  # Return
  #  prediction accuracies 
  
  fits <- create_fits(models, train_dat = train_dat, 
                      ks = ks, mtry = mtry, save_results = save_results)
  names(fits) <- models
  mcount <- length(models)
  
  # --- Compute Train Accuracy ---
  print("train_accuracy: ")
  print(train_accuracy <- sapply(1:mcount, function(n){max(fits[[n]]$results$Accuracy)}))
  names(train_accuracy) <- models
  
  print("train_mean:")
  print(train_mean <- mean(train_accuracy))
  
  train_results <- cbind(paste("Train Accuracy: ", test_note), 
                         train_accuracy, paste("train mean", train_mean))
  
  
  print("--- Compute True Value vs Pred Accuracy ---") 
  acc_results <- tibble(model="", accuracy=0)
  pred_accuracy <- sapply(fits, compute_accuracy, test_dat = test_dat,  
                          save_results = save_results, simplify = TRUE)
                                 
  saveRDS(train_results, "rdas/train_results.rda")

  return(pred_accuracy)
  
} #end run_models

# -----Function to compare the profiles of missed/hit predictions -------
compute_colmeans <- function(dat) {dat  %>% 
    select(education.num, age, fnlwgt, hours.per.week) %>% colMeans()}

compare_pred_hit_miss <- function(pred_dat) {
  # Diescription:
  #  Generate a table for comparing correct and incorrect predictions
  # Parameter:
  #  pred_dat:  an adult income test data set with additional prediction column.
  # Return:
  #  Averages of  important predictors: edunum, fnlwgt, age and hours per week
  
  high_income_avg <-  pred_dat %>% 
    filter(income %in% ">50K") %>% compute_colmeans()
  
  pred_right_avg <- pred_dat %>% 
    filter(income == ">50K" & pred == ">50K") %>%  compute_colmeans()
  
  pred_wrong_avg <- pred_dat %>% 
    filter(income == ">50K" & pred == "<=50K") %>% compute_colmeans()
  
  low_income_avg <-   pred_dat %>% 
    filter(income %in% "<=50K") %>% compute_colmeans()
  
  comps <- data.frame(group=c("high_income", "pred_right", 
                              "pred_wrong", "low_income"),
                      
                      edunum_avgs=c(high_income_avg[1], 
                                    pred_right_avg[1], 
                                    pred_wrong_avg[1], 
                                    low_income_avg[1]),
                      
                      age_avgs=c(high_income_avg[2], 
                                 pred_right_avg[2], 
                                 pred_wrong_avg[2], 
                                 low_income_avg[2]),
                      
                      fnlwgt_avgs=c(high_income_avg[3], 
                                    pred_right_avg[3], 
                                    pred_wrong_avg[3], 
                                    low_income_avg[3]),
                      
                      hwk_avgs=c(high_income_avg[4], 
                                 pred_right_avg[4], 
                                 pred_wrong_avg[4], 
                                 low_income_avg[4])
  )
  
  return(comps)
  
}


# ---global variables ---
options(digits = 5)

if (!dir.exists("rdas")) {dir.create("rdas")}

# -------- File download ---------
url <- "https://github.com/xwueng/ml-adult-income/blob/master/adult.csv?raw=true"
# url <- "https://www.kaggle.com/uciml/adult-census-income/download"

dest_file <- "adult.csv"
download.file(url, destfile = dest_file)

# --- read income csv file --
adult <- read_csv("adult.csv")

problems(adult)
glimpse(adult)

nquestion <- length( adult == "?" )
nquestion
questions <- sapply(adult, function(column) {sum(column == "?")}) 

adult <- as.data.frame(lapply(adult, function(x){replace(x, x == "?", NA)}))


# ---- Process NAs -----
# 1. add an "Unknown" level to workclass, occupation and native.country
# 2. Change NAs to unknown
levels(adult$workclass) <- c(levels(adult$workclass), "Unknown")
levels(adult$occupation) <- c(levels(adult$occupation), "Unknown")
levels(adult$native.country) <- c(levels(adult$native.country), "Unknown")
adult[is.na(adult)] <- 'Unknown'

options(knitr.kable.NA = '')
adult_summary1 <- summary(adult)[, 1:7] %>% 
  kable(caption="1994 Adult Census Income", "latex", booktabs = TRUE) %>% 
  kable_styling(latex_options = c("striped", "scale_down", "HOLD_position")) 


adult_summary2 <- summary(adult)[, 8:15] %>% 
  kable("latex", booktabs = TRUE) %>% 
  kable_styling(latex_options = c("striped", "scale_down", "HOLD_position")) 

# -- identify near zero value predictors ---
nzv <- nearZeroVar(adult)
# [1] 11 12 14

# r markdown
nzv_tbl <- names(adult)[nzv] %>% 
  kable(caption = "Near Zero Variables") %>% 
  kable_styling(latex_options = c("striped", "scale_down", "HOLD_position")) 

mean(adult$native.country =="United-States")
mean(adult$capital.gain ==0)
mean(adult$capital.loss ==0)

# --create nzv plots to visualize them
# capital gain
p_cgain <- adult %>% ggplot(aes(capital.gain)) +
  geom_histogram(bins = 3) 
p_cgain

# capital loss
p_closs <- adult %>% ggplot(aes(capital.loss)) +
  geom_histogram(bins = 3) 
p_closs

# native.country
p_country <- adult %>% mutate(country = ifelse(native.country =="United-States", "United-States", "Non U.S.")) %>% 
  ggplot(aes(country, fill=country)) +
  geom_histogram(stat="count") 
 
grid.arrange(p_cgain, p_closs, p_country, ncol=3,
                    top = textGrob("Near Zero Variables",gp=gpar(fontsize=12)))

# ----- Dimension Reduction ---
# remove capital gain, capital loss and native.conutry from adult 
# and save the result in adultr, r to denote it's a reduced adult data set

adultr <- adult %>% 
  select(-all_of(nzv)) %>% 
  mutate(education = as.numeric(education), 
         workclass = as.numeric(workclass), 
         occupation =  as.numeric(occupation),
         marital.status = as.numeric(marital.status),  
         relationship = as.numeric(relationship), 
         race = as.numeric(race),
         sex = as.numeric(sex)) 

# --- Split adult into dev and validation by 9:1 ratio
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = adult$age, times = 1, p = 0.1, list = FALSE)

# ----- dev is the train set
devr <- adultr[-test_index,] 
validationr <- adultr[test_index,]

rm(adult)

# ----- split dev into dev_train and dev_test -- 
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = devr$age, times = 1, p = 0.1, list = FALSE)

nc <- ncol(devr) 
devr_train <- devr[-test_index, ]
devr_test <- devr[test_index,]


# -- PCA Analysis ---
# --- adultrn PCA ---
adultrn <- adultr %>%
  mutate(income = as.numeric(income))
adultrn.pca <- prcomp(adultrn,  center = TRUE, scale. = TRUE)
adultrn.pca

colnames(adultrn.pca$rotation) <- colnames(adultrn)

# rmarkdown
adultrn.pca.summary <- summary(adultrn.pca)
adultrn.pca.summary

income_rotation <- round(adultrn.pca$rotation["income", -ncol(adultrn)], digits = 3)
income_comps <- income_rotation[income_rotation %in% range(income_rotation)]
cor(adultrn)

adultrn_small <- adultrn %>%  sample_n(1000, replace = FALSE)
adultrn_small.pca <- prcomp(adultrn_small,  center = TRUE, scale. = TRUE)
pca.out <- adultrn_small.pca
pca.out$rotation <- -pca.out$rotation
pca.out$x <- -pca.out$x

# r markdown
biplot(pca.out,scale=0, cex=.7)

rm(adultrn)

# ------- Create Training Modelings ------


# ---Use devr_train and devr_test data sets to train models  --
models <- c("glm", "lda", "knn", "gamLoess", "rf")
ks <- seq(34, 41, 1)
mtry <- c(3, 5)
devrtrain_devrtest_results<-run_models(models,
  train_dat = devr_train, test_dat = devr_test, ks = ks, mtry = mtry,
  test_note = "devr_train, devr_test", save_results = FALSE)


# # --- Use  validation data to run final model --
models <- c("rf")
mtry <- 3
devr_validationr_results<-run_models(models,
  train_dat = devr, test_dat = validationr, ks = ks, mtry = mtry,
  test_note = "devr, validationr", save_results = TRUE)

# --read rf confusionMatrix data to prepare stats for Rmd 
rfcm <- readRDS("rdas/rfcm.rda")
rfcm_acc_tbl <- rfcm$overall[1:2] 
# %>% kable() %>% 
#   kable_styling(latex_options = c("HOLD_position")) 
rfcm$byClass[1:2]
rf_ref_pred_tbl <- rfcm$table 

# --read rf training model data to prepare stats for Rmd 
rfm <- readRDS("rdas/rf.rda")
rfm$results
rfm$times
rf_varImp <- varImp(rfm)

# --read prediction data to compare prediction hit/miss profiles 
rf_pred_dat <- readRDS("rdas/rfpred_test_dat.rda")
comps <- compare_pred_hit_miss(pred_dat = rf_pred_dat) 

comps_tbl <- comps %>% 
  kable(caption = "Averages of rf Important Predictors") %>% 
  kable_styling(latex_options = c("HOLD_position")) 

#--- create plots to visualize what caused above $50K income earners 
# being miss classfiled as <=50K 

group_order <- factor(comps$group, 
                      level=c("high_income", "pred_right", "pred_wrong", "low_income"))

# -- average education years
p_edunum_avg <- comps %>% ggplot(aes(group_order, edunum_avgs, fill=group)) + 
  geom_col() +
  geom_hline(yintercept = comps$edunum_avgs[1], color="black", show.legend=TRUE) +
  labs(title = "Average Education Years", 
       y = "Education Years") 

# ---- Average age plot ---
p_age_avg <- comps %>% ggplot(aes(group_order, age_avgs, fill=group)) + 
  geom_col() +
  geom_hline(yintercept = comps$age_avgs[1], color="black", show.legend=TRUE) +
  labs(title = "Average Age", 
       y = "Age") 

# --- Average fnlwgt plot
p_fnlwgt_avg <- comps %>% ggplot(aes(group_order, fnlwgt_avgs, fill=group)) + 
  geom_col() +
  geom_hline(yintercept = comps$fnlwgt_avgs[1], color="black", show.legend=TRUE) +
  labs(title = "fnlwgt", 
       y = "fnlwgt") 

# -- Average Hours Per Week plot
p_hwk_avg <- comps %>% ggplot(aes(group_order, hwk_avgs, fill=group)) + 
  geom_col() +
  geom_hline(yintercept = comps$hwk_avgs[1], color="black", show.legend=TRUE) +
  labs(title = "Average Hours Per Week", 
       y = "Hours Per Week") 

grid.arrange(p_age_avg, p_fnlwgt_avg, p_edunum_avg, p_hwk_avg, nrow= 2, ncol = 2)


# -- get program run time --
print(ptm <- proc.time() - ptm)
