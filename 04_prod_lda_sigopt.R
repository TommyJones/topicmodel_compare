################################################################################
# This script runs the SigOpt optimization for ProdLDA
################################################################################

source("00_globals.R")


### Set global options etc. to work with SigOpt ----

# set environmental token to use sigopt API
# Note: I don't like doing this by setting an environmental variable
Sys.setenv(SIGOPT_API_TOKEN =
             scan("sigopt_api_key", what = "character", sep = "\n", quiet = TRUE)
)

# create an experiment
experiment <- create_experiment(list(
  name = "ProdLDA optimization",
  parameters = list(
    list(name = "k", type = "int", bounds = list(min = 100, max = 900)),
    list(name = "alpha", type = "double", bounds = list(min = 0.01, max = 1)),
    list(name = "hidden_nodes", type = "int", bounds = list(min = 50, max = 900))
  ),
  parallel_bandwidth = 1,
  observation_budget = 20,
  project = "topicmodel_compare"
))

### read in data ----
load("data_derived/20_newsgroups_formatted.RData")


### declare model creation and evaluation functions for SigOpt ----

create_model <- function(assignments) {
  
  # create a model sampled from from train1
  s <- sample(train1, 2000)
  
  prod_lda <- train_prodlda(dtm = dtm[s, ], # because batch_size must be divisible by 100
                            k = assignments$k,
                            alpha = assignments$alpha,
                            hidden_nodes = assignments$hidden_nodes,
                            batch_size = 100,
                            validation_split = 0.2,
                            epochs = 100)
  
  # predict on whole train1 set to calc R2 and coherence
  # also correct for missing values that happen with subsampling
  theta <- predict_prod_lda(model = prod_lda$model,
                           new_data = dtm[c(train1, train1[1:35]), ], # must be divisible by 100
                           batch_size = 100)
  
  theta <- theta[seq_along(train1), ]
  
  theta[is.nan(theta) | is.na(theta)] <- 0
  
  theta <- theta / rowSums(theta)
  
  coh <- CalcProbCoherence(prod_lda$phi, dtm[train1, ])
  
  r2 <- CalcTopicModelR2(dtm = dtm[train1, ],
                         phi = prod_lda$phi,
                         theta = theta)
  
  # # apply it to train2
  # prod_lda2 <- predict_prod_lda(prod_lda$model, 
  #                               dtm[c(train2, train2[1:35]), ], # because batch_size must be divisible by 100
  #                               100)
  # 
  # prod_lda2 <- prod_lda2[seq_along(train2), ]
  # 
  # prod_lda2[is.na(prod_lda2) | is.infinite(prod_lda2) ] <- 0
  # 
  # 
  # # train a classifier using train2
  # m <- train_classifier(y = doc_class[train2],
  #                       x = prod_lda2,
  #                       nodes = rep(assignments$nodes, 5),
  #                       dropout = 0.3,
  #                       activation = "relu")
  # 
  # # predict it on training data for optimization
  # p <- predict_classifier(object = m$net,
  #                         new_data = prod_lda2)
  # 
  # # return just the predicted values
  # p
  
  # get coherence and r2 to combine into metric
  mean(c(mean(coh, na.rm = TRUE), r2), na.rm = TRUE)
}

# evaluate_model <- function(assignments) {
#   
#   m <- create_model(assignments)
#   
#   accuracy <- apply(m, 1, function(x) names(x)[which.max(x)][1]) == doc_class[train2]
#   
#   mean(accuracy)
#   
# }

### run the optimization loop ----
for(j in 1:experiment$observation_budget) {
  
  set.seed(random_seed)
  
  suggestion <- create_suggestion(experiment$id)
  
  value <- create_model(suggestion$assignments)
  
  create_observation(experiment$id, list(
    suggestion=suggestion$id,
    value=value
  ))
}

### get the final results ----
prod_lda_experiment <- fetch_experiment(experiment$id)

prod_lda_best_assignments <- experiment$progress$best_observation$assignments

print(prod_lda_best_assignments)

save(prod_lda_experiment, prod_lda_best_assignments, file = "data_derived/prod_lda_sigopt.RData")


