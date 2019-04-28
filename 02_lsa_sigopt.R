################################################################################
# This script runs the SigOpt optimization for LSI
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
  name = "LSI optimization",
  parameters = list(
    list(name = "k", type = "int", bounds = list(min = 100, max = 900))
  ),
  parallel_bandwidth = 1,
  observation_budget = 20,
  project = "topicmodel_compare"
))

### read in data ----
load("data_derived/20_newsgroups_formatted.RData")

# prepare tf-idf
# calculate idf from train1
idf <- log(length(train1) / colSums(dtm[train1,] > 0))

idf_mat <- Matrix(0, nrow = length(idf), ncol = length(idf), sparse = TRUE)

diag(idf_mat) <- idf

rownames(idf_mat) <- names(idf)

colnames(idf_mat) <- names(idf)

# create tfidf for train1 and train2

tfidf1 <- dtm[train1, ] %*% idf_mat

# tfidf2 <- dtm[train2, ] %*% idf_mat
# 
# tfidf3 <- dtm[test, ] %*% idf_mat


### declare model creation and evaluation functions for SigOpt ----

create_model <- function(assignments) {
  
  # create an LSI model sampling from train1
  s <- sample(1:nrow(tfidf1), 2000)
  
  lsa <- FitLsaModel(tfidf1[s, ], assignments$k)
  
  # # apply it to train2
  # lsa2 <- predict(lsa, tfidf2)
  # 
  # lsa2[is.na(lsa2) | is.infinite(lsa2) ] <- 0
  
  
  # # train a classifier using train2
  # m_lsa <- train_classifier(y = doc_class[train2],
  #                           x = lsa2,
  #                           nodes = rep(assignments$nodes, 5),
  #                           dropout = 0.3,
  #                           activation = "relu")
  # 
  # # predict it on training data for optimization
  # p_lsa <- predict_classifier(object = m_lsa$net,
  #                             new_data = lsa2)
  
  # return coherence from full train1
  coh <- CalcProbCoherence(phi = lsa$phi, dtm = dtm[train1, ])
  
  mean(coh)
  
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
  
  # set.seed(random_seed)
  
  suggestion <- create_suggestion(experiment$id)
  
  value <- create_model(suggestion$assignments)
  
  create_observation(experiment$id, list(
    suggestion=suggestion$id,
    value=value
  ))
}

### get the final results ----
lsa_experiment <- fetch_experiment(experiment$id)

lsa_best_assignments <- experiment$progress$best_observation$assignments

print(lsa_best_assignments)

save(lsa_experiment, lsa_best_assignments, file = "data_derived/lsa_sigopt.RData")


