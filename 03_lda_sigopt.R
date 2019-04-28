################################################################################
# This script runs the SigOpt optimization for LDA
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
  name = "LDA optimization",
  parameters = list(
    list(name = "k", type = "int", bounds = list(min = 100, max = 900)),
    list(name = "alpha", type = "double", bounds = list(min = 0.01, max = 1)),
    list(name = "beta_sum", type = "double", bounds = list(min = 50, max = 500))
  ),
  parallel_bandwidth = 1,
  observation_budget = 20,
  project = "topicmodel_compare"
))

### read in data ----
load("data_derived/20_newsgroups_formatted.RData")


### declare model creation and evaluation functions for SigOpt ----

create_model <- function(assignments) {
  
  # create a model sampling from train1
  s <- sample(train1, 1000)
  
  lda <- FitLdaModel(dtm[s, ], k = assignments$k, 
                     iterations = 300, 
                     burnin = 250,
                     alpha = assignments$alpha, 
                     beta = (assignments$beta_sum) * (colSums(dtm[train1,]) / sum(dtm[train1, ])),
                     optimize_alpha = TRUE,
                     calc_likelihood = FALSE,
                     calc_r2 = FALSE,
                     calc_coherence = FALSE)
  
  # re-calculate R2 and coherence with the full set
  theta <- predict(object = lda, 
                   newdata = dtm[train1, ], 
                   method = "dot")
  
  coh <- CalcProbCoherence(phi = lda$phi,
                           dtm = dtm[train1, ],
                           M = 5)
  
  r2 <- CalcTopicModelR2(dtm = dtm[train1, ],
                         phi = lda$phi,
                         theta = theta)
  
  # apply it to train2
  # lda2 <- predict(lda, dtm[train2, ], method = "dot")
  # 
  # lda2[is.na(lda2) | is.infinite(lda2) ] <- 0
  # 
  # 
  # # train a classifier using train2
  # m_lda <- train_classifier(y = doc_class[train2],
  #                           x = lda2,
  #                           nodes = rep(assignments$nodes, 5),
  #                           dropout = 0.3,
  #                           activation = "relu")
  # 
  # # predict it on training data for optimization
  # p <- predict_classifier(object = m_lda$net,
  #                         new_data = lda2)
  # 
  # # return just the predicted values
  # p
  
  # return metric of mean of coherence and R-squared
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
lda_experiment <- fetch_experiment(experiment$id)

lda_best_assignments <- experiment$progress$best_observation$assignments

print(lda_best_assignments)

save(lda_experiment, lda_best_assignments, file = "data_derived/lda_sigopt.RData")


