library(textmineR)
library(SigOptR)

# get some sample data
dtm <- nih_sample_dtm

# set environmental token to use sigopt API
# Note: I don't like doing this by setting an environmental variable
Sys.setenv(SIGOPT_API_TOKEN="ZPVNRIDPYVGNLCMFTRLRIUJZLOPJADOASPAQTLAACIBVJWPC")

# create functions to train and evaluate the model
create_model <- function(assignments){
  
  set.seed(8675309) # remove random start as variable in experiment
  
  m <- FitLdaModel(dtm = dtm,
                   k = assignments$k,
                   iterations = 300,
                   burnin = 250,
                   alpha = assignments$alpha,
                   beta = (colSums(dtm) / sum(dtm)) * assignments$beta_sum,
                   optimize_alpha = TRUE,
                   calc_coherence = TRUE)
  
  m
}

evaluate_model <- function(assignments) {
  m <- create_model(assignments)
  
  return(mean(m$coherence))
}

# create an experiment
experiment <- create_experiment(list(
  name = "Tommy's first optimization",
  parameters = list(
    list(name = "k", type = "int", bounds = list(min = 2, max = 20)),
    list(name = "alpha", type = "double", bounds = list(min = 0.01, max = 1.0)),
    list(name = "beta_sum", type = "double", bounds = list(min = 50.0, max = 500.0))
  ),
  parallel_bandwidth = 1,
  observation_budget = 20,
  project = "examples"
))

# run the optimization loop
for(j in 1:experiment$observation_budget) {
  
  suggestion <- create_suggestion(experiment$id)
  
  value <- evaluate_model(suggestion$assignments)
  
  create_observation(experiment$id, list(
    suggestion=suggestion$id,
    value=value
  ))
}

# get the final results
experiment <- fetch_experiment(experiment$id)

best_assignments <- experiment$progress$best_observation$assignments

model <- create_model(best_assignments)

