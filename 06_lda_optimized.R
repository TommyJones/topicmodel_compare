################################################################################
# This script runs the LSI analysis using SigOpt optimized parameters
################################################################################

source("00_globals.R")

### declare parameters from SigOpt optimization ----
# note: these are hard-coded because of my use of Google Cloude ML and difficulty
# copying objects back and forth. Whatevs.

alpha <- round(0.1374685519102584, 3)

beta_sum <- round(233.2968725726665, 3)

k <- 317

nodes <- 862

### load data ----
load("data_derived/20_newsgroups_formatted.RData")


### create model and classify train2 and test ----

# create an LSI model from train1
lda <- FitLdaModel(dtm[train1, ], k = k, 
                   iterations = 300, 
                   burnin = 250,
                   alpha = alpha, 
                   beta = (beta_sum) * (colSums(dtm[train1,]) / sum(dtm[train1, ])),
                   optimize_alpha = TRUE,
                   calc_likelihood = TRUE,
                   calc_r2 = TRUE,
                   calc_coherence = TRUE)

# apply it to train2
lda2 <- predict(lda, dtm[train2, ], method = "dot")

lda2[is.na(lda2) | is.infinite(lda2) ] <- 0

# apply it to test
lda3 <- predict(lda, dtm[test, ], method = "dot")

lda3[is.na(lda3) | is.infinite(lda3) ] <- 0

### train a classifier and predict for test ----

set.seed(random_seed)

m_lda <- train_classifier(y = doc_class[train2],
                          x = lda2,
                          nodes = nodes,
                          dropout = 0.3,
                          activation = "relu")

p_lda <- predict_classifier(object = m_lda$net,
                            new_data = lda3)

### evaluate the result ----
e_lda <- evaluate(actual = doc_class[test],
                  predicted = apply(p_lda, 1, function(x) names(x)[which.max(x)][1]))

e_class_lda <- as.data.frame(do.call(rbind, lapply(e_lda$metrics, unlist)))


### save relevant objects ----
save(lda, lda2, lda3, m_lda, p_lda, e_lda, e_class_lda,
     file = "data_derived/lda_optimized_result.RData")
