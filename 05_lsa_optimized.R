################################################################################
# This script runs the LSI analysis using SigOpt optimized parameters
################################################################################

source("00_globals.R")

### declare parameters from SigOpt optimization ----
# note: these are hard-coded because of my use of Google Cloude ML and difficulty
# copying objects back and forth. Whatevs.

k <- 233

nodes <- 862

### load data and create tf-idf ----
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

tfidf2 <- dtm[train2, ] %*% idf_mat

tfidf3 <- dtm[test, ] %*% idf_mat

### create model and classify train2 and test ----

# create an LSI model from train1
lsa <- FitLsaModel(tfidf1, k = k)

# apply it to train2
lsa2 <- predict(lsa, tfidf2)

lsa2[is.na(lsa2) | is.infinite(lsa2) ] <- 0

# apply it to test
lsa3 <- predict(lsa, tfidf3)

lsa3[is.na(lsa3) | is.infinite(lsa3) ] <- 0

### train a classifier and predict for test ----

set.seed(random_seed)

m_lsa <- train_classifier(y = doc_class[train2],
                          x = lsa2,
                          nodes = nodes,
                          dropout = 0.3,
                          activation = "relu")

p_lsa <- predict_classifier(object = m_lsa$net,
                            new_data = lsa3)

### evaluate the result ----
e_lsa <- evaluate(actual = doc_class[test],
                  predicted = apply(p_lsa, 1, function(x) names(x)[which.max(x)][1]))

e_class_lsa <- as.data.frame(do.call(rbind, lapply(e_lsa$metrics, unlist)))


### save relevant objects ----
save(lsa, lsa2, lsa3, m_lsa, p_lsa, e_lsa, e_class_lsa,
     file = "data_derived/lsa_optimized_result.RData")
