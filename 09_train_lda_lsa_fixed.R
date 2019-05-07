################################################################################
# This script fits lda and lsi models on train1 and projects train2 into that space
################################################################################

rm(list = ls())

### load libraries ----
library(textmineR)

### load data ----
load("data_derived/20_newsgroups_formatted.RData")

### train lda ----

lda <- FitLdaModel(dtm[train1, ], 200, iterations = 300, burnin = 250,
                   alpha = 0.1, 
                   beta = (0.05 * ncol(dtm)) * (colSums(dtm[train1,]) / sum(dtm[train1, ])),
                   optimize_alpha = TRUE,
                   calc_likelihood = TRUE,
                   calc_r2 = TRUE)

# lda2 <- predict(lda, dtm[train2, colSums(dtm[train2, ]) > 0 ], iterations = 300, burnin = 250)

lda2 <- predict(lda, dtm[train2, ], method = "dot")

lda3 <- predict(lda, dtm[test, ], method = "dot")

### train lsa ----

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


# fit lsa model
lsa <- FitLsaModel(tfidf1, 200)

lsa2 <- predict(lsa, tfidf2)

lsa2[is.na(lsa2) | is.infinite(lsa2) ] <- 0

lsa3 <- predict(lsa, tfidf3)

lsa3[is.na(lsa3) | is.infinite(lsa3) ] <- 0


### save necessary results ----

save(lda, lda2, lda3, lsa, lsa2, lsa3, file = "data_derived/lda_lsa.RData")
