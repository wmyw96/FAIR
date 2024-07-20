require('CondIndTests')
require('nonlinearICP')

summaryresult = matrix(rep(0, 100*22), nrow=100)

for (eid in 0:99){
  data = read.csv(paste('chamber_tmp/data_tmp_', as.character(eid), '.csv', sep=""), header=FALSE)
  X = data[, 1:11]
  Y = data[, 12]
  E = as.factor(data[, 13])
  result <- nonlinearICP(X = X, Y = Y, environment = E, alpha=0.01)
  chosenIdx <- varSelectionRF(X = X, Y = Y, env = E, verbose = TRUE)
  
  for (i in chosenIdx){
    summaryresult[eid+1, i] = 1
  }
  for (i in result$retrievedCausalVars){
    summaryresult[eid+1, i + 11] = 1
  }
  
  print(eid)
}
write.table(summaryresult, 'saved_results/chamber_icp.csv', sep=",", col.names=FALSE, row.names=FALSE)
