require('CondIndTests')
require('nonlinearICP')

for (eid in 0:29){
  data = read.csv(paste('chamber_tmp/data_tmp_', as.character(eid), '.csv', sep=""), header=FALSE)
  X = data[, 1:11]
  Y = data[, 12]
  E = as.factor(data[, 13])
  result <- nonlinearICP(X = X, Y = Y, environment = E, alpha=0.01)
  chosenIdx <- varSelectionRF(X = X, Y = Y, env = E, verbose = TRUE)
  
  summaryresult = matrix(rep(0, 22), nrow=2)
  for (i in chosenIdx){
    summaryresult[1, i] = 1
  }
  for (i in result$retrievedCausalVars){
    summaryresult[2, i] = 1
  }
  
  print(summaryresult)
  write.table(summaryresult, paste('chamber_tmp/result_tmp_', as.character(eid), '.csv', sep=""), sep=",",  col.names=FALSE, row.names=FALSE)
}