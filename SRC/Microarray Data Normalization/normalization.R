setwd("../../Data")   # please set your own working directory
library(affy)

cel.path <- 'CELs' # move all the probe level data (CEL files) to this directory

all_cel_files <- list.files(cel.path)

CEL <- ReadAffy(filenames = all_cel_files, celfile.path = cel.path, compress = F)
saveRDS(object = CEL, file = 'CEL.rds')

RMA <- rma(CEL)
saveRDS(object = RMA, file = 'RMA.rds')

EXPRS <- exprs(RMA)
saveRDS(object = EXPRS, file = 'EXPRS.rds')