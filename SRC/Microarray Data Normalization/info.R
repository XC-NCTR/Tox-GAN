setwd("../../Data") # please set your own working directory
library(tidyverse)

zipfiles <- list.files(path = 'zip', pattern = '\\.zip$')

sampleInfo <- tibble()

for(zipfn in zipfiles){
  folder <- str_remove(string = zipfn, pattern = '\\.zip')
  drug <- str_remove(string = zipfn, pattern = '\\.Rat\\.in_vivo\\.Liver\\.Repeat\\.zip')
  info <- read_tsv(file = paste0('zip/', folder, '/Attribute.tsv'), col_names = T, col_types = cols(), na = 'NA', progress = F)
  info$DOSE_UNIT <- 'uM'
  sampleInfo <- bind_rows(sampleInfo, info)
}
sampleInfo <- mutate(sampleInfo, CEL_FILE = paste0(BARCODE,'.CEL'))

saveRDS(object = sampleInfo, file = 'info.rds')
write_tsv(x = sampleInfo, file = 'info.tsv', col_names = T)
write_excel_csv(x = sampleInfo, file = 'info.csv', col_names = T)
