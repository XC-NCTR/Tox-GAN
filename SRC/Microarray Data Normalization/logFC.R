setwd("/account/bgong/workspace/TGP2012/Rat/in_vivo/Liver/Repeat")

library(tidyverse)


sampleInfo <- readRDS(file = "info.rds") %>%
  filter(BARCODE != "No ChipData") %>%
  mutate(CEL_FILE == paste(BARCODE,'.CEL'),
         COMPOUND_NAME = factor(COMPOUND_NAME),
         `COMPOUND Abbr.` = factor(`COMPOUND Abbr.`),) %>%
  select(CEL_FILE, COMPOUND_NAME, `COMPOUND Abbr.`, SACRI_PERIOD, DOSE_LEVEL)

EXPRS <- readRDS(file = 'EXPRS.rds')
DATA <- rownames_to_column(as.data.frame(EXPRS), var = "probeset_ID")

write_tsv(DATA, file = "EXPRS.tsv", col_names = TRUE)

sampleInfo <- sampleInfo %>%
  filter(CEL_FILE %in% colnames(EXPRS))

logFC <- tibble(probeset_ID = rownames(EXPRS))

for(compound in levels(sampleInfo$COMPOUND_NAME)){
  Sinfo <- filter(sampleInfo, COMPOUND_NAME == compound) %>%
    mutate(SACRI_PERIOD = factor(SACRI_PERIOD))
  
  for(time in levels(Sinfo$SACRI_PERIOD)){
    info <- filter(Sinfo, SACRI_PERIOD == time) %>%
      mutate(DOSE_LEVEL = factor(DOSE_LEVEL))
    data_all <- EXPRS[, info$CEL_FILE %>% as_vector()]
    
    controls <- filter(info, DOSE_LEVEL == "Control")
    data_control <- EXPRS[,controls$CEL_FILE %>% as_vector()]
    
    logfc <- data_all - rowMeans(data_control)
    logfc <- rownames_to_column(as.data.frame(logfc), var = "probeset_ID")
    logFC <- left_join(logFC, logfc, by = "probeset_ID")
  }
}

saveRDS(logFC, file = "logFC.rds")
write_tsv(logFC, file = "logFC.tsv", col_names = T)


