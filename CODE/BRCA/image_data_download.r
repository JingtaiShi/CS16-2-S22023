library(TCGAbiolinks)
query <- GDCquery(project = "TCGA-LUAD",
                   data.category = "Biospecimen",
                   data.type = "Slide Image",
                   barcode = c("TCGA-4B-A93V"),
                   data.format = "SVS")

GDCdownload(query,
            directory = "F:/research/TCGA-LUAD",
            method = "api",
            files.per.chunk = 5)