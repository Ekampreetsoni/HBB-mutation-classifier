# Raw data sources

Cohort: TCGA Breast Cancer (BRCA) on the legacy UCSC Xena TCGA hub (https://tcga.xenahubs.net). Files are fetched from the hub's S3 download bucket.
Written by `src/download.py`. Do not edit by hand.

| Layer | Xena dataset ID | Title | Xena version | URL | Downloaded | Size | Count |
|---|---|---|---|---|---|---|---|
| rna | TCGA.BRCA.sampleMap/HiSeqV2.gz | TCGA breast invasive carcinoma (BRCA) gene expression by RNAseq (polyA+ IlluminaHiSeq) | 2017-10-13 | https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FHiSeqV2.gz | 2026-10-07 | 64.3 MB | 1218 samples (columns) |
| rppa | TCGA.BRCA.sampleMap/RPPA_RBN.gz | TCGA breast invasive carcinoma (BRCA) phospho- or total protein expression by reverse phase protein array (replicate-base normalization) | 2017-09-08 | https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FRPPA_RBN.gz | 2026-10-07 | 0.7 MB | 747 samples (columns) |
| cnv | TCGA.BRCA.sampleMap/Gistic2_CopyNumber_Gistic2_all_data_by_genes.gz | TCGA breast invasive carcinoma (BRCA) copy number gistic2 estimate | 2017-09-08 | https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FGistic2_CopyNumber_Gistic2_all_data_by_genes.gz | 2026-10-07 | 2.5 MB | 1080 samples (columns) |
| clinical | TCGA.BRCA.sampleMap/BRCA_clinicalMatrix | Phenotypes | 2019-12-06 | https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FBRCA_clinicalMatrix | 2026-10-07 | 1.7 MB | 1247 samples (rows) |
