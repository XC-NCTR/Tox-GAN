# Tox-GAN: An AI Approach Alternative to Animal Studies – a Case Study with Toxicogenomics
Animal studies are a critical component in biomedical research, pharmaceutical product development, and regulatory submissions. There is a worldwide effort in toxicology toward "reducing, refining and replacing" animal use. Here, we proposed a deep generative adversarial network (GAN)-based framework capable of deriving new animal results from existing animal studies without additional experiments. To prove the concept, we employed this Tox-GAN framework to generate both gene activities and expression profiles for multiple doses and treatment durations in toxicogenomics (TGx). Using the pre-existing rat liver TGx data from the Open TG-GATEs, we generated Tox-GAN transcriptomic profiles with high similarity (0.997±0.002 in intensity and 0.740±0.082 in fold change) to the corresponding real gene expression profiles. Consequently, Tox-GAN showed an outstanding performance in two critical TGx applications, gaining a molecular understanding of underlying toxicological mechanisms and gene expression-based biomarker development.<br>

This repository contains the programming code we developed for the paper: **Chen, X., Roberts, R., Tong, W., & Liu, Z. (2021). Tox-GAN: An AI Approach Alternative to Animal Studies-a Case Study with Toxicogenomics. Toxicol Sci. doi:10.1093/toxsci/kfab157.** For more details, please refer to our paper.
## Python library requirements:
python=3.8.5<br>
pytorch=1.4.0<br>
scikit-learn=0.23.2<br>
numpy=1.19.1<br>
pandas=1.1.2<br>
pubchempy=1.0.4<br>
## Codes description:
**Please NOTE that if you want to reuse our code, please follow our instructions below and comments in the scripts to set up your own working directory and specify parameters according to your data and environment.**
### Microarray Data Normalization
  1. Run normalization.R (Rscript normalization.R) to normalize microarray data.
  2. Run info.R to match treatment conditions and gene expression data.
  3. Run logFC.R to calculate fold change profiles for each treatment condition.
### Molecular representaiton
  1. Run getSMILES.py with python to retrieve SMILES strings for compounds.
  2. Run getMolecularDescriptors.py to calculate numeric molecular representations.
### Tox-GAN Development
  1. AE4Exp.py is the autoencoder we used to reduce the dimensions of transcriptomic profiles.
  2. Run getCode4Exp.py or getCode4FC.py to extract and incorporate transcriptomic representations.
  3. Run the model_sdtGAN4Exp.py or model_sdtGAN4FC.py to train Tox-GAN model (sepcify parameters for training, and the results will be stored in specified location.). The training process may take several days to complete, depending on the configuration.
  4. Once training is completed, you can use generateExp.py or generateFC.py to generate transcriptomic profiles.<br>
**NOTE: if you are interested, our trained models generator_Exp and generator_FC are available at https://drive.google.com/drive/folders/1hOcoiu72b_d4XVw72jPtKORf07UuXomm?usp=sharing due to the github file size limit.**
### liver necrosis prediction
  1. AE_DNN.py is the architecture of the classifier which is used to predcit liver necrosis.
  2. run eval_AEDNN_real.py/eval_AEDNN_gen.py to evaluate performance.
