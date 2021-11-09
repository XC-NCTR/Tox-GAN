# TGx-GAN: Inferencing Transcriptomic Profile from Chemical Information in Toxicogenomics using Artificial Intelligence
In vivo Toxicogenomics (TGx) has been widely applied in the toxicology field. However, profiling transcriptomic profiles treated with thousands of compounds on hundreds of thousands of animals are very expensive, time-consuming, and labor-intensive. Herein, we proposed an AI-powered TGx-based deep generative adversarial network (GAN) model named TGx-GAN to infer rat liver transcriptomic profiles in the Open Toxicogenomics Project-Genomics Assisted Toxicity Evaluation System (TG-GATEs) based on chemical information only.<br>

We further exemplified the potential utilities of the proposed TGx-GAN in facilitating 28-day repeated dose toxicity studies.<br>

This repository contains the programming code we developed for this paper. For more details, please refer to our paper.
## Python library requirements:
python=3.8.5<br>
pytorch=1.4.0<br>
scikit-learn=0.23.2<br>
numpy=1.19.1<br>
pandas=1.1.2<br>
pubchempy=1.0.4<br>
## Codes description:
### Microarray Data Normalization
  1. Run normalization.R (Rscript normalization.R) to normalize microarray data

### Molecular representaiton
  1. Run getSMILES.py with python to retrieve SMILES strings for compounds.
  2. Run getMolecularDescriptors.py to calculate numeric molecular representations.
### Tox-GAN Development
  1. AE4Exp.py is the autoencoder we used to reduce the dimensions of transcriptomic profiles.
  2. Run getCode4Exp.py or getCode4FC.py to extract and incorporate transcriptomic representations.
  3. Run the model.py to training Tox-GAN model (sepcify parameters for training, and the results will be stored in specified location.). The traiing process may take several days to complete, depending on the configuration.
  4. Once traiing is completed, you can use getExProfile.py/getFCprofile.py to generate transcriptomic profiles.
### liver necrosis prediction
  1. AE_DNN.py is the architecture of the classifier which is used to predcit liver necrosis.
  2. run eval_AEDNN_real.py/eval_AEDNN_gen.py to evaluate performance.
