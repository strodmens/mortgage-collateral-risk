# References & Resources

All external references, academic papers, dataset sources, and code examples collected for this project.
Kept in git so any session on any device has full context.

---

## Datasets

### SoCal House Prices and Images (CNN branch)
- **Kaggle**: https://www.kaggle.com/datasets/ted8080/house-prices-and-images-socal
- **Original author repo**: https://github.com/tncy67/House-Price-Prediction-via-Computer-Vision
- **Fork / related notebook**: https://www.kaggle.com/code/simple11/predict-house-price-from-images-tabular-data
- ~15,474 frontal house photos + CSV with `image_id`, `price`, `bed`, `bath`, `sqft`, etc.
- Southern California residential properties, prices $195k–$2M

### London Real Estate 2024 (RNN branch)
- **Kaggle**: https://www.kaggle.com/datasets/kanchana1990/real-estate-data-london-2024
- **Related notebook (CatBoost+SHAP)**: https://www.kaggle.com/code/dima806/london-properties-price-catboost-shap
- **Related repos**:
  - https://github.com/EricLeon/Real_Estate
  - https://github.com/sgawalsh/real-estate-data-analysis
- **London housing context report**: https://data.london.gov.uk/download/24rpx/8cdbb084-982c-44f3-a890-765f4002cbaa/Housing%20in%20London%202024%20report%20-%202nd%20edition.pdf
- 1,019 property listings with `descriptionHtml`, `price`, `propertyType`, etc.
- Prices £315k–£80M, descriptions avg 252 words

### Airbnb Open Data NYC (joint model pairing)
- **Kaggle**: https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata
- 102,599 listings — used for band-matching pairs in the joint model
- Note: no usable description column (`NAME` avg 6 words), so joint model pairs SoCal images with London text by price band instead

### GloVe Embeddings
- **Stanford NLP**: https://nlp.stanford.edu/data/glove.6B.zip
- Using `glove.6B.100d.txt` (400k vocabulary, 100 dimensions)

---

## Academic Papers

### Directly relevant — CNN-LSTM for risk / property valuation

1. **A hybrid model based on CNN-LSTM for assessing the risk of increasing claims in insurance companies**
   - Authors: (see paper)
   - Journal: PeerJ Computer Science, 2025
   - DOI: https://doi.org/10.7717/peerj-cs.2830
   - File: `academics/peerj-cs-2830.pdf`
   - Relevance: Hybrid CNN-LSTM architecture for risk classification (low/normal/high) — directly parallels our collateral risk flag approach

2. **A Hybrid CNN-LSTM Model for Enhancing Bond Default Risk Prediction**
   - Authors: Yao, J., Wang, J., Wang, B., Liu, B., & Jiang, M.
   - Journal: Journal of Computer Technology and Software, 3(6), 2024
   - DOI: https://doi.org/10.5281/zenodo.13910344
   - File: `academics/A+Hybrid+CNN-LSTM+Model+for+Enhancing+Bond+Default+Risk+Prediction.pdf`
   - Relevance: CNN extracts features from unstructured text, LSTM processes time-series — supports our CNN+LSTM dual-branch architecture for credit risk

3. **The Evaluation on the Credit Risk of Enterprises with the CNN-LSTM-ATT Model**
   - Journal: Computational Intelligence and Neuroscience, 2022, Article 6826573
   - DOI: https://doi.org/10.1155/2022/6826573
   - File: `academics/CIN2022-6826573.pdf`
   - Relevance: CNN-LSTM with attention mechanism for enterprise credit risk — relevant to our business integration layer

### Related — real estate / property valuation with deep learning

4. **Paper from `academics/2508.00415v1.pdf`**
   - File: `academics/2508.00415v1.pdf`
   - (Citation details to be confirmed from local copy)

5. **Paper from `academics/3675888.3676052.pdf`**
   - Likely DOI: https://doi.org/10.1145/3675888.3676052
   - File: `academics/3675888.3676052.pdf`
   - (ACM publication — citation details to be confirmed from local copy)

6. **Paper from `academics/j.ijsd.20241002.11.pdf`**
   - File: `academics/j.ijsd.20241002.11.pdf`
   - (Citation details to be confirmed from local copy)

---

## Code References

| Source | URL | What it shows |
|--------|-----|---------------|
| SoCal dataset author | https://github.com/tncy67/House-Price-Prediction-via-Computer-Vision | Original CNN house price prediction code |
| SoCal fork (Kaggle) | https://www.kaggle.com/code/simple11/predict-house-price-from-images-tabular-data | CNN + tabular approach on same dataset |
| London prices (Kaggle) | https://www.kaggle.com/code/dima806/london-properties-price-catboost-shap | CatBoost + SHAP on London 2024 data |
| London real estate repo | https://github.com/EricLeon/Real_Estate | General real estate analysis code |
| London real estate repo | https://github.com/sgawalsh/real-estate-data-analysis | Real estate data analysis |

---

## How to use these references in the notebook

The notebook's final section (cell 87 — "Contribution Table & References") should cite the academic papers above. The key citations for the academic framing are:

- **CNN-LSTM hybrid for risk**: peerj-cs-2830, Yao et al. 2024, CIN2022-6826573
- **Dataset sources**: Kaggle links for SoCal, London 2024, Airbnb, plus the original author's GitHub
- **Transfer learning**: standard ResNet-50 / ImageNet citations
- **GloVe**: Pennington et al. (2014), "GloVe: Global Vectors for Word Representation"
