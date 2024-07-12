# Papers on Interpretable Machine Learning Models

## NAM: Neural Additive Models
**Link:** [NAM Paper](https://proceedings.neurips.cc/paper/2021/file/251bd0442dfcc53b5a761e050f8022b8-Paper.pdf)

**Summary:**
Fitting a Neural netowkr (MLP) for each feature and summing their output for final model prediction.

## NodeGAM: Neural Generalized Additive Models
**Link:** [NodeGAM Paper](https://arxiv.org/pdf/2106.01613)

**Summary:**
NodeGAM introduces neural GAM (NODE-GAM) and neural GA2M (NODE-GA2M), which scale well on large datasets and maintain interpretability. Using NODE (Neural oblivious decision ensembles) as shape functions.

## NBM: Neural Basis Models
**Link:** [NBM Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/37da88965c016dca016514df0e420c72-Paper-Conference.pdf)

**Summary:**
Neural Basis Models (NBMs) use basis decomposition of shape functions, enabling scalable and interpretable models that excel in accuracy and efficiency for large-scale data with high-dimensional features.

## SPAM: Scalable Polynomial Additive Models
**Link:** [SPAM Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/ee81a23d6b83ac15fbeb5b7a30934e0b-Paper-Conference.pdf)

**Summary:**
Scalable Polynomial Additive Models (SPAM) leverage tensor rank decompositions of polynomials, outperforming current interpretable models and matching DNN/XGBoost performance.


## Sparse NAM: Sparse Neural Additive Models
**Link:** [Sparse NAM Paper](https://link.springer.com/chapter/10.1007/978-3-031-43418-1_21)

**Summary:**
Sparse Neural Additive Models (SNAM) enhance Neural Additive Models (NAMs) by incorporating group sparsity regularization for feature selection and improved generalization. SNAM provably achieves zero training loss and exact feature selection, demonstrating good accuracy and efficiency.

## SIAN: Sparse Interaction Additive Networks
**Link:** [SIAN Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/5a3674849d6d6d23ac088b9a2552f323-Paper-Conference.pdf)

**Summary:**
Sparse Interaction Additive Networks (SIAN) identifies necessary feature combinations. SIAN achieves competitive performance and finds an optimal tradeoff between neural network capacity and simpler model generalizability.

## Concurvity Regularization
**Link:** [Concurvity Regularization Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/3c6696d70d364337cf98dcb7c652a770-Paper-Conference.pdf)

**Summary:**
Concurvity Regularization addresses the issue of concurvity in Generalized Additive Models (GAMs) by penalizing pairwise correlations of non-linearly transformed feature variables. This improves interpretability without compromising prediction quality, reducing variance in feature importances.

## NATT: Neural Additive Tabular Transformer Networks
**Link:** [NATT Paper](https://openreview.net/pdf?id=TdJ7lpzAkD)

**Summary:**
Neural Additive Tabular Transformer Networks (NATT) combine the interpretability of additive neural networks with the predictive power of Transformer models. Categorical features are modelled with Transformer Encoders.

## NAMLSS: Neural Additive Models for Location Scale and Shape
**Link:** [NAMLSS Paper](https://proceedings.mlr.press/v238/frederik-thielmann24a/frederik-thielmann24a.pdf)

**Summary:**
Neural Additive Models for Location Scale and Shape (NAMLSS) integrate distributional regression with additive neural networks, extending beyond mean response predictions.

## NAIM: Neural Additive Image Models
**Link:** [NAIM Paper](https://arxiv.org/pdf/2405.02295)

**Summary:**
Neural Additive Image Models (NAIM) utilize Neural Additive Models and Diffusion Autoencoders to identify latent image semantics and their effects. NAIM demonstrates the ability to explore complex image effects, with a case study highlighting the impact of image characteristics on Airbnb pricing.



## SNAM: Structural Neural Additive Models
**Link:** [SNAM Paper](https://arxiv.org/pdf/2302.09275)

**Summary:**
Structural Neural Additive Models (SNAMs) enhance the interpretability of neural networks by combining classical statistical methods (Splines) with neural applications. Fitting NAMs with Splines instead of MLPs and optimizing knot locations.

## Semi-Structured Distributional Regression
**Link:** [Semi-Structured Distributional Regression Paper](https://www.tandfonline.com/doi/abs/10.1080/00031305.2022.2164054)

**Summary:**
This framework combines structured regression models with deep neural networks, addressing identifiability issues through an orthogonalization cell. It enables stable estimation and interpretability, demonstrated through numerical experiments and real-world applications.
