Research references
===================

These papers and methodological books describe the scientific foundations of
NAMpy's implemented models, losses, diagnostics and uncertainty procedures.
The tables identify the method associated with each reference. Stable paper
links also appear in the docstrings of the corresponding implementation modules.

A reference identifies mathematical provenance or explicitly labelled background;
it does not imply that every variant or guarantee in that paper is implemented.
Local capability declarations, task guides and validation evidence define the
supported scope. Existing upstream package/file/version notes identify numerical
port specifications separately from the papers.

.. contents:: Method groups
   :local:
   :depth: 1

GAMs, smooths and statistical inference
---------------------------------------

.. list-table:: Implemented methods and sources
   :header-rows: 1
   :widths: 48 52

   * - Method
     - References
   * - Additive models, likelihood and smoothing selection
     - [Hastie1986]_, [Wood2011]_, [Wood2016]_, [Wood2017]_
   * - Thin-plate, Duchon and spherical bases
     - [Wood2003]_, [Duchon1977]_, [Wahba1981]_
   * - P-splines and tensor-product penalties
     - [Eilers1996]_, [Wood2006]_
   * - Boundary-aware soap-film smooths
     - [Wood2008Soap]_
   * - Shape-constrained additive models
     - [Pya2015]_
   * - Extended Fellner-Schall and neighbourhood cross-validation
     - [WoodFasiolo2017]_, [WoodNCV2024]_
   * - Coefficient/function uncertainty
     - [Marra2012]_
   * - Distributed-lag cross-bases
     - [Gasparrini2010]_

.. [Hastie1986] Hastie, T., and Tibshirani, R. (1986).
   `Generalized Additive Models <https://doi.org/10.1214/ss/1177013604>`_.
   Statistical Science 1(3), 297–310.

.. [Wood2011] Wood, S. N. (2011).
   `Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models <https://doi.org/10.1111/j.1467-9868.2010.00749.x>`_.
   JRSS Series B 73(1), 3–36.

.. [Wood2016] Wood, S. N., Pya, N., and Säfken, B. (2016).
   `Smoothing Parameter and Model Selection for General Smooth Models <https://doi.org/10.1080/01621459.2016.1180986>`_.
   JASA 111(516), 1548–1563.

.. [Wood2017] Wood, S. N. (2017).
   `Generalized Additive Models: An Introduction with R <https://doi.org/10.1201/9781315370279>`_.
   Second edition. Chapman and Hall/CRC. Methodological book.

.. [Wood2003] Wood, S. N. (2003).
   `Thin plate regression splines <https://doi.org/10.1111/1467-9868.00374>`_.
   JRSS Series B 65(1), 95–114.

.. [Duchon1977] Duchon, J. (1977).
   `Splines minimizing rotation-invariant semi-norms in Sobolev spaces <https://doi.org/10.1007/BFb0086566>`_.
   Constructive Theory of Functions of Several Variables, Lecture Notes in Mathematics 571, 85–100.

.. [Wahba1981] Wahba, G. (1981).
   `Spline Interpolation and Smoothing on the Sphere <https://doi.org/10.1137/0902002>`_.
   SIAM Journal on Scientific and Statistical Computing 2(1), 5–16.

.. [Eilers1996] Eilers, P. H. C., and Marx, B. D. (1996).
   `Flexible smoothing with B-splines and penalties <https://doi.org/10.1214/ss/1038425655>`_.
   Statistical Science 11(2), 89–121.

.. [Wood2006] Wood, S. N. (2006).
   `Low-Rank Scale-Invariant Tensor Product Smooths for Generalized Additive Mixed Models <https://doi.org/10.1111/j.1541-0420.2006.00574.x>`_.
   Biometrics 62(4), 1025–1036.

.. [Wood2008Soap] Wood, S. N., Bravington, M. V., and Hedley, S. L. (2008).
   `Soap film smoothing <https://doi.org/10.1111/j.1467-9868.2008.00665.x>`_.
   JRSS Series B 70(5), 931–955.

.. [Pya2015] Pya, N., and Wood, S. N. (2015).
   `Shape constrained additive models <https://doi.org/10.1007/s11222-013-9448-7>`_.
   Statistics and Computing 25, 543–559.

.. [WoodFasiolo2017] Wood, S. N., and Fasiolo, M. (2017).
   `A generalized Fellner-Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models <https://doi.org/10.1111/biom.12666>`_.
   Biometrics 73(4), 1071–1081.

.. [WoodNCV2024] Wood, S. N. (2024).
   `On Neighbourhood Cross Validation <https://arxiv.org/abs/2404.16490>`_.
   arXiv:2404.16490; revised 2025, accepted for Statistical Science.

.. [Marra2012] Marra, G., and Wood, S. N. (2012).
   `Coverage Properties of Confidence Intervals for Generalized Additive Model Components <https://doi.org/10.1111/j.1467-9469.2011.00760.x>`_.
   Scandinavian Journal of Statistics 39(1), 53–74.

.. [Gasparrini2010] Gasparrini, A., Armstrong, B., and Kenward, M. G. (2010).
   `Distributed lag non-linear models <https://doi.org/10.1002/sim.3940>`_.
   Statistics in Medicine 29(21), 2224–2234.

Quantile and expectile regression
---------------------------------

.. list-table:: Implemented methods and sources
   :header-rows: 1
   :widths: 48 52

   * - Method
     - References
   * - Pinball loss and asymmetric least squares
     - [Koenker1978]_, [Newey1987]_
   * - qgam ELF loss, bandwidth and loss-scale calibration
     - [FasioloQuantile2021]_, [FasioloQgam2021]_

The qgam references concern loss-based quantile-function inference. QNAM
uses a neural additive representation with pinball loss; it does not inherit
qgam calibration merely by estimating quantiles.

.. [Koenker1978] Koenker, R., and Bassett, G., Jr. (1978).
   `Regression Quantiles <https://doi.org/10.2307/1913643>`_.
   Econometrica 46(1), 33–50.

.. [Newey1987] Newey, W. K., and Powell, J. L. (1987).
   `Asymmetric Least Squares Estimation and Testing <https://doi.org/10.2307/1911031>`_.
   Econometrica 55(4), 819–847.

.. [FasioloQuantile2021] Fasiolo, M., Wood, S. N., Zaffran, M., Nedellec, R., and Goude, Y. (2021).
   `Fast Calibrated Additive Quantile Regression <https://doi.org/10.1080/01621459.2020.1725521>`_.
   JASA 116(535), 1402–1412. Theory and calibration.

.. [FasioloQgam2021] Fasiolo, M., Wood, S. N., Zaffran, M., Nedellec, R., and Goude, Y. (2021).
   `qgam: Bayesian Nonparametric Quantile Regression Modeling in R <https://doi.org/10.18637/jss.v100.i09>`_.
   Journal of Statistical Software 100(9), 1–31. Software and implementation.

Fisher information diagnostics
------------------------------

.. list-table:: Implemented methods and sources
   :header-rows: 1
   :widths: 48 52

   * - Method
     - References
   * - Score-outer-product and negative-Hessian estimators and their variances
     - [Soen2021]_, [Soen2024]_

The two Soen--Sun papers motivate the estimator and variance diagnostics.
NAMpy applies the likelihood definitions across eligible neural and statistical
models, reports mean information per reference input, and validates each
likelihood/architecture explicitly. Information-estimation error is separate
from coefficient uncertainty.

.. [Soen2021] Soen, A., and Sun, K. (2021).
   `On the Variance of the Fisher Information for Deep Learning <https://proceedings.neurips.cc/paper_files/paper/2021/hash/2d290e496d16c9dcaa9b4ded5cac10cc-Abstract.html>`_.
   NeurIPS 34.

.. [Soen2024] Soen, A., and Sun, K. (2024).
   `Trade-Offs of Diagonal Fisher Information Matrix Estimators <https://proceedings.neurips.cc/paper_files/paper/2024/hash/0b43289db08ed60edc6451cb2132e203-Abstract-Conference.html>`_.
   NeurIPS 37.

Distributional regression and response families
-----------------------------------------------

.. list-table:: Implemented methods and sources
   :header-rows: 1
   :widths: 48 52

   * - Method
     - References
   * - GAMLSS framework and multi-parameter regression
     - [Rigby2005]_
   * - BCCG/LMS, BCPE and BCT
     - [Cole1992]_, [Rigby2004]_, [Rigby2006]_
   * - Beta, inflated beta and ordinal cumulative-link models
     - [Ferrari2004]_, [Ospina2010]_, [McCullagh1980]_
   * - Tweedie series, sinh-arcsinh and zero-inflated Poisson families
     - [Dunn2005]_, [Jones2009]_, [Lambert1992]_
   * - Censored and truncated observation likelihoods
     - [Lawless2003]_

.. [Rigby2005] Rigby, R. A., and Stasinopoulos, D. M. (2005).
   `Generalized additive models for location, scale and shape <https://doi.org/10.1111/j.1467-9876.2005.00510.x>`_.
   JRSS Series C (Applied Statistics) 54(3), 507–554.

.. [Cole1992] Cole, T. J., and Green, P. J. (1992).
   `Smoothing reference centile curves: the LMS method and penalized likelihood <https://doi.org/10.1002/sim.4780111005>`_.
   Statistics in Medicine 11(10), 1305–1319.

.. [Rigby2004] Rigby, R. A., and Stasinopoulos, D. M. (2004).
   `Smooth centile curves for skew and kurtotic data modelled using the Box-Cox power exponential distribution <https://doi.org/10.1002/sim.1861>`_.
   Statistics in Medicine 23(19), 3053–3076.

.. [Rigby2006] Rigby, R. A., and Stasinopoulos, D. M. (2006).
   `Using the Box-Cox t distribution in GAMLSS to model skewness and kurtosis <https://doi.org/10.1191/1471082X06st122oa>`_.
   Statistical Modelling 6(3), 209–229.

.. [Ferrari2004] Ferrari, S. L. P., and Cribari-Neto, F. (2004).
   `Beta Regression for Modelling Rates and Proportions <https://doi.org/10.1080/0266476042000214501>`_.
   Journal of Applied Statistics 31(7), 799–815.

.. [Ospina2010] Ospina, R., and Ferrari, S. L. P. (2010).
   `Inflated beta distributions <https://doi.org/10.1007/s00362-008-0125-4>`_.
   Statistical Papers 51, 111–126.

.. [McCullagh1980] McCullagh, P. (1980).
   `Regression Models for Ordinal Data <https://doi.org/10.1111/j.2517-6161.1980.tb01109.x>`_.
   JRSS Series B 42(2), 109–142.

.. [Dunn2005] Dunn, P. K., and Smyth, G. K. (2005).
   `Series evaluation of Tweedie exponential dispersion model densities <https://doi.org/10.1007/s11222-005-4070-y>`_.
   Statistics and Computing 15, 267–280.

.. [Jones2009] Jones, M. C., and Pewsey, A. (2009).
   `Sinh-arcsinh distributions <https://doi.org/10.1093/biomet/asp053>`_.
   Biometrika 96(4), 761–780.

.. [Lambert1992] Lambert, D. (1992).
   `Zero-Inflated Poisson Regression, With an Application to Defects in Manufacturing <https://doi.org/10.1080/00401706.1992.10485228>`_.
   Technometrics 34(1), 1–14.

.. [Lawless2003] Lawless, J. F. (2003).
   `Statistical Models and Methods for Lifetime Data <https://doi.org/10.1002/9781118033005>`_.
   Second edition. Wiley. Methodological book; censoring and truncation likelihoods.

Neural additive architectures and interactions
----------------------------------------------

.. list-table:: Implemented methods and sources
   :header-rows: 1
   :widths: 48 52

   * - Method
     - References
   * - NAM and group-sparse NAM
     - [Agarwal2021]_, [Xu2022]_
   * - NODE-GAM and sparse interaction networks
     - [Chang2022]_, [Enouen2023]_
   * - GP-NAM and IGANN
     - [Zhang2024]_, [Kraus2024]_
   * - Neural basis and polynomial additive models
     - [Radenovic2022]_, [Dubey2022]_
   * - NATT and NAMformer
     - [ThielmannNATT2024]_, [Thielmann2025]_
   * - Neural location/scale/shape models
     - [ThielmannLSS2024]_, [De2024]_
   * - Semi-structured distributional models
     - [Ruegamer2024]_
   * - Soft-tree background and Archipelago interaction detection
     - [Frosst2017]_, [Tsang2020]_

QNAM, SplineNAM, TreeNAM, EnsembleTreeNAM and NBM-SPAM combine the
referenced components according to their local architecture contracts. The
soft-tree paper is architectural background; NAMpy does not claim its teacher
distillation procedure. Preprint and workshop status are retained below.

.. [Agarwal2021] Agarwal, R., et al. (2021).
   `Neural Additive Models: Interpretable Machine Learning with Neural Nets <https://arxiv.org/abs/2004.13912>`_.
   NeurIPS 34.

.. [Xu2022] Xu, S., Bu, Z., Chaudhari, P., and Barnett, I. J. (2022).
   `Sparse Neural Additive Model: Interpretable Deep Learning with Feature Selection via Group Sparsity <https://arxiv.org/abs/2202.12482>`_.
   arXiv:2202.12482.

.. [Chang2022] Chang, C.-H., Caruana, R., and Goldenberg, A. (2022).
   `NODE-GAM: Neural Generalized Additive Model for Interpretable Deep Learning <https://arxiv.org/abs/2106.01613>`_.
   ICLR 2022.

.. [Enouen2023] Enouen, J., and Liu, Y. (2023).
   `Sparse Interaction Additive Networks via Feature Interaction Detection and Sparse Selection <https://arxiv.org/abs/2209.09326>`_.
   arXiv:2209.09326, revised 2023.

.. [Zhang2024] Zhang, W., Barr, B., and Paisley, J. (2024).
   `Gaussian Process Neural Additive Models <https://arxiv.org/abs/2402.12518>`_.
   arXiv:2402.12518.

.. [Kraus2024] Kraus, M., Tschernutter, D., Weinzierl, S., and Zschech, P. (2024).
   `Interpretable generalized additive neural networks <https://doi.org/10.1016/j.ejor.2023.06.032>`_.
   European Journal of Operational Research 317(2), 303–316.

.. [Radenovic2022] Radenovic, F., Dubey, A., and Mahajan, D. (2022).
   `Neural Basis Models for Interpretability <https://arxiv.org/abs/2205.14120>`_.
   arXiv:2205.14120.

.. [Dubey2022] Dubey, A., Radenovic, F., and Mahajan, D. (2022).
   `Scalable Interpretability via Polynomials <https://arxiv.org/abs/2205.14108>`_.
   arXiv:2205.14108.

.. [ThielmannNATT2024] Thielmann, A. F., Reuter, A., Kneib, T., Rügamer, D., and Säfken, B. (2024).
   `Interpretable Additive Tabular Transformer Networks <https://openreview.net/forum?id=TdJ7lpzAkD>`_.
   Transactions on Machine Learning Research.

.. [Thielmann2025] Thielmann, A., Reuter, A., and Säfken, B. (2025).
   `Beyond Black-Box Predictions: Identifying Marginal Feature Effects in Tabular Transformer Networks <https://arxiv.org/abs/2504.08712>`_.
   arXiv:2504.08712.

.. [ThielmannLSS2024] Thielmann, A. F., Kruse, R.-M., Kneib, T., and Säfken, B. (2024).
   `Neural Additive Models for Location Scale and Shape: A Framework for Interpretable Neural Regression Beyond the Mean <https://proceedings.mlr.press/v238/frederik-thielmann24a.html>`_.
   AISTATS, PMLR 238, 1783–1791.

.. [De2024] De, A., Thielmann, A., and Säfken, B. (2024).
   `NODE-GAMLSS: Interpretable Uncertainty Modelling via Deep Distributional Regression <https://openreview.net/forum?id=sWvTLlRkzy>`_.
   NeurIPS Workshop on Bayesian Decision-Making and Uncertainty.

.. [Ruegamer2024] Rügamer, D., Kolb, C., and Klein, N. (2024).
   `Semi-Structured Distributional Regression <https://doi.org/10.1080/00031305.2022.2164054>`_.
   The American Statistician 78(1), 88–99.

.. [Frosst2017] Frosst, N., and Hinton, G. (2017).
   `Distilling a Neural Network Into a Soft Decision Tree <https://arxiv.org/abs/1711.09784>`_.
   CEX workshop, AI*IA 2017; arXiv:1711.09784.

.. [Tsang2020] Tsang, M., Rambhatla, S., and Liu, Y. (2020).
   `How does this interaction affect me? Interpretable attribution for feature interactions <https://arxiv.org/abs/2006.10965>`_.
   NeurIPS 33.

Survival models, curves and evaluation
--------------------------------------

.. list-table:: Implemented methods and sources
   :header-rows: 1
   :widths: 48 52

   * - Method
     - References
   * - Cox partial likelihood and Breslow ties/baseline
     - [Cox1972]_, [Breslow1974]_
   * - Piecewise-exponential/Poisson representation
     - [Laird1981]_
   * - Neural Cox background
     - [Katzman2018]_
   * - Kaplan-Meier and cumulative-hazard estimation
     - [Kaplan1958]_, [Aalen1978]_
   * - IPCW Brier score and Harrell concordance
     - [Gerds2006]_, [Harrell1982]_
   * - Restricted mean survival time
     - [Royston2013]_
   * - Subject bootstrap
     - [Efron1979]_

The neural Cox route uses NAMpy’s registered architectures. DeepSurv is
background for neural Cox modeling, rather than a claim that NAMpy reproduces
that complete architecture or treatment-recommendation workflow. The implemented
IPCW evaluation assumes marginal censoring independence; the broader conditional
censoring results of Gerds and Schumacher do not expand that implementation.

.. [Cox1972] Cox, D. R. (1972).
   `Regression Models and Life-Tables <https://doi.org/10.1111/j.2517-6161.1972.tb00899.x>`_.
   JRSS Series B 34(2), 187–220 (with discussion).

.. [Breslow1974] Breslow, N. (1974).
   `Covariance Analysis of Censored Survival Data <https://www.jstor.org/stable/2529620>`_.
   Biometrics 30(1), 89–99.

.. [Laird1981] Laird, N., and Olivier, D. (1981).
   `Covariance Analysis of Censored Survival Data Using Log-Linear Analysis Techniques <https://doi.org/10.1080/01621459.1981.10477634>`_.
   JASA 76(374), 231–240.

.. [Katzman2018] Katzman, J. L., et al. (2018).
   `DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network <https://doi.org/10.1186/s12874-018-0482-1>`_.
   BMC Medical Research Methodology 18, 24. Neural Cox background.

.. [Kaplan1958] Kaplan, E. L., and Meier, P. (1958).
   `Nonparametric Estimation from Incomplete Observations <https://doi.org/10.2307/2281868>`_.
   JASA 53(282), 457–481.

.. [Aalen1978] Aalen, O. (1978).
   `Nonparametric Inference for a Family of Counting Processes <https://doi.org/10.1214/aos/1176344247>`_.
   The Annals of Statistics 6(4), 701–726.

.. [Gerds2006] Gerds, T. A., and Schumacher, M. (2006).
   `Consistent estimation of the expected Brier score in general survival models with right-censored event times <https://doi.org/10.1002/bimj.200610301>`_.
   Biometrical Journal 48(6), 1029–1040.

.. [Harrell1982] Harrell, F. E., Jr., et al. (1982).
   `Evaluating the Yield of Medical Tests <https://doi.org/10.1001/jama.1982.03320430047030>`_.
   JAMA 247(18), 2543–2546.

.. [Royston2013] Royston, P., and Parmar, M. K. B. (2013).
   `Restricted mean survival time: an alternative to the hazard ratio for the design and analysis of randomized trials with a time-to-event outcome <https://doi.org/10.1186/1471-2288-13-152>`_.
   BMC Medical Research Methodology 13, 152.

.. [Efron1979] Efron, B. (1979).
   `Bootstrap Methods: Another Look at the Jackknife <https://doi.org/10.1214/aos/1176344552>`_.
   The Annals of Statistics 7(1), 1–26.

Calibration and distribution diagnostics
----------------------------------------

.. list-table:: Implemented methods and sources
   :header-rows: 1
   :widths: 48 52

   * - Method
     - References
   * - Split-conformal and conformalized quantile regression
     - [Lei2018]_, [Romano2019]_
   * - Density-ratio and group-conditional constructions
     - [Tibshirani2019]_, [Vovk2012]_
   * - Censor-aware lower survival bounds
     - [Candes2023]_
   * - Quantile residuals and worm plots
     - [Dunn1996]_, [VanBuuren2001]_

.. [Lei2018] Lei, J., G’Sell, M., Rinaldo, A., Tibshirani, R. J., and Wasserman, L. (2018).
   `Distribution-Free Predictive Inference for Regression <https://doi.org/10.1080/01621459.2017.1307116>`_.
   JASA 113(523), 1094–1111.

.. [Romano2019] Romano, Y., Patterson, E., and Candès, E. J. (2019).
   `Conformalized Quantile Regression <https://arxiv.org/abs/1905.03222>`_.
   NeurIPS 32.

.. [Tibshirani2019] Tibshirani, R. J., Barber, R. F., Candès, E. J., and Ramdas, A. (2019).
   `Conformal Prediction Under Covariate Shift <https://arxiv.org/abs/1904.06019>`_.
   NeurIPS 32.

.. [Vovk2012] Vovk, V. (2012).
   `Conditional Validity of Inductive Conformal Predictors <https://proceedings.mlr.press/v25/vovk12.html>`_.
   ACML, PMLR 25, 475–490.

.. [Candes2023] Candès, E. J., Lei, L., and Ren, Z. (2023).
   `Conformalized Survival Analysis <https://arxiv.org/abs/2103.09763>`_.
   JRSS Series B 85(1), 24–45.

.. [Dunn1996] Dunn, P. K., and Smyth, G. K. (1996).
   `Randomized Quantile Residuals <https://doi.org/10.1080/10618600.1996.10474708>`_.
   Journal of Computational and Graphical Statistics 5(3), 236–244.

.. [VanBuuren2001] van Buuren, S., and Fredriks, M. (2001).
   `Worm plot: a simple diagnostic device for modelling growth reference curves <https://doi.org/10.1002/sim.746>`_.
   Statistics in Medicine 20(8), 1259–1277.
