Which information changes the time until an event?
==================================================

A customer who is still active at the end of follow-up is censored. That time is not an event time, and dropping the row removes information. The :doc:`censoring tutorial </notebooks/survival_censoring>` follows 480 synthetic independent subjects with one Weibull model. Time units are arbitrary. The scale depends on one predictor and the shape is constant, matching the simulation so the example can show censoring, splitting, and scoring.

What is the problem?
--------------------

The question is how a covariate shifts time-to-event when some subjects have not had the event yet. Each subject has one covariate ``x`` drawn uniformly on −1 to 1. The latent event time is ``exp(0.6 − 0.8 x)`` times a Weibull draw with shape 1.6. An independent exponential censoring time with mean 4 is drawn for every subject. The recorded time is the minimum of the two, and the event indicator is whether the event came first. Churn covariates beyond that single predictor would be added to the scale formula. This notebook does not fit a multi-covariate retention model.

Why is a linear model insufficient?
-----------------------------------

A linear regression of the recorded time treats a censored follow-up as if the event occurred then. That changes the outcome: a subject still active at time 3 is recorded as 3, and a regression of that number pulls the fitted mean toward the censoring time. The Weibull likelihood uses the density when the event is observed and the survival function when it is not. No separate censoring model is fit in this example. The censoring distribution is estimated afterwards, and only for the IPCW Brier score.

Why this model?
---------------

``SurvivalGAM(likelihood="weibull", formula={"scale": "~x", "shape": "~1"})`` is one parametric survival model. ``SurvivalTarget`` keeps the event indicator separate from ``x``. Smoothing is fixed because both parameter predictors are linear: the scale is a straight line in ``x``, and the shape is a single positive constant. Hazard and survival are different functions of the same fit. Hazard is a rate among subjects still at risk, and survival is the probability of remaining event-free. A hazard value can exceed one because it is a rate.

What the fit showed
-------------------

The split keeps every subject in one cohort: 320 subjects train the model and 160 are held out. About 33% of the held-out records are censored. The printed share on this fixed-seed run is 33.1%. The integrated IPCW Brier error is about 0.188 and Harrell concordance is about 0.669.

The Brier score is computed at horizons 0.5, 1.0, 1.5, and 2.0, which stay inside the observed training and test follow-up. The integrated score averages those horizons. The censoring distribution used for the inverse-probability weights is fit on the training outcomes and then supplied to the test score. Concordance uses negative restricted mean survival time at horizon 2 as the risk score, so a shorter expected remaining time is a higher risk. The two scores answer different questions: Brier error is about predicted event probabilities, and concordance is about ordering subjects.

.. image:: /_static/features/survival_profiles.png
   :alt: Twenty-eight held-out follow-up records, with events and right censoring marked, beside fitted survival curves at three covariate values
   :class: scientific-figure
   :align: center

The left panel is 28 of the held-out records, ordered by recorded time. A horizontal segment runs from time zero to the recorded time. A circle is an observed event and a cross is right censoring. Those censored times stay in the likelihood as survival probabilities. The right panel is the fitted survival curve for three profiles, ``x = −0.8``, ``0``, and ``0.8``. The vertical axis is ``P(T > time | x)``. At time 2 the notebook reports those three probabilities as approximately 0.65, 0.31, and 0.04. Larger ``x`` shortens event-free survival in this simulation. The curve is a probability for a profile, not a predicted event time for one person.

.. image:: /_static/features/survival_hazard.png
   :alt: Fitted Weibull hazard and survival curves for the same three covariate profiles
   :class: scientific-figure
   :align: center

The same three profiles are drawn again as a hazard and as a survival curve. The fitted Weibull shape is greater than one, so the hazard in the left panel rises with time while the survival curve in the right panel falls. The hazard axis is events per unit of the arbitrary time. A hazard is a rate, so a value above one is allowed. Survival stays between zero and one because it is the probability of still being event-free.

Limits of the interpretation
----------------------------

These scores are one simulated dataset. Marginal independent censoring holds in the simulation and is not a demonstration of covariate-dependent censoring. Concordance and Brier error answer different questions. The tutorial does not compare a Cox model. That comparison is a later analysis, described in the :ref:`survival guide <howto-user-guide-survival>`. Adding further covariates belongs in the scale formula. It is a different fit from the one-predictor example scored here.

:download:`Download notebook <../notebooks/survival_censoring.ipynb>`
