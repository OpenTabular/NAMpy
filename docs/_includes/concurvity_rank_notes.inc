Full concurvity follows mgcv's QR calculation without dropping dependent
columns. With repeated covariates across predictors, the joint design can be
rank deficient and these diagnostics can vary across numerical libraries.
Pairwise comparisons remain identifiable when the individual compared blocks
have full column rank; a full-design estimate is not a portable reference in
this case.
