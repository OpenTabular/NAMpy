**Model summary**

.. code-block:: text


   Family: gaussian
   Link function: identity

   Formula:
   y ~ s(x, bs='cr', k=12) + group

   Parametric coefficients:
                    Estimate   Std. Error      t value     Pr(>|t|)
   (Intercept)     -0.011074      0.01208     -0.91674         0.36
   group[b]       -0.0016233     0.016558    -0.098038        0.922
   ---
   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

   Approximate significance of smooth terms:
                         edf       Ref.df            F      p-value
   s(x)               9.5029       10.521       695.87       <2e-16 ***
   ---
   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

   R-sq.(adj) = 0.961   Deviance explained = 96.2%
   -REML = -136.28  Scale est. = 0.020088  n = 300

.. csv-table:: Additive components — first five observations
   :header-rows: 1

   x,group,intercept,group[b],"s(x, bs='cr', k=12)",prediction
   0.0000,b,-0.0111,-0.0016,-0.0512,-0.0639
   0.0033,b,-0.0111,-0.0016,-0.0280,-0.0407
   0.0067,b,-0.0111,-0.0016,-0.0048,-0.0175
   0.0100,b,-0.0111,-0.0016,0.0183,0.0056
   0.0134,b,-0.0111,-0.0016,0.0414,0.0287

For the first displayed row, intercept plus fitted contributions gives ``(-0.0111) + (-0.0016) + (-0.0512) = -0.0639``, up to displayed rounding. The x and group columns are inputs, not additional summands.
