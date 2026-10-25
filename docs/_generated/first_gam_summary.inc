.. code-block:: text


   Family: gaussian
   Link function: identity

   Formula:
   demand ~ s(temperature, bs="cr", k=12) + s(hour, bs="cc", k=12)

   Parametric coefficients:
                    Estimate   Std. Error      t value     Pr(>|t|)
   (Intercept)        61.745     0.078256       789.01       <2e-16 ***
   ---
   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

   Approximate significance of smooth terms:
                            edf       Ref.df            F      p-value
   s(temperature)         9.977        10.76       4077.7       <2e-16 ***
   s(hour)               7.1145           10       200.93       <2e-16 ***
   ---
   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

   R-sq.(adj) = 0.984   Deviance explained = 98.5%
   -REML = 1586.6  Scale est. = 4.4093  n = 720
