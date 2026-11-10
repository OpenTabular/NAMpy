Link power and variance are independent
---------------------------------------

The quasi model specifies a link and a mean-variance relation separately:

.. math::

   g(\mu)=\eta,\qquad\operatorname{Var}(Y\mid x)=\phi V(\mu).

Variance functions
~~~~~~~~~~~~~~~~~~

``variance`` selects one of these forms. Whitespace is removed before matching.

* ``'constant'``: :math:`V(\mu)=1`; real means are permitted.
* ``'mu'``: :math:`V(\mu)=\mu`; require :math:`\mu>0`.
* ``'mu(1-mu)'``: :math:`V(\mu)=\mu(1-\mu)`; require :math:`0<\mu<1`.
* ``'mu^2'``: :math:`V(\mu)=\mu^2`; require :math:`\mu>0`.
* ``'mu^3'``: :math:`V(\mu)=\mu^3`; require :math:`\mu>0`.

Link exponent
~~~~~~~~~~~~~

``power`` controls the link exponent independently of ``variance``.

.. math::

   g(\mu)=\begin{cases}
   \log\mu & a\leq0,\\
   \mu & a=1,\\
   \mu^a & a>0,\ a\ne1,
   \end{cases}\qquad a=\texttt{power}.

With ``power=None``, ``link='identity'`` is the general quasi default. The
nine named links are ``identity``, ``log``, ``inverse``, ``sqrt``, ``1/mu^2``,
``logit``, ``probit``, ``cloglog`` and ``cauchit``; see the
:doc:`complete link contracts </statistical_reference/links/index>` for
formulas, domains and inverse derivatives.

A supplied power must be a finite real scalar, excluding booleans. The
constructor accepts it with an omitted link, ``link='identity'`` or
``link='power'``; combining it with another named link raises ``ValueError``.
``link='power'`` alone is not a registered named link. Negative powers use the
implemented log-link dispatch; they do not request negative-power links.

For example, ``{'name':'quasi', 'variance':'mu^3', 'power':0.5}`` combines
:math:`\operatorname{Var}(Y\mid x)=\phi\mu^3` with
:math:`\eta=\sqrt\mu`. The runnable example below uses this configuration.
The separate ``quasipoisson`` and ``quasibinomial`` factory mappings accept
only ``link`` and retain their named variance functions; custom variance/power
options belong to general ``quasi``.

Constructor and factory options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Direct construction.** ``QuasiFamily(...)`` accepts inherited ``eps`` via
  ``**kwargs``, default ``1e-9``. It safeguards link calculations where used.
  The named quasi classes also forward ``eps`` and numeric ``power`` subject
  to the link compatibility rules.
* **Factory mappings.** General ``quasi`` accepts ``name``, ``link``,
  ``variance`` and ``power``. ``quasipoisson`` and ``quasibinomial`` accept
  ``name`` and ``link``.

The quasi objective supplies estimating equations rather than a normalized
response law. Its scale and supported extended-quasi smoothing criteria are
distinct from the link exponent. Likelihood, AIC and BIC outputs follow the
quasi conventions documented below, not an invented density.
