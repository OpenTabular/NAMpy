Link power and variance are independent
---------------------------------------

The quasi model specifies a link and a mean-variance relation separately:

.. math::

   g(\mu)=\eta,\qquad\operatorname{Var}(Y\mid x)=\phi V(\mu).

``variance`` selects exactly one of the following forms; whitespace is removed
from the option string before matching:

* ``'constant'``: :math:`V(\mu)=1`; real means are permitted.
* ``'mu'``: :math:`V(\mu)=\mu`; require :math:`\mu>0`.
* ``'mu(1-mu)'``: :math:`V(\mu)=\mu(1-\mu)`; require :math:`0<\mu<1`.
* ``'mu^2'``: :math:`V(\mu)=\mu^2`; require :math:`\mu>0`.
* ``'mu^3'``: :math:`V(\mu)=\mu^3`; require :math:`\mu>0`.

``power`` selects a **link exponent**, not a variance exponent. Its actual
constructor dispatch is

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

For direct ``QuasiFamily(...)`` construction, ``**kwargs`` accepts the inherited
``eps`` float, default ``1e-9``. It is a numerical safeguard, not dispersion or
response noise, and not every link uses it instead of machine epsilon. The
options-mapping factory accepts only ``name``, ``link``, ``variance`` and
``power`` for this family; it rejects an ``eps`` key. The named quasi classes
also forward inherited ``eps`` and numeric ``power`` in direct construction,
subject to the same link/power compatibility rules; their factory mappings
remain restricted to ``name`` and ``link``.

The quasi objective supplies estimating equations rather than a normalized
response law. Its scale and supported extended-quasi smoothing criteria are
distinct from the link exponent. Likelihood, AIC and BIC outputs follow the
quasi conventions documented below, not an invented density.
