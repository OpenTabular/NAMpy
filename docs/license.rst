License
=======

NAMpy is licensed under the MIT License.

Upstream numerical routines
---------------------------

The censored-Poisson implementation includes Python translations of R's
``dpois_raw``, ``ebd0``, integer ``stirlerr`` and log-space Poisson tail
routines, with constants from R 4.6's ``src/nmath``. Their upstream authors
include Catherine Loader, Morten Welinder, Ross Ihaka, the R Core Team and
the R Foundation. These translations retain the upstream GPL-2.0-or-later
license; the MIT statement does not replace that license. Copyright and
license notices are included beside the translated modules. The upstream
license text is available from
`R's license collection <https://www.R-project.org/Licenses/GPL-2>`_.

The generated analytic ``shash`` and ``gevlss`` derivative translations
retain the ``mgcv`` ``R/gamlss.r`` notice: Copyright (c) Simon N. Wood
(2013--2025), distributed under GPL version 2 or later. The upstream
``shash`` family credits Matteo Fasiolo and Simon N. Wood. The packaged
``nampy/gam/families/R_MATH_LICENSE.txt`` also supplies the GPL version 2 text
for these translations.

MIT License
-----------

Copyright (c) 2024 Ananyapam De, Anton Thielmann, and OpenTabular Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Summary
-------

The MIT License is a permissive free software license that allows you to:

* ✓ Use the software commercially
* ✓ Modify the software
* ✓ Distribute the software
* ✓ Sublicense the software
* ✓ Use the software privately

With the following conditions:

* Include the license and copyright notice in all copies or substantial portions
* The software is provided "as is", without warranty

For the complete license text, see the
`LICENSE <https://github.com/OpenTabular/NAMpy/blob/main/LICENSE>`_
file in the repository.
