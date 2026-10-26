"""Source-generated gevlss derivatives from mgcv 1.9-4 R/gamlss.r.

Mechanical translation of the source Maxima output, retaining its temporary
names, operation ordering and packed column order. Regeneration script:
nampy_notes/debug/translate_gevlss_derivatives.py (development source only).

Upstream Copyright (c) Simon N. Wood (2013-2025), GNU GPL version 2 or later.
Python translation modified for NumPy arrays. WITHOUT ANY WARRANTY;
see ../R_MATH_LICENSE.txt.
"""

import numpy as np


def gevlss_derivatives(y, mu, rho, xi, deriv):
    n = len(y)
    l1 = np.zeros((n, 3))
    l2 = np.zeros((n, 6))
    l3 = l4 = 0
    exp1 = np.exp(1)
    ymu = y - mu
    aa0 = (xi*ymu)/exp1**rho
    log_aa1 = np.log1p(aa0)
    aa1 = aa0 + 1
    aa2 = 1/xi
    l0 = (-aa2*(1+xi)*log_aa1)-1/aa1**aa2-rho
    if deriv:
        bb1 = 1/exp1**rho
        bb2 = bb1*xi*ymu+1
        l1[:, 0] = (bb1*(xi+1))/bb2-bb1*bb2**((-1/xi)-1)
        cc2 = ymu
        cc0 = bb1*xi*cc2
        log_cc3 = np.log1p(cc0)
        cc3 = cc0 + 1
        l1[:, 1] = (-bb1*cc2*cc3**((-1/xi)-1))+(bb1*(xi+1)*cc2)/cc3-1
        dd3 = xi+1
        dd6 = 1/cc3
        dd7 = log_cc3
        dd8 = 1/xi**2
        l1[:, 2] = (-(dd8*dd7-bb1*aa2*cc2*dd6)/cc3**aa2)+dd8*dd3*dd7- aa2*dd7-bb1*aa2*dd3*cc2*dd6
        ee1 = 1/exp1**(2*rho)
        ee3 = -1/xi
        l2[:, 0] = ee1*(ee3-1)*xi*aa1**(ee3-2)+(ee1*xi*(xi+1))/aa1**2
        ff7 = ee3-1
        l2[:, 1] = bb1*cc3**ff7+ee1*ff7*xi*cc2*cc3**(ee3-2)-(bb1*dd3)/cc3+ (ee1*xi*dd3*cc2)/cc3**2
        gg7 = -aa2
        l2[:, 2] = (-bb1*cc3**(gg7-1)*(log_cc3/xi**2-bb1*aa2*cc2*dd6))+ ee1*cc2*cc3**(gg7-2)+bb1*dd6-(ee1*(xi+1)*cc2)/cc3**2
        hh4 = cc2**2
        l2[:, 3] = bb1*cc2*cc3**ff7+ee1*ff7*xi*hh4*cc3**(ee3-2)- (bb1*dd3*cc2)/cc3+(ee1*xi*dd3*hh4)/cc3**2
        l2[:, 4] = (-bb1*cc2*cc3**(gg7-1)*(log_cc3/xi**2-bb1*aa2*cc2*dd6))+ ee1*hh4*cc3**(gg7-2)+bb1*cc2*dd6-(ee1*(xi+1)*hh4)/cc3**2
        jj08 = 1/cc3**2
        jj12 = 1/xi**3
        jj13 = 1/cc3**aa2
        l2[:, 5] = (-jj13*(dd8*dd7-bb1*aa2*cc2*dd6)**2)-jj13*(ee1*aa2*hh4*jj08+ 2*bb1*dd8*cc2*dd6-2*jj12*dd7)-2*jj12*dd3*dd7+2*dd8*dd7+ 2*bb1*dd8*dd3*cc2*dd6-2*bb1*aa2*cc2*dd6+ee1*aa2*dd3*hh4*jj08
    if deriv > 1:
        l3 = np.zeros((n, 10))
        kk1 = 1/exp1**(3*rho)
        kk2 = xi**2
        l3[:, 0] = (2*kk1*kk2*(xi+1))/aa1**3-kk1*(ee3-2)*(ee3-1)*kk2*aa1**(ee3-3)
        ll5 = (xi*cc2)/exp1**rho+1
        ll8 = ee3-2
        l3[:, 1] = (-2*ee1*ff7*xi*ll5**ll8)-kk1*ll8*ff7*kk2*cc2*ll5**(ee3-3)- (2*ee1*xi*dd3)/ll5**2+(2*kk1*kk2*dd3*cc2)/ll5**3
        mm10 = cc3**(gg7-3)
        mm11 = gg7-2
        mm12 = cc3**mm11
        l3[:, 2] = ee1*(gg7-1)*xi*mm12*(log_cc3/xi**2-(bb1*aa2*cc2)/cc3)-ee1*mm12- kk1*mm11*xi*cc2*mm10+kk1*cc2*mm10+ee1*dd3*jj08+ee1*xi*jj08- (2*kk1*xi*dd3*cc2)/cc3**3
        l3[:, 3] = (-bb1*cc3**ff7)-3*ee1*ff7*xi*cc2*cc3**ll8-kk1*ll8*ff7*kk2*hh4* cc3**(ee3-3)+(bb1*dd3)/cc3-(3*ee1*xi*dd3*cc2)/cc3**2+ (2*kk1*kk2*dd3*hh4)/cc3**3
        oo10 = gg7-1
        oo13 = log_cc3/xi**2
        l3[:, 4] = bb1*cc3**oo10*(bb1*oo10*cc2*dd6+oo13)+ee1*oo10*xi*cc2*mm12* (bb1*mm11*cc2*dd6+oo13)+ee1*aa2*cc2*mm12+ee1*oo10*cc2*mm12- bb1*dd6+2*ee1*dd3*cc2*jj08+ee1*xi*cc2*jj08- (2*xi*dd3*cc2**2)/(exp1**(3*rho)*cc3**3)
        pp07 = (-1/xi)-1
        pp08 = cc3**pp07
        l3[:, 5] = (-bb1*pp08*(bb1*pp07*cc2*dd6+dd8*dd7)**2)-bb1*pp08* ((-ee1*pp07*hh4*jj08)+2*bb1*dd8*cc2*dd6-(2*dd7)/xi**3)- 2*ee1*cc2*jj08+(2*(xi+1)*hh4)/(exp1**(3*rho)*cc3**3)
        qq05 = cc2**3
        l3[:, 6] = (-bb1*cc2*cc3**ff7)-3*ee1*ff7*xi*hh4*cc3**ll8- kk1*ll8*ff7*kk2*qq05*cc3**(ee3-3)+(bb1*dd3*cc2)/cc3- (3*ee1*xi*dd3*hh4)/cc3**2+(2*kk1*kk2*dd3*qq05)/cc3**3
        rr17 = log_cc3/xi**2-bb1*aa2*cc2*dd6
        l3[:, 7] = bb1*cc2*cc3**oo10*rr17+ee1*oo10*xi*hh4*mm12*rr17-2*ee1*hh4*mm12- kk1*mm11*xi*qq05*mm10+kk1*qq05*mm10-bb1*cc2*dd6+ 2*ee1*dd3*hh4*jj08+ee1*xi*hh4*jj08-(2*kk1*xi*dd3*qq05)/cc3**3
        l3[:, 8] = (-bb1*cc2*pp08*(bb1*pp07*cc2*dd6+dd8*dd7)**2)-bb1*cc2*pp08* ((-ee1*pp07*hh4*jj08)+2*bb1*dd8*cc2*dd6-(2*dd7)/xi**3)- 2*ee1*hh4*jj08+(2*(xi+1)*cc2**3)/(exp1**(3*rho)*cc3**3)
        tt08 = 1/cc3**3
        tt16 = 1/xi**4
        tt18 = dd8*dd7-bb1*aa2*cc2*dd6
        l3[:, 9] = (-jj13*tt18**3)-3*jj13*(ee1*aa2*hh4*jj08+2*bb1*dd8*cc2*dd6-2*jj12*dd7)* tt18-jj13*((-2*kk1*aa2*qq05*tt08)-3*ee1*dd8*hh4*jj08-6*bb1*jj12*cc2*dd6+ 6*tt16*dd7)+6*tt16*dd3*dd7-6*jj12*dd7-6*bb1*jj12*dd3*cc2*dd6+ 6*bb1*dd8*cc2*dd6-3*ee1*dd8*dd3*hh4*jj08+3*ee1*aa2*hh4*jj08- 2*kk1*aa2*dd3*qq05*tt08
    if deriv > 3:
        l4 = np.zeros((n, 15))
        uu1 = 1/exp1**(4*rho)
        uu2 = xi**3
        l4[:, 0] = uu1*(ee3-3)*(ee3-2)*(ee3-1)*uu2*aa1**(ee3-4)+(6*uu1*uu2*(xi+1))/aa1**4
        vv09 = ee3-3
        l4[:, 1] = 3*kk1*ll8*ff7*kk2*ll5**vv09+uu1*vv09*ll8*ff7*uu2*cc2*ll5**(ee3-4)- (6*kk1*kk2*dd3)/ll5**3+(6*uu1*uu2*dd3*cc2)/ll5**4
        ww11 = gg7-3
        ww12 = cc3**(gg7-4)
        ww15 = cc3**ww11
        l4[:, 2] = (-kk1*mm11*oo10*kk2*ww15*(log_cc3/kk2-(bb1*aa2*cc2)/cc3))+ 2*kk1*mm11*xi*ww15-kk1*ww15+uu1*ww11*mm11*kk2*cc2*ww12- uu1*oo10*xi*cc2*ww12-uu1*ww11*xi*cc2*ww12+2*kk1*kk2*tt08+ 4*kk1*xi*dd3*tt08-(6*uu1*kk2*dd3*cc2)/cc3**4
        l4[:, 3] = 4*ee1*ff7*xi*ll5**ll8+5*kk1*ll8*ff7*kk2*cc2*ll5**vv09+ uu1*vv09*ll8*ff7*uu2*hh4*ll5**(ee3-4)+(4*ee1*xi*dd3)/ll5**2- (10*kk1*kk2*dd3*cc2)/ll5**3+(6*uu1*uu2*dd3*hh4)/ll5**4
        yy18 = log_cc3/kk2
        l4[:, 4] = (-2*ee1*oo10*xi*mm12*(bb1*mm11*cc2*dd6+yy18))- kk1*mm11*oo10*kk2*cc2*ww15*(bb1*ww11*cc2*dd6+yy18)- 2*ee1*aa2*mm12-2*ee1*oo10*mm12-2*kk1*mm11*oo10*xi*cc2*ww15- kk1*oo10*cc2*ww15-kk1*mm11*cc2*ww15-2*ee1*dd3*jj08- 2*ee1*xi*jj08+2*kk1*kk2*cc2*tt08+8*kk1*xi*dd3*cc2*tt08- (6*kk2*dd3*cc2**2)/(exp1**(4*rho)*cc3**4)
        l4[:, 5] = ee1*oo10*xi*mm12*tt18**2-2*ee1*mm12*tt18-2*kk1*mm11*xi*cc2*ww15*tt18+ 2*kk1*cc2*ww15*tt18+ee1*oo10*xi*mm12*(ee1*aa2*hh4*jj08+2*bb1* dd8*cc2*dd6-(2*dd7)/xi**3)+4*kk1*cc2*ww15+2*uu1*ww11*xi*hh4*ww12- 4*uu1*hh4*ww12+2*ee1*jj08-4*kk1*dd3*cc2*tt08-4*kk1*xi*cc2*tt08+ (6*uu1*xi*dd3*hh4)/cc3**4
        l4[:, 6] = bb1*cc3**ff7+7*ee1*ff7*xi*cc2*cc3**ll8+6*kk1*ll8*ff7*kk2*hh4*cc3**vv09+ uu1*vv09*ll8*ff7*uu2*qq05*cc3**(ee3-4)-(bb1*dd3)/cc3+ (7*ee1*xi*dd3*cc2)/cc3**2-(12*kk1*kk2*dd3*hh4)/cc3**3+ (6*uu1*uu2*dd3*qq05)/cc3**4
        l4[:, 7] = (-bb1*cc3**oo10*(bb1*oo10*cc2*dd6+yy18))-3*ee1*oo10*xi*cc2*mm12* (bb1*mm11*cc2*dd6+yy18)-kk1*mm11*oo10*kk2*hh4*ww15* (bb1*ww11*cc2*dd6+yy18)-3*ee1*aa2*cc2*mm12-3*ee1*oo10*cc2*mm12- 2*kk1*mm11*oo10*xi*hh4*ww15-kk1*oo10*hh4*ww15-kk1*mm11*hh4*ww15+ bb1*dd6-4*ee1*dd3*cc2*jj08-3*ee1*xi*cc2*jj08+2*kk1*kk2*hh4*tt08+ 10*kk1*xi*dd3*hh4*tt08-(6*kk2*dd3*cc2**3)/(exp1**(4*rho)*cc3**4)
        ad17 = 2*bb1*dd8*cc2*dd6
        ad19 = -(2*dd7)/xi**3
        ad20 = cc3**oo10
        ad21 = dd8*dd7
        ad22 = ad21+bb1*mm11*cc2*dd6
        l4[:, 8] = bb1*ad20*(bb1*oo10*cc2*dd6+ad21)**2+ee1*oo10*xi*cc2*mm12*ad22**2+ 2*ee1*aa2*cc2*mm12*ad22+2*ee1*oo10*cc2*mm12*ad22+ bb1*ad20*((-ee1*oo10*hh4*jj08)+ad17+ad19)+ee1*oo10*xi*cc2*mm12* ((-ee1*mm11*hh4*jj08)+ad17+ad19)+4*ee1*cc2*jj08-6*kk1*dd3*hh4*tt08- 4*kk1*xi*hh4*tt08+(6*xi*dd3*cc2**3)/(exp1**(4*rho)*cc3**4)
        ae16 = dd8*dd7+bb1*pp07*cc2*dd6
        l4[:, 9] = (-bb1*pp08*ae16**3)-3*bb1*pp08*((-ee1*pp07*hh4*jj08)+ 2*bb1*dd8*cc2*dd6-2*jj12*dd7)*ae16-bb1*pp08*(2*kk1*pp07*qq05*tt08- 3*ee1*dd8*hh4*jj08-6*bb1*jj12*cc2*dd6+(6*dd7)/xi**4)+6*kk1*hh4*tt08- (6*(xi+1)*qq05)/(exp1**(4*rho)*cc3**4)
        af05 = cc2**4
        l4[:, 10] = bb1*cc2*cc3**ff7+7*ee1*ff7*xi*hh4*cc3**ll8+6*kk1*ll8*ff7*kk2*qq05* cc3**vv09+uu1*vv09*ll8*ff7*uu2*af05*cc3**(ee3-4)-(bb1*dd3*cc2)/cc3+ (7*ee1*xi*dd3*hh4)/cc3**2-(12*kk1*kk2*dd3*qq05)/cc3**3+ (6*uu1*uu2*dd3*af05)/cc3**4
        ag23 = log_cc3/kk2-bb1*aa2*cc2*dd6
        l4[:, 11] = (-bb1*cc2*cc3**oo10*ag23)-3*ee1*oo10*xi*hh4*mm12*ag23- kk1*mm11*oo10*kk2*qq05*ww15*ag23+4*ee1*hh4*mm12+ 5*kk1*mm11*xi*qq05*ww15-4*kk1*qq05*ww15+uu1*ww11*mm11*kk2*af05*ww12- uu1*oo10*xi*af05*ww12-uu1*ww11*xi*af05*ww12+bb1*cc2*dd6- 4*ee1*dd3*hh4*jj08-3*ee1*xi*hh4*jj08+2*kk1*kk2*qq05*tt08+ 10*kk1*xi*dd3*qq05*tt08-(6*uu1*kk2*dd3*af05)/cc3**4
        ah24 = (-(2*dd7)/xi**3)+2*bb1*dd8*cc2*dd6+ee1*aa2*hh4*jj08
        ah27 = tt18**2
        l4[:, 12] = bb1*cc2*ad20*ah27+ee1*oo10*xi*hh4*mm12*ah27-4*ee1*hh4*mm12*tt18- 2*kk1*mm11*xi*qq05*ww15*tt18+2*kk1*qq05*ww15*tt18+bb1*cc2*ad20*ah24+ ee1*oo10*xi*hh4*mm12*ah24+6*kk1*qq05*ww15+2*uu1*ww11*xi*af05*ww12- 4*uu1*af05*ww12+4*ee1*hh4*jj08-6*kk1*dd3*qq05*tt08- 4*kk1*xi*qq05*tt08+(6*uu1*xi*dd3*af05)/cc3**4
        l4[:, 13] = (-bb1*cc2*pp08*ae16**3)-3*bb1*cc2*pp08*((-ee1*pp07*hh4*jj08)+ 2*bb1*dd8*cc2*dd6-2*jj12*dd7)*ae16-bb1*cc2*pp08*(2*kk1*pp07*qq05* tt08-3*ee1*dd8*hh4*jj08-6*bb1*jj12*cc2*dd6+(6*dd7)/xi**4)+ 6*kk1*qq05*tt08-(6*(xi+1)*cc2**4)/(exp1**(4*rho)*cc3**4)
        aj08 = 1/cc3**4
        aj20 = 1/xi**5
        aj23 = (-2*jj12*dd7)+2*bb1*dd8*cc2*dd6+ee1*aa2*hh4*jj08
        l4[:, 14] = (-jj13*tt18**4)-6*jj13*aj23*tt18**2-3*jj13*aj23**2- 4*jj13*((-2*kk1*aa2*qq05*tt08)-3*ee1*dd8*hh4*jj08-6*bb1*jj12*cc2*dd6+ 6*tt16*dd7)*tt18-jj13*(6*uu1*aa2*af05*aj08+8*kk1*dd8*qq05*tt08+ 12*ee1*jj12*hh4*jj08+24*bb1*tt16*cc2*dd6-24*aj20*dd7)- 24*aj20*dd3*dd7+24*tt16*dd7+24*bb1*tt16*dd3*cc2*dd6- 24*bb1*jj12*cc2*dd6+12*ee1*jj12*dd3*hh4*jj08-12*ee1*dd8*hh4*jj08+ 8*kk1*dd8*dd3*qq05*tt08-8*kk1*aa2*qq05*tt08+6*uu1*aa2*dd3*af05*aj08
    return l0, l1, l2, l3, l4
