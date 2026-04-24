kwds = {'equal_nan': True, 'err_msg': '', 'header': 'Not equal to tolerance rtol=5e-06, atol=5e-06', 'verbose': True}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
                   ^^^^^^^^^^^^^^^^^^^
E           AssertionError: 
E           Not equal to tolerance rtol=5e-06, atol=5e-06
E           
E           Mismatched elements: 31 / 62 (50%)
E           Max absolute difference: 0.00158601
E           Max relative difference: 0.01021523
E            x: array([[0.141508, 0.086204],
E                  [0.15252 , 0.086204],
E                  [0.156846, 0.086204],...
E            y: array([[0.140157, 0.086204],
E                  [0.150988, 0.086204],
E                  [0.15526 , 0.086204],...

../../../miniconda3/envs/nampy/lib/python3.11/contextlib.py:81: AssertionError
________________________ test_general_family_newdata_unconditional_standard_errors_match_mgcv[response-shashlss_numeric_by] _________________________

case_id = 'shashlss_numeric_by', family = 'shashlss', formula = ['y ~ s(x, by=z, bs="cr", k=6)', '~ 1', '~ 1', '~ 1']
data_factory = <function _shashlss_by_data at 0x77e448667740>, method = 'ML', pred_atol = 8e-05, se_atol = 8e-05, check_response_se = True
pred_type = 'response'

    @pytest.mark.parametrize(
        (
            "case_id",
            "family",
            "formula",
            "data_factory",
            "method",
            "pred_atol",
            "se_atol",
            "check_response_se",
        ),
        GENERAL_SE_CASES,
        ids=[case[0] for case in GENERAL_SE_CASES],
    )
    @pytest.mark.parametrize("pred_type", ["link", "response", "terms"])
    def test_general_family_newdata_unconditional_standard_errors_match_mgcv(
        case_id,
        family,
        formula,
        data_factory,
        method,
        pred_atol,
        se_atol,
        check_response_se,
        pred_type,
    ):
        """Verify that general family new-data unconditional standard errors match mgcv."""
        _maybe_xfail_known_general_gap(case_id, surface=f"{pred_type} unconditional SE")
    
        select = "select_true" in case_id
        data = data_factory()
        newdata = _general_newdata(data)
        gam = _fit_nampy_model(data, formula, family, method, select=select)
        expected = _run_mgcv_predict_on_newdata(
            data,
            newdata,
            formula,
            family=family,
            method=method,
            type=pred_type,
            return_se=True,
            unconditional=True,
            select=select,
        )
    
        actual_cov = np.asarray(gam.vcov(unconditional=True), dtype=np.float64)
        actual_pred, actual_se = gam.predict(
            newdata,
            type=pred_type,
            return_se=True,
            cov=actual_cov,
        )
        _assert_general_prediction_close(actual_pred, expected["pred"], atol=pred_atol)
    
        if pred_type == "terms":
            _assert_general_term_labels_match(gam, expected.get("term_names", []))
            _assert_general_prediction_close(actual_se, expected["se"], atol=se_atol)
            return
    
        if pred_type == "response" and not check_response_se:
            assert (
                np.asarray(actual_se, dtype=np.float64).shape
                == np.asarray(actual_pred, dtype=np.float64).shape
            )
            return
    
>       _assert_general_prediction_close(actual_se, expected["se"], atol=se_atol)

tests/families/test_general_family_mgcv_parity.py:1546: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
tests/families/test_general_family_mgcv_parity.py:668: in _assert_general_prediction_close
    np.testing.assert_allclose(
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

args = (<function assert_allclose.<locals>.compare at 0x77e443fb4720>, array([[0.11406625, 0.11345394, 0.08532473, 0.14846148...1],
       [0.13384675, 0.11345396, 0.08532493, 0.14846211],
       [0.14033167, 0.11345396, 0.08532493, 0.14846211]]))
kwds = {'equal_nan': True, 'err_msg': '', 'header': 'Not equal to tolerance rtol=8e-05, atol=8e-05', 'verbose': True}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
                   ^^^^^^^^^^^^^^^^^^^
E           AssertionError: 
E           Not equal to tolerance rtol=8e-05, atol=8e-05
E           
E           Mismatched elements: 8 / 124 (6.45%)
E           Max absolute difference: 0.00010504
E           Max relative difference: 0.00086616
E            x: array([[0.114066, 0.113454, 0.085325, 0.148461],
E                  [0.110962, 0.113454, 0.085325, 0.148461],
E                  [0.107909, 0.113454, 0.085325, 0.148461],...
E            y: array([[0.114105, 0.113454, 0.085325, 0.148462],
E                  [0.110996, 0.113454, 0.085325, 0.148462],
E                  [0.10794 , 0.113454, 0.085325, 0.148462],...

../../../miniconda3/envs/nampy/lib/python3.11/contextlib.py:81: AssertionError
_________________________ test_general_family_newdata_unconditional_standard_errors_match_mgcv[terms-gaulss_select_true_cr] _________________________

case_id = 'gaulss_select_true_cr', family = 'gaulss', formula = ['y ~ s(x, bs="cr", k=6)', '~ 1']
data_factory = <function _gaulss_data at 0x77e479ad77e0>, method = 'ML', pred_atol = 5e-06, se_atol = 5e-06, check_response_se = True
pred_type = 'terms'

    @pytest.mark.parametrize(
        (
            "case_id",
            "family",
            "formula",
            "data_factory",
            "method",
            "pred_atol",
            "se_atol",
            "check_response_se",
        ),
        GENERAL_SE_CASES,
        ids=[case[0] for case in GENERAL_SE_CASES],
    )
    @pytest.mark.parametrize("pred_type", ["link", "response", "terms"])
    def test_general_family_newdata_unconditional_standard_errors_match_mgcv(
        case_id,
        family,
        formula,
        data_factory,
        method,
        pred_atol,
        se_atol,
        check_response_se,
        pred_type,
    ):
        """Verify that general family new-data unconditional standard errors match mgcv."""
        _maybe_xfail_known_general_gap(case_id, surface=f"{pred_type} unconditional SE")
    
        select = "select_true" in case_id
        data = data_factory()
        newdata = _general_newdata(data)
        gam = _fit_nampy_model(data, formula, family, method, select=select)
        expected = _run_mgcv_predict_on_newdata(
            data,
            newdata,
            formula,
            family=family,
            method=method,
            type=pred_type,
            return_se=True,
            unconditional=True,
            select=select,
        )
    
        actual_cov = np.asarray(gam.vcov(unconditional=True), dtype=np.float64)
        actual_pred, actual_se = gam.predict(
            newdata,
            type=pred_type,
            return_se=True,
            cov=actual_cov,
        )
        _assert_general_prediction_close(actual_pred, expected["pred"], atol=pred_atol)
    
        if pred_type == "terms":
            _assert_general_term_labels_match(gam, expected.get("term_names", []))
>           _assert_general_prediction_close(actual_se, expected["se"], atol=se_atol)

tests/families/test_general_family_mgcv_parity.py:1536: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
tests/families/test_general_family_mgcv_parity.py:668: in _assert_general_prediction_close
    np.testing.assert_allclose(
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

args = (<function assert_allclose.<locals>.compare at 0x77e443fb6020>, array([[0.12878051],
       [0.14079165],
       [0.14...0.10729693],
       [0.11547318],
       [0.12300734],
       [0.1261397 ],
       [0.1225594 ],
       [0.11305477]]))
kwds = {'equal_nan': True, 'err_msg': '', 'header': 'Not equal to tolerance rtol=5e-06, atol=5e-06', 'verbose': True}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
                   ^^^^^^^^^^^^^^^^^^^
E           AssertionError: 
E           Not equal to tolerance rtol=5e-06, atol=5e-06
E           
E           Mismatched elements: 31 / 31 (100%)
E           Max absolute difference: 0.0017115
E           Max relative difference: 0.01194077
E            x: array([[0.128781],
E                  [0.140792],
E                  [0.145467],...
E            y: array([[0.127294],
E                  [0.13913 ],
E                  [0.143755],...

../../../miniconda3/envs/nampy/lib/python3.11/contextlib.py:81: AssertionError
_____________________________ test_general_family_newdata_unconditional_standard_errors_match_mgcv[terms-gevlss_two_cr] _____________________________

case_id = 'gevlss_two_cr', family = 'gevlss', formula = ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', '~ 1', '~ 1']
data_factory = <function _gevlss_two_smooth_data at 0x77e448667560>, method = 'ML', pred_atol = 3e-05, se_atol = 3e-05, check_response_se = True
pred_type = 'terms'

    @pytest.mark.parametrize(
        (
            "case_id",
            "family",
            "formula",
            "data_factory",
            "method",
            "pred_atol",
            "se_atol",
            "check_response_se",
        ),
        GENERAL_SE_CASES,
        ids=[case[0] for case in GENERAL_SE_CASES],
    )
    @pytest.mark.parametrize("pred_type", ["link", "response", "terms"])
    def test_general_family_newdata_unconditional_standard_errors_match_mgcv(
        case_id,
        family,
        formula,
        data_factory,
        method,
        pred_atol,
        se_atol,
        check_response_se,
        pred_type,
    ):
        """Verify that general family new-data unconditional standard errors match mgcv."""
        _maybe_xfail_known_general_gap(case_id, surface=f"{pred_type} unconditional SE")
    
        select = "select_true" in case_id
        data = data_factory()
        newdata = _general_newdata(data)
        gam = _fit_nampy_model(data, formula, family, method, select=select)
        expected = _run_mgcv_predict_on_newdata(
            data,
            newdata,
            formula,
            family=family,
            method=method,
            type=pred_type,
            return_se=True,
            unconditional=True,
            select=select,
        )
    
        actual_cov = np.asarray(gam.vcov(unconditional=True), dtype=np.float64)
        actual_pred, actual_se = gam.predict(
            newdata,
            type=pred_type,
            return_se=True,
            cov=actual_cov,
        )
        _assert_general_prediction_close(actual_pred, expected["pred"], atol=pred_atol)
    
        if pred_type == "terms":
            _assert_general_term_labels_match(gam, expected.get("term_names", []))
>           _assert_general_prediction_close(actual_se, expected["se"], atol=se_atol)

tests/families/test_general_family_mgcv_parity.py:1536: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
tests/families/test_general_family_mgcv_parity.py:668: in _assert_general_prediction_close
    np.testing.assert_allclose(
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

args = (<function assert_allclose.<locals>.compare at 0x77e443fb7c40>, array([[0.07497618, 0.05913248],
       [0.07118225, 0...0612, 0.04515735],
       [0.0740757 , 0.04897953],
       [0.07269881, 0.05280213],
       [0.07277919, 0.05662501]]))
kwds = {'equal_nan': True, 'err_msg': '', 'header': 'Not equal to tolerance rtol=3e-05, atol=3e-05', 'verbose': True}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
                   ^^^^^^^^^^^^^^^^^^^
E           AssertionError: 
E           Not equal to tolerance rtol=3e-05, atol=3e-05
E           
E           Mismatched elements: 5 / 62 (8.06%)
E           Max absolute difference: 8.27802307e-05
E           Max relative difference: 0.02142662
E            x: array([[0.074976, 0.059132],
E                  [0.071182, 0.055249],
E                  [0.067581, 0.051371],...
E            y: array([[0.074976, 0.059121],
E                  [0.071182, 0.055239],
E                  [0.067581, 0.051363],...

../../../miniconda3/envs/nampy/lib/python3.11/contextlib.py:81: AssertionError
__________________________ test_general_family_newdata_unconditional_standard_errors_match_mgcv[terms-shashlss_numeric_by] __________________________

case_id = 'shashlss_numeric_by', family = 'shashlss', formula = ['y ~ s(x, by=z, bs="cr", k=6)', '~ 1', '~ 1', '~ 1']
data_factory = <function _shashlss_by_data at 0x77e448667740>, method = 'ML', pred_atol = 8e-05, se_atol = 8e-05, check_response_se = True
pred_type = 'terms'

    @pytest.mark.parametrize(
        (
            "case_id",
            "family",
            "formula",
            "data_factory",
            "method",
            "pred_atol",
            "se_atol",
            "check_response_se",
        ),
        GENERAL_SE_CASES,
        ids=[case[0] for case in GENERAL_SE_CASES],
    )
    @pytest.mark.parametrize("pred_type", ["link", "response", "terms"])
    def test_general_family_newdata_unconditional_standard_errors_match_mgcv(
        case_id,
        family,
        formula,
        data_factory,
        method,
        pred_atol,
        se_atol,
        check_response_se,
        pred_type,
    ):
        """Verify that general family new-data unconditional standard errors match mgcv."""
        _maybe_xfail_known_general_gap(case_id, surface=f"{pred_type} unconditional SE")
    
        select = "select_true" in case_id
        data = data_factory()
        newdata = _general_newdata(data)
        gam = _fit_nampy_model(data, formula, family, method, select=select)
        expected = _run_mgcv_predict_on_newdata(
            data,
            newdata,
            formula,
            family=family,
            method=method,
            type=pred_type,
            return_se=True,
            unconditional=True,
            select=select,
        )
    
        actual_cov = np.asarray(gam.vcov(unconditional=True), dtype=np.float64)
        actual_pred, actual_se = gam.predict(
            newdata,
            type=pred_type,
            return_se=True,
            cov=actual_cov,
        )
        _assert_general_prediction_close(actual_pred, expected["pred"], atol=pred_atol)
    
        if pred_type == "terms":
            _assert_general_term_labels_match(gam, expected.get("term_names", []))
>           _assert_general_prediction_close(actual_se, expected["se"], atol=se_atol)

tests/families/test_general_family_mgcv_parity.py:1536: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
tests/families/test_general_family_mgcv_parity.py:668: in _assert_general_prediction_close
    np.testing.assert_allclose(
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

args = (<function assert_allclose.<locals>.compare at 0x77e443fb4a40>, array([[0.13852153],
       [0.14212488],
       [0.14...0.24687536],
       [0.25264915],
       [0.25858249],
       [0.26467798],
       [0.27093788],
       [0.27736486]]))
kwds = {'equal_nan': True, 'err_msg': '', 'header': 'Not equal to tolerance rtol=8e-05, atol=8e-05', 'verbose': True}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
                   ^^^^^^^^^^^^^^^^^^^
E           AssertionError: 
E           Not equal to tolerance rtol=8e-05, atol=8e-05
E           
E           Mismatched elements: 11 / 31 (35.5%)
E           Max absolute difference: 0.00021684
E           Max relative difference: 0.00156296
E            x: array([[0.138522],
E                  [0.142125],
E                  [0.145736],...
E            y: array([[0.138738],
E                  [0.142328],
E                  [0.145926],...

../../../miniconda3/envs/nampy/lib/python3.11/contextlib.py:81: AssertionError
_____________________________ test_general_family_newdata_unconditional_standard_errors_match_mgcv[terms-ziplss_two_cr] _____________________________

case_id = 'ziplss_two_cr', family = 'ziplss', formula = ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', '~ 1']
data_factory = <function _ziplss_two_smooth_data at 0x77e448667a60>, method = 'ML', pred_atol = 2e-05, se_atol = 2e-05, check_response_se = False
pred_type = 'terms'

    @pytest.mark.parametrize(
        (
            "case_id",
            "family",
            "formula",
            "data_factory",
            "method",
            "pred_atol",
            "se_atol",
            "check_response_se",
        ),
        GENERAL_SE_CASES,
        ids=[case[0] for case in GENERAL_SE_CASES],
    )
    @pytest.mark.parametrize("pred_type", ["link", "response", "terms"])
    def test_general_family_newdata_unconditional_standard_errors_match_mgcv(
        case_id,
        family,
        formula,
        data_factory,
        method,
        pred_atol,
        se_atol,
        check_response_se,
        pred_type,
    ):
        """Verify that general family new-data unconditional standard errors match mgcv."""
        _maybe_xfail_known_general_gap(case_id, surface=f"{pred_type} unconditional SE")
    
        select = "select_true" in case_id
        data = data_factory()
        newdata = _general_newdata(data)
        gam = _fit_nampy_model(data, formula, family, method, select=select)
        expected = _run_mgcv_predict_on_newdata(
            data,
            newdata,
            formula,
            family=family,
            method=method,
            type=pred_type,
            return_se=True,
            unconditional=True,
            select=select,
        )
    
        actual_cov = np.asarray(gam.vcov(unconditional=True), dtype=np.float64)
        actual_pred, actual_se = gam.predict(
            newdata,
            type=pred_type,
            return_se=True,
            cov=actual_cov,
        )
        _assert_general_prediction_close(actual_pred, expected["pred"], atol=pred_atol)
    
        if pred_type == "terms":
            _assert_general_term_labels_match(gam, expected.get("term_names", []))
>           _assert_general_prediction_close(actual_se, expected["se"], atol=se_atol)

tests/families/test_general_family_mgcv_parity.py:1536: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
tests/families/test_general_family_mgcv_parity.py:668: in _assert_general_prediction_close
    np.testing.assert_allclose(
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

args = (<function assert_allclose.<locals>.compare at 0x77e443fb7240>, array([[0.16393386, 0.41500625],
       [0.15235713, 0...7555, 0.29577285],
       [0.1590622 , 0.27094741],
       [0.17055169, 0.25033674],
       [0.18204467, 0.23996635]]))
kwds = {'equal_nan': True, 'err_msg': '', 'header': 'Not equal to tolerance rtol=2e-05, atol=2e-05', 'verbose': True}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
                   ^^^^^^^^^^^^^^^^^^^
E           AssertionError: 
E           Not equal to tolerance rtol=2e-05, atol=2e-05
E           
E           Mismatched elements: 2 / 62 (3.23%)
E           Max absolute difference: 2.8224419e-05
E           Max relative difference: 0.00367145
E            x: array([[0.163934, 0.415006],
E                  [0.152357, 0.379389],
E                  [0.140789, 0.345346],...
E            y: array([[0.163932, 0.415006],
E                  [0.152355, 0.379389],
E                  [0.140788, 0.345346],...

../../../miniconda3/envs/nampy/lib/python3.11/contextlib.py:81: AssertionError
___________________________________ test_general_family_secondary_diagnostics_match_mgcv_snapshot[gevlss_two_cr] ____________________________________

family = 'gevlss', formula = ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', '~ 1', '~ 1']
data_factory = <function _gevlss_two_smooth_data at 0x77e448667560>, method = 'ML', pred_atol = 3e-05

    @pytest.mark.parametrize(
        (
            "case_id",
            "family",
            "formula",
            "data_factory",
            "method",
            "pred_atol",
            "_se_atol",
            "_check_response_se",
        ),
        GENERAL_TWO_CR_CASES,
        ids=[case[0] for case in GENERAL_TWO_CR_CASES],
    )
    def test_general_family_secondary_diagnostics_match_mgcv_snapshot(
        case_id,
        family,
        formula,
        data_factory,
        method,
        pred_atol,
        _se_atol,
        _check_response_se,
    ):
        """Verify that general family secondary diagnostics match mgcv snapshot."""
        del case_id, _se_atol, _check_response_se
        data = data_factory()
        expected = _run_mgcv_snapshot(data, formula, family, method)
        gam = _fit_nampy_model(data, formula, family, method)
        expected_diag = expected["parity"]["diagnostics"]
        diag_tol = _general_diag_tol(pred_atol)
    
        actual_full = gam.concurvity(full=True)
        actual_pairwise = gam.concurvity(full=False)
    
        assert [
            _normalize_mgcv_term_label(v) for v in actual_full["labels"]
        ] == expected_diag["concurvity_labels"]
>       np.testing.assert_allclose(
            np.asarray(actual_full["values"], dtype=np.float64),
            np.asarray(expected_diag["concurvity_full"], dtype=np.float64),
            atol=diag_tol,
            rtol=0.0,
        )

tests/families/test_general_family_mgcv_parity.py:1587: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

args = (<function assert_allclose.<locals>.compare at 0x77e443fb5b20>, array([[1.        , 0.40370096, 0.41822401],
       [1... , 0.14014113, 0.12399641],
       [1.        , 0.1181833 , 0.0111476 ],
       [1.        , 0.05468279, 0.04171239]]))
kwds = {'equal_nan': True, 'err_msg': '', 'header': 'Not equal to tolerance rtol=0, atol=0.0003', 'verbose': True}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
                   ^^^^^^^^^^^^^^^^^^^
E           AssertionError: 
E           Not equal to tolerance rtol=0, atol=0.0003
E           
E           Mismatched elements: 6 / 9 (66.7%)
E           Max absolute difference: 0.2942276
E           Max relative difference: 4.2580712
E            x: array([[1.      , 0.403701, 0.418224],
E                  [1.      , 0.083744, 0.058615],
E                  [1.      , 0.103451, 0.103797]])
E            y: array([[1.      , 0.140141, 0.123996],
E                  [1.      , 0.118183, 0.011148],
E                  [1.      , 0.054683, 0.041712]])

../../../miniconda3/envs/nampy/lib/python3.11/contextlib.py:81: AssertionError
__________________________________ test_general_family_secondary_diagnostics_match_mgcv_snapshot[shashlss_two_cr] ___________________________________

family = 'shashlss', formula = ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', '~ 1', '~ 1', '~ 1']
data_factory = <function _shashlss_two_smooth_data at 0x77e448667880>, method = 'ML', pred_atol = 8e-05

    @pytest.mark.parametrize(
        (
            "case_id",
            "family",
            "formula",
            "data_factory",
            "method",
            "pred_atol",
            "_se_atol",
            "_check_response_se",
        ),
        GENERAL_TWO_CR_CASES,
        ids=[case[0] for case in GENERAL_TWO_CR_CASES],
    )
    def test_general_family_secondary_diagnostics_match_mgcv_snapshot(
        case_id,
        family,
        formula,
        data_factory,
        method,
        pred_atol,
        _se_atol,
        _check_response_se,
    ):
        """Verify that general family secondary diagnostics match mgcv snapshot."""
        del case_id, _se_atol, _check_response_se
        data = data_factory()
        expected = _run_mgcv_snapshot(data, formula, family, method)
        gam = _fit_nampy_model(data, formula, family, method)
        expected_diag = expected["parity"]["diagnostics"]
        diag_tol = _general_diag_tol(pred_atol)
    
        actual_full = gam.concurvity(full=True)
        actual_pairwise = gam.concurvity(full=False)
    
        assert [
            _normalize_mgcv_term_label(v) for v in actual_full["labels"]
        ] == expected_diag["concurvity_labels"]
>       np.testing.assert_allclose(
            np.asarray(actual_full["values"], dtype=np.float64),
            np.asarray(expected_diag["concurvity_full"], dtype=np.float64),
            atol=diag_tol,
            rtol=0.0,
        )

tests/families/test_general_family_mgcv_parity.py:1587: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

args = (<function assert_allclose.<locals>.compare at 0x77e443fb6020>, array([[1.        , 0.82782921, 0.51843414],
       [1... , 1.        , 1.        ],
       [1.        , 0.33801352, 0.24906717],
       [1.        , 0.24628656, 0.32012956]]))
kwds = {'equal_nan': True, 'err_msg': '', 'header': 'Not equal to tolerance rtol=0, atol=0.0008', 'verbose': True}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
                   ^^^^^^^^^^^^^^^^^^^
E           AssertionError: 
E           Not equal to tolerance rtol=0, atol=0.0008
E           
E           Mismatched elements: 6 / 9 (66.7%)
E           Max absolute difference: 0.48156586
E           Max relative difference: 0.73676592
E            x: array([[1.      , 0.827829, 0.518434],
E                  [1.      , 0.58705 , 0.118451],
E                  [1.      , 0.316332, 0.143113]])
E            y: array([[1.      , 1.      , 1.      ],
E                  [1.      , 0.338014, 0.249067],
E                  [1.      , 0.246287, 0.32013 ]])

../../../miniconda3/envs/nampy/lib/python3.11/contextlib.py:81: AssertionError
___________________________________ test_general_family_secondary_diagnostics_match_mgcv_snapshot[ziplss_two_cr] ____________________________________

family = 'ziplss', formula = ['y ~ s(x, bs="cr", k=6) + s(z, bs="cr", k=6)', '~ 1']
data_factory = <function _ziplss_two_smooth_data at 0x77e448667a60>, method = 'ML', pred_atol = 2e-05

    @pytest.mark.parametrize(
        (
            "case_id",
            "family",
            "formula",
            "data_factory",
            "method",
            "pred_atol",
            "_se_atol",
            "_check_response_se",
        ),
        GENERAL_TWO_CR_CASES,
        ids=[case[0] for case in GENERAL_TWO_CR_CASES],
    )
    def test_general_family_secondary_diagnostics_match_mgcv_snapshot(
        case_id,
        family,
        formula,
        data_factory,
        method,
        pred_atol,
        _se_atol,
        _check_response_se,
    ):
        """Verify that general family secondary diagnostics match mgcv snapshot."""
        del case_id, _se_atol, _check_response_se
        data = data_factory()
        expected = _run_mgcv_snapshot(data, formula, family, method)
        gam = _fit_nampy_model(data, formula, family, method)
        expected_diag = expected["parity"]["diagnostics"]
        diag_tol = _general_diag_tol(pred_atol)
    
        actual_full = gam.concurvity(full=True)
        actual_pairwise = gam.concurvity(full=False)
    
        assert [
            _normalize_mgcv_term_label(v) for v in actual_full["labels"]
        ] == expected_diag["concurvity_labels"]
        np.testing.assert_allclose(
            np.asarray(actual_full["values"], dtype=np.float64),
            np.asarray(expected_diag["concurvity_full"], dtype=np.float64),
            atol=diag_tol,
            rtol=0.0,
        )
    
        assert [
            _normalize_mgcv_term_label(v) for v in actual_pairwise["labels"]
        ] == expected_diag["concurvity_pairwise"]["labels"]
        for name in actual_pairwise["measure_names"]:
            np.testing.assert_allclose(
                np.asarray(actual_pairwise["values"][name], dtype=np.float64),
                np.asarray(expected_diag["concurvity_pairwise"][name], dtype=np.float64),
                atol=diag_tol,
                rtol=0.0,
            )
    
>       np.testing.assert_allclose(
            np.asarray(gam.sp_vcov(edge_correct=False), dtype=np.float64),
            np.asarray(expected_diag["sp_vcov"], dtype=np.float64),
            atol=max(1e-4, diag_tol),
            rtol=0.0,
        )

tests/families/test_general_family_mgcv_parity.py:1605: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

args = (<function assert_allclose.<locals>.compare at 0x77e443fb6980>, array([[1006.40237988,  -46.34939148],
       [ -46.34939148,   48.16441996]]), array([[1006.72699841,  -46.36413958],
       [ -46.36413958,   48.16508883]]))
kwds = {'equal_nan': True, 'err_msg': '', 'header': 'Not equal to tolerance rtol=0, atol=0.0002', 'verbose': True}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
                   ^^^^^^^^^^^^^^^^^^^
E           AssertionError: 
E           Not equal to tolerance rtol=0, atol=0.0002
E           
E           Mismatched elements: 4 / 4 (100%)
E           Max absolute difference: 0.32461853
E           Max relative difference: 0.00032245
E            x: array([[1006.40238 ,  -46.349391],
E                  [ -46.349391,   48.16442 ]])
E            y: array([[1006.726998,  -46.36414 ],
E                  [ -46.36414 ,   48.165089]])

../../../miniconda3/envs/nampy/lib/python3.11/contextlib.py:81: AssertionError
============================================================== short test summary info ==============================================================
FAILED tests/families/test_general_family_mgcv_parity.py::test_general_family_fixed_sp_outer_derivatives_match_mgcv_across_surface[gammals_t2_full_true] - AssertionError: 
Not equal to tolerance rtol=2e-07, atol=2e-07

Mismatched elements: 1 / 1 (100%)
Max absolute difference: 0.00058682
Max relative difference: 2.3927907e-06
 x: array(245.243917)
 y: array(245.243331)
FAILED tests/families/test_general_family_mgcv_parity.py::test_general_family_fixed_sp_outer_derivatives_match_mgcv_across_surface[gevlss_t2_full_true] - AssertionError: 
Not equal to tolerance rtol=0.01, atol=0.01

Mismatched elements: 4 / 25 (16%)
Max absolute difference: 0.08467318
Max relative difference: 1.20424441
 x: array([[ 2.948384e-01,  8.728809e-06,  8.464672e-03,  1.490795e-02,
         1.051439e-03],
       [ 8.728809e-06,  9.308747e-05,  5.599162e-08, -2.248184e-06,...
 y: array([[ 2.951606e-01,  8.718385e-06,  8.481036e-03,  1.598462e-02,
        -6.558209e-02],
       [ 8.718385e-06,  9.309036e-05,  5.617337e-08, -2.290842e-06,...
FAILED tests/families/test_general_family_mgcv_parity.py::test_general_family_outer_fit_matches_mgcv_endpoint_across_surface[gevlss_two_cr] - AssertionError: 
Not equal to tolerance rtol=7.5e-05, atol=7.5e-05

Mismatched elements: 1 / 2 (50%)
Max absolute difference: 0.04701696
Max relative difference: 0.00300969
 x: array([ 7.067944, 15.574846])
 y: array([ 7.067944, 15.621863])
FAILED tests/families/test_general_family_mgcv_parity.py::test_general_family_newdata_prediction_surfaces_match_mgcv[lpmatrix-gaulss_t2_full_false] - AssertionError: 
Not equal to tolerance rtol=0.0002, atol=0.0002

Mismatched elements: 341 / 1147 (29.7%)
Max absolute difference: 4.40225039
Max relative difference: 2.
 x: array([[ 1.      , -0.104837, -0.075806, ..., -1.291889,  1.449428,
         1.      ],
       [ 1.      , -0.117788, -0.109623, ..., -1.206957,  1.267864,...
 y: array([[ 1.      , -0.104837, -0.075806, ..., -1.291889, -1.449428,
         1.      ],
       [ 1.      , -0.117788, -0.109623, ..., -1.206957, -1.267864,...
FAILED tests/families/test_general_family_mgcv_parity.py::test_general_family_newdata_prediction_surfaces_match_mgcv[lpmatrix-gaulss_t2_full_true] - AssertionError: 
Not equal to tolerance rtol=0.0002, atol=0.0002

Mismatched elements: 341 / 1147 (29.7%)
Max absolute difference: 4.40225039
Max relative difference: 2.
 x: array([[ 1.      , -0.104837, -0.075806, ..., -1.291889,  1.449428,
         1.      ],
       [ 1.      , -0.117788, -0.109623, ..., -1.206957,  1.267864,...
 y: array([[ 1.      , -0.104837, -0.075806, ..., -1.291889, -1.449428,
         1.      ],
       [ 1.      , -0.117788, -0.109623, ..., -1.206957, -1.267864,...
FAILED tests/families/test_general_family_mgcv_parity.py::test_general_family_newdata_prediction_surfaces_match_mgcv[lpmatrix-gammals_t2_full_false] - AssertionError: 
Not equal to tolerance rtol=0.0005, atol=0.0005

Mismatched elements: 186 / 1147 (16.2%)
Max absolute difference: 4.36592374
Max relative difference: 2.
 x: array([[ 1.      , -0.105977, -0.095266, ..., -1.238947,  1.393148,
         1.      ],
       [ 1.      , -0.113211, -0.12428 , ..., -1.153539,  1.220764,...
 y: array([[ 1.      , -0.105977, -0.095266, ..., -1.238947, -1.393148,
         1.      ],
       [ 1.      , -0.113211, -0.12428 , ..., -1.153539, -1.220764,...
FAILED tests/families/test_general_family_mgcv_parity.py::test_general_family_newdata_prediction_surfaces_match_mgcv[lpmatrix-gammals_t2_full_true] - AssertionError: 
Not equal to tolerance rtol=0.0005, atol=0.0005

Mismatched elements: 186 / 1147 (16.2%)
Max absolute difference: 4.36592374
Max relative difference: 2.
 x: array([[ 1.      , -0.105977, -0.095266, ..., -1.238947,  1.393148,
         1.      ],
       [ 1.      , -0.113211, -0.12428 , ..., -1.153539,  1.220764,...
 y: array([[ 1.      , -0.105977, -0.095266, ..., -1.238947, -1.393148,
         1.      ],
       [ 1.      , -0.113211, -0.12428 , ..., -1.153539, -1.220764,...
FAILED tests/families/test_general_family_mgcv_parity.py::test_general_family_newdata_prediction_surfaces_match_mgcv[lpmatrix-ziplss_t2_full_false] - AssertionError: 
Not equal to tolerance rtol=0.0006, atol=0.0006

Mismatched elements: 186 / 1147 (16.2%)
Max absolute difference: 3.69069162
Max relative difference: 2.
 x: array([[ 1.      , -0.11275 , -0.148091, ..., -1.157802,  1.359217,
         1.      ],
       [ 1.      , -0.10409 , -0.154713, ..., -1.072835,  1.193307,...
 y: array([[ 1.      , -0.11275 , -0.148091, ..., -1.157802,  1.359217,
         1.      ],
       [ 1.      , -0.10409 , -0.154713, ..., -1.072835,  1.193307,...
FAILED tests/families/test_general_family_mgcv_parity.py::test_general_family_newdata_prediction_surfaces_match_mgcv[lpmatrix-ziplss_t2_full_true] - AssertionError: 
Not equal to tolerance rtol=0.0006, atol=0.0006

Mismatched elements: 186 / 1147 (16.2%)
Max absolute difference: 3.69069162
Max relative difference: 2.
 x: array([[ 1.      , -0.11275 , -0.148091, ..., -1.157802,  1.359217,
         1.      ],
       [ 1.      , -0.10409 , -0.154713, ..., -1.072835,  1.193307,...
 y: array([[ 1.      , -0.11275 , -0.148091, ..., -1.157802,  1.359217,
         1.      ],
       [ 1.      , -0.10409 , -0.154713, ..., -1.072835,  1.193307,...
FAILED tests/families/test_general_family_mgcv_parity.py::test_general_family_newdata_unconditional_standard_errors_match_mgcv[link-gaulss_select_true_cr] - AssertionError: 
Not equal to tolerance rtol=5e-06, atol=5e-06

Mismatched elements: 31 / 62 (50%)
Max absolute difference: 0.00158601
Max relative difference: 0.01021523
 x: array([[0.141508, 0.060698],
       [0.15252 , 0.060698],
       [0.156846, 0.060698],...
 y: array([[0.140157, 0.060698],
       [0.150988, 0.060698],
       [0.15526 , 0.060698],...
FAILED tests/families/test_general_family_mgcv_parity.py::test_general_family_newdata_unconditional_standard_errors_match_mgcv[link-shashlss_numeric_by] - AssertionError: 
Not equal to tolerance rtol=8e-05, atol=8e-05

Mismatched elements: 8 / 124 (6.45%)
Max absolute difference: 0.00010504
Max relative difference: 0.00086616
 x: array([[0.114066, 0.115369, 0.085325, 0.148461],
       [0.110962, 0.115369, 0.085325, 0.148461],
       [0.107909, 0.115369, 0.085325, 0.148461],...
 y: array([[0.114105, 0.115369, 0.085325, 0.148462],
       [0.110996, 0.115369, 0.085325, 0.148462],
       [0.10794 , 0.115369, 0.085325, 0.148462],...
FAILED tests/families/test_general_family_mgcv_parity.py::test_general_family_newdata_unconditional_standard_errors_match_mgcv[response-gaulss_select_true_cr] - AssertionError: 
Not equal to tolerance rtol=5e-06, atol=5e-06

Mismatched elements: 31 / 62 (50%)
Max absolute difference: 0.00158601
Max relative difference: 0.01021523
 x: array([[0.141508, 0.086204],
       [0.15252 , 0.086204],
       [0.156846, 0.086204],...
 y: array([[0.140157, 0.086204],
       [0.150988, 0.086204],
       [0.15526 , 0.086204],...
FAILED tests/families/test_general_family_mgcv_parity.py::test_general_family_newdata_unconditional_standard_errors_match_mgcv[response-shashlss_numeric_by] - AssertionError: 
Not equal to tolerance rtol=8e-05, atol=8e-05

Mismatched elements: 8 / 124 (6.45%)
Max absolute difference: 0.00010504
Max relative difference: 0.00086616
 x: array([[0.114066, 0.113454, 0.085325, 0.148461],
       [0.110962, 0.113454, 0.085325, 0.148461],
       [0.107909, 0.113454, 0.085325, 0.148461],...
 y: array([[0.114105, 0.113454, 0.085325, 0.148462],
       [0.110996, 0.113454, 0.085325, 0.148462],
       [0.10794 , 0.113454, 0.085325, 0.148462],...
FAILED tests/families/test_general_family_mgcv_parity.py::test_general_family_newdata_unconditional_standard_errors_match_mgcv[terms-gaulss_select_true_cr] - AssertionError: 
Not equal to tolerance rtol=5e-06, atol=5e-06

Mismatched elements: 31 / 31 (100%)
Max absolute difference: 0.0017115
Max relative difference: 0.01194077
 x: array([[0.128781],
       [0.140792],
       [0.145467],...
 y: array([[0.127294],
       [0.13913 ],
       [0.143755],...
FAILED tests/families/test_general_family_mgcv_parity.py::test_general_family_newdata_unconditional_standard_errors_match_mgcv[terms-gevlss_two_cr] - AssertionError: 
Not equal to tolerance rtol=3e-05, atol=3e-05

Mismatched elements: 5 / 62 (8.06%)
Max absolute difference: 8.27802307e-05
Max relative difference: 0.02142662
 x: array([[0.074976, 0.059132],
       [0.071182, 0.055249],
       [0.067581, 0.051371],...
 y: array([[0.074976, 0.059121],
       [0.071182, 0.055239],
       [0.067581, 0.051363],...
FAILED tests/families/test_general_family_mgcv_parity.py::test_general_family_newdata_unconditional_standard_errors_match_mgcv[terms-shashlss_numeric_by] - AssertionError: 
Not equal to tolerance rtol=8e-05, atol=8e-05

Mismatched elements: 11 / 31 (35.5%)
Max absolute difference: 0.00021684
Max relative difference: 0.00156296
 x: array([[0.138522],
       [0.142125],
       [0.145736],...
 y: array([[0.138738],
       [0.142328],
       [0.145926],...
FAILED tests/families/test_general_family_mgcv_parity.py::test_general_family_newdata_unconditional_standard_errors_match_mgcv[terms-ziplss_two_cr] - AssertionError: 
Not equal to tolerance rtol=2e-05, atol=2e-05

Mismatched elements: 2 / 62 (3.23%)
Max absolute difference: 2.8224419e-05
Max relative difference: 0.00367145
 x: array([[0.163934, 0.415006],
       [0.152357, 0.379389],
       [0.140789, 0.345346],...
 y: array([[0.163932, 0.415006],
       [0.152355, 0.379389],
       [0.140788, 0.345346],...
FAILED tests/families/test_general_family_mgcv_parity.py::test_general_family_secondary_diagnostics_match_mgcv_snapshot[gevlss_two_cr] - AssertionError: 
Not equal to tolerance rtol=0, atol=0.0003

Mismatched elements: 6 / 9 (66.7%)
Max absolute difference: 0.2942276
Max relative difference: 4.2580712
 x: array([[1.      , 0.403701, 0.418224],
       [1.      , 0.083744, 0.058615],
       [1.      , 0.103451, 0.103797]])
 y: array([[1.      , 0.140141, 0.123996],
       [1.      , 0.118183, 0.011148],
       [1.      , 0.054683, 0.041712]])
FAILED tests/families/test_general_family_mgcv_parity.py::test_general_family_secondary_diagnostics_match_mgcv_snapshot[shashlss_two_cr] - AssertionError: 
Not equal to tolerance rtol=0, atol=0.0008

Mismatched elements: 6 / 9 (66.7%)
Max absolute difference: 0.48156586
Max relative difference: 0.73676592
 x: array([[1.      , 0.827829, 0.518434],
       [1.      , 0.58705 , 0.118451],
       [1.      , 0.316332, 0.143113]])
 y: array([[1.      , 1.      , 1.      ],
       [1.      , 0.338014, 0.249067],
       [1.      , 0.246287, 0.32013 ]])
FAILED tests/families/test_general_family_mgcv_parity.py::test_general_family_secondary_diagnostics_match_mgcv_snapshot[ziplss_two_cr] - AssertionError: 
Not equal to tolerance rtol=0, atol=0.0002

Mismatched elements: 4 / 4 (100%)
Max absolute difference: 0.32461853
Max relative difference: 0.00032245
 x: array([[1006.40238 ,  -46.349391],
       [ -46.349391,   48.16442 ]])
 y: array([[1006.726998,  -46.36414 ],
       [ -46.36414 ,   48.165089]])
============================================== 20 failed,

args = (<function assert_allclose.<locals>.compare at 0x703c8658f1a0>, array([   7.06556327,    2.53017532,   -8.91141651,  -...0965, -145.40238679, -157.89899454, -169.89625657,
       -171.72141371, -171.93280037, -171.93813275, -171.93813792]))
kwds = {'equal_nan': True, 'err_msg': 'mrf_lattice: outer_info score_hist mismatch', 'header': 'Not equal to tolerance rtol=0, atol=0.0002', 'verbose': True}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
                   ^^^^^^^^^^^^^^^^^^^
E           AssertionError: 
E           Not equal to tolerance rtol=0, atol=0.0002
E           mrf_lattice: outer_info score_hist mismatch
E           Mismatched elements: 18 / 20 (90%)
E           Max absolute difference: 1.00068465
E           Max relative difference: 0.21213095
E            x: array([   7.065563,    2.530175,   -8.911417,  -21.403151,  -33.903095,
E                   -46.403094,  -58.903094,  -71.403094,  -83.903094,  -96.403094,
E                  -108.903094, -121.403094, -133.903094, -146.403059, -158.897797,
E                  -170.619857, -171.894725, -171.937591, -171.938138, -171.938138])
E            y: array([   6.606764,    3.211416,   -7.913287,  -20.402483,  -32.90241 ,
E                   -45.40241 ,  -57.90241 ,  -70.40241 ,  -82.90241 ,  -95.40241 ,
E                  -107.90241 , -120.40241 , -132.90241 , -145.402387, -157.898995,
E                  -169.896257, -171.721414, -171.9328  , -171.938133, -171.938138])

../../../miniconda3/envs/nampy/lib/python3.11/contextlib.py:81: AssertionError
_______________________________________ test_gam_fit3_postprocessing_final_fit_matches_mgcv[factor_smooth_sz] _______________________________________

case = CaseSpec(case_id='factor_smooth_sz', formula='y ~ s(f1, f2, x, bs="sz", k=6)', family='gaussian', data_factory=<functi...703c8b5defc0>, select=False, weights_column=None, skip_coef_comparison=True, criterion_atol=0.0001, se_tol_scale=0.002)

    @pytest.mark.parametrize(
        "case", ORDINARY_CASES, ids=[c.case_id for c in ORDINARY_CASES]
    )
    def test_gam_fit3_postprocessing_final_fit_matches_mgcv(case: CaseSpec):
        """Verify that gam fit3 postprocessing final fit matches mgcv."""
        if case.case_id in _KNOWN_FAILING_OR_WARNING_CASE_IDS:
            pytest.xfail(
                "Known requested parity gap/warning case; post-proc coverage is kept "
                "visible without treating the existing model-level mismatch as fixed."
            )
        expected_snapshot = _run_mgcv_snapshot(
            data=case.data_factory(),
            formula=case.formula,
            family=case.family,
            method="REML",
            select=case.select,
            weights_column=case.weights_column,
        )
        optimizer = _nampy_optimizer_name(expected_snapshot)
        _data, gam, fit_warnings = _fit_requested_case(
            case,
            method="REML",
            optimizer=optimizer,
        )
    
        actual = _serialize_actual_final_fit(
            gam,
            fit_warnings,
            allow_synthetic_outer_info=False,
        )
        expected = _serialize_expected_final_fit(expected_snapshot)
        family_name = str(case.family).lower()
        if (
            family_name != "gaussian"
            and expected["Vc"] is not None
            and actual["Vc"] is None
        ):
            pytest.xfail(
                "Real implementation gap: non-Gaussian PIRLS final-fit objects do not "
                "yet carry mgcv-style unconditional covariance/edf2 post-processing."
            )
        cov_rtol = 3e-5
        if case.case_id == "binomial_separation":
            cov_rtol = 7e-5
>       _assert_final_fit_parity(
            case.case_id,
            actual,
            expected,
            full_covariance=final_fit_uses_exact_orientation_parity(
                case.formula,
                skip_coef_comparison=bool(case.skip_coef_comparison),
            ),
            compare_hat=True,
            compare_outer_info=True,
            cov_rtol=cov_rtol,
            cov_atol=5e-8,
            scalar_atol=2e-4,
            exact_outer_info_trace=(case.weights_column is None),
        )

tests/optimization/test_mgcv_postprocessing_final_fit_parity.py:788: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
tests/optimization/test_mgcv_postprocessing_final_fit_parity.py:461: in _assert_final_fit_parity
    _assert_covariance_close(
tests/optimization/test_mgcv_postprocessing_final_fit_parity.py:349: in _assert_covariance_close
    np.testing.assert_allclose(
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

args = (<function assert_allclose.<locals>.compare at 0x703c864a14e0>, array([1.51775338e-02, 3.85343563e+00, 7.96050449e-01,...30e-02, 2.29851849e+00,
       5.14827174e-01, 4.20042090e-01, 1.33964523e-01, 8.21875966e-01,
       6.86365410e-01]))
kwds = {'equal_nan': True, 'err_msg': 'factor_smooth_sz: Vp diagonal mismatch', 'header': 'Not equal to tolerance rtol=3e-05, atol=5e-08', 'verbose': True}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
                   ^^^^^^^^^^^^^^^^^^^
E           AssertionError: 
E           Not equal to tolerance rtol=3e-05, atol=5e-08
E           factor_smooth_sz: Vp diagonal mismatch
E           Mismatched elements: 12 / 25 (48%)
E           Max absolute difference: 1.53553109e-05
E           Max relative difference: 0.78309314
E            x: array([1.517753e-02, 3.853436e+00, 7.960504e-01, 5.796443e-01,
E                  1.709883e-01, 1.364706e+00, 1.286348e+00, 2.725075e-05,
E                  5.042315e-06, 3.513118e-06, 9.858485e-07, 5.415663e-02,...
E            y: array([1.517756e-02, 3.853451e+00, 7.960533e-01, 5.796466e-01,
E                  1.709890e-01, 1.364712e+00, 1.286352e+00, 1.528289e-05,
E                  2.827849e-06, 1.970239e-06, 5.528867e-07, 5.414920e-02,...

../../../miniconda3/envs/nampy/lib/python3.11/contextlib.py:81: AssertionError
____________________________________ test_gam_fit5_postprocessing_final_fit_matches_mgcv[gaulss_select_true_cr] _____________________________________

case = ('gaulss_select_true_cr', 'gaulss', ['y ~ s(x, bs="cr", k=6)', '~ 1'], <function _gaulss_data at 0x703c8b95af20>, 'ML', 5e-06, ...)

    @pytest.mark.parametrize(
        "case", GENERAL_SE_CASES, ids=[case[0] for case in GENERAL_SE_CASES]
    )
    def test_gam_fit5_postprocessing_final_fit_matches_mgcv(case):
        """Verify that gam fit5 postprocessing final fit matches mgcv."""
        case_id, family, formula, data_factory, method, pred_atol, sp_log_atol, _ = case
        if any(tag in case_id for tag in _GENERAL_POSTPROC_KNOWN_GAP_TAGS):
            pytest.xfail(
                "Known general-family post-proc gap: advanced/select/by/tensor "
                "surfaces do not yet have exact mgcv final-fit parity."
            )
        data = data_factory()
        select = "select_true" in case_id
        expected_snapshot = _run_mgcv_snapshot(
            data=data,
            formula=formula,
            family=family,
            method=method,
            select=select,
        )
        optimizer = _nampy_optimizer_name(expected_snapshot)
        _data, gam, fit_warnings = _fit_general_case(case, optimizer=optimizer)
    
        actual = _serialize_actual_final_fit(
            gam,
            fit_warnings,
            allow_synthetic_outer_info=False,
        )
        expected = _serialize_expected_final_fit(expected_snapshot)
    
>       _assert_final_fit_parity(
            case_id,
            actual,
            expected,
            full_covariance=final_fit_uses_exact_orientation_parity(
                formula,
                skip_coef_comparison=False,
            ),
            compare_hat=False,
            compare_outer_info=True,
            cov_rtol=max(5e-5, 10.0 * float(pred_atol)),
            cov_atol=max(5e-8, 10.0 * float(pred_atol)),
            scalar_atol=max(5e-4, 10.0 * float(pred_atol), float(sp_log_atol)),
            exact_outer_info_trace=False,
        )

tests/optimization/test_mgcv_postprocessing_final_fit_parity.py:835: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
tests/optimization/test_mgcv_postprocessing_final_fit_parity.py:461: in _assert_final_fit_parity
    _assert_covariance_close(
tests/optimization/test_mgcv_postprocessing_final_fit_parity.py:349: in _assert_covariance_close
    np.testing.assert_allclose(
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

args = (<function assert_allclose.<locals>.compare at 0x703c864a05e0>, array([0.00344003, 0.04212256, 0.0223107 , 0.02010375,...5614, 0.00368425]), array([0.00344003, 0.04131317, 0.02230009, 0.02010101, 0.02295665,
       0.02365413, 0.00368425]))
kwds = {'equal_nan': True, 'err_msg': 'gaulss_select_true_cr: Vc diagonal mismatch', 'header': 'Not equal to tolerance rtol=5e-05, atol=5e-05', 'verbose': True}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
                   ^^^^^^^^^^^^^^^^^^^
E           AssertionError: 
E           Not equal to tolerance rtol=5e-05, atol=5e-05
E           gaulss_select_true_cr: Vc diagonal mismatch
E           Mismatched elements: 1 / 7 (14.3%)
E           Max absolute difference: 0.00080939
E           Max relative difference: 0.01959151
E            x: array([0.00344 , 0.042123, 0.022311, 0.020104, 0.022957, 0.023656,
E                  0.003684])
E            y: array([0.00344 , 0.041313, 0.0223  , 0.020101, 0.022957, 0.023654,
E                  0.003684])

../../../miniconda3/envs/nampy/lib/python3.11/contextlib.py:81: AssertionError
____________________________________ test_gam_fit5_outer_info_trace_exact_known_gap[gaulss_cr_outer_info_exact] _____________________________________

case = ('gaulss_cr', 'gaulss', ['y ~ s(x, bs="cr", k=6)', '~ 1'], <function _gaulss_data at 0x703c8b95af20>, 'ML', 5e-06, ...)

    @pytest.mark.parametrize(
        "case",
        [
            next(case for case in ORDINARY_CASES if case.case_id == "gaussian_weights"),
            next(case for case in GENERAL_SE_CASES if case[0] == "gaulss_cr"),
            next(case for case in GENERAL_SE_CASES if case[0] == "gevlss_cr"),
        ],
        ids=[
            "gaussian_weights_outer_info_exact",
            "gaulss_cr_outer_info_exact",
            "gevlss_cr_outer_info_exact",
        ],
    )
    def test_gam_fit5_outer_info_trace_exact_known_gap(case):
        """Verify that gam fit5 outer info trace exact known gap."""
        if isinstance(case, CaseSpec):
            expected_snapshot = _run_mgcv_snapshot(
                data=case.data_factory(),
                formula=case.formula,
                family=case.family,
                method="REML",
                select=case.select,
                weights_column=case.weights_column,
            )
            optimizer = _nampy_optimizer_name(expected_snapshot)
            _data, gam, _fit_warnings = _fit_requested_case(
                case,
                method="REML",
                optimizer=optimizer,
            )
            actual = _serialize_actual_outer_info(gam, allow_synthetic=False)
            expected = _serialize_outer_info_block(
                expected_snapshot["fit"].get("outer_info")
            )
            _assert_outer_info_trace_close(case.case_id, actual, expected, atol=5e-6)
            return
    
        case_id, family, formula, data_factory, method, _pred_atol, _sp_log_atol, _ = case
        data = data_factory()
        expected_snapshot = _run_mgcv_snapshot(
            data=data,
            formula=formula,
            family=family,
            method=method,
            select=False,
        )
        optimizer = _nampy_optimizer_name(expected_snapshot)
        _data, gam, _fit_warnings = _fit_general_case(case, optimizer=optimizer)
        actual = _serialize_actual_outer_info(gam, allow_synthetic=False)
        expected = _serialize_outer_info_block(expected_snapshot["fit"].get("outer_info"))
>       _assert_outer_info_trace_close(case_id, actual, expected, atol=5e-6)

tests/optimization/test_mgcv_postprocessing_final_fit_parity.py:902: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
tests/optimization/test_mgcv_postprocessing_final_fit_parity.py:421: in _assert_outer_info_trace_close
    np.testing.assert_allclose(
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

args = (<function assert_allclose.<locals>.compare at 0x703c86978040>, array([168.8382225 , 160.23195037, 160.00416675, 159.99850759,
       159.99850205]), array([161.18224583, 160.0742561 , 159.99926799, 159.99850215,
       159.99850204]))
kwds = {'equal_nan': True, 'err_msg': 'gaulss_cr: outer_info score_hist mismatch', 'header': 'Not equal to tolerance rtol=0, atol=5e-06', 'verbose': True}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
                   ^^^^^^^^^^^^^^^^^^^
E           AssertionError: 
E           Not equal to tolerance rtol=0, atol=5e-06
E           gaulss_cr: outer_info score_hist mismatch
E           Mismatched elements: 4 / 5 (80%)
E           Max absolute difference: 7.65597667
E           Max relative difference: 0.04749888
E            x: array([168.838223, 160.23195 , 160.004167, 159.998508, 159.998502])
E            y: array([161.182246, 160.074256, 159.999268, 159.998502, 159.998502])

../../../miniconda3/envs/nampy/lib/python3.11/contextlib.py:81: AssertionError
____________________________________ test_gam_fit5_outer_info_trace_exact_known_gap[gevlss_cr_outer_info_exact] _____________________________________

case = ('gevlss_cr', 'gevlss', ['y ~ s(x, bs="cr", k=6)', '~ 1', '~ 1'], <function _gevlss_data at 0x703c8b698680>, 'ML', 2e-05, ...)

    @pytest.mark.parametrize(
        "case",
        [
            next(case for case in ORDINARY_CASES if case.case_id == "gaussian_weights"),
            next(case for case in GENERAL_SE_CASES if case[0] == "gaulss_cr"),
            next(case for case in GENERAL_SE_CASES if case[0] == "gevlss_cr"),
        ],
        ids=[
            "gaussian_weights_outer_info_exact",
            "gaulss_cr_outer_info_exact",
            "gevlss_cr_outer_info_exact",
        ],
    )
    def test_gam_fit5_outer_info_trace_exact_known_gap(case):
        """Verify that gam fit5 outer info trace exact known gap."""
        if isinstance(case, CaseSpec):
            expected_snapshot = _run_mgcv_snapshot(
                data=case.data_factory(),
                formula=case.formula,
                family=case.family,
                method="REML",
                select=case.select,
                weights_column=case.weights_column,
            )
            optimizer = _nampy_optimizer_name(expected_snapshot)
            _data, gam, _fit_warnings = _fit_requested_case(
                case,
                method="REML",
                optimizer=optimizer,
            )
            actual = _serialize_actual_outer_info(gam, allow_synthetic=False)
            expected = _serialize_outer_info_block(
                expected_snapshot["fit"].get("outer_info")
            )
            _assert_outer_info_trace_close(case.case_id, actual, expected, atol=5e-6)
            return
    
        case_id, family, formula, data_factory, method, _pred_atol, _sp_log_atol, _ = case
        data = data_factory()
        expected_snapshot = _run_mgcv_snapshot(
            data=data,
            formula=formula,
            family=family,
            method=method,
            select=False,
        )
        optimizer = _nampy_optimizer_name(expected_snapshot)
        _data, gam, _fit_warnings = _fit_general_case(case, optimizer=optimizer)
        actual = _serialize_actual_outer_info(gam, allow_synthetic=False)
        expected = _serialize_outer_info_block(expected_snapshot["fit"].get("outer_info"))
>       _assert_outer_info_trace_close(case_id, actual, expected, atol=5e-6)

tests/optimization/test_mgcv_postprocessing_final_fit_parity.py:902: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

case_id = 'gevlss_cr', actual = {'conv': 'full convergence', 'convergence': 0, 'counts': array([123.,   7.]), 'grad': array([-8.57332894e-09]), ...}
expected = {'conv': 'full convergence', 'convergence': None, 'counts': None, 'grad': array(-3.29739727e-06), ...}

    def _assert_outer_info_trace_close(case_id: str, actual, expected, *, atol: float):
        _assert_outer_info_close(case_id, actual, expected, atol=atol)
    
        if expected["iter"] is not None:
>           assert int(actual["iter"]) == int(expected["iter"]), (
                f"{case_id}: outer_info iter mismatch "
                f"{actual['iter']!r} != {expected['iter']!r}"
            )
E           AssertionError: gevlss_cr: outer_info iter mismatch 6 != 5
E           assert 6 == 5
E            +  where 6 = int(6)
E            +  and   5 = int(5)

tests/optimization/test_mgcv_postprocessing_final_fit_parity.py:416: AssertionError
_______________________________________________ test_endpoint_log_sp_seed_matrix[gaussian-ML-99-0.8] ________________________________________________

family = 'gaussian', method = 'ML', seed = 99, atol = 0.8

    @pytest.mark.parametrize(
        "family, method, seed, atol",
        [
            ("gaussian", "REML", 321, 0.75),
            ("gaussian", "ML", 99, 0.8),
            ("binomial", "REML", 456, 1.05),
            ("poisson", "REML", 789, 1.0),
        ],
    )
    def test_endpoint_log_sp_seed_matrix(family, method, seed, atol):
        """Verify that endpoint log smoothing parameters stay close across seed matrix."""
        maker = {
            "gaussian": _make_gaussian_data,
            "binomial": _make_binomial_data,
            "poisson": _make_poisson_data,
        }[family]
        data = maker(seed=seed)
        formula = 'y ~ s(x0, bs="cr", k=8) + s(x1, bs="cr", k=8)'
>       actual = _fit_nampy_trace(data, formula, family, method)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/optimization/test_mgcv_score_hist_trace_parity.py:158: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
tests/optimization/_trace_parity_helpers.py:183: in _fit_nampy_trace
    gam.fit(data=data)
nampy/gam/model/api.py:319: in fit
    fit_model_core(
nampy/gam/fit/orchestrator.py:114: in fit_model_core
    optimize_smoothing_params(
nampy/gam/fit/smoothing_params.py:167: in optimize_smoothing_params
    return _optimize(
nampy/gam/smoothing_selection/optimize/driver.py:739: in optimize_smoothing_params
    result = optimize_outer_newton_indefinite_hessian(
nampy/gam/smoothing_selection/optimize/newton.py:285: in optimize_outer_newton_indefinite_hessian
    return _optimize_outer_newton_mgcv(
nampy/gam/smoothing_selection/optimize/newton_mgcv.py:81: in _optimize_outer_newton_mgcv
    ) = eval_at(
nampy/gam/smoothing_selection/optimize/newton.py:185: in _eval_at
    np.asarray(objective.jac(x_eval), dtype=np.float64)
               ^^^^^^^^^^^^^^^^^^^^^
nampy/gam/smoothing_selection/optimize/objectives.py:110: in jac
    criterion_gradient(self.model, self.y, x, method=self.method),
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

model = <nampy.gam.model.api.GAM object at 0x703c8606e010>
y = array([ 0.12906092,  0.27507893,  0.09734556,  0.71698666,  0.69672186,
        0.77139948, -0.40727456,  0.45678169, ...122669,  0.33451964, -0.97701259,  0.01602414,
        1.18190233,  0.13051893, -0.34437188, -0.49607   ,  0.24328358])
log_sp = array([3.06536121, 3.61947197]), method = 'ml', eps_abs = 1e-05, eps_rel = 0.0001

    def criterion_gradient(
        model,
        y,
        log_sp,
        method="gcv",
        eps_abs=1e-5,
        eps_rel=1e-4,
    ):
        method = str(method).lower()
        if method == "ncv":
            return criterion_gradient_ncv(model, y, log_sp, qapprox=False)
        if method == "qncv":
            return criterion_gradient_ncv(model, y, log_sp, qapprox=True)
        if method in {"ml", "reml", "laml"}:
            backend = resolve_ml_reml_scoring_backend(model, method=method)
            if backend in {"gaussian_exact", "gaussian_dynamic"} and method in {
                "reml",
                "laml",
            }:
                exact_method = "REML" if method in {"reml", "laml"} else "ML"
                out = _gaussian_dynamic_reml_derivative_terms(
                    model, y, log_sp, exact_method
                )
                if bool(out.get("valid", False)):
                    return np.asarray(out["grad"], dtype=np.float64)
                raise NotImplementedError(
                    "Gaussian REML/LAML outer optimisation requires exact "
                    "mgcv-parity derivatives; finite-difference fallback removed."
                )
            if backend == GENERAL_FAMILY_BACKEND:
                exact_method = "REML" if method in {"reml", "laml"} else "ML"
                return criterion_gradient_ml_reml_general_family(
                    model, y, log_sp, exact_method
                )
            if (
                backend == "pirls_laplace"
                and (
                    getattr(model.family, "known_scale", None) is not None
                    or str(getattr(model.family, "name", "")).lower() == "gamma"
                )
                and bool(
                    getattr(model.family, "supports_exact_pirls_first_derivatives", False)
                )
            ):
                exact_method = "REML" if method in {"reml", "laml"} else "ML"
                return criterion_gradient_ml_reml_pirls_exact(
                    model, y, log_sp, exact_method
                )
>           raise NotImplementedError(
                "ML/REML/LAML outer optimisation requires an exact upstream-mirrored "
                "derivative path; numerical fallback removed."
            )
E           NotImplementedError: ML/REML/LAML outer optimisation requires an exact upstream-mirrored derivative path; numerical fallback removed.

nampy/gam/smoothing_selection/criteria/dispatch.py:181: NotImplementedError
_____________________________________ test_gam_vcomp_matches_mgcv_requested_surface[shashlss_ml_rescale_false] ______________________________________

data_factory = <function _shashlss_data at 0x703c8b698a40>, formula = ['y ~ s(x, bs="cr", k=6)', '~ 1', '~ 1', '~ 1'], family = 'shashlss'
method = 'ML', rescale = False, atol = 8e-05

    @pytest.mark.parametrize(
        ("data_factory", "formula", "family", "method", "rescale", "atol"),
        [
            (
                lambda: _make_gaussian_data(seed=41, n=120),
                'y ~ s(x0, bs="cr", k=8)',
                "gaussian",
                "GCV",
                False,
                5e-8,
            ),
            (
                lambda: _make_poisson_data(seed=789, n=140),
                'y ~ s(x0, bs="cr", k=8)',
                "poisson",
                "REML",
                False,
                2e-5,
            ),
            (
                lambda: _make_negbin_data(seed=77, n=140),
                'y ~ s(x0, bs="cr", k=8)',
                {"name": "negbin", "theta": 2.5, "estimate_theta": True},
                "REML",
                False,
                5e-5,
            ),
            (
                _gaulss_data,
                ['y ~ s(x, bs="cr", k=6)', "~ 1"],
                "gaulss",
                "ML",
                False,
                2e-5,
            ),
            (
                _gammals_data,
                ['y ~ s(x, bs="cr", k=6)', "~ 1"],
                "gammals",
                "ML",
                False,
                2e-5,
            ),
            (
                _gevlss_data,
                ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1"],
                "gevlss",
                "ML",
                False,
                3e-5,
            ),
            (
                _shashlss_data,
                ['y ~ s(x, bs="cr", k=6)', "~ 1", "~ 1", "~ 1"],
                "shashlss",
                "ML",
                False,
                8e-5,
            ),
            (
                _ziplss_data,
                ['y ~ s(x, bs="cr", k=6)', "~ 1"],
                "ziplss",
                "ML",
                False,
                5e-5,
            ),
        ],
        ids=[
            "gaussian_gcv_rescale_false",
            "poisson_reml_rescale_false",
            "negbin_est_reml_rescale_false",
            "gaulss_ml_rescale_false",
            "gammals_ml_rescale_false",
            "gevlss_ml_rescale_false",
            "shashlss_ml_rescale_false",
            "ziplss_ml_rescale_false",
        ],
    )
    def test_gam_vcomp_matches_mgcv_requested_surface(
        data_factory, formula, family, method, rescale, atol
    ):
        """Verify that gam vcomp matches mgcv requested surface."""
        data = data_factory()
        expected = _run_mgcv_gam_vcomp(
            data,
            formula,
            family,
            method,
            rescale=rescale,
        )
        gam = _fit_nampy_model(data, formula, family, method)
    
        actual = gam.gam_vcomp(rescale=rescale)
    
>       _assert_gam_vcomp_close(actual, expected, atol=atol)

tests/optimization/test_mgcv_vcomp_parity.py:287: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
tests/optimization/test_mgcv_vcomp_parity.py:37: in _assert_gam_vcomp_close
    np.testing.assert_allclose(
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

args = (<function assert_allclose.<locals>.compare at 0x703c865ae0c0>, array([[3.62645936e-04, 7.14692080e-71, 1.84012218e+63]]), array([[3.62646870e-04, 7.14976697e-71, 1.83939914e+63]]))
kwds = {'equal_nan': True, 'err_msg': '', 'header': 'Not equal to tolerance rtol=8e-05, atol=8e-05', 'verbose': True}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
                   ^^^^^^^^^^^^^^^^^^^
E           AssertionError: 
E           Not equal to tolerance rtol=8e-05, atol=8e-05
E           
E           Mismatched elements: 1 / 3 (33.3%)
E           Max absolute difference: 7.23042925e+59
E           Max relative difference: 0.00039808
E            x: array([[3.626459e-04, 7.146921e-71, 1.840122e+63]])
E            y: array([[3.626469e-04, 7.149767e-71, 1.839399e+63]])

../../../miniconda3/envs/nampy/lib/python3.11/contextlib.py:81: AssertionError
============================================================== short test summary info ==============================================================
FAILED tests/optimization/test_mgcv_ncv_qncv_parity.py::test_gaulss_outer_ncv_matches_mgcv - AssertionError: 
Not equal to tolerance rtol=0, atol=1e-06

Mismatched elements: 1 / 1 (100%)
Max absolute difference: 3.99465693e-06
Max relative difference: 2.69853319e-07
 x: array([14.803064])
 y: array(14.803068)
FAILED tests/optimization/test_mgcv_ncv_qncv_parity.py::test_gaulss_outer_qncv_matches_mgcv - AssertionError: 
Not equal to tolerance rtol=0, atol=1e-06

Mismatched elements: 1 / 2 (50%)
Max absolute difference: 8719.75691521
Max relative difference: 0.00032065
 x: array([1.806925e+01, 2.718551e+07])
 y: array([1.806925e+01, 2.719423e+07])
FAILED tests/optimization/test_mgcv_ncv_qncv_parity.py::test_general_family_outer_ncv_qncv_matches_mgcv_requested_families[gammals_ncv] - AssertionError: 
Not equal to tolerance rtol=0, atol=0.0002

Mismatched elements: 1 / 1 (100%)
Max absolute difference: 0.28941042
Max relative difference: 0.00018471
 x: array([1566.544725])
 y: array(1566.834136)
FAILED tests/optimization/test_mgcv_ncv_qncv_parity.py::test_general_family_outer_ncv_qncv_matches_mgcv_requested_families[gammals_qncv] - subprocess.CalledProcessError: Command '['/usr/bin/Rscript', '/home/ad32/projects/package/NAMpy/tests/parity/mgcv_snapshot.R', '/tmp/tmpmrlqz7d9/data.csv', '/tmp/tmpmrlqz7d9/snapshot.json', '[\'y ~ s(x, bs="cr", k=6)\', \'~ 1\']', 'gammals', 'QNCV', 'false']' returned non-zero exit status 1.
FAILED tests/optimization/test_mgcv_ncv_qncv_parity.py::test_general_family_outer_ncv_qncv_matches_mgcv_requested_families[gevlss_ncv] - KeyError: 'l1'
FAILED tests/optimization/test_mgcv_ncv_qncv_parity.py::test_general_family_outer_ncv_qncv_matches_mgcv_requested_families[gevlss_qncv] - subprocess.CalledProcessError: Command '['/usr/bin/Rscript', '/home/ad32/projects/package/NAMpy/tests/parity/mgcv_snapshot.R', '/tmp/tmpn7og1gi7/data.csv', '/tmp/tmpn7og1gi7/snapshot.json', '[\'y ~ s(x, bs="cr", k=6)\', \'~ 1\', \'~ 1\']', 'gevlss', 'QNCV', 'false']' returned non-zero exit status 1.
FAILED tests/optimization/test_mgcv_ncv_qncv_parity.py::test_general_family_outer_ncv_qncv_matches_mgcv_requested_families[shashlss_ncv] - AssertionError: 
Not equal to tolerance rtol=0, atol=0.002

Mismatched elements: 1 / 1 (100%)
Max absolute difference: 10.88538854
Max relative difference: 0.04500534
 x: array([252.754227])
 y: array(241.868838)
FAILED tests/optimization/test_mgcv_ncv_qncv_parity.py::test_general_family_outer_ncv_qncv_matches_mgcv_requested_families[shashlss_qncv] - subprocess.CalledProcessError: Command '['/usr/bin/Rscript', '/home/ad32/projects/package/NAMpy/tests/parity/mgcv_snapshot.R', '/tmp/tmppd0ihkfj/data.csv', '/tmp/tmppd0ihkfj/snapshot.json', '[\'y ~ s(x, bs="cr", k=6)\', \'~ 1\', \'~ 1\', \'~ 1\']', 'shash', 'QNCV', 'false']' returned non-zero exit status 1.
FAILED tests/optimization/test_mgcv_ncv_qncv_parity.py::test_general_family_outer_ncv_qncv_matches_mgcv_requested_families[ziplss_qncv] - subprocess.CalledProcessError: Command '['/usr/bin/Rscript', '/home/ad32/projects/package/NAMpy/tests/parity/mgcv_snapshot.R', '/tmp/tmpadi4vm_j/data.csv', '/tmp/tmpadi4vm_j/snapshot.json', '[\'y ~ s(x, bs="cr", k=6)\', \'~ 1\']', 'ziplss', 'QNCV', 'false']' returned non-zero exit status 1.
FAILED tests/optimization/test_mgcv_ncv_qncv_parity.py::test_gaulss_fixed_sp_ncv_jackknife_dd_matches_mgcv - AssertionError: 
Not equal to tolerance rtol=0, atol=1e-08

Mismatched elements: 1 / 1 (100%)
Max absolute difference: 1.63288621e-06
Max relative difference: 5.65769091e-09
 x: array(288.61354)
 y: array(288.613542)
FAILED tests/optimization/test_mgcv_optimization_lifecycle_parity.py::test_supported_optimization_lifecycle_matches_mgcv[poisson_reml_bfgs_two_cr] - AssertionError: assert 14 == 12
 +  where 14 = len([{'iter': 1, 'log_sp': [4.300068972240508, 5.516250610916343], 'log_scale': None, 'log_theta': None, 'criterion': 319.63478985883967, 'gradient': [0.4364762053398654, -0.28031514879295794], 'gradient_full': [0.4364762053398654, -0.28031514879295794], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 2.6537001955590647, 'n_fun': None, 'n_jac': None, 'n_hess': None, 'rank_info': {'source': 'outer_bfgs_mgcv', 'line_search_alpha': 1.0, 'converged_here': False, 'rolled_back': False}}, {'iter': 2, 'log_sp': [4.043183000845946, 7.268837939223341], 'log_scale': None, 'log_theta': None, 'criterion': 319.15206426427125, 'gradient': [0.0802889249098786, -0.16174398578764138], 'gradient_full': [0.0802889249098786, -0.16174398578764138], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 1.7713138473013728, 'n_fun': None, 'n_jac': None, 'n_hess': None, 'rank_info': {'source': 'outer_bfgs_mgcv', 'line_search_alpha': 1.0, 'converged_here': False, 'rolled_back': False}}, {'iter': 3, 'log_sp': [3.990136632090107, 9.67415394807762], 'log_scale': None, 'log_theta': None, 'criterion': 319.002260018268, 'gradient': [-0.01829882093994195, -0.007645704102510212], 'gradient_full': [-0.01829882093994195, -0.007645704102510212], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 2.4059008748676365, 'n_fun': None, 'n_jac': None, 'n_hess': None, 'rank_info': {'source': 'outer_bfgs_mgcv', 'line_search_alpha': 1.0, 'converged_here': False, 'rolled_back': False}}, {'iter': 4, 'log_sp': [4.004410471308293, 9.796328406485703], 'log_scale': None, 'log_theta': None, 'criterion': 319.00128545153086, 'gradient': [0.00037615374804866875, -0.006256273268422624], 'gradient_full': [0.00037615374804866875, -0.006256273268422624], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.1230054501773595, 'n_fun': None, 'n_jac': None, 'n_hess': None, 'rank_info': {'source': 'outer_bfgs_mgcv', 'line_search_alpha': 1.0, 'converged_here': False, 'rolled_back': False}}, {'iter': 5, 'log_sp': [4.017547178465264, 10.166019328739392], 'log_scale': None, 'log_theta': None, 'criterion': 318.9996867201288, 'gradient': [0.01666137790081823, -0.0032970631277388293], 'gradient_full': [0.01666137790081823, -0.0032970631277388293], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.3699242504509687, 'n_fun': None, 'n_jac': None, 'n_hess': None, 'rank_info': {'source': 'outer_bfgs_mgcv', 'line_search_alpha': 1.0, 'converged_here': False, 'rolled_back': False}}, {'iter': 6, 'log_sp': [4.014725264209582, 10.519483301412755], 'log_scale': None, 'log_theta': None, 'criterion': 318.9987861912106, 'gradient': [0.011729913332689357, -0.001732577754065552], 'gradient_full': [0.011729913332689357, -0.001732577754065552], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.353475237008341, 'n_fun': None, 'n_jac': None, 'n_hess': None, 'rank_info': {'source': 'outer_bfgs_mgcv', 'line_search_alpha': 1.0, 'converged_here': False, 'rolled_back': False}}, {'iter': 7, 'log_sp': [4.008316408334103, 10.911880437383026], 'log_scale': None, 'log_theta': None, 'criterion': 318.99826003421884, 'gradient': [0.0021468688834749283, -0.0008296690969740589], 'gradient_full': [0.0021468688834749283, -0.0008296690969740589], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.39244946904194383, 'n_fun': None, 'n_jac': None, 'n_hess': None, 'rank_info': {'source': 'outer_bfgs_mgcv', 'line_search_alpha': 1.0, 'converged_here': False, 'rolled_back': False}}, {'iter': 8, 'log_sp': [4.005885737052279, 11.261888732570103], 'log_scale': None, 'log_theta': None, 'criterion': 318.99804690548143, 'gradient': [-0.0017016839445478382, -0.0004284892708669892], 'gradient_full': [-0.0017016839445478382, -0.0004284892708669892], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.350016735117972, 'n_fun': None, 'n_jac': None, 'n_hess': None, 'rank_info': {'source': 'outer_bfgs_mgcv', 'line_search_alpha': 1.0, 'converged_here': False, 'rolled_back': False}}, {'iter': 9, 'log_sp': [4.006162611356391, 11.628068985747257], 'log_scale': None, 'log_theta': None, 'criterion': 318.99793285876177, 'gradient': [-0.0017303715089476146, -0.00021600724155669722], 'gradient_full': [-0.0017303715089476146, -0.00021600724155669722], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.36618035785151665, 'n_fun': None, 'n_jac': None, 'n_hess': None, 'rank_info': {'source': 'outer_bfgs_mgcv', 'line_search_alpha': 1.0, 'converged_here': False, 'rolled_back': False}}, {'iter': 10, 'log_sp': [4.007320107204666, 12.00273543544103], 'log_scale': None, 'log_theta': None, 'criterion': 318.99787318080155, 'gradient': [-0.00044592043861912956, -0.00010836540437384086], 'gradient_full': [-0.00044592043861912956, -0.00010836540437384086], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.37466823767538066, 'n_fun': None, 'n_jac': None, 'n_hess': None, 'rank_info': {'source': 'outer_bfgs_mgcv', 'line_search_alpha': 1.0, 'converged_here': False, 'rolled_back': False}}, {'iter': 11, 'log_sp': [4.008059626759502, 12.375891048074866], 'log_scale': None, 'log_theta': None, 'criterion': 318.99784377788876, 'gradient': [0.00036024601178419324, -5.5193434197281684e-05], 'gradient_full': [0.00036024601178419324, -5.5193434197281684e-05], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.373156345422807, 'n_fun': None, 'n_jac': None, 'n_hess': None, 'rank_info': {'source': 'outer_bfgs_mgcv', 'line_search_alpha': 1.0, 'converged_here': False, 'rolled_back': False}}, {'iter': 12, 'log_sp': [4.008234309354647, 12.755574318532474], 'log_scale': None, 'log_theta': None, 'criterion': 318.99782861017854, 'gradient': [0.00045771043065712114, -2.82177652474137e-05], 'gradient_full': [0.00045771043065712114, -2.82177652474137e-05], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.37968331064111094, 'n_fun': None, 'n_jac': None, 'n_hess': None, 'rank_info': {'source': 'outer_bfgs_mgcv', 'line_search_alpha': 1.0, 'converged_here': False, 'rolled_back': False}}, {'iter': 13, 'log_sp': [4.0081415336392885, 9.582585514105599], 'log_scale': None, 'log_theta': None, 'criterion': 319.002917176048, 'gradient': [0.0065916731904189785, -0.009047943129907546], 'gradient_full': [0.0065916731904189785, -0.009047943129907546], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 3.1729888057832203, 'n_fun': None, 'n_jac': None, 'n_hess': None, 'rank_info': {'source': 'outer_bfgs_mgcv', 'line_search_alpha': 1.0, 'converged_here': True, 'rolled_back': True}}, {'iter': 14, 'log_sp': [4.008311959258073, 14.582585514105599], 'log_scale': None, 'log_theta': None, 'criterion': 318.9978123769436, 'gradient': [0.0003070874699957482, -1.6857579837202454e-06], 'gradient_full': [0.0003070874699957482, -1.6857579837202454e-06], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 5.000000002904489, 'n_fun': None, 'n_jac': None, 'n_hess': None, 'rank_info': {'source': 'outer_bfgs_mgcv', 'line_search_alpha': 0.019363934246710457, 'converged_here': False, 'rolled_back': True}}])
 +  and   12 = len([{'iter': 1, 'log_sp': [4.300071694030156, 5.516256204956271], 'log_scale': {}, 'log_theta': {}, 'criterion': 319.63478949472216, 'gradient': [0.43648007502022823, -0.2803149702643024], 'gradient_full': [0.43648007502022823, -0.2803149702643024], 'hessian': {}, 'hessian_full': {}, 'accepted_step_norm': 0, 'rank_info': {'source': 'mgcv_bfgs', 'line_search_alpha': 1, 'converged_here': False, 'rolled_back': False}}, {'iter': 2, 'log_sp': [4.043183015121064, 7.268842012970138], 'log_scale': {}, 'log_theta': {}, 'criterion': 319.1520636065645, 'gradient': [0.0802888894934175, -0.16174353593494706], 'gradient_full': [0.0802888894934175, -0.16174353593494706], 'hessian': {}, 'hessian_full': {}, 'accepted_step_norm': 1.771312735742415, 'rank_info': {'source': 'mgcv_bfgs', 'line_search_alpha': 1, 'converged_here': False, 'rolled_back': False}}, {'iter': 3, 'log_sp': [3.990136380540041, 9.67414300577748], 'log_scale': {}, 'log_theta': {}, 'criterion': 319.0022601065308, 'gradient': [-0.01829910025095227, -0.007645846701105796], 'gradient_full': [-0.01829910025095227, -0.007645846701105796], 'hessian': {}, 'hessian_full': {}, 'accepted_step_norm': 2.4058858683321533, 'rank_info': {'source': 'mgcv_bfgs', 'line_search_alpha': 1, 'converged_here': False, 'rolled_back': False}}, {'iter': 4, 'log_sp': [4.004410410127232, 9.79631943134887], 'log_scale': {}, 'log_theta': {}, 'criterion': 319.0012855076588, 'gradient': [0.0003761152311314575, -0.006256370554316604], 'gradient_full': [0.0003761152311314575, -0.006256370554316604], 'hessian': {}, 'hessian_full': {}, 'accepted_step_norm': 0.12300742614190913, 'rank_info': {'source': 'mgcv_bfgs', 'line_search_alpha': 1, 'converged_here': False, 'rolled_back': False}}, {'iter': 5, 'log_sp': [4.0175471540626795, 10.166013282444263], 'log_scale': {}, 'log_theta': {}, 'criterion': 318.9996867396676, 'gradient': [0.01666136636265403, -0.003297098694214018], 'gradient_full': [0.01666136636265403, -0.003297098694214018], 'hessian': {}, 'hessian_full': {}, 'accepted_step_norm': 0.36992717875139725, 'rank_info': {'source': 'mgcv_bfgs', 'line_search_alpha': 1, 'converged_here': False, 'rolled_back': False}}, {'iter': 6, 'log_sp': [4.014725173585732, 10.519477823854725], 'log_scale': {}, 'log_theta': {}, 'criterion': 318.9987861996316, 'gradient': [0.01172980429060555, -0.0017325948723994067], 'gradient_full': [0.01172980429060555, -0.0017325948723994067], 'hessian': {}, 'hessian_full': {}, 'accepted_step_norm': 0.3534758062559876, 'rank_info': {'source': 'mgcv_bfgs', 'line_search_alpha': 1, 'converged_here': False, 'rolled_back': False}}, {'iter': 7, 'log_sp': [4.008316351410813, 10.911874836763838], 'log_scale': {}, 'log_theta': {}, 'criterion': 318.99826003873886, 'gradient': [0.0021468017243759796, -0.0008296776466956146], 'gradient_full': [0.0021468017243759796, -0.0008296776466956146], 'hessian': {}, 'hessian_full': {}, 'accepted_step_norm': 0.392449345446854, 'rank_info': {'source': 'mgcv_bfgs', 'line_search_alpha': 1, 'converged_here': False, 'rolled_back': False}}, {'iter': 8, 'log_sp': [4.005885728442085, 11.261883481624897], 'log_scale': {}, 'log_theta': {}, 'criterion': 318.9980469077447, 'gradient': [-0.0017016887543506165, -0.0004284934792595507], 'gradient_full': [-0.0017016887543506165, -0.0004284934792595507], 'hessian': {}, 'hessian_full': {}, 'accepted_step_norm': 0.35001708444801866, 'rank_info': {'source': 'mgcv_bfgs', 'line_search_alpha': 1, 'converged_here': False, 'rolled_back': False}}, {'iter': 9, 'log_sp': [4.0061626313108185, 11.628064224070947], 'log_scale': {}, 'log_theta': {}, 'criterion': 318.9979328597498, 'gradient': [-0.001730339996229624, -0.0002160091819733978], 'gradient_full': [-0.001730339996229624, -0.0002160091819733978], 'hessian': {}, 'hessian_full': {}, 'accepted_step_norm': 0.3661808471418716, 'rank_info': {'source': 'mgcv_bfgs', 'line_search_alpha': 1, 'converged_here': False, 'rolled_back': False}}, {'iter': 10, 'log_sp': [4.007320124192537, 12.002730509365145], 'log_scale': {}, 'log_theta': {}, 'criterion': 318.9978731813321, 'gradient': [-0.0004458941882519163, -0.00010836639833877726], 'gradient_full': [-0.0004458941882519163, -0.00010836639833877726], 'hessian': {}, 'hessian_full': {}, 'accepted_step_norm': 0.3746680732674255, 'rank_info': {'source': 'mgcv_bfgs', 'line_search_alpha': 1, 'converged_here': False, 'rolled_back': False}}, {'iter': 11, 'log_sp': [4.008059629535397, 12.375886046330184], 'log_scale': {}, 'log_theta': {}, 'criterion': 318.9978437781608, 'gradient': [0.00036025200233025245, -5.519393198483158e-05], 'gradient_full': [0.00036025200233025245, -5.519393198483158e-05], 'hessian': {}, 'hessian_full': {}, 'accepted_step_norm': 0.37315626972599325, 'rank_info': {'source': 'mgcv_bfgs', 'line_search_alpha': 1, 'converged_here': False, 'rolled_back': False}}, {'iter': 12, 'log_sp': [4.008234303306564, 12.755569401939546], 'log_scale': {}, 'log_theta': {}, 'criterion': 318.99782861029377, 'gradient': [0.0004577036980606408, -2.8218004940239183e-05], 'gradient_full': [0.0004577036980606408, -2.8218004940239183e-05], 'hessian': {}, 'hessian_full': {}, 'accepted_step_norm': 0.3796833957887965, 'rank_info': {'source': 'mgcv_bfgs', 'line_search_alpha': 1, 'converged_here': False, 'rolled_back': False}}])
FAILED tests/optimization/test_mgcv_optimization_lifecycle_parity.py::test_supported_optimization_lifecycle_matches_mgcv[poisson_reml_efs_two_cr] - AssertionError: poisson_reml_efs_two_cr: Vc expected None, got matrix
assert array([[ 4.42616929e-03,  4.09289971e-03,  2.21790200e-03,\n         1.57593473e-03,  4.98580514e-04, -1.54892176e-03,\n        -2.07985559e-04,  6.58467941e-04,  1.12991130e-04,\n        -6.71815575e-04,  3.62820564e-04,  7.26152235e-04,\n         9.45949757e-04,  2.51077523e-03,  4.97811806e-03],\n       [ 4.09289971e-03,  1.18498413e-01,  3.96934867e-02,\n         6.18326196e-02,  6.18143823e-02, -7.68198399e-03,\n         1.04235770e-01,  1.66956199e-02, -5.95484369e-03,\n        -2.90950263e-03, -4.66252266e-03, -4.16743529e-03,\n        -9.56491657e-04, -8.23304509e-04,  1.71297996e-02],\n       [ 2.21790200e-03,  3.96934867e-02,  2.04285492e-02,\n         2.43249084e-02,  1.98002726e-02, -5.29002989e-03,\n         3.20247375e-02,  2.79916441e-03, -2.92964605e-03,\n        -1.06574972e-03, -2.42934844e-03, -2.37257173e-03,\n        -9.34934045e-04, -1.65756856e-03,  5.75780758e-03],\n       [ 1.57593473e-03,  6.18326196e-02,  2.43249084e-02,\n         4.65829567e-02,  4.42475503e-02, -2.38291034e-03,\n         6.48163226e-02,  1.12713031e-02, -3.54541763e-03,\n        -1.88488454e-03, -2.71631483e-03, -2.34294357e-03,\n        -3.72877885e-04,  3.67736036e-05,  1.13101895e-02],\n       [ 4.98580514e-04,  6.18143823e-02,  1.98002726e-02,\n         4.42475503e-02,  4.95385338e-02,  2.45650841e-03,\n         7.06758166e-02,  1.33379033e-02, -2.40575707e-03,\n        -1.73854652e-03, -1.67272968e-03, -1.18495815e-03,\n         3.30310994e-04,  1.60042018e-03,  1.10376689e-02],\n       [-1.54892176e-03, -7.68198399e-03, -5.29002989e-03,\n        -2.38291034e-03,  2.45650841e-03,  7.52084472e-03,\n         3.26783061e-03, -3.88394226e-03,  1.43330451e-03,\n         3.02461771e-04,  1.27079751e-03,  1.35489717e-03,\n         7.36333422e-04,  1.56346496e-03, -1.21548335e-03],\n       [-2.07985559e-04,  1.04235770e-01,  3.20247375e-02,\n         6.48163226e-02,  7.06758166e-02,  3.26783061e-03,\n         1.21965566e-01,  5.95880708e-03, -6.74219776e-03,\n        -3.50565013e-03, -5.19293152e-03, -4.52216509e-03,\n        -8.06737757e-04, -1.99259377e-04,  2.09164188e-02],\n       [ 6.58467941e-04,  1.66956199e-02,  2.79916441e-03,\n         1.12713031e-02,  1.33379033e-02, -3.88394226e-03,\n         5.95880708e-03,  3.22354519e-01,  2.11039173e-02,\n         4.41610627e-03,  1.86003257e-02,  1.98233444e-02,\n         1.07652485e-02,  2.29906486e-02, -1.73435995e-02],\n       [ 1.12991130e-04, -5.95484369e-03, -2.92964605e-03,\n        -3.54541763e-03, -2.40575707e-03,  1.43330451e-03,\n        -6.74219776e-03,  2.11039173e-02,  2.44583854e-02,\n         7.70102112e-03,  2.05948787e-02,  2.06919234e-02,\n         9.19303391e-03,  1.77989919e-02, -3.89473180e-02],\n       [-6.71815575e-04, -2.90950263e-03, -1.06574972e-03,\n        -1.88488454e-03, -1.73854652e-03,  3.02461771e-04,\n        -3.50565013e-03,  4.41610627e-03,  7.70102112e-03,\n         3.70376049e-03,  6.02376071e-03,  5.40050953e-03,\n         1.27899948e-03,  1.25382286e-03, -2.15233513e-02],\n       [ 3.62820564e-04, -4.66252266e-03, -2.42934844e-03,\n        -2.71631483e-03, -1.67272968e-03,  1.27079751e-03,\n        -5.19293152e-03,  1.86003257e-02,  2.05948787e-02,\n         6.02376071e-03,  1.75978304e-02,  1.79366830e-02,\n         8.40418654e-03,  1.66529217e-02, -2.96406886e-02],\n       [ 7.26152235e-04, -4.16743529e-03, -2.37257173e-03,\n        -2.34294357e-03, -1.18495815e-03,  1.35489717e-03,\n        -4.52216509e-03,  1.98233444e-02,  2.06919234e-02,\n         5.40050953e-03,  1.79366830e-02,  1.86218592e-02,\n         9.29377795e-03,  1.89675594e-02, -2.51817883e-02],\n       [ 9.45949757e-04, -9.56491657e-04, -9.34934045e-04,\n        -3.72877885e-04,  3.30310994e-04,  7.36333422e-04,\n        -8.06737757e-04,  1.07652485e-02,  9.19303391e-03,\n         1.27899948e-03,  8.40418654e-03,  9.29377795e-03,\n         5.58967625e-03,  1.22829844e-02, -3.24369239e-03],\n       [ 2.51077523e-03, -8.23304509e-04, -1.65756856e-03,\n         3.67736036e-05,  1.60042018e-03,  1.56346496e-03,\n        -1.99259377e-04,  2.29906486e-02,  1.77989919e-02,\n         1.25382286e-03,  1.66529217e-02,  1.89675594e-02,\n         1.22829844e-02,  2.79102406e-02,  3.06069666e-03],\n       [ 4.97811806e-03,  1.71297996e-02,  5.75780758e-03,\n         1.13101895e-02,  1.10376689e-02, -1.21548335e-03,\n         2.09164188e-02, -1.73435995e-02, -3.89473180e-02,\n        -2.15233513e-02, -2.96406886e-02, -2.51817883e-02,\n        -3.24369239e-03,  3.06069666e-03,  1.31787681e-01]]) is None
FAILED tests/optimization/test_mgcv_optimization_lifecycle_parity.py::test_supported_optimization_lifecycle_matches_mgcv[poisson_reml_optim_two_cr] - AssertionError: assert 23 == 9
 +  where 23 = len([{'iter': 0, 'log_sp': [2.935425365778347, 3.2403172737440595], 'log_scale': None, 'log_theta': None, 'criterion': 321.0440273557411, 'gradient': [-1.1295541781544693, -0.534017408370719], 'gradient_full': [-1.1295541781544693, -0.534017408370719], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.0, 'n_fun': 1, 'n_jac': 1, 'n_hess': 0, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 1, 'log_sp': [3.65984431859691, 3.5827995904319234], 'log_scale': None, 'log_theta': None, 'criterion': 320.29812714551997, 'gradient': [-0.42447449626747935, -0.45391856281534326], 'gradient_full': [-0.42447449626747935, -0.45391856281534326], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.8012970463234151, 'n_fun': 2, 'n_jac': 2, 'n_hess': 0, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 2, 'log_sp': [4.228696290781205, 4.3552262471137], 'log_scale': None, 'log_theta': None, 'criterion': 319.95705731709756, 'gradient': [0.305468740825104, -0.33491002353564214], 'gradient_full': [0.305468740825104, -0.33491002353564214], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.9592890629057277, 'n_fun': 3, 'n_jac': 3, 'n_hess': 0, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 3, 'log_sp': [4.234602432060098, 5.032621219893999], 'log_scale': None, 'log_theta': None, 'criterion': 319.7480952614307, 'gradient': [0.3356638370741716, -0.2940076648500367], 'gradient_full': [0.3356638370741716, -0.2940076648500367], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.6774207198284001, 'n_fun': 4, 'n_jac': 4, 'n_hess': 0, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 4, 'log_sp': [3.6053393678390657, 10.317829746279795], 'log_scale': None, 'log_theta': None, 'criterion': 319.1034233809872, 'gradient': [-0.5088385616778168, -0.0010921637392508482], 'gradient_full': [-0.5088385616778168, -0.0010921637392508482], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 5.322537099107339, 'n_fun': 5, 'n_jac': 5, 'n_hess': 0, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 5, 'log_sp': [4.383338380702089, 11.487453859315782], 'log_scale': None, 'log_theta': None, 'criterion': 319.09704074507204, 'gradient': [0.5334131210169342, -0.0006227793683336764], 'gradient_full': [0.5334131210169342, -0.0006227793683336764], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 1.4047430476108633, 'n_fun': 6, 'n_jac': 6, 'n_hess': 0, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 6, 'log_sp': [3.986134730445479, 10.890307899687375], 'log_scale': None, 'log_theta': None, 'criterion': 318.99856354684016, 'gradient': [-0.027874105164421348, -0.0008220527214665395], 'gradient_full': [-0.027874105164421348, -0.0008220527214665395], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.717184799670005, 'n_fun': 7, 'n_jac': 7, 'n_hess': 0, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 7, 'log_sp': [4.009153773639365, 10.940641308186498], 'log_scale': None, 'log_theta': None, 'criterion': 318.9982390311369, 'gradient': [0.003233628165520308, -0.0007878754621811791], 'gradient_full': [0.003233628165520308, -0.0007878754621811791], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.055347342851320015, 'n_fun': 8, 'n_jac': 8, 'n_hess': 0, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 8, 'log_sp': [4.008036035312223, 10.940399535808819], 'log_scale': None, 'log_theta': None, 'criterion': 318.99823645622484, 'gradient': [0.0017142903666795828, -0.00078624267439888], 'gradient_full': [0.0017142903666795828, -0.00078624267439888], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.0011435877100474507, 'n_fun': 9, 'n_jac': 9, 'n_hess': 0, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 9, 'log_sp': [4.006905713522768, 10.945725191958454], 'log_scale': None, 'log_theta': None, 'criterion': 318.9982312310102, 'gradient': [0.00016844772520707707, -0.0007764748252485874], 'gradient_full': [0.00016844772520707707, -0.0007764748252485874], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.005444285147920129, 'n_fun': 10, 'n_jac': 10, 'n_hess': 0, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 10, 'log_sp': [4.004274620042285, 10.97217743978736], 'log_scale': None, 'log_theta': None, 'criterion': 318.99821557192763, 'gradient': [-0.0034523029067650146, -0.0007345103767328089], 'gradient_full': [-0.0034523029067650146, -0.0007345103767328089], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.02658277765969795, 'n_fun': 11, 'n_jac': 11, 'n_hess': 0, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 11, 'log_sp': [4.0020058434084, 11.022969109689566], 'log_scale': None, 'log_theta': None, 'criterion': 318.9981915069652, 'gradient': [-0.0066183200576015455, -0.000664072583967442], 'gradient_full': [-0.0066183200576015455, -0.000664072583967442], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.05084231582913024, 'n_fun': 12, 'n_jac': 12, 'n_hess': 0, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 12, 'log_sp': [3.9999774181360563, 11.197475378124091], 'log_scale': None, 'log_theta': None, 'criterion': 318.9981094857285, 'gradient': [-0.009636400998848105, -0.0004752415235234411], 'gradient_full': [-0.009636400998848105, -0.0004752415235234411], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.17451805703716763, 'n_fun': 13, 'n_jac': 13, 'n_hess': 0, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 13, 'log_sp': [4.013339579125828, 11.746717588213627], 'log_scale': None, 'log_theta': None, 'criterion': 318.9979317428211, 'gradient': [0.007932222517291088, -0.00017898401563805344], 'gradient_full': [0.007932222517291088, -0.00017898401563805344], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.5494047257626704, 'n_fun': 14, 'n_jac': 14, 'n_hess': 0, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 14, 'log_sp': [4.008887006372796, 11.984819626256623], 'log_scale': None, 'log_theta': None, 'criterion': 318.99787614367085, 'gradient': [0.001696923218222679, -0.00011302081226099818], 'gradient_full': [0.001696923218222679, -0.00011302081226099818], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.23814366656358785, 'n_fun': 15, 'n_jac': 15, 'n_hess': 0, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 15, 'log_sp': [4.007007763194171, 12.329686385097956], 'log_scale': None, 'log_theta': None, 'criterion': 318.997846788712, 'gradient': [-0.0010494560292300115, -5.949696567782853e-05], 'gradient_full': [-0.0010494560292300115, -5.949696567782853e-05], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.34487187897630944, 'n_fun': 16, 'n_jac': 16, 'n_hess': 0, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 16, 'log_sp': [4.0084570012641745, 12.750554865751258], 'log_scale': None, 'log_theta': None, 'criterion': 318.99782888842105, 'gradient': [0.0007622141292240592, -2.8533574364496793e-05], 'gradient_full': [0.0007622141292240592, -2.8533574364496793e-05], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.42087097583273997, 'n_fun': 17, 'n_jac': 17, 'n_hess': 0, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 17, 'log_sp': [4.005469736062162, 13.277505916220901], 'log_scale': None, 'log_theta': None, 'criterion': 318.9978231468166, 'gradient': [-0.003426042550651376, -1.1166767103458833e-05], 'gradient_full': [-0.003426042550651376, -1.1166767103458833e-05], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.5269595177472816, 'n_fun': 18, 'n_jac': 18, 'n_hess': 0, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 18, 'log_sp': [4.0075127169021325, 13.519713366895985], 'log_scale': None, 'log_theta': None, 'criterion': 318.9978166758704, 'gradient': [-0.0006868912806763383, -7.795133236266748e-06], 'gradient_full': [-0.0006868912806763383, -7.795133236266748e-06], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.2422160666290234, 'n_fun': 19, 'n_jac': 19, 'n_hess': 0, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 19, 'log_sp': [4.0083128930723255, 13.845828699973076], 'log_scale': None, 'log_theta': None, 'criterion': 318.9978145394668, 'gradient': [0.0003620477040586767, -4.781067310464203e-06], 'gradient_full': [0.0003620477040586767, -4.781067310464203e-06], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.32611631475577135, 'n_fun': 20, 'n_jac': 20, 'n_hess': 0, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 20, 'log_sp': [4.008535552772691, 14.189524225710397], 'log_scale': None, 'log_theta': None, 'criterion': 318.99781335871097, 'gradient': [0.0006350733408126974, -2.909126742146233e-06], 'gradient_full': [0.0006350733408126974, -2.909126742146233e-06], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.34369559786124027, 'n_fun': 21, 'n_jac': 21, 'n_hess': 0, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 21, 'log_sp': [4.004988634901054, 15.165435612359754], 'log_scale': None, 'log_theta': None, 'criterion': 318.99781824721015, 'gradient': [-0.00423448898985157, -7.125086322947458e-07], 'gradient_full': [-0.00423448898985157, -7.125086322947458e-07], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.9759178322063085, 'n_fun': 22, 'n_jac': 22, 'n_hess': 0, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 22, 'log_sp': [4.00766085030434, 14.430192921656278], 'log_scale': None, 'log_theta': None, 'criterion': 318.9978127428589, 'gradient': [-0.0005706364781747908, -2.0321783352178057e-06], 'gradient_full': [-0.0005706364781747908, -2.0321783352178057e-06], 'hessian': None, 'hessian_full': None, 'accepted_step_norm': 0.7352475467269843, 'n_fun': 23, 'n_jac': 23, 'n_hess': 0, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}])
 +  and   9 = len([{'iter': 0, 'log_sp': [2.9354253657783467, 3.2403172737440595], 'log_scale': {}, 'log_theta': {}, 'criterion': 321.04402735583307, 'gradient': [-1.1295541780905254, -0.5340174083814895], 'gradient_full': [-1.1295541780905254, -0.5340174083814895], 'hessian': {}, 'hessian_full': {}, 'accepted_step_norm': 0, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 1, 'log_sp': [3.839483300513346, 3.667727206528368], 'log_scale': {}, 'log_theta': {}, 'criterion': 320.20289481741355, 'gradient': [-0.21343450898869776, -0.4345975283173309], 'gradient_full': [-0.21343450898869776, -0.4345975283173309], 'hessian': {}, 'hessian_full': {}, 'accepted_step_norm': 0.9999999999999998, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 2, 'log_sp': [4.182009738713196, 4.320647472572469], 'log_scale': {}, 'log_theta': {}, 'criterion': 319.95594149849114, 'gradient': [0.2421666847330819, -0.3398429628103927], 'gradient_full': [0.2421666847330819, -0.3398429628103927], 'hessian': {}, 'hessian_full': {}, 'accepted_step_norm': 0.737312169082388, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 3, 'log_sp': [4.252604198737371, 5.447547742661423], 'log_scale': {}, 'log_theta': {}, 'criterion': 319.634998515048, 'gradient': [0.36965216538254975, -0.28265531902076857], 'gradient_full': [0.36965216538254975, -0.28265531902076857], 'hessian': {}, 'hessian_full': {}, 'accepted_step_norm': 1.1291092934311815, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 4, 'log_sp': [3.708413984574331, 11.77418208841133], 'log_scale': {}, 'log_theta': {}, 'criterion': 319.0568789042569, 'gradient': [-0.38774552664759554, 9.161334576468505e-05], 'gradient_full': [-0.38774552664759554, 9.161334576468505e-05], 'hessian': {}, 'hessian_full': {}, 'accepted_step_norm': 6.3499956798420865, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 5, 'log_sp': [4.29377743523702, 12.593876220064427], 'log_scale': {}, 'log_theta': {}, 'criterion': 319.0547942740766, 'gradient': [0.4026016286169001, -0.00012885731400258393], 'gradient_full': [0.4026016286169001, -0.00012885731400258393], 'hessian': {}, 'hessian_full': {}, 'accepted_step_norm': 1.0072481515685476, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 6, 'log_sp': [3.9956113624621, 12.176349331709915], 'log_scale': {}, 'log_theta': {}, 'criterion': 318.9979566057946, 'gradient': [-0.016444741473737867, -7.247964050938194e-05], 'gradient_full': [-0.016444741473737867, -7.247964050938194e-05], 'hessian': {}, 'hessian_full': {}, 'accepted_step_norm': 0.5130611166839875, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 7, 'log_sp': [4.008418468047368, 12.196172184408558], 'log_scale': {}, 'log_theta': {}, 'criterion': 318.99785574459025, 'gradient': [0.0009354299090120755, -7.659395337400454e-05], 'gradient_full': [0.0009354299090120755, -7.659395337400454e-05], 'hessian': {}, 'hessian_full': {}, 'accepted_step_norm': 0.02360015768134441, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 1, 'n_jac': 1}}, {'iter': 8, 'log_sp': [4.0078163461034695, 12.195408114834141], 'log_scale': {}, 'log_theta': {}, 'criterion': 318.99785548625994, 'gradient': [0.00011678092105604776, -7.638199535842105e-05], 'gradient_full': [0.00011678092105604776, -7.638199535842105e-05], 'hessian': {}, 'hessian_full': {}, 'accepted_step_norm': 0.0009728068409887361, 'rank_info': {'source': 'mgcv_optim', 'n_fun': 2, 'n_jac': 1}}])
FAILED tests/optimization/test_mgcv_optimization_lifecycle_parity.py::test_supported_optimization_lifecycle_matches_mgcv[gaulss_reml_efs_two_cr] - AssertionError: 
Not equal to tolerance rtol=5e-05, atol=5e-08
gaulss_reml_efs_two_cr: Vc diagonal mismatch
Mismatched elements: 11 / 12 (91.7%)
Max absolute difference: 0.00496487
Max relative difference: 7.85791406
 x: array([0.002962, 0.023656, 0.018977, 0.018   , 0.023059, 0.067115,
       0.001026, 0.003218, 0.007297, 0.015458, 0.013375, 0.003699])
 y: array([0.002962, 0.023047, 0.018156, 0.017677, 0.022439, 0.063194,
       0.000451, 0.000363, 0.002332, 0.010521, 0.012946, 0.003696])
FAILED tests/optimization/test_mgcv_outer_optimization_parity.py::test_poisson_outer_bfgs_trace_matches_mgcv - assert 2 == 0
 +  where 2 = int(2.95158706060724)
FAILED tests/optimization/test_mgcv_outer_optimization_parity.py::test_gaulss_outer_efs_trace_matches_mgcv - AssertionError: assert {'log_scale', 'message', 'lsp1', 'hessian', 'gradient', 'hess1', 'edge_correct', 'convergence', 'score_hist', 'iter', 'hessian_full', 'counts', 'optimizer', 'gradient_full', 'conv', 'log_theta'} <= {'grad', 'hess', 'message', 'conv', 'convergence', 'score_hist', 'iter'}
  
  Extra items in the left set:
  'log_scale'
  'lsp1'
  'hessian_full'
  'hessian'
  'gradient'
  'counts'
  'hess1'
  'optimizer'
  'gradient_full'
  'edge_correct'
  'log_theta'
FAILED tests/optimization/test_mgcv_outer_optimization_parity.py::test_gamma_outer_newton_joint_scale_history_matches_mgcv - AssertionError: 
Not equal to tolerance rtol=0, atol=5e-07

Mismatched elements: 1 / 1 (100%)
Max absolute difference: 5.50055039e-07
Max relative difference: 4.96463121e-07
 x: array(-1.107947)
 y: array(-1.107947)
FAILED tests/optimization/test_mgcv_outer_optimization_parity.py::test_negbin_est_outer_newton_trace_matches_mgcv_joint_theta - TypeError: float() argument must be a string or a real number, not 'list'
FAILED tests/optimization/test_mgcv_outer_optimization_parity.py::test_poisson_tensor_vector_fx_outer_newton_trace_matches_mgcv[te_fx_vector] - TypeError: float() argument must be a string or a real number, not 'list'
FAILED tests/optimization/test_mgcv_outer_optimization_parity.py::test_poisson_tensor_vector_fx_outer_newton_trace_matches_mgcv[ti_fx_vector_mc] - TypeError: float() argument must be a string or a real number, not 'list'
FAILED tests/optimization/test_mgcv_outer_optimization_parity.py::test_poisson_outer_optim_endpoint_and_metadata_match_mgcv - AssertionError: assert {'log_scale', 'accepted_step_norm', 'criterion', 'hessian_full', 'hessian', 'gradient', 'rank_info', 'gradient_full', 'log_sp', 'iter', 'log_theta'} <= {'accepted_step_norm', 'n_hess', 'criterion', 'hessian_full', 'n_fun', 'hessian', 'gradient', 'rank_info', 'n_jac', 'gradient_full', 'log_sp', 'iter', 'log_theta'}
  
  Extra items in the left set:
  'log_scale'
FAILED tests/optimization/test_mgcv_postprocessing_final_fit_parity.py::test_gam_fit3_non_gaussian_unconditional_postproc_matches_mgcv[gamma_log] - AssertionError: 
Not equal to tolerance rtol=3e-05, atol=5e-08
gamma_log: Vc diagonal mismatch
Mismatched elements: 12 / 12 (100%)
Max absolute difference: 0.00127348
Max relative difference: 0.00734783
 x: array([0.001541, 0.021422, 0.032167, 0.00321 , 0.009908, 0.002163,
       0.008394, 0.001239, 0.007373, 0.001036, 0.174588, 0.030247])
 y: array([0.001541, 0.021286, 0.032027, 0.003194, 0.009857, 0.002149,
       0.008345, 0.001232, 0.007325, 0.00103 , 0.173314, 0.030081])
FAILED tests/optimization/test_mgcv_postprocessing_final_fit_parity.py::test_gam_fit3_postprocessing_final_fit_matches_mgcv[gamma_log] - AssertionError: 
Not equal to tolerance rtol=3e-05, atol=5e-08
gamma_log: Vc diagonal mismatch
Mismatched elements: 12 / 12 (100%)
Max absolute difference: 0.00127348
Max relative difference: 0.00734783
 x: array([0.001541, 0.021422, 0.032167, 0.00321 , 0.009908, 0.002163,
       0.008394, 0.001239, 0.007373, 0.001036, 0.174588, 0.030247])
 y: array([0.001541, 0.021286, 0.032027, 0.003194, 0.009857, 0.002149,
       0.008345, 0.001232, 0.007325, 0.00103 , 0.173314, 0.030081])
FAILED tests/optimization/test_mgcv_postprocessing_final_fit_parity.py::test_gam_fit3_postprocessing_final_fit_matches_mgcv[gaussian_random_intercept_re] - AssertionError: gaussian_random_intercept_re: outer_info iter mismatch 8 != 7
assert 8 == 7
 +  where 8 = int(8)
 +  and   7 = int(7)
FAILED tests/optimization/test_mgcv_postprocessing_final_fit_parity.py::test_gam_fit3_postprocessing_final_fit_matches_mgcv[mrf_lattice] - AssertionError: 
Not equal to tolerance rtol=0, atol=0.0002
mrf_lattice: outer_info score_hist mismatch
Mismatched elements: 18 / 20 (90%)
Max absolute difference: 1.00068465
Max relative difference: 0.21213095
 x: array([   7.065563,    2.530175,   -8.911417,  -21.403151,  -33.903095,
        -46.403094,  -58.903094,  -71.403094,  -83.903094,  -96.403094,
       -108.903094, -121.403094, -133.903094, -146.403059, -158.897797,
       -170.619857, -171.894725, -171.937591, -171.938138, -171.938138])
 y: array([   6.606764,    3.211416,   -7.913287,  -20.402483,  -32.90241 ,
        -45.40241 ,  -57.90241 ,  -70.40241 ,  -82.90241 ,  -95.40241 ,
       -107.90241 , -120.40241 , -132.90241 , -145.402387, -157.898995,
       -169.896257, -171.721414, -171.9328  , -171.938133, -171.938138])
FAILED tests/optimization/test_mgcv_postprocessing_final_fit_parity.py::test_gam_fit3_postprocessing_final_fit_matches_mgcv[factor_smooth_sz] - AssertionError: 
Not equal to tolerance rtol=3e-05, atol=5e-08
factor_smooth_sz: Vp diagonal mismatch
Mismatched elements: 12 / 25 (48%)
Max absolute difference: 1.53553109e-05
Max relative difference: 0.78309314
 x: array([1.517753e-02, 3.853436e+00, 7.960504e-01, 5.796443e-01,
       1.709883e-01, 1.364706e+00, 1.286348e+00, 2.725075e-05,
       5.042315e-06, 3.513118e-06, 9.858485e-07, 5.415663e-02,...
 y: array([1.517756e-02, 3.853451e+00, 7.960533e-01, 5.796466e-01,
       1.709890e-01, 1.364712e+00, 1.286352e+00, 1.528289e-05,
       2.827849e-06, 1.970239e-06, 5.528867e-07, 5.414920e-02,...
FAILED tests/optimization/test_mgcv_postprocessing_final_fit_parity.py::test_gam_fit5_postprocessing_final_fit_matches_mgcv[gaulss_select_true_cr] - AssertionError: 
Not equal to tolerance rtol=5e-05, atol=5e-05
gaulss_select_true_cr: Vc diagonal mismatch
Mismatched elements: 1 / 7 (14.3%)
Max absolute difference: 0.00080939
Max relative difference: 0.01959151
 x: array([0.00344 , 0.042123, 0.022311, 0.020104, 0.022957, 0.023656,
       0.003684])
 y: array([0.00344 , 0.041313, 0.0223  , 0.020101, 0.022957, 0.023654,
       0.003684])
FAILED tests/optimization/test_mgcv_postprocessing_final_fit_parity.py::test_gam_fit5_outer_info_trace_exact_known_gap[gaulss_cr_outer_info_exact] - AssertionError: 
Not equal to tolerance rtol=0, atol=5e-06
gaulss_cr: outer_info score_hist mismatch
Mismatched elements: 4 / 5 (80%)
Max absolute difference: 7.65597667
Max relative difference: 0.04749888
 x: array([168.838223, 160.23195 , 160.004167, 159.998508, 159.998502])
 y: array([161.182246, 160.074256, 159.999268, 159.998502, 159.998502])
FAILED tests/optimization/test_mgcv_postprocessing_final_fit_parity.py::test_gam_fit5_outer_info_trace_exact_known_gap[gevlss_cr_outer_info_exact] - AssertionError: gevlss_cr: outer_info iter mismatch 6 != 5
assert 6 == 5
 +  where 6 = int(6)
 +  and   5 = int(5)
FAILED tests/optimization/test_mgcv_score_hist_trace_parity.py::test_endpoint_log_sp_seed_matrix[gaussian-ML-99-0.8] - NotImplementedError: ML/REML/LAML outer optimisation requires an exact upstream-mirrored derivative path; numerical fallback removed.
FAILED tests/optimization/test_mgcv_vcomp_parity.py::test_gam_vcomp_matches_mgcv_requested_surface[shashlss_ml_rescale_false] - AssertionError: 
Not equal to tolerance rtol=8e-05, atol=8e-05

Mismatched elements: 1 / 3 (33.3%)
Max absolute difference: 7.23042925e+59
Max relative difference: 0.00039808
 x: array([[3.626459e-04, 7.146921e-71, 1.840122e+63]])
 y: array([[3.626469e-04, 7.149767e-71, 1.839399e+63]])
============================================== 31 failed, 