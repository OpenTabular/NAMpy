====================================================================================================== FAILURES ======================================================================================================
_______________________________________________________________________________ test_requested_mgcv_parity_20_models[binomial_probit] ________________________________________________________________________________

case = CaseSpec(case_id='binomial_probit', formula='y ~ s(x, bs="tp", k=12)', family={'name': 'binomial', 'link': 'probit'}, ...y=<function _data_binomial_univariate at 0x7248bbdbf1a0>, select=False, weights_column=None, skip_coef_comparison=True)

    @pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
    def test_requested_mgcv_parity_20_models(case: CaseSpec):
        data = case.data_factory()
    
        actual = _fit_nampy_snapshot(case, data)
        expected = _run_mgcv_snapshot(
            data=data,
            formula=case.formula,
            family=case.family,
            method="REML",
            select=case.select,
            weights_column=case.weights_column,
        )
    
>       _assert_requested_parity(case, actual, expected)

tests/parity/test_mgcv_parity.py:349: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

case = CaseSpec(case_id='binomial_probit', formula='y ~ s(x, bs="tp", k=12)', family={'name': 'binomial', 'link': 'probit'}, ...y=<function _data_binomial_univariate at 0x7248bbdbf1a0>, select=False, weights_column=None, skip_coef_comparison=True)
actual_snapshot = {'fit': {'coef_full': [0.039634968911913544, -0.0227561710266948, -0.012210272681300963, -0.0007746618194689261, -0.00...9, 0.22917800523100917, 0.11445928472591407, 0.11155594072683804, 0.09332270786033822, 0.10917880282369628, ...], ...}}
expected_snapshot = {'fit': {'coef_full': [0.03963503474094759, -0.02276416047126724, 0.012215725153550435, -0.0007754207881552114, -0.002...8, 0.22918917603307754, 0.11446023091287812, 0.11155697345362998, 0.09332461414938464, 0.10917968582475679, ...], ...}}

    def _assert_requested_parity(
        case: CaseSpec,
        actual_snapshot: dict,
        expected_snapshot: dict,
    ) -> None:
        if case.skip_coef_comparison:
            link_actual = np.asarray(actual_snapshot["predictions"]["link"], dtype=np.float64)
            link_expected = np.asarray(
                expected_snapshot["predictions"]["link"], dtype=np.float64
            )
            link_tol = 1e-4 * (1.0 + np.abs(link_actual))
            link_err = np.abs(link_actual - link_expected)
            assert np.all(link_err <= link_tol), (
                f"{case.case_id}: |link-link_mgcv| exceeded tolerance; "
                f"max_err={link_err.max():.3e}, max_tol={link_tol.max():.3e}"
            )
        else:
            beta = np.asarray(actual_snapshot["fit"]["coef_full"], dtype=np.float64)
            beta_mgcv = np.asarray(expected_snapshot["fit"]["coef_full"], dtype=np.float64)
            assert beta.shape == beta_mgcv.shape, f"{case.case_id}: beta shape mismatch"
            beta_tol = 1e-6 * (1.0 + np.abs(beta))
            beta_err = np.abs(beta - beta_mgcv)
            assert np.all(beta_err <= beta_tol), (
                f"{case.case_id}: |beta-beta_mgcv| exceeded tolerance; max_err={beta_err.max():.3e}, "
                f"max_tol={beta_tol.max():.3e}"
            )
    
        edf = float(actual_snapshot["fit"]["edf_total"])
        edf_mgcv = float(expected_snapshot["fit"]["edf_total"])
>       assert abs(edf - edf_mgcv) < 1e-4, (
            f"{case.case_id}: |edf-edf_mgcv|={abs(edf - edf_mgcv):.3e} >= 1e-4"
        )
E       AssertionError: binomial_probit: |edf-edf_mgcv|=1.051e-04 >= 1e-4
E       assert 0.00010514303693076599 < 0.0001
E        +  where 0.00010514303693076599 = abs((2.3423768071613305 - 2.342481950198261))

tests/parity/test_mgcv_parity.py:312: AssertionError
________________________________________________________________________________ test_requested_mgcv_parity_20_models[gaussian_ti_mc] ________________________________________________________________________________

case = CaseSpec(case_id='gaussian_ti_mc', formula='y ~ ti(x1, x2, bs=["cr", "cr"], k=[8, 8], mc=c(True, True))', family='gaus...tory=<function _data_gaussian_tensor at 0x7248bbdbf380>, select=False, weights_column=None, skip_coef_comparison=False)

    @pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
    def test_requested_mgcv_parity_20_models(case: CaseSpec):
        data = case.data_factory()
    
        actual = _fit_nampy_snapshot(case, data)
        expected = _run_mgcv_snapshot(
            data=data,
            formula=case.formula,
            family=case.family,
            method="REML",
            select=case.select,
            weights_column=case.weights_column,
        )
    
>       _assert_requested_parity(case, actual, expected)

tests/parity/test_mgcv_parity.py:349: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

case = CaseSpec(case_id='gaussian_ti_mc', formula='y ~ ti(x1, x2, bs=["cr", "cr"], k=[8, 8], mc=c(True, True))', family='gaus...tory=<function _data_gaussian_tensor at 0x7248bbdbf380>, select=False, weights_column=None, skip_coef_comparison=False)
actual_snapshot = {'fit': {'coef_full': [0.11204633430178246, -0.010573831163477234, -0.002725675745937753, 0.005607444994918153, 0.0108...4, 0.05626071873314734, 0.08063638003726369, 0.05104883779478717, 0.05286116608317879, 0.06659453455303244, ...], ...}}
expected_snapshot = {'fit': {'coef_full': [0.11204511743347498, -0.0105741120248448, -0.0027255446883753323, 0.005607899751622138, 0.01087...3, 0.056266416583180406, 0.08064329228107214, 0.05109018577027405, 0.0528651800760871, 0.06659577248639019, ...], ...}}

    def _assert_requested_parity(
        case: CaseSpec,
        actual_snapshot: dict,
        expected_snapshot: dict,
    ) -> None:
        if case.skip_coef_comparison:
            link_actual = np.asarray(actual_snapshot["predictions"]["link"], dtype=np.float64)
            link_expected = np.asarray(
                expected_snapshot["predictions"]["link"], dtype=np.float64
            )
            link_tol = 1e-4 * (1.0 + np.abs(link_actual))
            link_err = np.abs(link_actual - link_expected)
            assert np.all(link_err <= link_tol), (
                f"{case.case_id}: |link-link_mgcv| exceeded tolerance; "
                f"max_err={link_err.max():.3e}, max_tol={link_tol.max():.3e}"
            )
        else:
            beta = np.asarray(actual_snapshot["fit"]["coef_full"], dtype=np.float64)
            beta_mgcv = np.asarray(expected_snapshot["fit"]["coef_full"], dtype=np.float64)
            assert beta.shape == beta_mgcv.shape, f"{case.case_id}: beta shape mismatch"
            beta_tol = 1e-6 * (1.0 + np.abs(beta))
            beta_err = np.abs(beta - beta_mgcv)
>           assert np.all(beta_err <= beta_tol), (
                f"{case.case_id}: |beta-beta_mgcv| exceeded tolerance; max_err={beta_err.max():.3e}, "
                f"max_tol={beta_tol.max():.3e}"
            )
E           AssertionError: gaussian_ti_mc: |beta-beta_mgcv| exceeded tolerance; max_err=5.349e-05, max_tol=1.112e-06
E           assert False
E            +  where False = <function all at 0x724a1857adb0>(array([1.21686831e-06, 2.80861368e-07, 1.31057562e-07, 4.54756704e-07,\n       5.49287216e-07, 7.41575184e-07, 9.55683845e-07, 8.52664293e-07,\n       8.20719526e-06, 2.13407349e-06, 4.36347878e-06, 8.47769398e-06,\n       1.71721857e-05, 2.49472688e-05, 2.68217622e-05, 1.62206976e-05,\n       4.12502588e-06, 8.68973258e-06, 1.67484381e-05, 3.39722131e-05,\n       4.95200242e-05, 5.34913320e-05, 1.48820658e-05, 3.54990665e-06,\n       8.36343632e-06, 1.57944261e-05, 3.15085178e-05, 4.55206732e-05,\n       4.84474020e-05, 8.19996632e-06, 1.82356336e-06, 4.78996671e-06,\n       8.82386906e-06, 1.73947297e-05, 2.51382238e-05, 2.68185040e-05,\n       1.54690563e-06, 8.31720265e-08, 1.57214893e-06, 2.32875527e-06,\n       3.78081909e-06, 4.96604058e-06, 4.42814666e-06, 1.40632275e-05,\n       3.82727820e-06, 7.11835640e-06, 1.41043171e-05, 2.91849658e-05,\n       4.28688507e-05, 4.68386676e-05]) <= array([1.11204633e-06, 1.01057383e-06, 1.00272568e-06, 1.00560744e-06,\n       1.01087128e-06, 1.02212885e-06, 1.03226805e-06, 1.03486154e-06,\n       1.00372138e-06, 1.00095925e-06, 1.00197357e-06, 1.00382614e-06,\n       1.00778811e-06, 1.01135648e-06, 1.01226918e-06, 1.00663334e-06,\n       1.00171000e-06, 1.00351760e-06, 1.00681981e-06, 1.01388216e-06,\n       1.02024291e-06, 1.02187000e-06, 1.01429666e-06, 1.00368543e-06,\n       1.00758155e-06, 1.01469871e-06, 1.02991993e-06, 1.04362900e-06,\n       1.04713571e-06, 1.02078775e-06, 1.00535865e-06, 1.01102389e-06,\n       1.02137245e-06, 1.04350452e-06, 1.06343788e-06, 1.06853660e-06,\n       1.02918368e-06, 1.00752285e-06, 1.01547644e-06, 1.03000463e-06,\n       1.06107555e-06, 1.08905974e-06, 1.09621770e-06, 1.03169508e-06,\n       1.00817017e-06, 1.01680836e-06, 1.03258675e-06, 1.06633141e-06,\n       1.09672375e-06, 1.10449770e-06]))
E            +    where <function all at 0x724a1857adb0> = np.all

tests/parity/test_mgcv_parity.py:305: AssertionError
____________________________________________________________________________ test_requested_mgcv_parity_20_models[gaussian_t2_full_false] ____________________________________________________________________________

case = CaseSpec(case_id='gaussian_t2_full_false', formula='y ~ t2(x1, x2, bs=["cr", "cr"], k=[8, 8], full=False)', family='ga...tory=<function _data_gaussian_tensor at 0x7248bbdbf380>, select=False, weights_column=None, skip_coef_comparison=False)

    @pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
    def test_requested_mgcv_parity_20_models(case: CaseSpec):
        data = case.data_factory()
    
        actual = _fit_nampy_snapshot(case, data)
        expected = _run_mgcv_snapshot(
            data=data,
            formula=case.formula,
            family=case.family,
            method="REML",
            select=case.select,
            weights_column=case.weights_column,
        )
    
>       _assert_requested_parity(case, actual, expected)

tests/parity/test_mgcv_parity.py:349: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

case = CaseSpec(case_id='gaussian_t2_full_false', formula='y ~ t2(x1, x2, bs=["cr", "cr"], k=[8, 8], full=False)', family='ga...tory=<function _data_gaussian_tensor at 0x7248bbdbf380>, select=False, weights_column=None, skip_coef_comparison=False)
actual_snapshot = {'fit': {'coef_full': [0.19259495605753163, -6.462290746725123e-06, -6.347527349839193e-06, 2.1862101726926838e-05, -5..., 0.06044401618884327, 0.051915960945396696, 0.0454156668114429, 0.03690165949078343, 0.039126178118077594, ...], ...}}
expected_snapshot = {'fit': {'coef_full': [0.11077300354984514, -6.3463087153115905e-06, 2.1874174380186215e-05, 5.406239060066891e-07, 8...., 0.06044412728379317, 0.05191600187517769, 0.045415741755165336, 0.03690172663790195, 0.03912618158797386, ...], ...}}

    def _assert_requested_parity(
        case: CaseSpec,
        actual_snapshot: dict,
        expected_snapshot: dict,
    ) -> None:
        if case.skip_coef_comparison:
            link_actual = np.asarray(actual_snapshot["predictions"]["link"], dtype=np.float64)
            link_expected = np.asarray(
                expected_snapshot["predictions"]["link"], dtype=np.float64
            )
            link_tol = 1e-4 * (1.0 + np.abs(link_actual))
            link_err = np.abs(link_actual - link_expected)
            assert np.all(link_err <= link_tol), (
                f"{case.case_id}: |link-link_mgcv| exceeded tolerance; "
                f"max_err={link_err.max():.3e}, max_tol={link_tol.max():.3e}"
            )
        else:
            beta = np.asarray(actual_snapshot["fit"]["coef_full"], dtype=np.float64)
            beta_mgcv = np.asarray(expected_snapshot["fit"]["coef_full"], dtype=np.float64)
            assert beta.shape == beta_mgcv.shape, f"{case.case_id}: beta shape mismatch"
            beta_tol = 1e-6 * (1.0 + np.abs(beta))
            beta_err = np.abs(beta - beta_mgcv)
>           assert np.all(beta_err <= beta_tol), (
                f"{case.case_id}: |beta-beta_mgcv| exceeded tolerance; max_err={beta_err.max():.3e}, "
                f"max_tol={beta_tol.max():.3e}"
            )
E           AssertionError: gaussian_t2_full_false: |beta-beta_mgcv| exceeded tolerance; max_err=1.281e+00, max_tol=1.641e-06
E           assert False
E            +  where False = <function all at 0x724a1857adb0>(array([8.18219525e-02, 1.15982031e-07, 2.82217017e-05, 2.13214778e-05,\n       8.33503115e-05, 8.20681438e-05, 1.34580054e-05, 2.28210049e-05,\n       1.63434334e-05, 1.91206830e-06, 7.77667192e-05, 1.18857501e-04,\n       1.58500224e-04, 1.87297715e-05, 2.44836130e-05, 8.06393212e-05,\n       1.44449182e-04, 1.30146705e-04, 4.78199609e-05, 2.79362724e-05,\n       1.02134476e-05, 4.76376880e-05, 5.71181424e-05, 2.58368337e-04,\n       3.62757380e-04, 4.95631220e-05, 1.09433721e-04, 4.92367199e-04,\n       1.83482572e-04, 4.37939630e-04, 2.25048716e-04, 1.82075682e-04,\n       1.87640296e-04, 1.19957805e-03, 1.13894584e-03, 1.41620395e-03,\n       4.66683514e-03, 1.53469028e-02, 1.65452003e-02, 1.06205516e-01,\n       8.47229255e-02, 1.21684814e-01, 8.93869090e-02, 3.17568687e-02,\n       3.93451893e-03, 9.22256862e-03, 1.85629192e-02, 9.09679931e-03,\n       9.46124784e-02, 1.88925666e-01, 1.36692046e-02, 1.18098235e-01,\n       4.55532342e-03, 3.73847090e-02, 1.05422688e-03, 4.22527490e-02,\n       2.68431070e-01, 2.85847934e-01, 5.65842215e-03, 2.23280874e-03,\n       1.07094118e-01, 1.33771662e-02, 1.28129904e+00, 5.64537209e-04]) <= array([1.19259496e-06, 1.00000646e-06, 1.00000635e-06, 1.00002186e-06,\n       1.00000055e-06, 1.00008280e-06, 1.00000068e-06, 1.00001414e-06,\n       1.00000866e-06, 1.00002502e-06, 1.00002694e-06, 1.00005087e-06,\n       1.00016963e-06, 1.00001114e-06, 1.00000759e-06, 1.00001691e-06,\n       1.00006377e-06, 1.00008068e-06, 1.00004938e-06, 1.00000155e-06,\n       1.00002638e-06, 1.00001618e-06, 1.00003143e-06, 1.00008860e-06,\n       1.00034688e-06, 1.00001584e-06, 1.00003375e-06, 1.00007583e-06,\n       1.00041643e-06, 1.00023303e-06, 1.00020510e-06, 1.00001998e-06,\n       1.00020211e-06, 1.00001427e-06, 1.00118455e-06, 1.00004540e-06,\n       1.00146286e-06, 1.00612966e-06, 1.00921724e-06, 1.02576226e-06,\n       1.08044283e-06, 1.00427993e-06, 1.12596187e-06, 1.03657498e-06,\n       1.00481821e-06, 1.00088349e-06, 1.00833911e-06, 1.01022372e-06,\n       1.00112713e-06, 1.09348534e-06, 1.09544051e-06, 1.10910981e-06,\n       1.00898860e-06, 1.01354380e-06, 1.02384077e-06, 1.02278681e-06,\n       1.01946610e-06, 1.28789723e-06, 1.00204921e-06, 1.00361191e-06,\n       1.00584531e-06, 1.00636917e-06, 1.64095753e-06, 1.02672382e-06]))
E            +    where <function all at 0x724a1857adb0> = np.all

tests/parity/test_mgcv_parity.py:305: AssertionError
_____________________________________________________________________________ test_requested_mgcv_parity_20_models[binomial_separation] ______________________________________________________________________________

case = CaseSpec(case_id='binomial_separation', formula='y ~ s(x, bs="tp", k=12)', family='binomial', data_factory=<function _data_binomial_separation at 0x7248bbdbf600>, select=False, weights_column=None, skip_coef_comparison=False)

    @pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
    def test_requested_mgcv_parity_20_models(case: CaseSpec):
        data = case.data_factory()
    
        actual = _fit_nampy_snapshot(case, data)
>       expected = _run_mgcv_snapshot(
            data=data,
            formula=case.formula,
            family=case.family,
            method="REML",
            select=case.select,
            weights_column=case.weights_column,
        )

tests/parity/test_mgcv_parity.py:340: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/mgcv_parity_utils.py:426: in _run_mgcv_snapshot
    subprocess.run(
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

input = None, capture_output = True, timeout = None, check = True
popenargs = (['/usr/bin/Rscript', '/home/ad32/projects/package/NAMpy/tests/parity/mgcv_snapshot.R', '/tmp/tmproge3yg1/data.csv', '/tmp/tmproge3yg1/snapshot.json', 'y ~ s(x, bs="tp", k=12)', 'binomial', ...],)
kwargs = {'cwd': PosixPath('/home/ad32/projects/package/NAMpy'), 'stderr': -1, 'stdout': -1, 'text': True}, process = <Popen: returncode: 1 args: ['/usr/bin/Rscript', '/home/ad32/projects/packag...>, stdout = ''
stderr = 'Error in gam.reparam(UrS, sp, grderiv) : \n  NA/NaN/Inf in foreign function call (arg 3)\nCalls: do.call ... estimate.gam -> gam.outer -> newton -> gam.fit3 -> gam.reparam\nExecution halted\n'
retcode = 1

    def run(*popenargs,
            input=None, capture_output=False, timeout=None, check=False, **kwargs):
        """Run command with arguments and return a CompletedProcess instance.
    
        The returned instance will have attributes args, returncode, stdout and
        stderr. By default, stdout and stderr are not captured, and those attributes
        will be None. Pass stdout=PIPE and/or stderr=PIPE in order to capture them,
        or pass capture_output=True to capture both.
    
        If check is True and the exit code was non-zero, it raises a
        CalledProcessError. The CalledProcessError object will have the return code
        in the returncode attribute, and output & stderr attributes if those streams
        were captured.
    
        If timeout is given, and the process takes too long, a TimeoutExpired
        exception will be raised.
    
        There is an optional argument "input", allowing you to
        pass bytes or a string to the subprocess's stdin.  If you use this argument
        you may not also use the Popen constructor's "stdin" argument, as
        it will be used internally.
    
        By default, all communication is in bytes, and therefore any "input" should
        be bytes, and the stdout and stderr will be bytes. If in text mode, any
        "input" should be a string, and stdout and stderr will be strings decoded
        according to locale encoding, or by "encoding" if set. Text mode is
        triggered by setting any of text, encoding, errors or universal_newlines.
    
        The other arguments are the same as for the Popen constructor.
        """
        if input is not None:
            if kwargs.get('stdin') is not None:
                raise ValueError('stdin and input arguments may not both be used.')
            kwargs['stdin'] = PIPE
    
        if capture_output:
            if kwargs.get('stdout') is not None or kwargs.get('stderr') is not None:
                raise ValueError('stdout and stderr arguments may not be used '
                                 'with capture_output.')
            kwargs['stdout'] = PIPE
            kwargs['stderr'] = PIPE
    
        with Popen(*popenargs, **kwargs) as process:
            try:
                stdout, stderr = process.communicate(input, timeout=timeout)
            except TimeoutExpired as exc:
                process.kill()
                if _mswindows:
                    # Windows accumulates the output in a single blocking
                    # read() call run on child threads, with the timeout
                    # being done in a join() on those threads.  communicate()
                    # _after_ kill() is required to collect that and add it
                    # to the exception.
                    exc.stdout, exc.stderr = process.communicate()
                else:
                    # POSIX _communicate already populated the output so
                    # far into the TimeoutExpired exception.
                    process.wait()
                raise
            except:  # Including KeyboardInterrupt, communicate handled that.
                process.kill()
                # We don't call process.wait() as .__exit__ does that for us.
                raise
            retcode = process.poll()
            if check and retcode:
>               raise CalledProcessError(retcode, process.args,
                                         output=stdout, stderr=stderr)
E               subprocess.CalledProcessError: Command '['/usr/bin/Rscript', '/home/ad32/projects/package/NAMpy/tests/parity/mgcv_snapshot.R', '/tmp/tmproge3yg1/data.csv', '/tmp/tmproge3yg1/snapshot.json', 'y ~ s(x, bs="tp", k=12)', 'binomial', 'REML', 'false']' returned non-zero exit status 1.

../../../miniconda3/envs/nampy/lib/python3.11/subprocess.py:571: CalledProcessError
_________________________________________________________________________________ test_requested_mgcv_parity_20_models[mrf_lattice] __________________________________________________________________________________

case = CaseSpec(case_id='mrf_lattice', formula='y ~ s(region, bs="mrf", k=3, xt=list(nb=list(A=c("B"), B=c("A","C"), C=c("B")..._factory=<function _data_mrf_lattice at 0x7248bbdbf7e0>, select=False, weights_column=None, skip_coef_comparison=False)

    @pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
    def test_requested_mgcv_parity_20_models(case: CaseSpec):
        data = case.data_factory()
    
        actual = _fit_nampy_snapshot(case, data)
        expected = _run_mgcv_snapshot(
            data=data,
            formula=case.formula,
            family=case.family,
            method="REML",
            select=case.select,
            weights_column=case.weights_column,
        )
    
>       _assert_requested_parity(case, actual, expected)

tests/parity/test_mgcv_parity.py:349: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

case = CaseSpec(case_id='mrf_lattice', formula='y ~ s(region, bs="mrf", k=3, xt=list(nb=list(A=c("B"), B=c("A","C"), C=c("B")..._factory=<function _data_mrf_lattice at 0x7248bbdbf7e0>, select=False, weights_column=None, skip_coef_comparison=False)
actual_snapshot = {'fit': {'coef_full': [0.26249999999999996, -1.0501957591237951, -0.15429717274919688], 'cov_bayes': [[1.7718555488362...5812624304e-17, 8.418682910850739e-17, 6.873825812624305e-17, 6.873825812624304e-17, 8.418682910850739e-17, ...], ...}}
expected_snapshot = {'fit': {'coef_full': [0.2625, -1.0501957591237954, -0.15429717274919674], 'cov_bayes': [[4.314083075427407e-33, -1.28...758063e-16, 1.3136335981433191e-16, 1.0725773414758065e-16, 1.0725773414758063e-16, 1.3136335981433191e-16, ...], ...}}

    def _assert_requested_parity(
        case: CaseSpec,
        actual_snapshot: dict,
        expected_snapshot: dict,
    ) -> None:
        if case.skip_coef_comparison:
            link_actual = np.asarray(actual_snapshot["predictions"]["link"], dtype=np.float64)
            link_expected = np.asarray(
                expected_snapshot["predictions"]["link"], dtype=np.float64
            )
            link_tol = 1e-4 * (1.0 + np.abs(link_actual))
            link_err = np.abs(link_actual - link_expected)
            assert np.all(link_err <= link_tol), (
                f"{case.case_id}: |link-link_mgcv| exceeded tolerance; "
                f"max_err={link_err.max():.3e}, max_tol={link_tol.max():.3e}"
            )
        else:
            beta = np.asarray(actual_snapshot["fit"]["coef_full"], dtype=np.float64)
            beta_mgcv = np.asarray(expected_snapshot["fit"]["coef_full"], dtype=np.float64)
            assert beta.shape == beta_mgcv.shape, f"{case.case_id}: beta shape mismatch"
            beta_tol = 1e-6 * (1.0 + np.abs(beta))
            beta_err = np.abs(beta - beta_mgcv)
            assert np.all(beta_err <= beta_tol), (
                f"{case.case_id}: |beta-beta_mgcv| exceeded tolerance; max_err={beta_err.max():.3e}, "
                f"max_tol={beta_tol.max():.3e}"
            )
    
        edf = float(actual_snapshot["fit"]["edf_total"])
        edf_mgcv = float(expected_snapshot["fit"]["edf_total"])
        assert abs(edf - edf_mgcv) < 1e-4, (
            f"{case.case_id}: |edf-edf_mgcv|={abs(edf - edf_mgcv):.3e} >= 1e-4"
        )
    
        reml = float(actual_snapshot["fit"]["criterion_value"])
        reml_mgcv = float(expected_snapshot["fit"]["criterion_value"])
>       assert (
            abs(reml - reml_mgcv) < 1e-4
        ), f"{case.case_id}: |REML-REML_mgcv|={abs(reml - reml_mgcv):.3e} >= 1e-4"
E       AssertionError: mrf_lattice: |REML-REML_mgcv|=3.494e-01 >= 1e-4
E       assert 0.349404855932562 < 0.0001
E        +  where 0.349404855932562 = abs((-171.58873305950243 - -171.938137915435))

tests/parity/test_mgcv_parity.py:318: AssertionError
_______________________________________________________________________________ test_requested_mgcv_parity_20_models[factor_smooth_sz] _______________________________________________________________________________

case = CaseSpec(case_id='factor_smooth_sz', formula='y ~ s(f1, f2, x, bs="sz", k=6)', family='gaussian', data_factory=<function _data_sz_interaction at 0x7248bbdbf880>, select=False, weights_column=None, skip_coef_comparison=False)

    @pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
    def test_requested_mgcv_parity_20_models(case: CaseSpec):
        data = case.data_factory()
    
        actual = _fit_nampy_snapshot(case, data)
        expected = _run_mgcv_snapshot(
            data=data,
            formula=case.formula,
            family=case.family,
            method="REML",
            select=case.select,
            weights_column=case.weights_column,
        )
    
>       _assert_requested_parity(case, actual, expected)

tests/parity/test_mgcv_parity.py:349: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

case = CaseSpec(case_id='factor_smooth_sz', formula='y ~ s(f1, f2, x, bs="sz", k=6)', family='gaussian', data_factory=<function _data_sz_interaction at 0x7248bbdbf880>, select=False, weights_column=None, skip_coef_comparison=False)
actual_snapshot = {'fit': {'coef_full': [1.0117168032495676, 1.2419734134444396, 0.1212394126614088, -0.18018058185676422, 0.03523613801...38, 0.32402210949435734, 0.25392455712046275, 0.40187680639524415, 0.3913022690160016, 0.34925759706980525, ...], ...}}
expected_snapshot = {'fit': {'coef_full': [1.0117173210547805, 1.2419614162177588, 0.12123916835135488, -0.18018447970175494, -0.035235630...28016, 0.3240221540540324, 0.2539247395656044, 0.4018768780345336, 0.3913025340485307, 0.34925765407527515, ...], ...}}

    def _assert_requested_parity(
        case: CaseSpec,
        actual_snapshot: dict,
        expected_snapshot: dict,
    ) -> None:
        if case.skip_coef_comparison:
            link_actual = np.asarray(actual_snapshot["predictions"]["link"], dtype=np.float64)
            link_expected = np.asarray(
                expected_snapshot["predictions"]["link"], dtype=np.float64
            )
            link_tol = 1e-4 * (1.0 + np.abs(link_actual))
            link_err = np.abs(link_actual - link_expected)
            assert np.all(link_err <= link_tol), (
                f"{case.case_id}: |link-link_mgcv| exceeded tolerance; "
                f"max_err={link_err.max():.3e}, max_tol={link_tol.max():.3e}"
            )
        else:
            beta = np.asarray(actual_snapshot["fit"]["coef_full"], dtype=np.float64)
            beta_mgcv = np.asarray(expected_snapshot["fit"]["coef_full"], dtype=np.float64)
            assert beta.shape == beta_mgcv.shape, f"{case.case_id}: beta shape mismatch"
            beta_tol = 1e-6 * (1.0 + np.abs(beta))
            beta_err = np.abs(beta - beta_mgcv)
>           assert np.all(beta_err <= beta_tol), (
                f"{case.case_id}: |beta-beta_mgcv| exceeded tolerance; max_err={beta_err.max():.3e}, "
                f"max_tol={beta_tol.max():.3e}"
            )
E           AssertionError: factor_smooth_sz: |beta-beta_mgcv| exceeded tolerance; max_err=7.047e-02, max_tol=2.960e-06
E           assert False
E            +  where False = <function all at 0x724a1857adb0>(array([5.17805213e-07, 1.19972267e-05, 2.44310054e-07, 3.89784499e-06,\n       7.04717686e-02, 7.99147486e-06, 1.33013118e-07, 1.17559329e-05,\n       5.27154015e-09, 3.46260993e-07, 1.15469006e-07, 1.17772716e-05,\n       9.13046343e-07, 1.27058399e-06, 1.58531272e-06, 1.32901104e-07,\n       8.37963660e-08, 1.60143467e-06, 1.58854360e-06, 2.34354919e-06,\n       1.82446367e-06, 3.07792885e-07, 3.33288780e-02, 3.90203640e-06,\n       2.69270587e-06]) <= array([2.01171680e-06, 2.24197341e-06, 1.12123941e-06, 1.18018058e-06,\n       1.03523614e-06, 2.10779523e-06, 1.19522298e-06, 1.00000870e-06,\n       1.00000200e-06, 1.00000110e-06, 1.00000003e-06, 1.40887817e-06,\n       1.19888682e-06, 1.00000000e-06, 1.00000000e-06, 1.00000000e-06,\n       1.00000000e-06, 1.69684283e-06, 1.50394778e-06, 2.74176330e-06,\n       1.25472146e-06, 1.13896300e-06, 1.01666425e-06, 2.95959049e-06,\n       1.52166701e-06]))
E            +    where <function all at 0x724a1857adb0> = np.all

tests/parity/test_mgcv_parity.py:305: AssertionError
___________________________________________________________________________ test_requested_mgcv_parity_20_models[negbin_theta_estimation] ____________________________________________________________________________

case = CaseSpec(case_id='negbin_theta_estimation', formula='y ~ s(x, bs="tp", k=12)', family={'name': 'negbin', 'theta': 1.0,...nction _data_negbin_theta_estimation at 0x7248bbdbf920>, select=False, weights_column=None, skip_coef_comparison=False)

    @pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
    def test_requested_mgcv_parity_20_models(case: CaseSpec):
        data = case.data_factory()
    
        actual = _fit_nampy_snapshot(case, data)
        expected = _run_mgcv_snapshot(
            data=data,
            formula=case.formula,
            family=case.family,
            method="REML",
            select=case.select,
            weights_column=case.weights_column,
        )
    
>       _assert_requested_parity(case, actual, expected)

tests/parity/test_mgcv_parity.py:349: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

case = CaseSpec(case_id='negbin_theta_estimation', formula='y ~ s(x, bs="tp", k=12)', family={'name': 'negbin', 'theta': 1.0,...nction _data_negbin_theta_estimation at 0x7248bbdbf920>, select=False, weights_column=None, skip_coef_comparison=False)
actual_snapshot = {'fit': {'coef_full': [0.34516755813168776, -1.612926167586921, 0.20747156879400594, -0.10167290173977504, 0.102077536...0855218, 0.11921133654994202, 0.1285948701156659, 0.3166192509583321, 0.13322097033117086, 0.3830364618177, ...], ...}}
expected_snapshot = {'fit': {'coef_full': [0.34516903079711425, -1.6128912855387976, 0.20745862442403146, -0.10166689140486028, 0.10207230...28, 0.11920796926014407, 0.1285910037777453, 0.31661056729242765, 0.13321699793716074, 0.38302380844243167, ...], ...}}

    def _assert_requested_parity(
        case: CaseSpec,
        actual_snapshot: dict,
        expected_snapshot: dict,
    ) -> None:
        if case.skip_coef_comparison:
            link_actual = np.asarray(actual_snapshot["predictions"]["link"], dtype=np.float64)
            link_expected = np.asarray(
                expected_snapshot["predictions"]["link"], dtype=np.float64
            )
            link_tol = 1e-4 * (1.0 + np.abs(link_actual))
            link_err = np.abs(link_actual - link_expected)
            assert np.all(link_err <= link_tol), (
                f"{case.case_id}: |link-link_mgcv| exceeded tolerance; "
                f"max_err={link_err.max():.3e}, max_tol={link_tol.max():.3e}"
            )
        else:
            beta = np.asarray(actual_snapshot["fit"]["coef_full"], dtype=np.float64)
            beta_mgcv = np.asarray(expected_snapshot["fit"]["coef_full"], dtype=np.float64)
            assert beta.shape == beta_mgcv.shape, f"{case.case_id}: beta shape mismatch"
            beta_tol = 1e-6 * (1.0 + np.abs(beta))
            beta_err = np.abs(beta - beta_mgcv)
>           assert np.all(beta_err <= beta_tol), (
                f"{case.case_id}: |beta-beta_mgcv| exceeded tolerance; max_err={beta_err.max():.3e}, "
                f"max_tol={beta_tol.max():.3e}"
            )
E           AssertionError: negbin_theta_estimation: |beta-beta_mgcv| exceeded tolerance; max_err=4.117e-05, max_tol=2.613e-06
E           assert False
E            +  where False = <function all at 0x724a1857adb0>(array([1.47266543e-06, 3.48820481e-05, 1.29443700e-05, 6.01033491e-06,\n       5.22930521e-06, 7.08641348e-06, 1.28777067e-06, 5.25375543e-06,\n       2.82693579e-06, 3.22032597e-06, 4.11745255e-05, 3.37722951e-05]) <= array([1.34516756e-06, 2.61292617e-06, 1.20747157e-06, 1.10167290e-06,\n       1.10207754e-06, 1.12948790e-06, 1.09557115e-06, 1.11917306e-06,\n       1.07621568e-06, 1.09997699e-06, 1.80592917e-06, 2.18321615e-06]))
E            +    where <function all at 0x724a1857adb0> = np.all

tests/parity/test_mgcv_parity.py:305: AssertionError
_______________________________________________________________________ TestAdditionalScenarioParity.test_gaussian_fs_select_reml_matches_mgcv _______________________________________________________________________

self = <test_mgcv_additional_scenarios.TestAdditionalScenarioParity object at 0x7248bbccc510>

    def test_gaussian_fs_select_reml_matches_mgcv(self):
        data = _make_fs_data()
        formula = 'y ~ s(f, x, bs="fs", k=6)'
    
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML", select=True)
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML", select=True)
    
>       _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=1e-6,
            pred_rtol=1e-6,
            sp_log_atol=2.0,
            criterion_atol=1e-3,
        )

tests/_mgcv_snapshot_parity_shared.py:2835: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/mgcv_parity_utils.py:1063: in _assert_basic_mgcv_parity
    np.testing.assert_allclose(
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

args = (<function assert_allclose.<locals>.compare at 0x724a190a0f40>, array([-14.15211412, -20.41112285, -15.26347055]), array([-18.1929885 , -24.45200616, -19.30433512]))
kwds = {'equal_nan': True, 'err_msg': '', 'header': 'Not equal to tolerance rtol=0, atol=2', 'verbose': True}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
                   ^^^^^^^^^^^^^^^^^^^
E           AssertionError: 
E           Not equal to tolerance rtol=0, atol=2
E           
E           Mismatched elements: 3 / 3 (100%)
E           Max absolute difference: 4.04088332
E           Max relative difference: 0.22211163
E            x: array([-14.152114, -20.411123, -15.263471])
E            y: array([-18.192989, -24.452006, -19.304335])

../../../miniconda3/envs/nampy/lib/python3.11/contextlib.py:81: AssertionError
____________________________________________________________________ TestAdditionalScenarioParity.test_gaussian_fs_ps_marginal_reml_matches_mgcv _____________________________________________________________________

self = <test_mgcv_additional_scenarios.TestAdditionalScenarioParity object at 0x7248bbcce810>

    def test_gaussian_fs_ps_marginal_reml_matches_mgcv(self):
        data = _make_fs_data()
        formula = 'y ~ s(f, x, bs="fs", xt=list(bs="ps", m=2, k=7))'
    
        actual = _fit_nampy_snapshot(data, formula, "gaussian", "REML")
        expected = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
    
        # Both NAMpy and mgcv converge to near-zero smoothing on a flat
        # landscape; sp values differ substantially in log-space but both
        # represent effectively-unpenalized fits.  Check predictions only.
>       _assert_basic_mgcv_parity(
            actual,
            expected,
            pred_atol=5e-3,
            pred_rtol=0.0,
            sp_log_atol=5.0,
            criterion_atol=2.0,
        )

tests/_mgcv_snapshot_parity_shared.py:3632: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/mgcv_parity_utils.py:1063: in _assert_basic_mgcv_parity
    np.testing.assert_allclose(
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

args = (<function assert_allclose.<locals>.compare at 0x7248ed0dfa60>, array([-12.56681676, -18.9677729 , -18.02349855]), array([-17.83939528, -23.29615725, -24.24070382]))
kwds = {'equal_nan': True, 'err_msg': '', 'header': 'Not equal to tolerance rtol=0, atol=5', 'verbose': True}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
                   ^^^^^^^^^^^^^^^^^^^
E           AssertionError: 
E           Not equal to tolerance rtol=0, atol=5
E           
E           Mismatched elements: 2 / 3 (66.7%)
E           Max absolute difference: 6.21720528
E           Max relative difference: 0.29555814
E            x: array([-12.566817, -18.967773, -18.023499])
E            y: array([-17.839395, -23.296157, -24.240704])

../../../miniconda3/envs/nampy/lib/python3.11/contextlib.py:81: AssertionError
___________________________________________________________________________________ TestKCheckParity.test_fs_k_prime_and_edf_match ___________________________________________________________________________________

self = <test_mgcv_k_check_parity.TestKCheckParity object at 0x7248bbcf1850>

    def test_fs_k_prime_and_edf_match(self):
        """fs() smooth: k_prime and edf match mgcv.
    
        k_index is NaN in NAMpy because _numeric_feature_block does not yet extract
        the metric feature from an fs() RuntimeTerm (known limitation — the fs term
        stores both a factor and a numeric feature; the current introspection path
        only handles terms with _feature_index or _feature_indices).  mgcv's k.check
        does compute a finite k_index for fs().  This mismatch is documented in
        PARITY_SUMMARY.md section 9 (item "k_check fs/sz feature extraction").
    
        edf_atol=5e-5: EDF accumulates small differences across factor-level columns.
        """
        data = _make_fs_data()
        formula = 'y ~ s(f, x, bs="fs", k=6)'
    
        snap = _run_mgcv_snapshot(data, formula, "gaussian", "REML")
        model = _fit_nampy_model(data, formula, "gaussian", "REML")
    
        r_block = _r_k_check(snap)
        assert r_block is not None
        py_labels, py_values = _nampy_k_check(model)
        r_labels, r_values = r_block
        assert len(py_labels) == len(r_labels) == 1
    
        # k_prime: exact
        assert int(py_values[0, 0]) == int(
            round(r_values[0, 0])
        ), f"k_prime mismatch: NAMpy={int(py_values[0,0])} R={int(round(r_values[0,0]))}"
        # edf: tight
>       np.testing.assert_allclose(
            py_values[0, 1],
            r_values[0, 1],
            atol=5e-5,
            rtol=0.0,
            err_msg="edf mismatch for fs() term",
        )

tests/test_mgcv_k_check_parity.py:466: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

args = (<function assert_allclose.<locals>.compare at 0x7248b75f39c0>, array(16.99843423), array(16.99997246))
kwds = {'equal_nan': True, 'err_msg': 'edf mismatch for fs() term', 'header': 'Not equal to tolerance rtol=0, atol=5e-05', 'verbose': True}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
                   ^^^^^^^^^^^^^^^^^^^
E           AssertionError: 
E           Not equal to tolerance rtol=0, atol=5e-05
E           edf mismatch for fs() term
E           Mismatched elements: 1 / 1 (100%)
E           Max absolute difference: 0.00153823
E           Max relative difference: 9.04843008e-05
E            x: array(16.998434)
E            y: array(16.999972)

../../../miniconda3/envs/nampy/lib/python3.11/contextlib.py:81: AssertionError
____________________________________________________________________________________ test_output_parity_terms_standard_errors[fs] ____________________________________________________________________________________

case = {'case_id': 'fs', 'data_factory': <function _make_fs_data at 0x7248bbdbd6c0>, 'formula': 'y ~ s(f, x, bs="fs", k=6)', 'method': 'REML', ...}

    @pytest.mark.parametrize(
        "case",
        [case for case in TERMS_PARITY_CASES if "se_atol" in case],
        ids=[case["case_id"] for case in TERMS_PARITY_CASES if "se_atol" in case],
    )
    def test_output_parity_terms_standard_errors(case):
        train = case["data_factory"]()
        model = _fit_nampy_model(train, case["formula"], "gaussian", case["method"])
    
        actual_terms, actual_se = model.predict(X=train, type="terms", return_se=True)
        r_result = _run_mgcv_predict_on_newdata(
            train,
            train,
            case["formula"],
            family="gaussian",
            method=case["method"],
            type="terms",
            return_se=True,
        )
    
        expected_terms = np.asarray(r_result["pred"], dtype=np.float64)
        expected_se = np.asarray(r_result["se"], dtype=np.float64)
        actual_terms = np.asarray(actual_terms, dtype=np.float64)
        actual_se = np.asarray(actual_se, dtype=np.float64)
    
        assert (
            actual_terms.shape
            == expected_terms.shape
            == actual_se.shape
            == expected_se.shape
        )
        assert np.atleast_1d(r_result["term_names"]).size == 1
    
        np.testing.assert_allclose(
            actual_terms,
            expected_terms,
            atol=case["pred_atol"],
            rtol=case["pred_rtol"],
        )
>       np.testing.assert_allclose(
            actual_se,
            expected_se,
            atol=case["se_atol"],
            rtol=case["se_rtol"],
        )

tests/test_mgcv_output_parity.py:424: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

args = (<function assert_allclose.<locals>.compare at 0x7248b750e020>, array([[0.00540912],
       [0.00540912],
       [0.00...0.13671191],
       [0.13671192],
       [0.13671191],
       [0.13671192],
       [0.13671192],
       [0.13671192]]))
kwds = {'equal_nan': True, 'err_msg': '', 'header': 'Not equal to tolerance rtol=0.001, atol=0.0002', 'verbose': True}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
                   ^^^^^^^^^^^^^^^^^^^
E           AssertionError: 
E           Not equal to tolerance rtol=0.001, atol=0.0002
E           
E           Mismatched elements: 18 / 18 (100%)
E           Max absolute difference: 0.13130279
E           Max relative difference: 0.96043415
E            x: array([[0.005409],
E                  [0.005409],
E                  [0.005409],...
E            y: array([[0.136712],
E                  [0.136712],
E                  [0.136712],...

../../../miniconda3/envs/nampy/lib/python3.11/contextlib.py:81: AssertionError
================================================================================================== warnings summary ==================================================================================================
tests/parity/test_mgcv_parity.py::test_requested_mgcv_parity_20_models[gaussian_random_intercept_re]
tests/test_mgcv_additional_scenarios.py::TestAdditionalScenarioParity::test_gaussian_t2_ts_cr_reml_matches_mgcv
  /home/ad32/miniconda3/envs/nampy/lib/python3.11/site-packages/scipy/optimize/_optimize.py:2358: RuntimeWarning: invalid value encountered in scalar subtract
    p = (xf - fulc) * q - (xf - nfc) * r

tests/parity/test_mgcv_parity.py::test_requested_mgcv_parity_20_models[gaussian_random_intercept_re]
tests/test_mgcv_additional_scenarios.py::TestAdditionalScenarioParity::test_gaussian_t2_ts_cr_reml_matches_mgcv
  /home/ad32/miniconda3/envs/nampy/lib/python3.11/site-packages/scipy/optimize/_optimize.py:2359: RuntimeWarning: invalid value encountered in scalar subtract
    q = 2.0 * (q - r)

tests/test_mgcv_additional_scenarios.py::TestAdditionalScenarioParity::test_gaussian_fs_select_reml_matches_mgcv
tests/test_mgcv_k_check_parity.py::TestKCheckParity::test_fs_k_prime_and_edf_match
tests/test_mgcv_output_parity.py::test_output_parity_terms_all_smooth_types[fs]
tests/test_mgcv_output_parity.py::test_output_parity_terms_standard_errors[fs]
tests/test_mgcv_snapshot_parity.py::TestMgcvParity::test_gaussian_fs_reml_matches_mgcv
  /home/ad32/miniconda3/envs/nampy/lib/python3.11/site-packages/scipy/optimize/_numdiff.py:686: RuntimeWarning: invalid value encountered in subtract
    df = [f_eval - f0 for f_eval in f_evals]

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============================================================================================== short test summary info ===============================================================================================
FAILED tests/parity/test_mgcv_parity.py::test_requested_mgcv_parity_20_models[binomial_probit] - AssertionError: binomial_probit: |edf-edf_mgcv|=1.051e-04 >= 1e-4
FAILED tests/parity/test_mgcv_parity.py::test_requested_mgcv_parity_20_models[gaussian_ti_mc] - AssertionError: gaussian_ti_mc: |beta-beta_mgcv| exceeded tolerance; max_err=5.349e-05, max_tol=1.112e-06
FAILED tests/parity/test_mgcv_parity.py::test_requested_mgcv_parity_20_models[gaussian_t2_full_false] - AssertionError: gaussian_t2_full_false: |beta-beta_mgcv| exceeded tolerance; max_err=1.281e+00, max_tol=1.641e-06
FAILED tests/parity/test_mgcv_parity.py::test_requested_mgcv_parity_20_models[binomial_separation] - subprocess.CalledProcessError: Command '['/usr/bin/Rscript', '/home/ad32/projects/package/NAMpy/tests/parity/mgcv_snapshot.R', '/tmp/tmproge3yg1/data.csv', '/tmp/tmproge3yg1/snapshot.json', 'y ~ s(x, bs="tp", ...
FAILED tests/parity/test_mgcv_parity.py::test_requested_mgcv_parity_20_models[mrf_lattice] - AssertionError: mrf_lattice: |REML-REML_mgcv|=3.494e-01 >= 1e-4
FAILED tests/parity/test_mgcv_parity.py::test_requested_mgcv_parity_20_models[factor_smooth_sz] - AssertionError: factor_smooth_sz: |beta-beta_mgcv| exceeded tolerance; max_err=7.047e-02, max_tol=2.960e-06
FAILED tests/parity/test_mgcv_parity.py::test_requested_mgcv_parity_20_models[negbin_theta_estimation] - AssertionError: negbin_theta_estimation: |beta-beta_mgcv| exceeded tolerance; max_err=4.117e-05, max_tol=2.613e-06
FAILED tests/test_mgcv_additional_scenarios.py::TestAdditionalScenarioParity::test_gaussian_fs_select_reml_matches_mgcv - AssertionError: 
FAILED tests/test_mgcv_additional_scenarios.py::TestAdditionalScenarioParity::test_gaussian_fs_ps_marginal_reml_matches_mgcv - AssertionError: 
FAILED tests/test_mgcv_k_check_parity.py::TestKCheckParity::test_fs_k_prime_and_edf_match - AssertionError: 
FAILED tests/test_mgcv_output_parity.py::test_output_parity_terms_standard_errors[fs] - AssertionError: 
============================================================================== 11 failed, 241 passed, 9 warnings in 6106.05s (1:41:46)