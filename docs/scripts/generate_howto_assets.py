"""Run the how-to demonstrations and regenerate their fitted figures/tables.

The literalinclude markers expose the same code that produces the outputs.
Run from the repository root with the installed documentation environment.
"""
from pathlib import Path
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from generate_detailed_assets import csv_table, save

ROOT = Path(__file__).resolve().parents[2]


def training():
    # TRAINING START
    from pathlib import Path
    import tempfile
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch
    from sklearn.model_selection import train_test_split
    from lightning.pytorch.loggers import CSVLogger
    from nampy.models import NAMRegressor
    from nampy.neural.configs.nam_config import DefaultNAMConfig
    from nampy.neural.configuration import GradientTrainingConfig, PreprocessingConfig

    rng = np.random.default_rng(61)
    X = pd.DataFrame({"x": rng.uniform(-1, 1, 480),
                      "group": rng.choice(["A", "B"], 480)})
    y = np.sin(3 * X.x) + .5 * (X.group == "B") + rng.normal(0, .2, len(X))
    X_fit, X_test, y_fit, y_test = train_test_split(X, y, test_size=.2, random_state=61)
    X_train, X_val, y_train, y_val = train_test_split(
        X_fit, y_fit, test_size=.25, random_state=62
    )
    # TRAINING FIT
    with tempfile.TemporaryDirectory() as directory:
        logger = CSVLogger(directory, name="losses", version=0)
        model = NAMRegressor(
            architecture_config=DefaultNAMConfig(layer_sizes=[16, 8], dropout=0.),
            preprocessing=PreprocessingConfig(numerical_preprocessing="minmax"),
            training=GradientTrainingConfig(
                max_epochs=200, patience=12, lr=.01, random_state=61, batch_size=64,
                checkpoint_path=directory,
                trainer_kwargs={"logger": logger, "log_every_n_steps": 1,
                    "enable_progress_bar": False, "enable_model_summary": False},
            ),
        )
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        losses = pd.read_csv(Path(logger.log_dir) / "metrics.csv")
        checkpoint = torch.load(model.best_model_path_, map_location="cpu", weights_only=False)
        retained_epoch = int(checkpoint["epoch"])
    # TRAINING PLOTTING
    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for column, label in [("train_loss_epoch", "Training"), ("val_loss", "Validation")]:
        values = losses.dropna(subset=[column]).groupby("epoch")[column].last()
        axis.plot(values.index, values, label=label)
    axis.axvline(retained_epoch, color="black", linestyle="--", label="Retained checkpoint")
    axis.set(xlabel="Epoch (zero-based)", ylabel="Logged squared-error loss")
    axis.legend()
    # TRAINING END
    assert retained_epoch == int(losses.dropna(subset=['val_loss']).loc[lambda d: d.val_loss.idxmin(), 'epoch'])
    save(fig, 'howto_training')
    return locals()


def preprocessing():
    # PREPROCESSING START
    import tempfile
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from nampy.models import NAMRegressor
    from nampy.neural.configs.nam_config import DefaultNAMConfig
    from nampy.neural.configuration import GradientTrainingConfig, PreprocessingConfig

    X_train = pd.DataFrame({"age": [20., 30., 40., 50., 60., 70.],
                            "group": ["A", "B", "A", "B", "A", "B"]})
    y_train = np.array([1., 2., 1.8, 3., 2.5, 3.8])
    X_val = pd.DataFrame({"age": [35., 55.], "group": ["A", "B"]})
    y_val = np.array([1.6, 3.2])
    X_new = pd.DataFrame({"age": [45.], "group": ["B"]})
    with tempfile.TemporaryDirectory() as directory:
        model = NAMRegressor(
            architecture_config=DefaultNAMConfig(layer_sizes=[4]),
            preprocessing=PreprocessingConfig(numerical_preprocessing="minmax",
                categorical_preprocessing="one-hot", cat_cutoff=0.),
            training=GradientTrainingConfig(max_epochs=2, random_state=63,
                checkpoint_path=directory, trainer_kwargs={"logger": False,
                    "enable_progress_bar": False, "enable_model_summary": False}),
        )
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    # All six training rows fit PreTab; validation and new rows reuse its state.
    preprocessor = model.fitted_preprocessing_.preprocessor
    transformed = preprocessor.transform(pd.concat([X_train, X_new], ignore_index=True))
    blocks = {name: np.asarray(block) for name, block in transformed.items()}
    new_blocks = preprocessor.transform(X_new)
    for name, block in blocks.items():
        np.testing.assert_allclose(block[-1:], np.asarray(new_blocks[name]))
    matrix = np.column_stack(list(blocks.values()))
    columns = [f"{name}[{i}]" for name, block in blocks.items()
               for i in range(block.shape[1])]
    block_table = pd.DataFrame(matrix, columns=columns)
    block_table.insert(0, "Rows", ["Training input"] * len(X_train) + ["New row"])
    # PREPROCESSING PLOTTING
    fig, axis = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap="Blues")
    axis.set(xticks=np.arange(len(columns)), xticklabels=columns,
             yticks=np.arange(len(matrix)), yticklabels=[*map(str, range(1, 7)), "New"])
    boundary = 0
    for block in blocks.values():
        boundary += block.shape[1]
        axis.axvline(boundary - .5, color="white", linewidth=2)
    axis.axhline(len(X_train) - .5, color="black", linestyle="--")
    fig.colorbar(image, ax=axis, label="Transformed value")
    # PREPROCESSING END
    csv_table('howto_preprocessing_input', 'Mixed-type inputs (last row is new)',
              pd.concat([X_train, X_new], ignore_index=True))
    csv_table('howto_preprocessing_blocks', 'Reused feature blocks', block_table)
    save(fig, 'howto_preprocessing')


def interpretability():
    # INTERPRETATION START
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from nampy.models import GAMRegressor

    rng = np.random.default_rng(64)
    data = pd.DataFrame({"x": rng.uniform(-1, 1, 600), "z": rng.uniform(-1, 1, 600)})
    data["log_exposure"] = np.log(rng.uniform(.5, 2, len(data)))
    data["y"] = rng.poisson(np.exp(.3 + np.sin(3 * data.x) + .4 * data.z
                                    + .7 * data.x * data.z + data.log_exposure))
    model = GAMRegressor(family="poisson", formula=
        "y ~ s(x, bs='cr', k=8) + s(z, bs='cr', k=8) + ti(x,z,k=c(5,5)) + offset(log_exposure)"
    ).fit(data)
    row = data.iloc[[10]]
    parts = model.predict_components(row)
    parts.validate_additive_reconstruction()
    names = ["Intercept", *parts.terms, "Known offset"]
    values = [float(np.asarray(parts.intercept).reshape(-1)[0]),
              *[float(np.asarray(v).reshape(-1)[0]) for v in parts.terms.values()],
              float(np.asarray(parts.offset).reshape(-1)[0])]
    total = np.cumsum([0., *values])
    assert np.isclose(total[-1], np.asarray(parts.link).item())
    assert np.isclose(np.exp(total[-1]), model.predict(row).item())
    # INTERPRETATION PLOTTING
    fig, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for i, value in enumerate(values):
        axis.bar(i, abs(value), bottom=min(total[i], total[i+1]),
                 color="#2878ad" if value >= 0 else "#d97732")
        axis.plot([i-.4, i+.4], [total[i+1]]*2, color="black", linewidth=1)
    axis.bar(len(values), total[-1], color="#335c49")
    span = max(float(np.ptp(total)), .1)
    axis.set_ylim(float(total.min()) - .1*span, float(total.max()) + .1*span)
    axis.axhline(0, color="gray", linewidth=.7)
    axis.set(xticks=range(len(names)+1), xticklabels=names+["Total η"], ylabel="Log expected count")
    axis.tick_params(axis="x", labelrotation=25)
    axis.set_title(f"One row: η={total[-1]:.3f}; exp(η)={np.exp(total[-1]):.3f} expected counts")
    # INTERPRETATION END
    save(fig, 'howto_waterfall')
    # CENTERING START
    # Re-express the same fitted terms using two reference distributions.
    grid = pd.DataFrame({"x": np.linspace(-1, 1, 160), "z": 0., "log_exposure": 0.})
    raw = model.predict_components(grid)
    references = {"All training rows": data, "Training rows with x > 0": data[data.x > 0]}
    feature = next(name for name in raw.terms if name.startswith("s(x"))
    representations = []
    for label, reference in references.items():
        reference_parts = model.predict_components(reference)
        shifts = {name: np.asarray(value).mean(axis=0)
                  for name, value in reference_parts.terms.items()}
        centered = {name: np.asarray(value) - shifts[name] for name, value in raw.terms.items()}
        intercept = np.asarray(raw.intercept) + sum(shifts.values())
        reconstructed = intercept + sum(centered.values()) + np.asarray(raw.offset)
        np.testing.assert_allclose(reconstructed, raw.link, atol=1e-10)
        representations.append((label, float(intercept), centered[feature], np.exp(reconstructed)))
    # CENTERING PLOTTING
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for style, (label, intercept, effect, prediction) in zip(["-", "--"], representations):
        axes[0].plot(grid.x, effect, style, label=f"{label}; intercept {intercept:.2f}")
        axes[1].plot(grid.x, prediction, style, label=label)
    axes[0].set(xlabel="x", ylabel="Centered log-count contribution")
    axes[1].set(xlabel="x (z=0, exposure=1)", ylabel="Expected count")
    for axis in axes: axis.legend(fontsize=8)
    # CENTERING END
    save(fig, 'howto_centering')


def interaction():
    # INTERACTION START
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from nampy.gam import GAM

    rng = np.random.default_rng(65)
    data = pd.DataFrame(rng.uniform(-1, 1, (600, 2)), columns=["x", "z"])
    data["y"] = np.sin(3*data.x) + .5*data.z + 2*data.x*data.z + rng.normal(0, .2, len(data))
    model = GAM(formula="y ~ s(x,bs='cr',k=8) + s(z,bs='cr',k=8) + ti(x,z,k=c(6,6))",
                optimize_smoothing=True, smoothing_method="reml").fit(data)
    values = np.linspace(-1, 1, 70)
    gx, gz = np.meshgrid(values, values)
    grid = pd.DataFrame({"x": gx.ravel(), "z": gz.ravel()})
    terms = np.asarray(model.predict(grid, type="terms"))
    column = next(i for i, term in enumerate(model.fit_result().term_results)
                  if term.label.startswith("ti("))
    surface = terms[:, column].reshape(gx.shape)
    # INTERACTION PLOTTING
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    image = axes[0].pcolormesh(gx, gz, surface, cmap="coolwarm", shading="auto")
    fig.colorbar(image, ax=axes[0], label="Interaction contribution (response units)")
    axes[0].set(xlabel="x", ylabel="z", title="Interaction only")
    for index in [10, 35, 59]:
        axes[1].plot(values, surface[index], label=f"z={values[index]:.2f}")
    axes[1].set(xlabel="x", ylabel="Interaction contribution", title="Slices at fixed z")
    axes[1].legend()
    # INTERACTION END
    save(fig, 'howto_interaction')


def offsets():
    # OFFSETS START
    import numpy as np
    import pandas as pd
    from nampy.gam import GAM

    rng = np.random.default_rng(66)
    data = pd.DataFrame({"age": np.linspace(20, 70, 160)})
    data["exposure"] = rng.uniform(.5, 2, len(data))
    data["count"] = rng.poisson(data.exposure * np.exp(.3 + np.sin(data.age / 10)))
    model = GAM(formula="count ~ s(age,bs='cr',k=8)", family="poisson",
                optimize_smoothing=True, smoothing_method="reml").fit(
                    data, offset=np.log(data.exposure))
    new_data = pd.DataFrame({"age": [25., 45., 65.]})
    link_one = model.predict(new_data, type="link", offset=np.zeros(3))
    link_two = model.predict(new_data, type="link", offset=np.full(3, np.log(2)))
    prediction_one = model.predict(new_data, type="response", offset=np.zeros(3))
    prediction_two = model.predict(new_data, type="response", offset=np.full(3, np.log(2)))
    np.testing.assert_allclose(link_two - link_one, np.log(2))
    np.testing.assert_allclose(prediction_two, 2 * prediction_one)
    offset_table = pd.DataFrame({"Age": new_data.age, "Link (exposure 1)": link_one,
        "Link (exposure 2)": link_two, "Mean (exposure 1)": prediction_one,
        "Mean (exposure 2)": prediction_two})
    # OFFSETS END
    csv_table('howto_offset_comparison', 'Same fitted effects, two supplied exposures', offset_table)


def prediction_outputs():
    # PREDICTIONS START
    import numpy as np
    import pandas as pd
    from nampy.models import GAMClassifier, GAMLSS

    rng = np.random.default_rng(67)
    X = pd.DataFrame({"x": rng.uniform(-1, 1, 300)})
    y = rng.binomial(1, 1 / (1 + np.exp(-2 * X.x)))
    classifier = GAMClassifier(k=6, basis="cr").fit(X, y)
    labels = classifier.predict(X.iloc[:3])
    probabilities = classifier.predict_proba(X.iloc[:3])
    classification_table = pd.DataFrame({"Label": labels,
        **{f"P(class={label})": probabilities[:, i]
           for i, label in enumerate(classifier.classes_)}})
    response = rng.normal(np.sin(3 * X.x), .3 + .2 * (X.x + 1))
    distribution = GAMLSS(family="normal", k=6, basis="cr").fit(X, response)
    parameters = distribution.predict_parameters(X.iloc[:3])
    parameter_table = pd.DataFrame(parameters, columns=distribution.parameter_names_)
    parameter_table["Point prediction"] = distribution.predict_point(X.iloc[:3])
    np.testing.assert_allclose(parameter_table["mu"], parameter_table["Point prediction"])
    # PREDICTIONS END
    csv_table('howto_classification_outputs', 'Labels and probabilities in class order', classification_table)
    csv_table('howto_distribution_outputs', 'Normal parameters and conditional mean', parameter_table)


def gamma_parameters():
    import nbformat
    namespace = {}
    book = nbformat.read(ROOT / 'docs/notebooks/gamma_location_scale.ipynb', 4)
    for cell in book.cells:
        if cell.cell_type == 'code':
            exec(compile(cell.source, 'gamma_location_scale', 'exec'), namespace)
    save(namespace['figure'], 'howto_gamma_parameters')


def quick_recipes():
    from generate_practical_assets import execute
    result = execute(ROOT / 'docs/examples/fit_models.rst')
    for name in ['regression', 'classification', 'distribution']:
        csv_table('howto_recipe_' + name, 'Example returned ' + name + ' rows',
                  result[name + '_preview'])
    plt.close('all')


def quantile_validation():
    """Fit the tutorial's unchanged procedure and render its validation output."""
    import runpy
    import textwrap
    path = ROOT / 'examples/example_qnam.py'
    namespace = runpy.run_path(str(path))
    source = path.read_text().split('def main():', 1)[1].split('    # Predictions', 1)[0]
    exec(compile(textwrap.dedent(source), str(path), 'exec'), namespace)
    model = namespace['model']
    prediction = np.asarray(model.predict(namespace['X_val']))
    coverage = pd.DataFrame({
        'Quantile': namespace['quantiles'],
        'Fraction below': [np.mean(namespace['y_val'] <= prediction[:, j])
                           for j in range(len(namespace['quantiles']))],
    })
    csv_table('detailed_quantile_validation_coverage',
              'Quantile coverage on validation rows', coverage)


def main():
    torch.set_num_threads(1)
    logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)
    for run in [training, preprocessing, interpretability, interaction, offsets, prediction_outputs, gamma_parameters, quick_recipes, quantile_validation]:
        run()
        print('PASS', run.__name__, flush=True)

if __name__ == '__main__':
    main()
