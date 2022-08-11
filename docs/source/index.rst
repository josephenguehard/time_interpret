========================================
Time Interpret (tint)
========================================

This package expands the `Captum library <https://captum.ai>`_ with a
specific focus on time-series. As such, it includes various interpretability
methods specifically designed to handle time series data. More documentation
about these methods can be found here:

.. toctree::
    :glob:
    :maxdepth: 1
    :caption: Attribution

    attr


In addition, tint also provides some time series datasets which have been used
as benchmark in recent publications. These datasets are listed here:

.. toctree::
    :glob:
    :maxdepth: 1
    :caption: Datasets

    datasets


We also provide some metrics to evaluate different attribution methods.
These metrics differ depending on if the true saliency is known:

.. toctree::
    :glob:
    :maxdepth: 1
    :caption: Metrics

    metrics


.. toctree::
    :glob:
    :maxdepth: 1
    :caption: White box metrics

    white_box_metrics


Finally, a few general deep learning models, as well as a network to be used along with the
`Pytorch Lightning <https://pytorch-lightning.rtfd.io/en/latest/>`_ framework.
These models can easily be used and trained with this framework.

.. toctree::
    :glob:
    :maxdepth: 1
    :caption: White box metrics

    models


Installation
========================================

.. toctree::
    :glob:
    :maxdepth: 1
    :caption: Installation

    install


Quick-start
========================================

First, let's load an Arma dataset:

.. code-block:: python

    from tint.datasets import Arma

    arma = Arma()
    arma.download()  #This methods generates the dataset


We then load some test data from the dataset and the
corresponding true saliency:

.. code-block:: python

    x = arma.preprocess()["x"][0]
    true_saliency = arma.true_saliency(dim=rare_dim)[0]


We can now load an attribution method and use it to compute the saliency:

.. code-block:: python

    from tint.attr import TemporalIntegratedGradients

    explainer = TemporalIntegratedGradients(arma.get_white_box)

    baseline = inputs * 0
    attr = explainer.attribute(
        inputs,
        baselines=inputs * 0,
        additional_forward_args=(true_saliency,),
        temporal_additional_forward_args=(True,),
    ).abs()


Finally, we evaluate our method using the true saliency and a white box metric:

.. code-block:: python

    from tint.metrics.white_box import aup

    print(f"{aup(attr, true_saliency):.4})
