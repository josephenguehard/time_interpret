========================================
Time Interpret (tint)
========================================

This package expands the `Captum library <https://captum.ai>`_ with a
specific focus on time-series. As such, it includes various interpretability
methods specifically designed to handle time series data.


Installation
========================================

.. toctree::
    :glob:
    :maxdepth: 1

    install


Quick-start
========================================

First, let's load an Arma dataset:

.. code-block:: python

    from tint.datasets import Arma

    arma = Arma()
    arma.download()  # This method generates the dataset


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


API
========================================

Each of the implemented interpretability methods can be found here:

.. autosummary::

    tint.attr.AugmentedOcclusion
    tint.attr.BayesKernelShap
    tint.attr.BayesLime
    tint.attr.BayesMask
    tint.attr.DiscretetizedIntegratedGradients
    tint.attr.DynaMask
    tint.attr.Fit
    tint.attr.LofKernelShap
    tint.attr.LofLime
    tint.attr.Occlusion
    tint.attr.Retain
    tint.attr.SmoothGrad
    tint.attr.SequentialIntegratedGradients
    tint.attr.TemporalAugmentedOcclusion
    tint.attr.TemporalIntegratedGradients
    tint.attr.TemporalOcclusion
    tint.attr.TimeForwardTunnel


Some of these attributions use specific models which are listed here:

.. autosummary::

    tint.attr.models.BayesMaskNet
    tint.attr.models.BLRRegression
    tint.attr.models.BLRRidge
    tint.attr.models.JointFeatureGeneratorNet
    tint.attr.models.MaskNet
    tint.attr.models.RetainNet
    tint.attr.models.scale_inputs

In addition, tint also provides some time series datasets which have been used
as benchmark in recent publications. These datasets are listed here:

.. autosummary::

    tint.datasets.Arma
    tint.datasets.BioBank
    tint.datasets.Hawkes
    tint.datasets.HMM
    tint.datasets.Mimic3


We also provide some metrics to evaluate different attribution methods.
These metrics differ depending on if the true saliency is known:

.. autosummary::

    tint.metrics.accuracy
    tint.metrics.comprehensiveness
    tint.metrics.cross_entropy
    tint.metrics.lipschitz_max
    tint.metrics.log_odds
    tint.metrics.sufficiency


.. autosummary::

    tint.metrics.white_box.aup
    tint.metrics.white_box.auprc
    tint.metrics.white_box.aur
    tint.metrics.white_box.entropy
    tint.metrics.white_box.information
    tint.metrics.white_box.mae
    tint.metrics.white_box.mse
    tint.metrics.white_box.rmse
    tint.metrics.white_box.roc_auc


Finally, a few general deep learning models, as well as a network to be used along with the
`Pytorch Lightning <https://pytorch-lightning.rtfd.io/en/latest/>`_ framework.
These models can easily be used and trained with this framework.

.. autosummary::

    tint.models.Bert
    tint.models.DistilBert
    tint.models.CNN
    tint.models.MLP
    tint.models.Net
    tint.models.RNN
    tint.models.Roberta
    tint.models.TransformerEncoder


More details about each of these categories can be found here:

.. toctree::
    :glob:
    :maxdepth: 1

    attr
    attr_models
    datasets
    metrics
    white_box_metrics
    models
