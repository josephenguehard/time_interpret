========================================
Metrics Weights
========================================

Most of the metrics in time_interpret are computed by perturbing an input and
computing the difference between the output of the model given the original
and this perturbed inputs. In time_interpret, it is also possible to
weight the results according to some method. For instance, ``lime_weights``
weights the results by how close the perturbed input is compared with the
original one.


Summary
========================================

.. autosummary::

    tint.metrics.weights.lime_weights
    tint.metrics.weights.lof_weights


Detailed classes and methods
========================================

.. automodule:: tint.metrics.weights
    :members:
