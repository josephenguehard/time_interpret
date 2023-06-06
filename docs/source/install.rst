========================================
Installation
========================================


PIP
========================================

Time interpret can be installed with pip:

.. code-block:: console

    pip install time_interpret


From source
========================================

Conda
----------------------------------------

.. code-block:: console

    git clone git@github.com:josephenguehard/time_interpret.git
    cd time_interpret
    conda env create
    source activate tint
    pip install --no-deps .


Pip
----------------------------------------

.. code-block:: console

    git clone git@github.com:josephenguehard/time_interpret.git
    python -m venv <myvenv>
    source <myvenv>/bin/activate
    cd time_interpret
    pip install .


Docker
----------------------------------------

.. code-block:: console

    git clone git@github.com:josephenguehard/time_interpret.git
    cd time_interpret
    docker build -t {your_image} .
    docker push {your_image}