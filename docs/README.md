# Documentation

This package provides documentation using Sphinx and Read the Docs.

## Installation

In order to generate the documentation, install the required packages to your
environment:

```shell script
cd docs/
source activate app  # If using conda env
pip install -r requirements.txt
```

## Docs generation

To generate the documentation itself, run, inside of the `docs` folder:

```shell script
make html
```

You can also modify the source docs by changing the files in `docs/source`.

## Server hosting
To start an online server hosting the docs, run this command:

```shell script
cd build/html/
python -m http.server {PORT}
```

The server will be accessible at the specified PORT: `http://localhost:{PORT}`.