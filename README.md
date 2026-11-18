# Introduction to Machine Learning

## What is this?

This repository holds the online pages of my course Introduction to Machine Learning.

The course uses [Jupyter-Book](https://jupyterbook.org/en/stable/intro.html) as a wrapper for displaying lecture content and exercises.

You can access the published version [here](https://clairedavid.github.io/intro_to_ml).

This course has been mostly delivered at the [African Institute for Mathematical Sciences (AIMS)](https://nexteinstein.org) in Senegal, Cameroon, Ghana and South Africa. For the latter, it is taught as part of the [AI for Science](https://ai.aims.ac.za/) Master's program.

## Local preview

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
these commands from the repository root:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
uv run --no-project jupyter-book build . --builder html
uv run --no-project python -m http.server 8000 --bind 127.0.0.1 --directory _build/html
```

Open <http://localhost:8000> to preview the site. After editing content, rerun
the build command and refresh your browser. Press `Ctrl+C` to stop the server.

On Intel Macs, use Python 3.12 as shown above: `requirements.txt` automatically
selects PyTorch 2.2.2 (the last release with Intel macOS wheels) and NumPy 1.x
for compatibility. These wheels do not support Python 3.13 or newer. Other
platforms use PyTorch 2.8.0.

The preview executes only the four PCA pages and the autoencoder page in
`unsupervised/`, caching their outputs for subsequent builds. These pages have
per-page `mystnb` settings with a five-minute timeout per cell; execution errors
stop the build. The PCA walkthrough embeds its interactive Altair chart as HTML.
Other pages use saved notebook outputs rather than executing the notebooks.
To execute notebooks with missing outputs, set `execute.execute_notebooks` to
`auto` in `_config.yml` before building.
Jupyter Book is constrained to version 1 because this site uses its Sphinx-based
configuration format.

## Acknowledgments

Thanks to the Jupyter universe for providing a great educational platform and making my life easier.

Thanks to Matthew Feickert to point out the convenient Jupyter-Book tool.

Thanks to Sam Meehan for insisting ^^

Thanks to Michaël Ughetto for the added knowledge, content and frankness.

Thanks to Harrison Prosper for being an amazing guide and mentor, as well as for sharing curated tutorials that saved me numerous hours.
