# Mathematical Modeling of Transcriptional Bursting

In molecular biology, genes don’t politely churn out RNA at a constant rate. No, they **burst.** This project attempts to make sense of that unruly behavior using mathematics.

We model that behavior using two approaches:

1. **Poisson + two-state Markov model** – Transcripts are produced when the promoter is in an "On" state, with detection modeled probabilistically.
2. **Negative binomial + hierarchical two-state model** – A more flexible version that accounts for variation across cells using a Gamma–Poisson mixture.

Both models are derived from first principles using standard probability tools such as exponential waiting times, Poisson thinning, compound distributions, and moment calculations. This project was developed as part of an IB Mathematics AA HL Internal Assessment, which means the code prioritizes educational value over computational elegance. It may not run fast, but with any luck, the math behaves itself.

These models are implemented in Python. All derivations and mathematical justifications are documented in [`docs/model_A_poisson.md`](docs/model_A_poisson.md) and[`docs/model_B_negbin.md`](docs/model_B_negbin.md). Figures for promoter dynamics, burst size distributions, and overdispersion metrics are in [`docs/figures/`](docs/figures/).

To run the code, you will first need to install the dependencies listed in [`requirements.txt`](requirements.txt):

```bash
pip install -r requirements.txt
```

Then, open JupyterLab or Jupyter Notebook:

```bash
jupyter notebook
```

Then explore [`notebooks/poisson_model_demo.ipynb`](notebooks/poisson_model_demo.ipynb) and [`notebooks/negbin_model_demo.ipynb`](notebooks/negbin_model_demo.ipynb). Alternatively, you can run an example script:

```bash
python examples/simulate_poisson_model.py
```

To test the code (just in case), you can run unit tests using `pytest`:

```bash
pytest tests/
```

## License

This project is licensed under the [MIT License](LICENSE). This is only because licensing things makes people feel like adults and gives the illusion that someone, somewhere, knows what they’re doing with the code.

You’re free to use or modify this code—though if you do copy it wholesale for your IA, please be aware that the IB takes plagiarism about as seriously as your next-door neighbor cares about the late-night drilling noises you make in the walls. You’ve been warned.
