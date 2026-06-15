# Scientific Interest, in Plain English

This project uses **Scientific Interest** (`SI`) to measure whether a pathogen-antibiotic pair is receiving more or less AMR publication attention than expected.

The key idea is simple:

> Compare the observed number of AMR papers for a pair with the number we would expect after accounting for how much the pathogen and antibiotic are studied in general.

For example, *Escherichia coli* and ciprofloxacin may have many AMR papers partly because both *E. coli* and ciprofloxacin are already heavily studied. Scientific Interest tries to separate this broad popularity from pair-specific extra attention.

## What Goes Into The Measure

For each pathogen-antibiotic pair in each year, the workflow starts with three counts:

- `count`: AMR-positive papers for that exact pair in that year
- `bg_pathogen_n`: all papers mentioning the pathogen in that year
- `bg_antibiotic_n`: all papers mentioning the antibiotic in that year

The background counts are not resistance counts. They are publication-attention counts.

So if the row is:

| pathogen | antibiotic | year | count | bg_pathogen_n | bg_antibiotic_n |
|---|---:|---:|---:|---:|---:|
| Escherichia coli | Ciprofloxacin | 2020 | 45 | 12,000 | 2,000 |

the model asks:

> Given that *E. coli* had 12,000 papers and ciprofloxacin had 2,000 papers in 2020, how many AMR papers would we expect for this pair?

## Why The Counts Are Logged

Publication counts are very uneven. Some pairs have one or two papers; others have hundreds.

To make these numbers easier to compare, the workflow uses:

```text
log_count = log10(1 + count)
```

The `1 + count` part means zero-count rows can still be handled. The `log10` part compresses very large values.

## The Expected Count Baseline

The workflow fits one pooled model across all pathogen-antibiotic-year rows.

In plain English, the model learns:

- which pathogens usually attract more AMR papers
- which antibiotics usually attract more AMR papers
- how general pathogen publication volume relates to AMR publication volume
- how general antibiotic publication volume relates to AMR publication volume

The model then predicts an expected log-count:

```text
expected_log_count
```

This is the expected AMR publication level on the log scale after accounting for broad publication attention.

## The Scientific Interest Formula

Scientific Interest is the gap between observed attention and expected attention on the log scale:

```text
SI = log10(1 + observed AMR count) - expected_log_count
```

Because this is a difference between logs, it is best interpreted as a **ratio**, not as an absolute difference in papers.

If:

```text
expected_log_count = log10(1 + expected AMR count)
```

then:

```text
SI = log10((1 + observed AMR count) / (1 + expected AMR count))
```

To convert SI back into an observed-versus-expected multiplier:

```text
attention_ratio = 10^SI
```

Interpretation:

- `SI > 0`: more AMR papers than expected
- `SI = 0`: about as many AMR papers as expected
- `SI < 0`: fewer AMR papers than expected

For example:

- `SI = 0.30` means about `10^0.30 ≈ 2.0` times expected
- `SI = 0.40` means about `10^0.40 ≈ 2.5` times expected
- `SI = -0.30` means about `10^-0.30 ≈ 0.5` times expected

## A Simple Example

Suppose a pair has:

```text
observed AMR count = 99 papers
```

Then:

```text
log10(1 + 99) = log10(100) = 2.00
```

Now suppose the baseline model expected:

```text
expected_log_count = 1.60
```

Then:

```text
SI = 2.00 - 1.60 = 0.40
```

That pair has positive Scientific Interest. It is receiving more AMR publication attention than expected.

On the original count scale, `expected_log_count = 1.60` means:

```text
10^1.60 - 1 ≈ 39 expected papers
```

But the SI value itself should be read as a log-ratio:

```text
10^0.40 ≈ 2.5
```

So the plain-English version is:

> The pair had 99 observed AMR papers, while the model expected about 39. That is about 2.5 times the expected level, so Scientific Interest is positive.

## A Negative Example

Suppose another pair has:

```text
observed AMR count = 9 papers
log10(1 + 9) = 1.00
expected_log_count = 1.60
```

Then:

```text
SI = 1.00 - 1.60 = -0.60
```

On the original count scale, the model again expected about 39 papers, but only 9 were observed.

As a multiplier:

```text
10^-0.60 ≈ 0.25
```

So this pair has negative Scientific Interest:

> It is receiving about one quarter of the AMR attention expected given the background popularity of the pathogen and antibiotic.

## What SI Is Not

Scientific Interest is not a direct measure of clinical resistance burden.

It does not mean:

- the pathogen is more dangerous
- the antibiotic is failing more often
- resistance prevalence is higher

It means:

> The literature is paying unusually high or low attention to that pathogen-antibiotic resistance pair, compared with a background publication baseline.

## Why It Is Useful

Raw publication counts are biased toward famous pathogens and famous antibiotics.

Scientific Interest helps answer a more focused question:

> Which pairs are getting more AMR attention than their general publication popularity would suggest?

That makes it useful for ranking, comparison, and early-warning exploration.

## How Forecasted SI Is Calculated

For future years, the workflow does not forecast SI directly.

Instead, it:

1. fits a smooth trend through the observed yearly AMR counts
2. predicts future AMR counts on the log scale
3. compares the predicted log-count with the expected baseline

The forecasted version is:

```text
predicted_SI = predicted_log_count - expected_log_count_baseline
```

So the same ratio idea still applies:

> predicted Scientific Interest is the log-ratio between predicted publication attention and expected background attention.
