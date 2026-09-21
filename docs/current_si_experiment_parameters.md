# Current Scientific Interest Experiment Parameters

This document records the main SI (Scientific Interest) parameters used by the current experiments. The source of truth is `outputs/pipeline_config.yaml`, which is written by `notebooks/00_configure_pipeline.ipynb`.

## Fit window and forecast horizon

| Parameter | Current value | Meaning |
|---|---:|---|
| `fit_start_year` | `2000` | First year included in the SI fit. |
| `fit_end_year` | `2023` | Last year used for the validation/ranking fit. |
| `publication_filter_end_year` | `2025` | End of the publication-filter window. |
| `curve_fit_end_year` | `2025` | End year for the displayed/interpolated curve. |
| `num_future_years` | `5` | Forecast years beyond `curve_fit_end_year`; the forecast horizon is 2030. |
| `loess_frac` | `0.6` | Linear LOESS smoothing fraction. |
| `loess_degree` | `1` | LOESS degree. |
| `tail_damping_years` | `null` | No additional tail damping. |

## AMR publication filters and usable counts

| Parameter | Current value | Meaning |
|---|---:|---|
| `min_avg_amr_publications_per_year` | `1.0` | Legacy/alternative mean accepted AMR papers-per-year threshold for SI model inputs. |
| `min_tot_amr_publications` | `50` | Minimum accepted AMR publications in the publication-filter window for a pair to enter SI modeling. |
| `min_amr_pair_publication_count` | `3` | Yearly pair counts below this threshold are unavailable to SI modeling and interpolation. |
| `min_num_years_with_valid_counts` | `10` | Minimum number of valid yearly AMR pair observations required for interpolation. |
| `fill_missing_amr_counts_with_zero` | `false` | Missing/low yearly pair counts remain unavailable rather than being converted to zero. |

Counts below `min_amr_pair_publication_count` are represented as unavailable values. They do not contribute to the interpolation or SI curve fit, and the plots show the resulting information gaps.

## Interesting-pair ranking criteria

| Parameter | Current value | Meaning |
|---|---:|---|
| `min_current_predicted_count` | `10` | Minimum predicted pair count at `fit_end_year` for ranking. |
| `min_current_amr_count` | `15` | Minimum observed/materialized AMR pair count at `fit_end_year` for ranking. |
| `min_future_predicted_count` | `10` | Minimum predicted pair count at the future horizon. |
| `min_si_change` | `0.01` | Minimum projected SI increase required for an interesting-pair candidate. |
| `min_post_fit_observed_years` | `1` | Minimum number of observed post-fit years required for validation. |
| `max_post_fit_si_abs_pct_error_mean` | `100.0` | Maximum mean absolute post-fit SI percentage error. |
| `max_si_loess_residual_variance_mean` | `null` | No additional residual-variance cutoff. |
| `fallback_top_n` | `10` | Number of fallback-ranked pairs retained when the strict selection needs fallback candidates. Set to `0` when every output must satisfy the strict ranking criteria. |

The eligible-pair set is identified before downstream plotting. The main SI plots and the decreasing, convex U-shaped, and concave inverted-U-shaped screens operate on that eligible set.

## Shape-screen settings

| Parameter | Current value | Meaning |
|---|---:|---|
| `shape_screen_plot_count` | `10` | Maximum plots saved per shape category. |
| `figure_format` | `pdf` | Output figure format. |
| `display_plots` | `true` | Display generated figures in the notebook. |

The U-shaped and inverted-U-shaped tests use the predicted SI curve from the configured fit/forecast window. A U-shaped curve requires both endpoints to be above the midpoint and a positive quadratic coefficient. An inverted-U-shaped curve requires both endpoints to be below the midpoint and a negative quadratic coefficient.
