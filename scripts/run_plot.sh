device=chest
exp_name=collossl
type=multi
python3 ../plot_results.py --${type} /mnt/data/gsl/runs/${device}/${exp_name}/${type}/logs/hparam_tuning_*/results_summary.csv --show_error_bar=False --out_file ${device}_${exp_name}_${type}.png 