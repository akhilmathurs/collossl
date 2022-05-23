# ColloSSL: Collaborative Self-Supervised Learning for Human Activity Recognition

ColloSSL (pronounced colossal) is a technique for collaborative self-supervised contrastive learning among a group of devices by utlizing the time-synchronocity of their data. 


![gsl_architecture_revised-1](https://user-images.githubusercontent.com/34444901/163887816-98981dc3-96e3-41d4-83bc-243359990756.png)



This repo is a Tensorflow implementation of the [ColloSSL paper](https://arxiv.org/pdf/2202.00758.pdf). 
```
@article{jain2022collossl,
  title={ColloSSL: Collaborative Self-Supervised Learning for Human Activity Recognition},
  author={Jain, Yash and Tang, Chi Ian and Min, Chulhong and Kawsar, Fahim and Mathur, Akhil},
  journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume={6},
  number={1},
  pages={1--28},
  year={2022},
  publisher={ACM New York, NY, USA}
}
```

## Environment setup

The code works with the latest docker image of tensorflow. Run  `bash run_container.sh` to start a docker container, make changes in the script according to your filesystem. Subsequently, run `pip install -r requirements.txt` to install extra dependencies in the docker container.

## Directory Structure
We create a directory for each `train_device` as follows:

`args.working_directory / args.train_device / args.exp_name / args.training_mode`

e.g., /mnt/data/gsl/runs/thigh/my_exp/single/


Inside each directory, there are three subdirs:

* `/models`
* `/logs`
* `/results`

Each hyperparam runs is assigned a `run_name` and all logs, models and result files share the same name. 

## Running instruction
The `scripts/` directory has example scripts for each type of experiment. Before running any script, you need to change the `working_directory` and `dataset_path` in the scripts.

* `collossl_single_run.sh` - Single ColloSSL run for a particular `train_device` and `eval_device` with a fixed hyperparameter settings
* `collossl.sh` - Multiple ColloSSL runs for all devices in the dataset. Runs happen across *multiple-gpus* and are automatically scheduled one after other. Please refer to `hparam_tuning_mp.py` to change/add any hyperparameters.
* `supervised.sh` - Supervised baseline for each device in the dataset. Runs happen across *multiple-gpus* and are automatically scheduled one after other. Please refer to `hparam_tuning_mp.py` to change/add any hyperparameters.
* `supervised_all_devices.sh` - Supervised baseline using all device data during training.
* `other_ssl_baselines.sh` - Running configurations for other baselines in the paper. Runs happen across *multiple-gpus* and are automatically scheduled one after other. Please refer to `hparam_tuning_mp_ssl_baseline.py` to change/add any hyperparameters.

The results of all the runs are stored in `args.working_directory / args.train_device / args.exp_name / args.training_mode/ logs /result_summary.csv` file which can later be plotted using `plot_results.py`. 

Scripts also generate plots of completed runs in `scripts/args.exp_name` directory. Example plots are shown in `/results` directory


## Steps to retrieve a certain model (manually) 

1) Open Tensorboard with `--logdir=<args.working_directory/args.train_device/args.exp_name/args.training_mode/logs/hparam_tuning_*>`
2) Go to the Scalars tab and pick the run of your choice (e.g., the one with the best F1 score). Copy its run_id. 
3) (Optionally) You can now go to the HParams Table view and find the hyperparams corresponding to it. Unfortunately, the ID in Hparams is system generated and not the same as run_id. We will have to match the runs based on other metrics, e.g., F-1 score.
4) The model and result file for the selected run should be in `args.working_directory/args.train_device/args.exp_name/args.training_mode/models/run_id.hdf5` and `args.working_directory/args.train_device/args.exp_name/args.training_mode/results/run_id.txt`   

## Models


## License
