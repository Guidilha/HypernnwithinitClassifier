import argparse
import sklearn.metrics as mtr
import src.torchsetup as tsetup
from src.dataset.datasetcollector import DatasetCollector
from src.evaluation.modelevaluator import ModelEvaluator
from src.models.hypernnwithinitclassifier import HyperNNWithInitClassifier


def parse_experiments_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",default="hypernn",type=str,help="which model to use: prim, hypernn.",)
    parser.add_argument("--results_directory",default="results",type=str,help="where to store results.",)
    parser.add_argument("--max_train_epochs",default=10000,type=int,help="maximum number of epochs.",)
    parser.add_argument("--max_number_of_boxes",default=100,type=int,help="maximum number of boxes.",)
    parser.add_argument("--patience", default=200, type=int, help="early stopping count.")
    parser.add_argument("--tau", default=1.0, type=float, help="initial smooth sigmoid factor.")
    parser.add_argument("--alpha", default=1.0, type=float, help="initial smoothmax factor.")
    parser.add_argument("--use_cuda",default="True",type=str,help="set device as cuda if available.",)
    parser.add_argument("--init_strategy",default="som",type=str,choices=["som", "random", "kmeans", "dbscan", "xavier", "all"],help="Estratégia de inicialização a ser testada.",)
    parser.add_argument("--test_size",default=0.3,type=float,help="split ratio for the test dataset.",)
    parser.add_argument("--num_cv_folds", default=5, type=int, help="number of cv folds.")
    parser.add_argument("--num_seeds", default=3, type=int, help="number of random seeds.")
    parser.add_argument("--scaling",default=True,type=bool,help="apply feature standard scaling.",)
    parser.add_argument("--use_small_data",default=True,type=bool,help="use only small datasets.",)
    args, _ = parser.parse_known_args()
    if isinstance(args.use_cuda, str):
        args.use_cuda = args.use_cuda.lower() in ["true", "1", "yes"]
    return args


def get_hypernn_and_hyperparameters(experiment_options):
    device = tsetup.set_device(cuda=experiment_options.use_cuda)
    strategy = experiment_options.init_strategy
    model = HyperNNWithInitClassifier(init_strategy=strategy)

    if strategy == "som":
        init_param_grid = {
            "init_strategy": ["som"],
            "som_sigma": [0.5,1.0,1.5],
            "som_lr": [0.5,0.1,1.5],
            "nboxes": [4, 8, 12, 16],
        }
        hypernn_param_grid = {
            "init_strategy": ["som"],
            "device": [device],
            "training_epochs": [experiment_options.max_train_epochs],
            "patience_early_stopping": [experiment_options.patience],
            "learning_rate": [0.05, 0.01, 0.1],
            "reg_warmup_epochs": [400],
            "reg_warmup_start": [1e-5],
            "l1_reg": [1e-3],
            "l2_reg": [1e-3],
            "alpha_tau_decay_step": [10],
            "alpha": [experiment_options.alpha],
            "tau": [experiment_options.tau],
        }
    elif strategy == "kmeans":
        init_param_grid = {
            "init_strategy": ["kmeans"],
            "nboxes": [4, 8, 12, 16],
        }
        hypernn_param_grid = {
            "init_strategy": ["kmeans"],
            "device": [device],
            "training_epochs": [experiment_options.max_train_epochs],
            "patience_early_stopping": [experiment_options.patience],
            "learning_rate": [0.05, 0.01, 0.1],
            "reg_warmup_epochs": [400],
            "reg_warmup_start": [1e-5],
            "l1_reg": [1e-3],
            "l2_reg": [1e-3],
            "alpha_tau_decay_step": [10],
            "alpha": [experiment_options.alpha],
            "tau": [experiment_options.tau],
        }
    elif strategy == "dbscan":
        init_param_grid = {
            "init_strategy": ["dbscan"],
            "dbscan_eps": [0.1, 0.5, 1, 1.5, 2],
            "dbscan_min_samples": [1, 10, 20, 50],
            "dbscan_eps_by_dim":[True],
            "dbscan_cap_boxes":[True],
            "dbscan_min_coverage":[0.9],
            "nboxes": [4, 8, 12, 16]
        }
        hypernn_param_grid = {
            "init_strategy": ["dbscan"],
            "device": [device],
            "training_epochs": [experiment_options.max_train_epochs],
            "patience_early_stopping": [experiment_options.patience],
            "learning_rate": [0.05, 0.01, 0.1],
            "reg_warmup_epochs": [400],
            "reg_warmup_start": [1e-5],
            "l1_reg": [1e-3],
            "l2_reg": [1e-3],
            "alpha_tau_decay_step": [10],
            "alpha": [experiment_options.alpha],
            "tau": [experiment_options.tau],
        }
    elif strategy in ["random", "xavier"]:
        init_param_grid = None
        hypernn_param_grid = {
            "init_strategy": [strategy],
            "nboxes": [4, 8, 12, 16], #nboxes pode ser fixo contanto que seja grande o suficiente para o dataset, reduzindo tempo de execução
            "device": [device],
            "training_epochs": [experiment_options.max_train_epochs],
            "patience_early_stopping": [experiment_options.patience],
            "learning_rate": [0.05,0.01, 0.1],
            "reg_warmup_epochs": [400],
            "reg_warmup_start": [1e-5],
            "l1_reg": [1e-3], 
            "l2_reg": [1e-3],
            "alpha_tau_decay_step": [10],
            "alpha": [experiment_options.alpha],
            "tau": [experiment_options.tau],
        }
    else:
        raise ValueError(f"Estratégia desconhecida: {strategy}")

    return model, init_param_grid, hypernn_param_grid


def main(experiment_options):
    model_name = experiment_options.model_name
    if model_name != "hypernn":
        raise NotImplementedError(f"{model_name} is not implemented.")

    collector = DatasetCollector()
    if not experiment_options.use_small_data:
        datasets = collector.get_all_datasets()
    else:
        datasets = [
            collector.load_iris(),
            collector.load_wine(),
            collector.load_breast_cancer(),
            collector.load_digits(),
            collector.load_vehicle(),
        ]

    strategies = (
        ["som", "kmeans", "dbscan", "random", "xavier"]
        if experiment_options.init_strategy == "all"
        else [experiment_options.init_strategy]
    )

    for strat in strategies:
        print(f"\n============================================================")
        print(f"   INICIANDO EXPERIMENTO - ESTRATÉGIA: {strat.upper()}")
        print(f"============================================================\n")

        opts = argparse.Namespace(**vars(experiment_options))
        opts.init_strategy = strat

        model, init_param_grid, hypernn_param_grid = get_hypernn_and_hyperparameters(opts)
        model.init_strategy = strat

        evaluator = ModelEvaluator(
            model=model,
            init_hyperparameter_space=init_param_grid,
            hypernn_hyperparameter_space=hypernn_param_grid,
            test_size=opts.test_size,
            cross_validation_folds=opts.num_cv_folds,
            scale_data=opts.scaling,
            num_seeds=opts.num_seeds,
            evaluation_metric=mtr.f1_score,
            results_logging_directory=opts.results_directory,
            experiment_name=strat,
        )
        evaluator.evaluate(datasets)


if __name__ == "__main__":
    main(parse_experiments_parameters())
