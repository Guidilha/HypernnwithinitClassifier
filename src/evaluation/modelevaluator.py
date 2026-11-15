import hashlib
import itertools
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.metrics as mtr
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import trange

from ..torchsetup import set_seeds
from ..utils import binarize_labels
from .hyperparametersearch import HyperparameterGridSearch

CHECKPOINT_FILE = "experiment_checkpoint.json"

# ======================================================================
# RECURSOS OPCIONAIS DE DIAGNÓSTICO
# True  = liga (útil localmente, para depurar e gerar figuras)
# False = desliga
# Desligados, não gravam nada em disco, não calculam o F1 por época e nem
# exigem matplotlib instalado. O CSV e o log sempre são gerados.
# ======================================================================
ENABLE_FIGURES = True       # figuras PNG (só da semente 0)
# Datasets que geram figuras (None = todos).
FIGURE_DATASETS = None

if ENABLE_FIGURES:
    from src.evaluation.visualization import plot_boxes_figure, som_grid_shape

# ======================================================================
# CONTROLE DO TERMINAL
# O arquivo .log SEMPRE recebe tudo. Aqui você escolhe o que também aparece
# no terminal: comente uma linha para ESCONDER, descomente para MOSTRAR.
# ======================================================================
CONSOLE_SHOW = {
    "exec",         # [EXECUÇÃO] Dataset | Estratégia | Seed
    "result",       # [RESULTADO] F1, caixas e tempos de cada classe
    # "run",        # Seed / Dataset / Estratégia / Label (n/total)
    # "seed",       # [SEED] fingerprint dos dados de treino
    # "init_grid",  # [INIT GRID] uma linha por combinação do grid de init
    # "init_best",  # [INIT GRID] Melhor: ...
    # "init_lock",  # Init travada: N caixas
    # "viz",        # [VIZ] figura salva em ...
    # "init_warn",  # [DBSCAN Warning] nenhum cluster encontrado
}


class _ConsoleCategoryFilter(logging.Filter):
    def filter(self, record):
        cat = getattr(record, "log_cat", None)
        return cat is None or cat in CONSOLE_SHOW


class ModelEvaluator:
    def __init__(
        self,
        model,
        test_size,
        cross_validation_folds,
        scale_data,
        num_seeds,
        evaluation_metric,
        results_logging_directory,
        init_hyperparameter_space=None,
        hypernn_hyperparameter_space=None,
        model_hyperparameter_space=None,
        experiment_name=None,
        random_state=42,
        visualize=True,
        visualize_seeds=(0,),
    ):
        self.model = model
        self.cross_validation_folds = cross_validation_folds
        self.evaluation_metric = evaluation_metric
        self.test_size = test_size
        self.num_seeds = num_seeds
        self.visualize = visualize and ENABLE_FIGURES
        self.visualize_seeds = tuple(visualize_seeds)

        self.init_hyperparameter_space = init_hyperparameter_space
        self.hypernn_hyperparameter_space = hypernn_hyperparameter_space

        if init_hyperparameter_space is not None and hypernn_hyperparameter_space is not None:
            self.model_hyperparameter_space = {**init_hyperparameter_space, **hypernn_hyperparameter_space}
        else:
            self.model_hyperparameter_space = model_hyperparameter_space or hypernn_hyperparameter_space or {}

        self.strategy_name = self._determine_strategy_name()

        self.scale_data = scale_data
        self.results = []
        self.logger = None
        self.results_filepath = ""
        self.random_state = random_state
        self.experiment_name = experiment_name or self.strategy_name
        self.configure_logging_and_results(results_logging_directory)

    @staticmethod
    def _seed_everything(seed):
        """Fixa todas as fontes de aleatoriedade (random, numpy, torch) na mesma seed."""
        set_seeds(seed)
        try:
            import torch
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

    def _determine_strategy_name(self) -> str:
        for space in [self.init_hyperparameter_space, self.hypernn_hyperparameter_space, self.model_hyperparameter_space]:
            if space and "init_strategy" in space:
                val = space["init_strategy"]
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    return str(val[0])
                elif isinstance(val, str):
                    return val
        return str(getattr(self.model, "init_strategy", "default"))

    # Checkpoint

    def _config_hash(self) -> str:
        """Hash da configuração: evita reaproveitar checkpoint de um grid diferente."""
        cfg = json.dumps(
            {
                "init": self.init_hyperparameter_space,
                "hypernn": self.hypernn_hyperparameter_space,
                "test_size": self.test_size,
                "cv": self.cross_validation_folds,
                "scale": self.scale_data,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.md5(cfg.encode()).hexdigest()[:8]

    def _checkpoint_key(self, dataset_name: str, strategy_name: str, seed: int) -> str:
        return f"{dataset_name}_{strategy_name}_seed_{seed}_{self._config_hash()}"

    def check_and_load_checkpoint(self, dataset_name: str, strategy_name: str, seed: int):
        if os.path.exists(CHECKPOINT_FILE):
            try:
                with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                key = self._checkpoint_key(dataset_name, strategy_name, seed)
                if key in data:
                    print(f"[Checkpoint] Recuperado do disco: {key}")
                    return True, data[key]
            except Exception as e:
                print(f"[Checkpoint Warning] Erro ao ler checkpoint: {e}")
        return False, None

    def save_checkpoint(self, dataset_name: str, strategy_name: str, seed: int, result_metrics: dict):
        data = {}
        if os.path.exists(CHECKPOINT_FILE):
            try:
                with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        key = self._checkpoint_key(dataset_name, strategy_name, seed)
        data[key] = result_metrics

        try:
            temp_file = CHECKPOINT_FILE + ".tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, default=str)
            os.replace(temp_file, CHECKPOINT_FILE)
        except Exception as e:
            print(f"[Checkpoint Warning] Erro ao salvar checkpoint: {e}")

    # Logging / resultados

    def configure_logging_and_results(self, results_logging_directory):
        directory_path = Path(results_logging_directory)
        assert directory_path.exists(), f"The provided path {results_logging_directory} does not exist."

        model_name = type(self.model).__name__
        datetime_str = time.strftime("%Y%m%d-%H%M%S")
        exp_name = f"_{self.experiment_name}" if self.experiment_name is not None else ""
        filename = f"results_{model_name}_{datetime_str}{exp_name}"

        logging_filepath = directory_path.joinpath(f"{filename}.log")
        console_handler = logging.StreamHandler()
        console_handler.addFilter(_ConsoleCategoryFilter())
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            handlers=[
                logging.FileHandler(filename=logging_filepath),
                console_handler,
            ],
            force=True,
        )
        self.logger = logging.getLogger("logger")
        self.results_filepath = directory_path.joinpath(f"{filename}.csv")

        # As figuras ficam junto dos resultados
        self.figures_dir = directory_path.joinpath("figures")
        if ENABLE_FIGURES:
            self.figures_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment_report(
        self,
        dataset_name,
        dataset_label,
        best_model_hyperparameters,
        training_scores,
        test_scores,
        hyperparameter_search_time,
        training_time,
        initialization_time,
        prediction_time,
        final_nboxes,
        seed,
        epochs_run=None,
    ):
        """Cria um dicionário com métricas calculadas e relatórios de execução."""
        report = {
            "model": type(self.model).__name__,
            "hyperparameter_space": self.model_hyperparameter_space,
            "dataset": dataset_name,
            "dataset_label": dataset_label,
            "scale_data": self.scale_data,
            "evaluation_metric": self.evaluation_metric.__name__,
            "cross_validation_folds": self.cross_validation_folds,
            "test_size": self.test_size,
            "final_nboxes": final_nboxes,
            **best_model_hyperparameters,
            **training_scores,
            **test_scores,
            "training_time": training_time,
            "initialization_time": initialization_time,
            "init_time": initialization_time,
            "hyperparameter_search_time": hyperparameter_search_time,
            "prediction_time": prediction_time,
            "seed": seed,
            "epochs_run": epochs_run,
        }
        return report

    def compute_metric_scores(self, y_true, y_pred, prefix=""):
        """Calcula as métricas de avaliação."""
        scores = {
            f"{prefix}accuracy": mtr.accuracy_score(y_true, y_pred),
            f"{prefix}precision": mtr.precision_score(y_true, y_pred, zero_division=0),
            f"{prefix}recall": mtr.recall_score(y_true, y_pred, zero_division=0),
            f"{prefix}f1_score": mtr.f1_score(y_true, y_pred, zero_division=0),
            f"{prefix}balanced_accuracy": mtr.balanced_accuracy_score(y_true, y_pred),
        }
        return scores

    # ESTÁGIO 1: busca dos parâmetros de inicialização

    @staticmethod
    def _predict_with_init_boxes(model, X):
        """Retorna 1 se o ponto está dentro de alguma caixa inicial (usado só no desempate)."""
        mins = model.mins_tensor.detach().cpu().numpy()
        maxs = mins + np.abs(model.lengths_tensor.detach().cpu().numpy())
        X = np.asarray(X)
        inside = (X[:, None, :] >= mins[None]) & (X[:, None, :] <= maxs[None])
        return inside.all(axis=2).any(axis=1).astype(int)

    def _score_init_candidate(self, params, X_train, y_train, seed):
        """
        Avalia uma combinação de parâmetros de init.
        Retorna (score onde MAIOR é melhor, modelo com init já calculada).

        - SOM:            CE = QE * (TE + eps)  (menor é melhor -> score = -CE)
        - KMeans/DBSCAN:  Silhouette sobre os pontos positivos (os mesmos que
                          geram as caixas); já calculada em fit_init.
        """
        strat = params.get("init_strategy", self.strategy_name)
        cand = self.model.set_hyperparameters({**params, "random_state": seed})
        cand.fit_init(X_train, y_train)

        if cand.mins_tensor is None:  # ex.: DBSCAN sem clusters
            return -np.inf, cand

        if strat == "som":
            return -float(cand.init_metric_score), cand

        return float(cand.init_metric_score), cand

    def _secondary_cluster_metrics(self, cand, X_train, y_train, seed):
        X_pos = X_train[y_train == 1]
        try:
            if cand.init_strategy == "kmeans":
                labels = KMeans(
                    n_clusters=min(cand.nboxes, len(X_pos)),
                    random_state=seed,
                    n_init="auto",
                ).fit_predict(X_pos)
                mask = np.ones(len(labels), dtype=bool)
            elif cand.init_strategy == "dbscan":
                labels = cand.dbscan_labels(X_pos)
                mask = labels != -1
            else:
                return {}
            if len(set(labels[mask])) < 2:
                return {}
            return {
                "DB": davies_bouldin_score(X_pos[mask], labels[mask]),
                "CH": calinski_harabasz_score(X_pos[mask], labels[mask]),
            }
        except Exception:
            return {}

    def search_best_init(self, X_train, y_train, seed):
        """Estágio 1: testa todas as combinações de init e escolhe pela métrica do método."""
        keys = list(self.init_hyperparameter_space.keys())
        combos = [
            dict(zip(keys, vals))
            for vals in itertools.product(*self.init_hyperparameter_space.values())
        ]
        metric_name = "-CE" if self.strategy_name == "som" else "Silhouette"

        results = []
        for params in combos:
            score, cand = self._score_init_candidate(
                params, X_train, y_train, seed
            )
            n_boxes = cand.mins_tensor.shape[0] if cand.mins_tensor is not None else 0
            extra = ""
            if self.strategy_name == "som":
                extra = f" | QE={cand.init_qe:.4f} TE={cand.init_te:.4f}"
            elif cand.mins_tensor is not None:
                sec = self._secondary_cluster_metrics(cand, X_train, y_train, seed)
                if sec:
                    extra = " | " + " ".join(f"{k}={v:.4f}" for k, v in sec.items())
            self.logger.info(
                f"[INIT GRID] {params} -> {metric_name}={score:.6f} | caixas={n_boxes}{extra}",
                extra={"log_cat": "init_grid"},
            )
            results.append((score, params, cand))

        best_score = max(r[0] for r in results)
        if not np.isfinite(best_score):
            raise RuntimeError("Nenhuma combinação de init gerou caixas válidas.")

        # gamma_escala só muda o tamanho das caixas, não o clustering, então a
        # métrica empata entre valores de gamma. Desempate: F1 das caixas no treino.
        tied = [r for r in results if r[0] >= best_score - 1e-9]
        if len(tied) > 1:
            def box_f1(r):
                y_pred = self._predict_with_init_boxes(r[2], X_train)
                return mtr.f1_score(y_train, y_pred, zero_division=0)
            chosen = max(tied, key=box_f1)
        else:
            chosen = tied[0]

        score, params, cand = chosen
        self.logger.info(
            f"[INIT GRID] Melhor: {params} ({metric_name}={score:.6f})",
            extra={"log_cat": "init_best"},
        )
        return params, cand

    def _build_and_save_figure(self, best_model, X_train, y_train, dataset_name, seed, label):
        try:
            block = best_model.clf.model.block
            final_mins = block.mins.detach().cpu().numpy()
            final_len = block.length.detach().cpu().numpy()
            init_mins = init_len = None
            if best_model.mins_tensor is not None:
                init_mins = best_model.mins_tensor.detach().cpu().numpy()
                init_len = best_model.lengths_tensor.detach().cpu().numpy()
            is_som = self.strategy_name == "som"
            fig = plot_boxes_figure(
                X_train=X_train,
                y_train=y_train,
                init_mins=init_mins,
                init_lengths=init_len,
                final_mins=final_mins,
                final_lengths=final_len,
                strategy=self.strategy_name,
                grid_shape=som_grid_shape(best_model.nboxes) if is_som else None,
                qe=getattr(best_model, "init_qe", None) if is_som else None,
                te=getattr(best_model, "init_te", None) if is_som else None,
                title=f"{dataset_name} | {self.strategy_name.upper()} | seed {seed} | label {label}",
            )
            if fig is None:
                return None
            path = self.figures_dir / f"{dataset_name}_{self.strategy_name}_seed{seed}_label{label}.png"
            fig.savefig(path, dpi=130, bbox_inches="tight")
            self.logger.info(f"[VIZ] figura salva em {path}", extra={"log_cat": "viz"})
            return fig
        except Exception as e:
            self.logger.info(f"[VIZ Warning] não foi possível gerar a figura: {e}")
            return None

    # Avaliação por dataset

    def evaluate_model_on_dataset(self, X, y, dataset_name, seed):
        """Avalia o modelo executando a busca em grid separada em 2 estágios."""
        results = []
        distinct_labels = np.unique(y)
        strategy_name = self.strategy_name

        has_checkpoint, cached_metrics = self.check_and_load_checkpoint(
            dataset_name, strategy_name, seed
        )
        if has_checkpoint:
            return cached_metrics

        for i, label in enumerate(distinct_labels):
            self.logger.info(
                f"Seed: {seed}, Dataset: {dataset_name}, Estratégia: {strategy_name.upper()}, Label: {label} ({i+1}/{len(distinct_labels)}).",
                extra={"log_cat": "run"},
            )

            y_current = binarize_labels(y, label)

            # Split por índice (os mesmos índices alimentam a impressão digital abaixo)
            idx = np.arange(len(y_current))
            idx_train, idx_test = train_test_split(
                idx,
                test_size=self.test_size,
                stratify=y_current,
                random_state=seed,
            )
            X_arr = np.asarray(X)
            X_train, X_test = X_arr[idx_train], X_arr[idx_test]
            y_train, y_test = y_current[idx_train], y_current[idx_test]

            if self.scale_data:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            # Reinicia as seeds e registra uma "impressão digital" dos dados de treino:
            # ela deve ser IDÊNTICA entre SOM, KMeans, DBSCAN, Random e Xavier.
            self._seed_everything(seed)
            fp = hashlib.md5(
                idx_train.tobytes() + np.ascontiguousarray(X_train).tobytes()
            ).hexdigest()[:8]
            self.logger.info(
                f"[SEED] seed={seed} | label={label} | fingerprint treino={fp}",
                extra={"log_cat": "seed"},
            )

            t0_search = time.time()
            best_params_combined = {}

            if self.init_hyperparameter_space and self.hypernn_hyperparameter_space:
                # --- ESTÁGIO 1: melhor init pela métrica (CE ou ) ---
                best_init_params, best_init_model = self.search_best_init(
                    X_train, y_train, seed
                )  # best_init_model já carrega mins_tensor / lengths_tensor

                self.logger.info(
                    f"Init travada: {best_init_model.mins_tensor.shape[0]} caixas",
                    extra={"log_cat": "init_lock"},
                )

                # --- ESTÁGIO 2: grid da HyperNN partindo dessas caixas ---
                hypernn_gridsearch = HyperparameterGridSearch(
                    best_init_model,
                    self.hypernn_hyperparameter_space,
                    num_cross_validation_folds=self.cross_validation_folds,
                    evaluation_metric=self.evaluation_metric,
                    random_state=seed,
                )
                best_model = hypernn_gridsearch.fit(X_train, y_train)
                best_params_combined = {
                    **best_init_params,
                    **hypernn_gridsearch.best_model_hyperparameters,
                }
            else:
                # random / xavier: grid único (não há estágio de init)
                grid_to_use = self.hypernn_hyperparameter_space or self.model_hyperparameter_space
                base_model = self.model.set_hyperparameters({"random_state": seed})
                gridsearch = HyperparameterGridSearch(
                    base_model,
                    grid_to_use,
                    num_cross_validation_folds=self.cross_validation_folds,
                    evaluation_metric=self.evaluation_metric,
                    random_state=seed,
                )
                best_model = gridsearch.fit(X_train, y_train)
                best_params_combined = gridsearch.best_model_hyperparameters

            grid_search_time = time.time() - t0_search

            # Treino final (mesma seed dos folds de validação).
            self._seed_everything(seed)
            start_train_time = time.time()
            best_model.fit(X_train, y_train)
            # épocas realmente executadas no treino final (confirma o efeito do patience)
            epochs_run = getattr(getattr(best_model, "clf", None), "epochs_run", None)
            training_time = time.time() - start_train_time

            # Avaliação em treino
            y_pred_train = best_model.predict(X_train)
            train_scores = self.compute_metric_scores(
                y_train, y_pred_train, prefix="train_"
            )

            # Avaliação em teste
            start_prediction_time = time.time()
            y_pred_test = best_model.predict(X_test)
            prediction_time = time.time() - start_prediction_time

            test_scores = self.compute_metric_scores(
                y_test, y_pred_test, prefix="test_"
            )
            initialization_time = getattr(best_model, "init_time", 0.0)
            final_nboxes = best_model.get_num_boxes()

            # Relatório do experimento
            current_result_report = self.create_experiment_report(
                dataset_name=dataset_name,
                dataset_label=label,
                best_model_hyperparameters=best_params_combined,
                training_scores=train_scores,
                test_scores=test_scores,
                hyperparameter_search_time=grid_search_time,
                training_time=training_time,
                initialization_time=initialization_time,
                prediction_time=prediction_time,
                final_nboxes=final_nboxes,
                seed=seed,
                epochs_run=epochs_run,
            )

            if (
                self.visualize
                and seed in self.visualize_seeds
                and (FIGURE_DATASETS is None or dataset_name in FIGURE_DATASETS)
            ):
                self._build_and_save_figure(
                    best_model, X_train, y_train, dataset_name, seed, label
                )

            self.logger.info(
                f"[RESULTADO] {dataset_name} | {strategy_name.upper()} | seed {seed} | "
                f"label {label} | F1 teste={test_scores['test_f1_score']:.4f} | "
                f"caixas={final_nboxes:.2f} | busca={grid_search_time:.1f}s | "
                f"treino={training_time:.1f}s | épocas={epochs_run}",
                extra={"log_cat": "result"},
            )

            results.append(current_result_report)

        self.save_checkpoint(dataset_name, strategy_name, seed, results)
        return results

    def evaluate(self, datasets):
        num_experiments = self.num_seeds * len(datasets)
        self.logger.info(f"Starting {num_experiments} experiments for strategy: {self.strategy_name.upper()}")
        self.results = []
        seeds = list(range(self.num_seeds))
        self.logger.info(f"Seeds utilizadas: {seeds}")
        with trange(num_experiments) as exp_pbar:
            exp_pbar.set_description(f"Experimentos [{self.strategy_name.upper()}]")
            for dt in datasets:
                for seed in seeds:
                    self.logger.info(
                        f"[EXECUÇÃO] Dataset: {dt.name} | Estratégia: {self.strategy_name.upper()} | Seed: {seed}",
                        extra={"log_cat": "exec"},
                    )
                    self._seed_everything(seed)
                    exp_pbar.set_postfix(dataset=dt.name, seed=seed)
                    results = self.evaluate_model_on_dataset(
                        X=dt.X, y=dt.y, dataset_name=dt.name, seed=seed
                    )
                    self.results.extend(results)
                    self.save_results()  # CSV atualizado a cada (dataset, seed)
                    exp_pbar.update(1)

        print(f"\n=== Resumo dos Resultados [{self.strategy_name.upper()}] ===")
        print(self.show_summary())
        self.save_results()
        print(f"[CSV] {self.results_filepath}")
        if ENABLE_FIGURES:
            print(f"[Figuras] {self.figures_dir}")

    def save_results(self):
        experiment_results = pd.DataFrame(self.results)
        # Colunas mais usadas primeiro; o dicionário longo do espaço de busca vai para o fim.
        front = [
            c for c in [
                "dataset", "dataset_label", "seed", "init_strategy", "final_nboxes",
                "test_f1_score", "train_f1_score", "test_precision", "test_recall",
                "hyperparameter_search_time", "training_time",
            ] if c in experiment_results.columns
        ]
        back = [c for c in ["hyperparameter_space"] if c in experiment_results.columns]
        rest = [c for c in experiment_results.columns if c not in front + back]
        experiment_results = experiment_results[front + rest + back]
        experiment_results.to_csv(self.results_filepath)
        return experiment_results

    def show_summary(self):
        experiment_results = pd.DataFrame(self.results)
        if experiment_results.empty:
            return "Nenhum resultado registrado."
        df_pivot = experiment_results.pivot_table(
            columns=["dataset"],
            values=[
                c for c in (
                    "final_nboxes", "test_f1_score", "training_time",
                    "prediction_time", "init_time", "epochs_run",
                ) if c in experiment_results.columns
            ],
        )
        return df_pivot
