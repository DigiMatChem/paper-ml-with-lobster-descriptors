import os
import json
import shutil
import subprocess
import pandas as pd
import numpy as np
import warnings
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import get_scorer
from sklearn import metrics
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate
from deap import base, creator, tools, algorithms
from sissopp.postprocess.load_models import load_model
from tqdm.autonotebook import tqdm
from feature_engine.selection import SmartCorrelatedSelection, DropConstantFeatures
from arfs.feature_selection import GrootCV

import logging

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_relevant_features(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    const_feat_tol: float = 0.95,
    collinearity_tol: float = 0.9,
    grootcv_nfolds: int = 5,
    grootcv_n_iter: int = 20,
    grootcv_lgbm_objective: str = "mae",
    **pipeline_kwargs,
) -> tuple[Pipeline, pd.DataFrame]:
    """
    Build and apply a feature selection pipeline to remove correlated and irrelevant features.

    The pipeline applies the following steps:
    1. **DropConstantFeatures**: Removes features with low variance (near-constant).
    2. **SmartCorrelatedSelection**: Removes highly correlated features based on Pearson correlation.
    3. **GrootCV**: Selects relevant features using cross-validation with a LightGBM-based model.

    Parameters
    ----------
    X_train : pd.DataFrame | np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Target values corresponding to `X_train`. 1D numpy array
    const_feat_tol : float, default=0.95
        Threshold for removing near-constant features. A feature is removed if
        a single value accounts for at least this proportion of observations.
    collinearity_tol : float, default=0.9
        Correlation threshold for removing highly correlated features.
    grootcv_nfolds : int, default=5
        Number of folds for cross-validation in `GrootCV`.
    grootcv_n_iter : int, default=20
        Number of iterations for feature selection in `GrootCV`.
    grootcv_lgbm_objective : str, default="mae"
        Objective function for the LightGBM model inside `GrootCV`.
    **pipeline_kwargs : dict
        Additional keyword arguments passed to the underlying `Pipeline`.

    Returns
    -------
    pipeline, pd.DataFrame
        Pipeline instance and a transformed training set with only the relevant features retained.

    """

    pipeline = Pipeline(
        [
            ("zero_variance", DropConstantFeatures(tol=const_feat_tol)),
            (
                "collinearity",
                SmartCorrelatedSelection(
                    threshold=collinearity_tol,
                    cv=5,
                    method="pearson",
                    selection_method="variance",
                ),
            ),
            (
                "all_rel_feats",
                GrootCV(
                    objective=grootcv_lgbm_objective,
                    n_jobs=8,
                    n_iter=grootcv_n_iter,
                    n_folds=grootcv_nfolds,
                    rf=False,
                ),
            ),
        ],
        verbose=True,
    )

    if pipeline_kwargs is not None and isinstance(pipeline_kwargs, dict):
        pipeline.set_params(**pipeline_kwargs)

    return pipeline, pipeline.fit_transform(X_train, y_train)


# ----------SISSO_GAFeatureSelector----------


def init_valid_individual(icls, n_features: int, num_selected_features: int):
    """
    Initialize an individual with a fixed number of selected features.
    Ensures that exactly `num_selected_features` in a binary vector are switched to 1.

    Parameters
    ----------
    icls : Individual creator class
        The individual class to instantiate.
    n_features : int
        Total number of features.
    num_selected_features : int
        Number of features to select (element of binary vector set to 1).
    """
    individual = [0] * n_features
    selected_indices = np.random.choice(
        range(n_features), num_selected_features, replace=False
    )
    for idx in selected_indices:
        individual[idx] = 1
    return icls(individual)


def mutate_and_fix(individual: np.ndarray, indpb: float, num_selected_features: int):
    """
    Mutate an individual and ensure it has exactly `num_selected_features` selected.

    Parameters
    ----------
    individual : np.ndarray
        The individual to mutate.
    indpb : float
        Probability for individual bits to be flipped.
    num_selected_features : int
        Number of features to select (element of binary vector set to 1).
    """
    tools.mutFlipBit(individual, indpb)
    while sum(individual) > num_selected_features:
        ones_indices = [i for i, bit in enumerate(individual) if bit == 1]
        individual[np.random.choice(ones_indices)] = 0
    while sum(individual) < num_selected_features:
        zeros_indices = [i for i, bit in enumerate(individual) if bit == 0]
        individual[np.random.choice(zeros_indices)] = 1
    return (individual,)


def hamming_distance(ind1: np.ndarray, ind2: np.ndarray) -> int:
    """
    Calculate the Hamming distance between two individuals.

    Parameters
    ----------
    ind1 : np.ndarray
        First individual.
    ind2 : np.ndarray
        Second individual.

    Returns
    -------
    int
        Hamming distance between the two individuals.
    """
    return sum(x != y for x, y in zip(ind1, ind2))


def is_diverse(
    individual: np.ndarray,
    population: list[np.ndarray],
    min_distance: int = 5,
) -> bool:
    """
    Check if an individual is diverse enough from a population based on Hamming distance.

    Parameters
    ----------
    individual : np.ndarray
        The individual to check.
    population : list of np.ndarray
        The population to compare against.
    min_distance : int, default=5
        Minimum Hamming distance required for diversity.

    Returns
    -------
    bool
        True if the individual is diverse enough, False otherwise.
    """
    return all(
        hamming_distance(individual, other) >= min_distance for other in population
    )


def population_entropy(population: list[np.ndarray]) -> float:
    """
    Calculate the entropy of a population of binary individuals.

    Parameters
    ----------
    population : list of np.ndarray
        The population of individuals.

    Returns
    -------
    float
        Entropy of the population
    """

    pop_array = np.array(population)
    p1 = np.mean(pop_array, axis=0)
    p1 = np.clip(p1, 1e-6, 1 - 1e-6)
    entropy = -np.mean(p1 * np.log2(p1) + (1 - p1) * np.log2(1 - p1))
    return entropy


def mixed_selection(population: list[np.ndarray], k: int) -> list[np.ndarray]:
    """
    Mixed selection strategy: combines elitism and random selection.

    Parameters
    ----------
    population : list of np.ndarray
        The population of individuals.
    k : int
        Number of individuals to select.

    Returns
    -------
    list of np.ndarray
        Selected individuals.

    """
    elite_count = int(0.2 * k)
    random_count = k - elite_count
    elites = tools.selBest(population, elite_count)
    randoms = tools.selRandom(population, random_count)
    return elites + randoms


def cxTwoPointAndFix(
    ind1: np.ndarray,
    ind2: np.ndarray,
    num_selected_features: int,
):
    """
    Crossover two individuals and fix them to have a specific number of selected features.

    Parameters
    ----------
    ind1 : np.ndarray
        First individual.
    ind2 : np.ndarray
        Second individual.
    num_selected_features : int
        Number of features to select (element of binary vector set to 1).

    Returns
    -------
    tuple of np.ndarray
        The two modified individuals after crossover and fixing.
    """
    tools.cxTwoPoint(ind1, ind2)
    mutate_and_fix(ind1, indpb=0, num_selected_features=num_selected_features)
    mutate_and_fix(ind2, indpb=0, num_selected_features=num_selected_features)
    return ind1, ind2


class GAFeatureSelector:
    """Genetic Algorithm Feature Selector using DEAP.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : pd.Series or np.ndarray
        Target values.
    model : sklearn-like estimator
        Model to evaluate feature subsets.
    num_features : int, default=25
        Number of features to select.
    population_size : int, default=50
        Size of the GA population.
    generations : int, default=100
        Number of generations to run.
    feature_names : list of str, optional
        Names of the features. If None, indices will be used.
    cxpb : float, default=0.5
        Crossover probability.
    mutpb : float, default=0.2
        Mutation probability.
    early_stop_patience : int, default=5
        Generations to wait for improvement before stopping.
    mutation_boost_threshold : int, default=3
        Generations without improvement to trigger mutation boost.
    mutation_boost_factor : float, default=2.0
        Factor to increase mutation probability when boosting.
    cv : int, default=5
        Number of cross-validation folds.
    scoring : str, default="r2"
        Scoring metric for evaluation.
    min_diversity : int, default=5
        Minimum Hamming distance for diversity.
    entropy_threshold : float, default=0.3
        Entropy threshold to trigger mutation boost.
    X_test : pd.DataFrame or np.ndarray, optional
        Test feature matrix for diagnostic evaluation.
    y_test : pd.Series or np.ndarray, optional
        Test target values for diagnostic evaluation.
    test_scoring : str or callable, optional
        Scoring metric for test evaluation. Defaults to `scoring`.
    n_jobs : int, default=1
        Number of parallel jobs for evaluation.
    return_train_score : bool, default=True
        Whether to return training scores in cross-validation.
    error_score : float, default=np.nan
        Value to assign to failed evaluations.
    sissopp_binary_path : str, optional
        Path to SISSO++ binary for model evaluation.
    mpi_tasks : int, default=8
        Number of MPI tasks for SISSO++.
    sissopp_inputs : dict, optional
        Additional inputs for SISSO++.
    """

    def __init__(
        self,
        X: pd.DataFrame or np.ndarray,
        y: pd.Series or np.ndarray,
        model: any,
        num_features: int = 25,
        population_size: int = 50,
        generations: int = 100,
        feature_names: list[str] | None = None,
        cxpb: float = 0.5,
        mutpb: float = 0.2,
        early_stop_patience: int = 5,
        mutation_boost_threshold: int = 3,
        mutation_boost_factor: float = 2.0,
        cv: int = 5,
        scoring: str = "r2",
        min_diversity: int = 5,
        entropy_threshold: float = 0.3,
        X_test: pd.DataFrame | np.ndarray | None = None,
        y_test: pd.DataFrame | np.ndarray | None = None,
        test_scoring: str | None = None,
        n_jobs: int = 1,
        return_train_score: bool = True,
        error_score=np.nan,
        sissopp_binary_path: str = None,
        mpi_tasks: int = 8,
        sissopp_inputs: dict | None = None,
    ):
        self.X = X
        self.y = y
        self.model = model
        self.num_features = num_features
        self.population_size = population_size
        self.generations = generations
        self.cxpb = cxpb
        self.base_mutpb = mutpb
        self.early_stop_patience = early_stop_patience
        self.mutation_boost_threshold = mutation_boost_threshold
        self.mutation_boost_factor = mutation_boost_factor
        self.cv = cv
        self.scoring = scoring
        self.n_features = X.shape[1]
        self.min_diversity = min_diversity
        self.entropy_threshold = entropy_threshold
        self.n_jobs = n_jobs
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.sissopp_binary_path = sissopp_binary_path
        self.mpi_tasks = mpi_tasks
        self.fitness_history = []
        self.sissopp_inputs = sissopp_inputs
        self.operator_usage_counts = defaultdict(int)
        self.feature_usage_counts = defaultdict(int)

        self.test_score_history = []

        self.X_test = X_test
        self.y_test = y_test
        self.test_scoring = test_scoring or scoring

        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(self.X, pd.DataFrame):
            self.feature_names = list(self.X.columns)
        else:
            self.feature_names = [f"feat_{i}" for i in range(self.n_features)]

        self._setup_deap()

    def _setup_deap(self):
        """Setup DEAP toolbox and creator."""

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register(
            "individual",
            init_valid_individual,
            creator.Individual,
            self.n_features,
            self.num_features,
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register(
            "mate", cxTwoPointAndFix, num_selected_features=self.num_features
        )
        self.toolbox.register(
            "mutate",
            mutate_and_fix,
            indpb=0.01,
            num_selected_features=self.num_features,
        )
        self.toolbox.register("select", mixed_selection)

    def evaluate_individual(self, individual):
        """Evaluate an individual using cross-validation."""
        selected_indices = [i for i, bit in enumerate(individual) if bit == 1]
        X_selected = (
            self.X.iloc[:, selected_indices]
            if isinstance(self.X, pd.DataFrame)
            else self.X[:, selected_indices]
        )

        if isinstance(self.y, np.ndarray):
            y_df = pd.DataFrame(
                index=X_selected.index,
                columns=[self.sissopp_inputs["property_key"]],
                data=self.y,
            )
        else:
            y_df = self.y

        df = pd.concat([X_selected, y_df], axis=1)

        kf = KFold(random_state=18012019, n_splits=self.cv, shuffle=True)

        if self.model is None and self.sissopp_binary_path:
            kf_splits = kf.split(X_selected)

            base_path = os.getcwd()

            cv_scores = []

            for i, (train, test) in enumerate(kf_splits):
                _ = X_selected.iloc[train, :]
                _ = self.y[train]
                X_test = X_selected.iloc[test, :]
                y_test = self.y[test]

                cv_dir = f"cv_{i+1}"
                os.makedirs(cv_dir, exist_ok=True)
                os.chdir(cv_dir)

                inputs_cv = self.sissopp_inputs.copy()
                inputs_cv["leave_out_inds"] = [str(i) for i in test]
                with open("sisso.json", "w") as f:
                    json.dump(inputs_cv, f, indent=4)
                # json.dumps(inputs, indent=4)
                df.to_csv("data.csv")

                cmd = (
                    f"OMP_NUM_THREADS=64 OMP_PLACES=cores mpirun -n {self.mpi_tasks} "
                    + self.sissopp_binary_path
                )

                _ = subprocess.run(
                    cmd, shell=True, check=True, capture_output=True, text=True
                )
                max_dim = inputs_cv["desc_dim"]

                # Handle inf prediction models with nan scores
                try:
                    model = load_model(
                        f"{os.getcwd()}/models/train_dim_{max_dim}_model_0.dat"
                    )

                    y_pred = model.eval_many(X_test.values)

                    self.parse_postfix(model=model, selected_indices=selected_indices)

                    score = getattr(metrics, self.scoring.split("neg_")[-1])(
                        y_test, y_pred
                    )
                except Exception:
                    score = np.nan

                cv_scores.append(score * -1 if "neg_" in self.scoring else score)

                os.chdir(base_path)

        else:

            model_clone = clone(self.model)

            cv_scores = cross_validate(
                model_clone,
                X_selected,
                self.y,
                cv=kf,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                error_score=self.error_score,
                return_train_score=self.return_train_score,
            )

        base_path = os.getcwd()
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)

            # Remove all cv models
            if os.path.isdir(item_path) and (
                item.startswith("SISSO_") or item.startswith("cv")
            ):
                shutil.rmtree(item_path)

        val_scores = (
            cv_scores["test_score"] if isinstance(cv_scores, dict) else cv_scores
        )

        if np.isnan(val_scores).any():
            return (-1e10,)
        else:
            return (np.mean(val_scores),)

    def evaluate_on_test(self, individual):
        """Evaluate the best individual on the test set."""
        if self.X_test is None or self.y_test is None:
            return None

        selected_indices = [i for i, bit in enumerate(individual) if bit == 1]
        X_train_sel = (
            self.X.iloc[:, selected_indices]
            if isinstance(self.X, pd.DataFrame)
            else self.X[:, selected_indices]
        )
        X_test_sel = (
            self.X_test.iloc[:, selected_indices]
            if isinstance(self.X_test, pd.DataFrame)
            else self.X_test[:, selected_indices]
        )

        model_clone = clone(self.model)

        base_path = os.getcwd()
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)

            # Remove test set models
            if os.path.isdir(item_path) and (
                item.startswith("SISSO_") or item.startswith("cv")
            ):
                shutil.rmtree(item_path)

        model_clone.fit(X_train_sel, self.y)
        y_pred = model_clone.predict(X_test_sel)

        if isinstance(self.test_scoring, str):
            scorer = get_scorer(self.test_scoring)
            score = scorer(model_clone, X_test_sel, self.y_test)
        else:
            score = self.test_scoring(self.y_test, y_pred)

        return score

    def differential_evolution_ga(self, pop, F=0.5, mutpb=0.1):
        """Differential Evolution GA offspring generation.

        Parameters
        ----------
        pop : list of Individuals
            Current population.
        F : float, default=0.5
            Differential weight.
        mutpb : float, default=0.1
            Mutation probability.
        Returns
        -------
        list of np.ndarray
            New list of population after DE operation.
        """
        new_pop = []
        for target in pop:
            a, b, c = random.sample(pop, 3)
            trial = creator.Individual([0] * len(target))
            for i in range(len(target)):
                if random.random() < F:
                    trial[i] = int(a[i] ^ (b[i] ^ c[i]))
                else:
                    trial[i] = target[i]
            (trial,) = mutate_and_fix(
                trial, indpb=mutpb, num_selected_features=sum(target)
            )
            trial.fitness.values = self.toolbox.evaluate(trial)

            if not target.fitness.valid:
                target.fitness.values = self.toolbox.evaluate(target)

            new_pop.append(
                trial if trial.fitness.values[0] > target.fitness.values[0] else target
            )
        return new_pop

    def parse_postfix(self, model, selected_indices):
        """Extracts features and operators from postfix expression string.

        Parameters
        ----------
        model : SISSO model object
            The SISSO model containing features.
        selected_indices : list of int
            Indices of selected features.

        """
        for feat in model.feats:
            expression = feat.postfix_expr.split("|")
            for exp in expression:
                if exp.isdigit():
                    self.feature_usage_counts[selected_indices[int(exp)]] += 1
                else:
                    self.operator_usage_counts[exp] += 1

    def run(self, plot=True, strategy="standard"):
        """Run the Genetic Algorithm for feature selection.

        Parameters
        ----------
        plot : bool, default=True
            Whether to plot the fitness history.
        strategy : str, default="standard"
            Offspring generation strategy: "standard" or "de" (differential evolution).

        Returns
        -------
        list of str
            Selected feature names.
        """
        pop = self.toolbox.population(n=self.population_size)
        hall_of_fame = tools.HallOfFame(1)
        no_improvement_counter = 0
        best_fitness = -np.inf

        for gen in range(self.generations):
            entropy = population_entropy(pop)

            if (
                no_improvement_counter >= self.mutation_boost_threshold
                or entropy < self.entropy_threshold
            ):
                mutpb = self.base_mutpb * self.mutation_boost_factor
                indpb = 0.05
            else:
                mutpb = self.base_mutpb
                indpb = 0.01

            self.toolbox.unregister("mutate")
            self.toolbox.register(
                "mutate",
                mutate_and_fix,
                indpb=indpb,
                num_selected_features=self.num_features,
            )

            # Choose offspring generation strategy
            if strategy == "standard":
                offspring = algorithms.varAnd(pop, self.toolbox, self.cxpb, mutpb)
            elif strategy == "de":
                offspring = self.differential_evolution_ga(pop, F=0.7, mutpb=0.05)
            else:
                raise ValueError("Unknown strategy")

            for ind in tqdm(offspring, desc=f"Evaluating Gen {gen}"):
                ind.fitness.values = self.toolbox.evaluate(ind)

            # Keep only diverse individuals
            diverse_offspring = []
            for ind in offspring:
                if is_diverse(
                    ind, diverse_offspring + pop, min_distance=self.min_diversity
                ):
                    diverse_offspring.append(ind)
            if len(diverse_offspring) < len(pop):
                needed = len(pop) - len(diverse_offspring)
                diverse_offspring.extend(self.toolbox.select(offspring, k=needed))

            pop[:] = diverse_offspring
            hall_of_fame.update(pop)

            current_best = max(ind.fitness.values[0] for ind in pop)
            current_avg = np.mean([ind.fitness.values[0] for ind in pop])
            self.fitness_history.append((current_best, current_avg))

            test_score = None
            if self.X_test is not None:
                test_score = self.evaluate_on_test(hall_of_fame[0])
                self.test_score_history.append(test_score)

            if current_best > best_fitness + 1e-8:
                best_fitness = current_best
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            if no_improvement_counter >= self.early_stop_patience:
                print(f"Converged at generation {gen}")
                break

            logging.info(
                f"Gen {gen}: Best Fitness ({self.scoring}) = {current_best:.4f}, Entropy = {entropy:.4f}, Mutation = {mutpb:.2f}"
                + (f", Test Score = {test_score:.4f}" if test_score is not None else "")
            )
            logging.info(f"All time best : {best_fitness:.4f}")

        self.best_individual = hall_of_fame[0]
        self.selected_features = [
            self.feature_names[i]
            for i, bit in enumerate(self.best_individual)
            if bit == 1
        ]

        if plot:
            self.plot_fitness(strategy)

        return self.selected_features

    def plot_fitness(self, strategy) -> None:
        """Plot the fitness history of the GA.

        Parameters
        ----------
        strategy : str
            Offspring generation strategy used.

        """
        gens = range(len(self.fitness_history))
        best_fitness_vals = [f[0] for f in self.fitness_history]

        plt.plot(gens, best_fitness_vals, label=f"Best Fitness ({self.scoring})")

        if self.test_score_history:
            plt.plot(
                gens,
                self.test_score_history,
                label=f"Test {self.test_scoring} (diagnostic)",
                linestyle="--",
            )

        plt.xlabel("Generation")
        plt.ylabel(f"{self.scoring} Score")
        plt.legend()
        plt.title(f"GA Convergence Plot : {strategy.capitalize()}")
        plt.savefig(f"GA_result_{strategy.capitalize()}.png")
        plt.show()
        plt.close()

    def get_selected_feature_indices(self) -> list[int]:
        """
        Get indices of selected features from the best individual.

        Returns
        -------
        list of int
            Indices of selected features.
        """
        return [i for i, bit in enumerate(self.best_individual) if bit == 1]
