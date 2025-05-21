import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random
from rich import print as print
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


class GeneticFeatureSelector:
    """
    Genetic algorithm to select a subset of features that maximizes predictive
    accuracy, minimizes the number of features, and reduces correlation among features.

    Each individual is represented as a bitstring of length n_features, where 1 indicates
    the feature is selected.
    """
    def __init__(self,
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 population_size: int = 50,
                 generations: int = 30,
                 crossover_prob: float = 0.7,
                 mutation_prob: float = 0.2,
                 accuracy_weight: float = 0.6,
                 feature_count_weight: float = 0.2,
                 correlation_weight: float = 0.2,
                 random_state: int = None):
        self.X_train = X_train.reset_index(drop=True)
        self.y_train = y_train.reset_index(drop=True)
        self.n_features = X_train.shape[1]
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.accuracy_weight = accuracy_weight
        self.feature_count_weight = feature_count_weight
        self.correlation_weight = correlation_weight
        self.random_state = random_state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        # Precompute correlation matrix and feature-target correlations, filling NaNs
        self.feature_corr_matrix = X_train.corr().abs().fillna(0)
        self.feat_target_corr = X_train.apply(lambda f: abs(np.corrcoef(f, y_train)[0,1]))
        self.feat_target_corr = self.feat_target_corr.fillna(0)

    def _initialize_population(self):
        # Random initial population of bitstrings
        return [np.random.choice([0,1], size=self.n_features).astype(int) for _ in range(self.population_size)]

    def _fitness(self, individual: np.ndarray) -> float:
        # If no feature selected, return minimal fitness
        if individual.sum() == 0:
            return -np.inf
        # Subset data
        selected_idx = np.where(individual == 1)[0]
        X_tr = self.X_train.iloc[:, selected_idx]

        # 1. Accuracy term: train fast classifier
        clf = LogisticRegression(solver='liblinear', max_iter=200)
        # Use cross-validation on training data only
        cv_score = 0
        try:
            cv_score = np.mean(cross_val_score(clf, X_tr, self.y_train, cv=5))
        except Exception:
            # In case of singular matrix or other issues
            cv_score = 0.0

        # 2. Feature count term: ratio of selected features
        feat_ratio = individual.sum() / self.n_features

        # 3. Correlation term: average correlation among selected features + feature-target corr
        if len(selected_idx) > 1:
            sub_corr = self.feature_corr_matrix.iloc[selected_idx, selected_idx]
            # sum off-diagonal entries
            corr_ff = (sub_corr.values.sum() - len(selected_idx)) / (len(selected_idx)*(len(selected_idx)-1))
        else:
            corr_ff = 0.0
        corr_ft = self.feat_target_corr.iloc[selected_idx].mean() if len(selected_idx) > 0 else 0.0
        corr_term = corr_ff + corr_ft
        if np.isnan(corr_term):
            corr_term = 0.0

        # Fitness: weighted sum (higher is better)
        fitness = (
            self.accuracy_weight * cv_score
            - self.feature_count_weight * feat_ratio
            - self.correlation_weight * corr_term
        )
        return fitness

    def _select_parents(self, population, fitnesses):
        # Tournament selection
        parents = []
        for _ in range(self.population_size):
            i, j = random.sample(range(self.population_size), 2)
            winner = population[i] if fitnesses[i] > fitnesses[j] else population[j]
            parents.append(winner.copy())
        return parents

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple:
        if random.random() < self.crossover_prob:
            point = random.randint(1, self.n_features-1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()

    def _mutate(self, individual: np.ndarray):
        for i in range(self.n_features):
            if random.random() < self.mutation_prob:
                individual[i] = 1 - individual[i]
        return individual

    def run(self):
        # Main GA loop
        population = self._initialize_population()
        best_individual = None
        best_fitness = -np.inf
        
        # Track fitness over generations
        self.generation_history = list(range(1, self.generations+1))
        self.best_fitness_history = []
        self.avg_fitness_history = []

        for gen in range(1, self.generations+1):
            # Evaluate fitness
            fitnesses = np.array([self._fitness(ind) for ind in population])

            # Track best
            # filter out -inf
            valid_idx = np.where(fitnesses != -np.inf)[0]
            if valid_idx.size > 0:
                gen_best_idx = valid_idx[np.argmax(fitnesses[valid_idx])]
                gen_best_fit = fitnesses[gen_best_idx]
            else:
                gen_best_idx = None
                gen_best_fit = -np.inf
            gen_avg_fit = np.mean(fitnesses[np.isfinite(fitnesses)]) if np.any(np.isfinite(fitnesses)) else -np.inf
            
            # Store fitness values for plotting
            self.best_fitness_history.append(gen_best_fit)
            self.avg_fitness_history.append(gen_avg_fit)

            if gen_best_fit > best_fitness and gen_best_idx is not None:
                best_fitness = gen_best_fit
                best_individual = population[gen_best_idx].copy()

            # Logging every 10 generations (and first)
            if gen % 10 == 0 or gen == 1:
                print(f"[bold cyan]Generation {gen}/{self.generations}: Avg Fitness = {gen_avg_fit:.4f}, Best Fitness = {gen_best_fit:.4f}[/bold cyan]")

            # Create next generation
            parents = self._select_parents(population, fitnesses)
            next_pop = []
            for i in range(0, self.population_size, 2):
                p1 = parents[i]
                p2 = parents[i+1] if i+1 < self.population_size else parents[0]
                c1, c2 = self._crossover(p1, p2)
                next_pop.append(self._mutate(c1))
                next_pop.append(self._mutate(c2))
            population = next_pop[:self.population_size]

        # Final report
        if best_individual is None:
            print("GA completed but no valid individual found.")
            return []
        print(f"[bold green]GA completed. Best Fitness = {best_fitness:.4f}, Features selected = {best_individual.sum()}[/bold green]")

        # Return list of selected feature names
        return self.X_train.columns[np.where(best_individual == 1)[0]].tolist()
        
    def plot_fitness_history(self, figsize=(10, 6), save_path=None):
        """
        Plot the fitness history across generations.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height) in inches
        save_path : str, optional
            If provided, saves the plot to this path
        """
        plt.figure(figsize=figsize)
        plt.plot(self.generation_history, self.best_fitness_history, 'b-', label='Best Fitness')
        plt.plot(self.generation_history, self.avg_fitness_history, 'r--', label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness Value')
        plt.title('Fitness Evolution Over Generations')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()