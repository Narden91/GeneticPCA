import numpy as np
import pandas as pd
import random
from rich import print as rprint # Use rprint to avoid conflict if 'print' is a variable
from rich.console import Console
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Local import (assuming clustering_utils.py is in the same directory or accessible)
try:
    from clustering import perform_dbscan, evaluate_clustering
except ImportError:
    raise ImportError("clustering module not found. Ensure clustering_utils.py is accessible.")


console = Console()

class GeneticPCAOptimizer:
    def __init__(self,
                 X_scaled: pd.DataFrame,
                 dbscan_eps: float,
                 dbscan_min_samples: int,
                 population_size: int = 20,
                 generations: int = 10,
                 crossover_prob: float = 0.7,
                 mutation_prob: float = 0.2,
                 silhouette_weight: float = 0.8,
                 components_penalty_weight: float = 0.2,
                 min_pca_components_ratio: float = 0.0, 
                 max_pca_components_ratio: float = 0.5, 
                 random_state: int = None):

        if X_scaled is None or X_scaled.empty:
            raise ValueError("Input data X_scaled cannot be None or empty for GeneticPCAOptimizer.")
        
        self.X_scaled = X_scaled
        self.original_n_features = X_scaled.shape[1]
        
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        
        self.silhouette_weight = silhouette_weight
        self.components_penalty_weight = components_penalty_weight

        # Determine min and max number of components based on ratios
        self.min_n_components = 0 # Always allow 'no PCA' option (controlled by min_pca_components_ratio if it's > 0)
        
        # If min_pca_components_ratio > 0, it could set a floor higher than 0.
        # Let's refine this slightly for clarity based on the config:
        min_from_ratio_config = int(self.original_n_features * min_pca_components_ratio)
        self.min_n_components = max(0, min_from_ratio_config) # Ensures min_n_components is at least 0

        # Max components should not exceed original_n_features - 1 (if original_n_features > 1)
        # and should respect max_pca_components_ratio
        max_from_ratio_config = int(self.original_n_features * max_pca_components_ratio)
        upper_bound_for_max = self.original_n_features -1 if self.original_n_features > 1 else (1 if self.original_n_features == 1 else 0)

        self.max_n_components = min(max_from_ratio_config, upper_bound_for_max)
        
        if self.max_n_components < self.min_n_components and self.original_n_features > 0:
             self.max_n_components = self.min_n_components 
        if self.original_n_features <=1 :
            self.min_n_components = 0
            self.max_n_components = 0 # No PCA if 0 or 1 original feature
        
        # Ensure min_n_components is not greater than max_n_components after all calculations
        if self.min_n_components > self.max_n_components :
            self.min_n_components = self.max_n_components # Or handle as an error/warning

        console.print(f"[GA] Original features: {self.original_n_features}. PCA components search range: [{self.min_n_components}, {self.max_n_components}]")


        self.random_state = random_state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.generation_history = list(range(1, self.generations + 1))


    def _initialize_population(self):
        """ Initialize population with random integers representing n_components for PCA.
            0 means no PCA.
        """
        if self.min_n_components == self.max_n_components: # if range is just a single value (e.g. 0 to 0)
            return [self.min_n_components] * self.population_size
        return [random.randint(self.min_n_components, self.max_n_components) for _ in range(self.population_size)]

    def _fitness(self, n_pca_components: int) -> float:
        """
        Fitness function:
        1. Applies PCA if n_pca_components > 0.
        2. Runs DBSCAN.
        3. Calculates Silhouette Score.
        4. Fitness = (silhouette_weight * silhouette) - (components_penalty_weight * component_ratio)
           Silhouette is in [-1, 1]. component_ratio is in [0, 1].
           Aims to maximize silhouette while penalizing too many components.
        """
        if self.X_scaled.shape[0] < self.dbscan_min_samples: # Not enough samples for DBSCAN
            return -float('inf')

        X_current = self.X_scaled.copy()
        
        if n_pca_components > 0 and self.original_n_features > 1:
            if n_pca_components >= self.original_n_features: # Should not happen with proper init/mutation
                n_pca_components = self.original_n_features -1 
            
            if n_pca_components <= 0: # Fallback if bad value
                pass # No PCA
            else:
                try:
                    pca = PCA(n_components=n_pca_components, random_state=self.random_state)
                    X_current_array = pca.fit_transform(X_current)
                    X_current = pd.DataFrame(X_current_array, index=X_current.index)
                except Exception as e:
                    return -float('inf')
        
        # Perform DBSCAN and evaluate silently within GA
        labels = perform_dbscan(X_current, eps=self.dbscan_eps, min_samples=self.dbscan_min_samples, verbose=0) 
        
        eval_metrics = evaluate_clustering(X_current, labels, verbose=0) 
        silhouette = eval_metrics.get('silhouette_score')
        n_clusters = eval_metrics.get('n_clusters', 0)

        if silhouette is None or n_clusters < 2: # Adjusted condition: n_clusters must be >= 2 for silhouette
            silhouette = -1 
        if n_clusters == 1 and eval_metrics.get('n_noise_points', len(labels)) < len(labels) : # Only one actual cluster found
            silhouette = -0.5 # Penalize if only one cluster and not all noise

        # Penalty for number of components (normalized)
        if self.original_n_features > 1:
            component_ratio = n_pca_components / (self.original_n_features -1) if n_pca_components > 0 else 0
        elif self.original_n_features == 1 and n_pca_components > 0:
            component_ratio = 1 
        else: 
            component_ratio = 0

        fitness_val = (self.silhouette_weight * silhouette) - \
                      (self.components_penalty_weight * component_ratio)
        
        return fitness_val

    def _select_parents(self, population, fitnesses):
        # Tournament selection
        parents = []
        for _ in range(self.population_size):
            k = min(2, len(population)) 
            if k < 2 and len(population) > 0: # if only one individual, it's the winner
                parents.append(population[0])
                continue
            elif len(population) == 0: # Should not happen
                return []

            idx1, idx2 = random.sample(range(len(population)), k)
            winner_idx = idx1 if fitnesses[idx1] > fitnesses[idx2] else idx2
            parents.append(population[winner_idx]) # appending the value, not a copy
        return parents

    def _crossover(self, parent1: int, parent2: int) -> tuple:
        if random.random() < self.crossover_prob:
            child1 = int((parent1 + parent2) / 2)
            child2 = random.choice([parent1, parent2]) 
            
            # Ensure children are within bounds
            child1 = max(self.min_n_components, min(child1, self.max_n_components))
            child2 = max(self.min_n_components, min(child2, self.max_n_components))
            return child1, child2
        return parent1, parent2 # No crossover, return parents

    def _mutate(self, individual: int) -> int:
        if random.random() < self.mutation_prob:
            # Small random change, or flip to/from 0 (no PCA)
            if self.min_n_components == self.max_n_components: # No range to mutate
                return individual

            if random.random() < 0.3 and self.min_n_components == 0: # Chance to flip to/from "no PCA"
                individual = 0 if individual > 0 else random.randint(max(1, self.min_n_components), self.max_n_components) # if it was 0, pick a random valid >0
            else:
                # Add/subtract a small random number of components
                mutation_val = random.randint(-max(1,int(self.original_n_features * 0.1)), max(1,int(self.original_n_features * 0.1)))
                individual += mutation_val
            
            individual = max(self.min_n_components, min(individual, self.max_n_components))
        return individual

    def run(self):
        if self.original_n_features == 0:
            console.print("[GA] No features in input data. GA cannot run. Defaulting to no PCA (0 components).")
            return 0 # No PCA

        population = self._initialize_population()
        best_individual = population[0] # Initialize with first
        best_fitness = -float('inf')

        for gen in range(1, self.generations + 1):
            fitnesses = [self._fitness(ind) for ind in population]
            
            current_gen_best_idx = np.argmax(fitnesses)
            current_gen_best_fitness = fitnesses[current_gen_best_idx]
            current_gen_avg_fitness = np.mean([f for f in fitnesses if f != -float('inf')] or [-float('inf')])


            self.best_fitness_history.append(current_gen_best_fitness)
            self.avg_fitness_history.append(current_gen_avg_fitness)

            if current_gen_best_fitness > best_fitness:
                best_fitness = current_gen_best_fitness
                best_individual = population[current_gen_best_idx]

            if gen % (self.generations // 5 or 1) == 0 or gen == 1: # Log periodically
                console.print(f"[GA Gen {gen}/{self.generations}] Avg Fitness: {current_gen_avg_fitness:.4f}, Best Fitness: {best_fitness:.4f} (Best n_comp: {best_individual})")

            if len(set(fitnesses)) == 1 and current_gen_avg_fitness == -float('inf') and gen > 3:
                console.print("[GA] All individuals have very low fitness. Stopping early. Consider DBSCAN params or fitness function.")
                break # Early stopping if no progress or all bad

            parents = self._select_parents(population, fitnesses)
            if not parents: # handles empty population if select_parents fails
                 console.print("[GA] No parents selected, re-initializing population for next generation.")
                 population = self._initialize_population()
                 continue


            next_population = []
            for i in range(0, self.population_size, 2):
                parent1 = parents[i]
                # Ensure parent2 index is valid
                parent2_idx = i + 1 if (i + 1) < len(parents) else 0 
                parent2 = parents[parent2_idx]
                
                child1, child2 = self._crossover(parent1, parent2)
                next_population.append(self._mutate(child1))
                if len(next_population) < self.population_size : 
                    next_population.append(self._mutate(child2))
            
            population = next_population[:self.population_size] 

        console.print(f"[GA] Optimization Complete. Best n_components: [bold green]{best_individual}[/bold green], Best Fitness: {best_fitness:.4f}")
        return best_individual

    def plot_fitness_history(self, save_path=None):
        if not self.best_fitness_history or not self.avg_fitness_history:
            console.print("[GA] No fitness history to plot.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.generation_history[:len(self.best_fitness_history)], self.best_fitness_history, 'b-', label='Best Fitness')
        plt.plot(self.generation_history[:len(self.avg_fitness_history)], self.avg_fitness_history, 'r--', label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness Value')
        plt.title('GA Fitness Evolution for PCA Component Selection')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            console.print(f"[GA] Fitness plot saved to {save_path}")
        plt.show()