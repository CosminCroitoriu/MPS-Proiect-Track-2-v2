"""
Model Training with Genetic Algorithm Feature Selection
Uses Genetic Algorithm to find optimal feature subset for employee burnout prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score
import joblib
import os
import time
import warnings
import random

warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create output directories
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42
CV_FOLDS = 5

# Genetic Algorithm Parameters
GA_POPULATION_SIZE = 20
GA_GENERATIONS = 15
GA_CROSSOVER_RATE = 0.8
GA_MUTATION_RATE = 0.2
GA_ELITE_SIZE = 2

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# =============================================================================
# LOAD DATA
# =============================================================================

print("="*80)
print("EMPLOYEE BURNOUT PREDICTION - GENETIC ALGORITHM FEATURE SELECTION")
print("="*80)

print("\n1. Loading Preprocessed Data...")

# Load scaled data for training
X_train = pd.read_csv('processed_data/X_train_scaled.csv')
X_test = pd.read_csv('processed_data/X_test_scaled.csv')
y_train = pd.read_csv('processed_data/y_train.csv').values.ravel()
y_test = pd.read_csv('processed_data/y_test.csv').values.ravel()

print(f"   ✓ Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"   ✓ Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

# Handle any remaining NaN values
if X_train.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
    print(f"   ⚠ Found missing values, imputing with column means...")
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())
    print(f"   ✓ Missing values imputed")

# Store feature names
feature_names = X_train.columns.tolist()
print(f"\n   Available features ({len(feature_names)}):")
for i, feat in enumerate(feature_names, 1):
    print(f"      {i}. {feat}")

# =============================================================================
# GENETIC ALGORITHM FOR FEATURE SELECTION
# =============================================================================

class GeneticAlgorithmFeatureSelector:
    """
    Genetic Algorithm for optimal feature subset selection
    """
    
    def __init__(self, X_train, y_train, feature_names, 
                 population_size=20, generations=15, 
                 crossover_rate=0.8, mutation_rate=0.2, elite_size=2):
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        
        self.best_individual = None
        self.best_fitness = -np.inf
        self.fitness_history = []
        
    def create_individual(self):
        """Create a random individual (feature subset)"""
        # Ensure at least 2 features are selected
        individual = np.random.randint(0, 2, self.n_features)
        while individual.sum() < 2:
            individual = np.random.randint(0, 2, self.n_features)
        return individual
    
    def create_population(self):
        """Create initial population"""
        return [self.create_individual() for _ in range(self.population_size)]
    
    def evaluate_fitness(self, individual):
        """
        Evaluate fitness of an individual using cross-validation
        Fitness = CV R² score with penalty for too many features
        """
        selected_features = individual.astype(bool)
        n_selected = selected_features.sum()
        
        if n_selected == 0:
            return -1000  # Invalid individual
        
        # Get selected features
        X_subset = self.X_train.iloc[:, selected_features]
        
        # Train a fast model for fitness evaluation (Ridge Regression)
        model = Ridge(alpha=10, random_state=RANDOM_STATE)
        
        try:
            # Cross-validation score
            cv_scores = cross_val_score(model, X_subset, self.y_train, 
                                       cv=3, scoring='r2', n_jobs=1)
            cv_score = cv_scores.mean()
            
            # Penalty for using too many features (encourage parsimony)
            feature_penalty = 0.01 * (n_selected / self.n_features)
            
            fitness = cv_score - feature_penalty
            
        except Exception as e:
            fitness = -1000
        
        return fitness
    
    def selection(self, population, fitness_scores):
        """Tournament selection"""
        tournament_size = 3
        selected = []
        
        for _ in range(len(population)):
            # Random tournament
            tournament_idx = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_idx]
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
        return selected
    
    def crossover(self, parent1, parent2):
        """Single-point crossover"""
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.n_features - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            
            # Ensure at least 2 features
            if child1.sum() < 2:
                child1[np.random.choice(self.n_features, 2, replace=False)] = 1
            if child2.sum() < 2:
                child2[np.random.choice(self.n_features, 2, replace=False)] = 1
            
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()
    
    def mutate(self, individual):
        """Bit-flip mutation"""
        mutated = individual.copy()
        for i in range(self.n_features):
            if random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]
        
        # Ensure at least 2 features
        if mutated.sum() < 2:
            mutated[np.random.choice(self.n_features, 2, replace=False)] = 1
        
        return mutated
    
    def evolve(self):
        """Run the genetic algorithm"""
        print("\n2. Running Genetic Algorithm for Feature Selection...")
        print(f"   Population Size: {self.population_size}")
        print(f"   Generations: {self.generations}")
        print(f"   Crossover Rate: {self.crossover_rate}")
        print(f"   Mutation Rate: {self.mutation_rate}")
        print(f"   Elite Size: {self.elite_size}")
        
        # Initialize population
        population = self.create_population()
        
        start_time = time.time()
        
        for generation in range(self.generations):
            gen_start = time.time()
            
            # Evaluate fitness
            fitness_scores = [self.evaluate_fitness(ind) for ind in population]
            
            # Track best individual
            max_fitness_idx = np.argmax(fitness_scores)
            max_fitness = fitness_scores[max_fitness_idx]
            
            if max_fitness > self.best_fitness:
                self.best_fitness = max_fitness
                self.best_individual = population[max_fitness_idx].copy()
            
            self.fitness_history.append({
                'generation': generation + 1,
                'best_fitness': max_fitness,
                'avg_fitness': np.mean(fitness_scores),
                'n_features': population[max_fitness_idx].sum()
            })
            
            gen_time = time.time() - gen_start
            
            print(f"\n   Generation {generation + 1}/{self.generations}:")
            print(f"      Best Fitness: {max_fitness:.4f}")
            print(f"      Avg Fitness: {np.mean(fitness_scores):.4f}")
            print(f"      Features: {population[max_fitness_idx].sum()}/{self.n_features}")
            print(f"      Time: {gen_time:.2f}s")
            
            # Elitism: Keep best individuals
            elite_idx = np.argsort(fitness_scores)[-self.elite_size:]
            elite = [population[i].copy() for i in elite_idx]
            
            # Selection
            selected = self.selection(population, fitness_scores)
            
            # Crossover and Mutation
            offspring = []
            for i in range(0, len(selected) - self.elite_size, 2):
                if i + 1 < len(selected):
                    child1, child2 = self.crossover(selected[i], selected[i+1])
                    offspring.append(self.mutate(child1))
                    offspring.append(self.mutate(child2))
            
            # New population = elite + offspring
            population = elite + offspring[:self.population_size - self.elite_size]
        
        total_time = time.time() - start_time
        
        print(f"\n   ✓ Genetic Algorithm completed in {total_time:.2f} seconds")
        print(f"   ✓ Best Fitness: {self.best_fitness:.4f}")
        print(f"   ✓ Selected {self.best_individual.sum()} out of {self.n_features} features")
        
        # Get selected feature names
        selected_features = [self.feature_names[i] for i, val in enumerate(self.best_individual) if val == 1]
        
        return selected_features, self.best_individual
    
    def plot_evolution(self):
        """Plot fitness evolution over generations"""
        df = pd.DataFrame(self.fitness_history)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        
        # Fitness evolution
        axes[0].plot(df['generation'], df['best_fitness'], 'o-', label='Best Fitness', linewidth=2)
        axes[0].plot(df['generation'], df['avg_fitness'], 's-', label='Avg Fitness', alpha=0.7)
        axes[0].set_xlabel('Generation', fontsize=12)
        axes[0].set_ylabel('Fitness (CV R²)', fontsize=12)
        axes[0].set_title('Genetic Algorithm - Fitness Evolution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Feature count evolution
        axes[1].plot(df['generation'], df['n_features'], 'o-', color='green', linewidth=2)
        axes[1].set_xlabel('Generation', fontsize=12)
        axes[1].set_ylabel('Number of Features', fontsize=12)
        axes[1].set_title('Selected Features Over Generations', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/ga_evolution.png', dpi=300, bbox_inches='tight')
        print("\n   ✓ Saved: results/ga_evolution.png")
        plt.close()

# =============================================================================
# RUN GENETIC ALGORITHM
# =============================================================================

ga = GeneticAlgorithmFeatureSelector(
    X_train, y_train, feature_names,
    population_size=GA_POPULATION_SIZE,
    generations=GA_GENERATIONS,
    crossover_rate=GA_CROSSOVER_RATE,
    mutation_rate=GA_MUTATION_RATE,
    elite_size=GA_ELITE_SIZE
)

selected_features, feature_mask = ga.evolve()

print(f"\n{'='*80}")
print("SELECTED FEATURES BY GENETIC ALGORITHM:")
print('='*80)
for i, feat in enumerate(selected_features, 1):
    print(f"   {i}. {feat}")

# Plot evolution
ga.plot_evolution()

# Filter datasets to selected features
X_train_ga = X_train[selected_features]
X_test_ga = X_test[selected_features]

print(f"\n   ✓ Filtered training set: {X_train_ga.shape}")
print(f"   ✓ Filtered test set: {X_test_ga.shape}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Comprehensive model evaluation"""
    print(f"\n   Evaluating {model_name}...")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Train R²': r2_score(y_train, y_train_pred),
        'Test R²': r2_score(y_test, y_test_pred),
        'Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'Train MAE': mean_absolute_error(y_train, y_train_pred),
        'Test MAE': mean_absolute_error(y_test, y_test_pred),
        'Test MAPE': mean_absolute_percentage_error(y_test, y_test_pred) * 100,
    }
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring='r2', n_jobs=1)
    metrics['CV R² Mean'] = cv_scores.mean()
    metrics['CV R² Std'] = cv_scores.std()
    
    # Print results
    print(f"      Train R²: {metrics['Train R²']:.4f} | Test R²: {metrics['Test R²']:.4f}")
    print(f"      Train RMSE: {metrics['Train RMSE']:.4f} | Test RMSE: {metrics['Test RMSE']:.4f}")
    print(f"      CV R²: {metrics['CV R² Mean']:.4f} ± {metrics['CV R² Std']:.4f}")
    
    return metrics, y_train_pred, y_test_pred

def save_model(model, model_name):
    """Save trained model to disk"""
    filename = f"models/{model_name.replace(' ', '_').lower()}_ga.pkl"
    joblib.dump(model, filename)
    print(f"      ✓ Model saved: {filename}")

# =============================================================================
# DEFINE MODELS
# =============================================================================

print("\n" + "="*80)
print("3. TRAINING MODELS WITH GA-SELECTED FEATURES")
print("="*80)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=10, random_state=RANDOM_STATE),
    'Lasso Regression': Lasso(alpha=0.01, random_state=RANDOM_STATE, max_iter=10000),
    'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=10000),
    'Decision Tree': DecisionTreeRegressor(max_depth=15, min_samples_split=5, 
                                           min_samples_leaf=2, random_state=RANDOM_STATE),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=20, 
                                           min_samples_split=2, min_samples_leaf=1,
                                           random_state=RANDOM_STATE, n_jobs=-1),
    'SVR': SVR(C=100, gamma='scale', epsilon=0.1, kernel='rbf'),
    'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', alpha=0.001,
                        learning_rate='adaptive', max_iter=500, early_stopping=True,
                        random_state=RANDOM_STATE, verbose=False)
}

# =============================================================================
# TRAIN AND EVALUATE MODELS
# =============================================================================

results_ga = []
predictions_ga = {}
trained_models_ga = {}

for model_name, model in models.items():
    print(f"\n{'='*80}")
    print(f"Training: {model_name}")
    print('='*80)
    
    start_time = time.time()
    
    # Train model with GA-selected features
    print(f"   Training with {len(selected_features)} GA-selected features...")
    model.fit(X_train_ga, y_train)
    
    training_time = time.time() - start_time
    print(f"   ✓ Training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    metrics, y_train_pred, y_test_pred = evaluate_model(
        model, X_train_ga, X_test_ga, y_train, y_test, model_name
    )
    metrics['Training Time (s)'] = training_time
    metrics['N Features'] = len(selected_features)
    
    # Save results
    results_ga.append(metrics)
    predictions_ga[model_name] = {
        'train': y_train_pred,
        'test': y_test_pred
    }
    trained_models_ga[model_name] = model
    
    # Save model
    save_model(model, model_name)

# =============================================================================
# RESULTS COMPARISON
# =============================================================================

print("\n" + "="*80)
print("4. RESULTS COMPARISON")
print("="*80)

results_ga_df = pd.DataFrame(results_ga).sort_values('Test R²', ascending=False)

print("\n" + "="*80)
print("MODEL PERFORMANCE WITH GA-SELECTED FEATURES")
print("="*80)
print("\n" + results_ga_df.to_string(index=False))

# Save results
results_ga_df.to_csv('results/model_comparison_ga.csv', index=False)
print("\n✓ Results saved to 'results/model_comparison_ga.csv'")

best_model_ga = results_ga_df.iloc[0]['Model']
best_r2_ga = results_ga_df.iloc[0]['Test R²']

print(f"\n🏆 Best Model with GA Features: {best_model_ga} (Test R² = {best_r2_ga:.4f})")
print(f"   Using {len(selected_features)} out of {len(feature_names)} features")

# Save selected features
with open('results/ga_selected_features.txt', 'w') as f:
    f.write("GENETIC ALGORITHM - SELECTED FEATURES\n")
    f.write("="*50 + "\n\n")
    f.write(f"Total features selected: {len(selected_features)}/{len(feature_names)}\n")
    f.write(f"Best fitness: {ga.best_fitness:.4f}\n\n")
    f.write("Selected features:\n")
    for i, feat in enumerate(selected_features, 1):
        f.write(f"  {i}. {feat}\n")

print("✓ Selected features saved to 'results/ga_selected_features.txt'")

# =============================================================================
# VISUALIZATIONS
# =============================================================================

print("\n" + "="*80)
print("5. GENERATING VISUALIZATIONS")
print("="*80)

# 1. Model Comparison
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(results_ga_df))
width = 0.35

ax.bar(x - width/2, results_ga_df['Train R²'], width, label='Train R²', alpha=0.8, color='steelblue')
ax.bar(x + width/2, results_ga_df['Test R²'], width, label='Test R²', alpha=0.8, color='coral')

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('R² Score', fontsize=12)
ax.set_title(f'Model Comparison with GA-Selected Features ({len(selected_features)} features)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(results_ga_df['Model'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/model_comparison_ga_r2.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: results/model_comparison_ga_r2.png")
plt.close()

# 2. Feature Selection Visualization
fig, ax = plt.subplots(figsize=(12, 8))
y_pos = np.arange(len(feature_names))
colors = ['green' if feature_mask[i] else 'lightgray' for i in range(len(feature_names))]

ax.barh(y_pos, np.ones(len(feature_names)), color=colors, alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(feature_names)
ax.set_xlabel('Selected (Green) / Not Selected (Gray)', fontsize=12)
ax.set_title(f'Genetic Algorithm Feature Selection ({len(selected_features)}/{len(feature_names)} features)', 
             fontsize=14, fontweight='bold')
ax.set_xlim([0, 1.2])

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', alpha=0.7, label=f'Selected ({len(selected_features)})'),
    Patch(facecolor='lightgray', alpha=0.7, label=f'Not Selected ({len(feature_names) - len(selected_features)})')
]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig('results/ga_feature_selection.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: results/ga_feature_selection.png")
plt.close()

# 3. Best Model Predictions
best_preds_ga = predictions_ga[best_model_ga]
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].scatter(y_train, best_preds_ga['train'], alpha=0.3, s=10)
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Burn Rate', fontsize=12)
axes[0].set_ylabel('Predicted Burn Rate', fontsize=12)
axes[0].set_title(f'{best_model_ga} (GA) - Training Set', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_test, best_preds_ga['test'], alpha=0.3, s=10, color='orange')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Burn Rate', fontsize=12)
axes[1].set_ylabel('Predicted Burn Rate', fontsize=12)
axes[1].set_title(f'{best_model_ga} (GA) - Test Set (R² = {best_r2_ga:.4f})', 
                  fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/best_model_predictions_ga.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: results/best_model_predictions_ga.png")
plt.close()

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*80)
print("TRAINING WITH GA FEATURE SELECTION COMPLETED!")
print("="*80)

print(f"\n📊 Summary:")
print(f"   • Features Selected by GA: {len(selected_features)}/{len(feature_names)}")
print(f"   • Best Model: {best_model_ga}")
print(f"   • Best Test R²: {best_r2_ga:.4f}")
print(f"   • Best Test RMSE: {results_ga_df.iloc[0]['Test RMSE']:.4f}")
print(f"   • Best Test MAE: {results_ga_df.iloc[0]['Test MAE']:.4f}")

print(f"\n📁 Output Files:")
print(f"   Models saved in: models/ (*_ga.pkl)")
print(f"   Results saved in: results/")
print(f"      • model_comparison_ga.csv")
print(f"      • model_comparison_ga_r2.png")
print(f"      • ga_evolution.png")
print(f"      • ga_feature_selection.png")
print(f"      • ga_selected_features.txt")
print(f"      • best_model_predictions_ga.png")

print("\n" + "="*80)
print("🎉 Genetic Algorithm Feature Selection Complete!")
print("="*80)
