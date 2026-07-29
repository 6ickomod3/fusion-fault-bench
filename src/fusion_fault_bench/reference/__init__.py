"""Independent scientific reference calculations."""

from fusion_fault_bench.reference.analytic import (
    ANALYTIC_GAUSSIAN_METHODS,
    AnalyticCondition,
    AnalyticPopulationPoint,
    CrossoverReference,
    GaussianPopulation,
    continuous_crossover_root,
    expand_conditions,
    first_zero_grid_root,
    gaussian_population,
    grid_crossover_root,
    pava_non_decreasing,
    population_contrast,
    population_crossover_references,
    population_points,
)

__all__ = [
    "ANALYTIC_GAUSSIAN_METHODS",
    "AnalyticCondition",
    "AnalyticPopulationPoint",
    "CrossoverReference",
    "GaussianPopulation",
    "continuous_crossover_root",
    "expand_conditions",
    "first_zero_grid_root",
    "gaussian_population",
    "grid_crossover_root",
    "pava_non_decreasing",
    "population_contrast",
    "population_crossover_references",
    "population_points",
]
