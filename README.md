# Genetic Algorithm for Feature Selection

This repository contains a Python implementation of a Genetic Algorithm (GA) for feature selection. The GA is designed to select the most relevant features from a dataset to improve the performance of a machine learning model.

## Genetic Algorithm (GA)

A Genetic Algorithm is a heuristic search algorithm inspired by the process of natural selection. It is commonly used to find approximate solutions to optimization and search problems. Here's a brief overview of how it works:

1. **Initialization**: Create an initial population of candidate solutions (individuals) randomly.

2. **Evaluation**: Evaluate the fitness of each individual in the population using a fitness function, which measures how well each individual solves the problem.

3. **Selection**: Select individuals from the current population to be parents based on their fitness. The probability of selection is proportional to the individual's fitness.

4. **Crossover**: Create new individuals (offspring) by combining the genetic material (chromosomes) of selected parents through crossover (recombination) operations.

5. **Mutation**: Introduce random changes (mutations) to the offspring's genetic material to maintain genetic diversity.

6. **Replacement**: Replace the current population with the new population of offspring.

7. **Termination**: Repeat steps 2-6 until a termination condition is met, such as a maximum number of generations or satisfactory fitness level achieved.

## Results
In a spam email detection dataset containing 6000 data points, I utilized the power of Genetic feature selection to increase the accuracy of my model. 
To prepare the data for the genetic algorithm, I performed a vector embedding of my text data using openAI's embedding API, which turns each email into a vector of 1536 features. 
Without any feature selection, my decision tree algorithm attained 95% accuracy. After the application of my genetic algorithm for dimension reduction, I managed to increase my performance to 98%. It is not a significant increase, but it still shows that the algorithm performs as it should. 

<img width="987" alt="Screen Shot 2024-02-29 at 3 02 23 PM" src="https://github.com/julesgransden/Genetic_Dimmensionality_Reduction/assets/78057184/3afb9d8f-774d-49e7-a013-86dc4568eb8c">
Figure 1: Accuracy after 100 iterations using a multiple-point crossover function, 1% mutation factor
<img width="993" alt="Screen Shot 2024-02-28 at 5 41 30 PM" src="https://github.com/julesgransden/Genetic_Dimmensionality_Reduction/assets/78057184/9e889c2c-e10d-4885-8c96-d29f3ec094c9">
Figure 2: Accuracy after 10 iterations of the single-point crossover function, 1% mutation factor



## Usage

To use this Genetic Algorithm, follow these steps:

1. Ensure you have Python installed on your system.

2. Clone this repository to your local machine.

3. Navigate to the cloned directory.

4. Install the required dependencies:
   pip install numpy pandas scikit-learn

5. Prepare your dataset in CSV format and save it as `vectorizedData.csv` in the repository folder.

6. Run the `geneticFeatureSelection.py` script:


## Description

### Files

- `geneticFeatureSelection.py`: Main Python script containing the implementation of the Genetic Algorithm for feature selection.
- `vectorizedData.csv`: Example dataset in CSV format. You can replace this with your own dataset.

### GeneticAlgorithm Class

- `Generation`: This class represents a generation in the Genetic Algorithm. It includes methods for parent selection, producing new generations, and sorting the population based on performance.

### Methods

- `initialPopulation`: Initializes the population of solutions with random genetic representations and evaluates their performance using a machine learning model.
- `splitData`: Splits the dataset into training and testing sets.
- `generateGeneticSLN`: Generates a genetic representation of a solution (binary representation).
- `generateDFSolution`: Generates a solution DataFrame based on the genetic representation.
- `genHeuristicValue`: Calculates the heuristic value (e.g., accuracy) of a solution using a machine learning model.
- `ParentSelection1`: Selects parents for mutation and child formation based on performance.
- `ParentSelection2`: Selects parents for mutation and child formation, using the top-performing solution as the parent for all new children.
- `ProducenewGen`: Produces a new generation of children from selected parents.
- `MultiplePointCrossoverFct`: Performs multiple-point crossover between two parent genes.
- `singlePointCrossoverFct`: Performs single-point crossover between two parent genes.
- `mutation`: Performs genetic mutation on children genes.
- `geneticFeatureSelection`: Executes the genetic feature selection process for a specified number of generations.



