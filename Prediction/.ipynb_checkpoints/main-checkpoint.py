import numpy as np
from scipy import stats
import pandas as pd
import random
import secrets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from keras.layers import Input, Dense, Dropout
from keras.models import Sequential
from keras.models import Model
from keras.optimizers import Adam
from collections import Counter  # <-- Import the Counter class
from deap import algorithms, base, creator, tools
from itertools import combinations
from collections import defaultdict

# Define constants
HISTORICAL_FILE_PATH = 'historical.txt'
PREDICTION_FILE_PATH = 'prediction.txt'
EPOCHS = 20
BATCH_SIZE = 64
NUM_PREDICTIONS = 20
ENCODING_DIM = 738

# Load sequences from a file
def load_sequences(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        sequences = [list(map(int, line.strip()[1:-1].replace(']', '').split(', '))) for line in lines]
        return sequences

# Feature Engineering
def feature_engineering(sequences):
    # Initialize a dictionary to store the frequency of each number
    number_frequency = {}

    # Initialize variables to store the most recent occurrence of each number
    last_occurrence = {}

    # Initialize the current sequence number
    sequence_number = 0

    # Calculate the frequency of each number and the last occurrence of each number
    for seq in sequences:
        sequence_number += 1
        for num in seq:
            if num in number_frequency:
                number_frequency[num] += 1
            else:
                number_frequency[num] = 1
            last_occurrence[num] = sequence_number

    # Calculate the features for each sequence
    features = []
    for seq in sequences:
        # Calculate the mean and standard deviation of the sequence
        mean_value = np.mean(seq)
        std_dev = np.std(seq)

        # Calculate the average frequency and last occurrence of the numbers in the sequence
        avg_frequency = np.mean([number_frequency[num] for num in seq])
        avg_last_occurrence = np.mean([sequence_number - last_occurrence[num] for num in seq])

        # Add the features to the list
        features.append([mean_value, std_dev, avg_frequency, avg_last_occurrence])

    return np.array(features)

# Advanced Feature Engineering
def advanced_feature_engineering(data):
    # Flatten the list of lists to work with individual elements
    flattened_data = [num for sublist in data for num in sublist]

    # Determine the maximum length of subsequences to consider
    max_subsequence_length = min(len(sublist) for sublist in data)

    # Initialize a Counter to store the frequency of each subsequence
    subsequence_counter = Counter()

    # Iterate over all possible subsequence lengths
    for length in range(2, max_subsequence_length + 1):
        # Generate all combinations of subsequences of the current length
        subsequences = [tuple(subseq) for sublist in data for subseq in combinations(sublist, length)]

        # Update the Counter with the frequency of each subsequence
        subsequence_counter.update(subsequences)

    return subsequence_counter

# Function to calculate the fitness of an individual
def evaluate_combinations(individual, historical_sequences, advanced_features):
    # Extract the selected features based on the individual
    selected_indices = tuple(sorted(i for i, select in enumerate(individual) if select))
    selected_combinations = [tuple(seq[i] for i in selected_indices) for seq in historical_sequences]

    # Perform some evaluation on the selected combinations
    # Modify this based on your specific task
    fitness_value = sum(sum(combination) for combination in selected_combinations)

    # Incorporate advanced feature engineering
    if isinstance(advanced_features, Counter):
        # Check if advanced_features is a Counter object
        advanced_feature_value = sum(advanced_features.values())
    else:
        # Handle the case when advanced_features is an integer
        advanced_feature_value = advanced_features

    # Weight the advanced feature value
    fitness_value += 0.5 * advanced_feature_value

    return fitness_value,


    # Flatten the list of lists to a single list of numbers
    all_numbers = [number for sublist in sequences for number in sublist]

    # Calculate the frequency of each number
    number_frequency = Counter(all_numbers)

    # Find the most common numbers
    most_common_numbers = number_frequency.most_common()

    # Find the least common numbers

    # Flatten the list of lists to work with individual elements
    flattened_data = [num for sublist in data for num in sublist]
    
    # Determine the maximum length of subsequences to consider
    max_subsequence_length = min(len(sublist) for sublist in data)

    # Initialize a Counter to store the frequency of each subsequence
    subsequence_counter = Counter()

    # Iterate over all possible subsequence lengths
    for length in range(2, max_subsequence_length + 1):
        # Generate all combinations of subsequences of the current length
        subsequences = [tuple(subseq) for sublist in data for subseq in combinations(sublist, length)]

        # Update the Counter with the frequency of each subsequence
        subsequence_counter.update(subsequences)

    return subsequence_counter

# Modified analyze_numbers function to include odd and even number analysis
def analyze_numbers(numbers):
    analysis = {
        "Mean": np.mean(numbers),
        "Median": np.median(numbers),
        "Mode": stats.mode(numbers)[0][0],
        "Range": np.ptp(numbers),
        "Standard Deviation": np.std(numbers),
        "Variance": np.var(numbers),
        "Skewness": stats.skew(numbers),
        "Kurtosis": stats.kurtosis(numbers),
        "Frequency": dict(Counter(numbers)),
        "Odds": sum(n % 2 != 0 for n in numbers),
        "Evens": sum(n % 2 == 0 for n in numbers)
    }
    return analysis

# Function to analyze each column in a dataset
def analyze_dataset_columns(dataset):
    column_analysis = {}
    for column_name, column_data in dataset.items():
        column_analysis[column_name] = analyze_numbers(column_data)
    return column_analysis

def track_patterns(sequences):
    # Convert sequences to DataFrame for statistical analysis
    df = pd.DataFrame(sequences)

    # Perform complex statistical analysis
    stats = df.describe()
    all_numbers = df.values.flatten()
    frequency = Counter(all_numbers)
    most_common_numbers = frequency.most_common(5)
    least_common_numbers = frequency.most_common()[:-6:-1]
    odd_even_counts = df.applymap(lambda x: 'Odd' if x % 2 != 0 else 'Even').apply(pd.Series.value_counts, axis=1)

    # Initialize variables for pattern tracking
    total_sequences = len(sequences)
    total_occurrences = 0
    odd_count = 0
    even_count = 0

    # Define patterns to search for
    target_pattern = [8, 19, 32, 41, 42]

    for seq in sequences:
        # Count the total occurrences of the target pattern
        if all(item in seq for item in target_pattern):
            total_occurrences += 1

        # Count odd and even numbers
        odd_numbers = [num for num in seq if num % 2 != 0]
        even_numbers = [num for num in seq if num % 2 == 0]

        # Update odd and even counts
        odd_count += len(odd_numbers)
        even_count += len(even_numbers)

    # Calculate statistics for pattern tracking
    average_occurrences = total_occurrences / total_sequences
    odd_percentage = (odd_count / (odd_count + even_count)) * 100
    even_percentage = (even_count / (odd_count + even_count)) * 100

    # Return combined statistics
    return {
        "DataFrame Stats": stats,
        "Most Common Numbers": most_common_numbers,
        "Least Common Numbers": least_common_numbers,
        "Odd Even Counts": odd_even_counts.head(),
        "Average Occurrences of Pattern": average_occurrences,
        "Odd Percentage": odd_percentage,
        "Even Percentage": even_percentage
    }

# Compare line [8, 19, 32, 41, 42] to historical data and track patterns, algorithm, odds, and evens
def compare_and_track(historical_sequences):
    # Define the line to compare
    line_to_compare = [8, 19, 32, 41, 42]
    
    # Search for matches in historical data
    matching_indices = [i for i, seq in enumerate(historical_sequences) if set(line_to_compare) <= set(seq)]
    
    # Iterate through matched lines and track patterns, algorithm, odds, and evens
    for idx in matching_indices:
        print("Matched Line Index:", idx)
        print("Matched Line:", historical_sequences[idx])
        
        # Track patterns, algorithm, odds, and evens of 3 lines before and after
        for i in range(max(0, idx - 3), min(len(historical_sequences), idx + 4)):
            track_patterns(historical_sequences[i])
            print()
        print()

# Build and compile the autoencoder model
def build_autoencoder(input_dim, encoding_dim):
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation='relu')(input_layer)
    decoder = Dense(input_dim, activation='sigmoid')(encoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer=Adam(), loss='mse')
    return autoencoder

# Train the autoencoder
def train_autoencoder(autoencoder, X_train, X_val, epochs, batch_size):
    autoencoder.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=epochs, batch_size=batch_size, verbose=0)

# Train and predict using LinearRegression
def train_and_predict_linear_regression(X_train, X_val, y_train):
    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)
    predictions = linear_reg.predict(X_val)
    return predictions

# Train and predict using RandomForestRegressor
def train_and_predict_random_forest(X_train, X_val, y_train):
    random_forest = RandomForestRegressor(n_estimators=100, random_state=64)
    random_forest.fit(X_train, y_train)
    predictions = random_forest.predict(X_val)
    return predictions

# Predict and print sequences
def predict_and_print(autoencoder, sequences, num_predictions):
    random_indices = secrets.SystemRandom().sample(range(len(sequences)), num_predictions)
    for idx in random_indices:
        input_sequence = np.array(sequences[idx]).reshape(1, -1)
        predicted_sequence = autoencoder.predict(input_sequence)
        print("Input Sequence:", sequences[idx])
        print("Predicted Sequence from Autoencoder:", np.round(predicted_sequence).astype(int).tolist()[0])
        print()

def find_pattern(sequence):
    # Convert the sequence to a NumPy array for advanced numerical operations
    arr = np.array(sequence)

    # Attempt polynomial fitting to the sequence
    try:
        # Fit a polynomial of degree equal to the length of the sequence - 1
        coefficients = np.polyfit(np.arange(len(sequence)), arr, len(sequence) - 1)
        poly = np.poly1d(coefficients)

        # Generate predicted values based on the polynomial fit
        predicted_values = poly(np.arange(len(sequence)))

        # Check the similarity between the predicted values and the original sequence
        similarity = np.corrcoef(arr, predicted_values)[0, 1]

        # If the correlation is high, return the polynomial coefficients
        if similarity > 0.8:
            return coefficients
        else:
            return None

    except np.linalg.LinAlgError:
        return None

def identify_pattern(sequence):
    # Prepare the data for linear regression
    X = np.arange(len(sequence)).reshape(-1, 1)
    y = np.array(sequence)

    # Fit linear regression model
    model = LinearRegression().fit(X, y)

    # Check if the coefficient is close to an integer
    coefficient = model.coef_[0]
    if abs(round(coefficient) - coefficient) < 0.01:
        pattern = round(coefficient)
        return f"Linear pattern with a difference of {pattern}"
    else:
        return "Pattern not recognized"

# Load historical sequences
historical_sequences = load_sequences(HISTORICAL_FILE_PATH)

# Assign historical_sequences to extracted_numbers for further processing
extracted_numbers = historical_sequences

# Initialize a dictionary to store the frequency of each number
frequency = defaultdict(int)

# Initialize variables to store the most recent occurrence of each number
last_occurrence = defaultdict(int)

# Initialize the current sequence number
sequence_number = 1

# Initialize lists to store features for each sequence
mean_list = []
std_dev_list = []
avg_frequency_list = []
avg_last_occurrence_list = []

# Iterating through each sequence to calculate the required features
for sequence in extracted_numbers:
    if len(sequence) == 5:
        # Update frequency and last occurrence of each number
        for number in sequence:
            frequency[number] += 1
            last_occurrence[number] = sequence_number

        # Calculate mean and standard deviation
        mean_list.append(np.mean(sequence))
        std_dev_list.append(np.std(sequence))

        # Calculate average frequency and last occurrence for the sequence
        avg_frequency = np.mean([frequency[num] for num in sequence])
        avg_last_occurrence = np.mean([last_occurrence[num] for num in sequence])
        avg_frequency_list.append(avg_frequency)
        avg_last_occurrence_list.append(avg_last_occurrence)

    sequence_number += 1

# Preparing the data for analysis
analysis_data = {
    "Mean": mean_list,
    "Standard Deviation": std_dev_list,
    "Average Frequency": avg_frequency_list,
    "Average Last Occurrence": avg_last_occurrence_list
}

# Data Preprocessing Function
def preprocess_data(data):
    scaler = MinMaxScaler()
    processed_data = scaler.fit_transform(data)
    return processed_data

# Model Training Function
def train_model(data, labels, model_type='linear_regression'):
    if model_type == 'linear_regression':
        model = LinearRegression()
        model.fit(data, labels)
    elif model_type == 'random_forest':
        model = RandomForestRegressor()
        model.fit(data, labels)
    # Add more model types as needed
    return model

# Model Evaluation Function
def evaluate_model(model, test_data, test_labels):
    predictions = model.predict(test_data)
    # Add evaluation metrics as needed
    return predictions

# Hyperparameter Tuning for RandomForest
def tune_random_forest(train_data, train_labels):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4, 6, 8],
        'criterion' :['gini', 'entropy']
    }
    rfr = RandomForestRegressor()
    grid_search = GridSearchCV(estimator = rfr, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)
    grid_search.fit(train_data, train_labels)
    return grid_search.best_params_

# Advanced Neural Network Model
def create_advanced_model(input_shape):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
if __name__ == "__main__":
    # Load historical and prediction sequences
    historical_sequences = load_sequences(HISTORICAL_FILE_PATH)
    prediction_sequences = load_sequences(PREDICTION_FILE_PATH)
    print("historical_sequences:", historical_sequences)
    print("prediction_sequences:", prediction_sequences)

    # Reverse the order of historical data so that they start from the left side
    historical_sequences.reverse()
    
    # Insert [16, 29, 32, 36, 41] at the beginning
    historical_sequences.insert(0, [16, 29, 32, 36, 41])
    
    # Append [8, 19, 32, 41, 42] at the end
    historical_sequences.append([8, 19, 32, 41, 42])
    
    prediction_sequences = load_sequences(PREDICTION_FILE_PATH)

    # Feature Engineering for historical data
    historical_features = feature_engineering(historical_sequences)
    
    # Track patterns, statistics, odds, and evens in historical data
    track_patterns(historical_sequences)
    
    # Append [8, 19, 32, 41, 42] at the beginning when the condition is met
    for idx, seq in enumerate(historical_sequences):
        if len(set(seq) & {8, 19, 32, 41, 42}) >= 2:
            historical_sequences.insert(idx, [8, 19, 32, 41, 42])
            break
    
    # Preprocess data
    X_historical = np.array(historical_sequences)
    X_train, X_val = train_test_split(X_historical, test_size=1706, random_state=64)
    
    # Build and train autoencoder
    autoencoder = build_autoencoder(X_train.shape[1], ENCODING_DIM)
    train_autoencoder(autoencoder, X_train, X_val, EPOCHS, BATCH_SIZE)
    
    # Train and predict using LinearRegression
    y_train = X_train
    predictions_linear_reg = train_and_predict_linear_regression(X_train, X_val, y_train)
    
    # Train and predict using RandomForestRegressor
    predictions_random_forest = train_and_predict_random_forest(X_train, X_val, y_train)
    
    # Load prediction sequences
    prediction_sequences = load_sequences(PREDICTION_FILE_PATH)
    
    # Choose 5 lines from the prediction file
    selected_prediction_lines = random.sample(prediction_sequences, NUM_PREDICTIONS)
    
    # Predict and print sequences
    predict_and_print(autoencoder, selected_prediction_lines, NUM_PREDICTIONS)

    # Advanced Feature Engineering for historical data
    advanced_features = advanced_feature_engineering(historical_sequences)

    # Genetic Algorithm setup for combinations
    num_features = len(historical_sequences[0])
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Custom initialization function for individuals
    def initialize_individual():
        return random.sample(range(num_features), num_features)

    toolbox.register("individual", tools.initIterate, creator.Individual, initialize_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_combinations, historical_sequences=historical_sequences, advanced_features=advanced_features)

    population_size = 50
    num_generations = 10

    population = toolbox.population(n=population_size)

    algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, lambda_=2*population_size,
                              cxpb=0.7, mutpb=0.2, ngen=num_generations, stats=None, halloffame=None)

    best_individual = tools.selBest(population, k=1)[0]

    # Extract the selected features based on the best individual
    selected_indices = [i for i, select in enumerate(best_individual) if select]
    selected_combinations = [tuple(seq[i] for i in selected_indices) for seq in historical_sequences]

    # Print the best individual and selected combinations
    print("Best Individual:", best_individual)
    print("Selected Combinations:", selected_combinations)

    # Select 20 lines from the prediction data
    selected_prediction_lines = random.sample(prediction_sequences, 20)
    print("Selected Prediction Lines:", selected_prediction_lines)


    # Print predictions from LinearRegression
    print("Predictions from Linear Regression:")
    print(predictions_linear_reg)
    
    # Print predictions from RandomForestRegressor
    print("Predictions from RandomForestRegressor:")
    print(predictions_random_forest)
