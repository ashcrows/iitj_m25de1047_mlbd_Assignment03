# iitj_m25de1047_mlbd_Assignment03

## Requirements
Install the following Python packages:
- pandas
- numpy
- scikit-learn
- scipy
- matplotlib
- surprise
- tensorflow
- shap

### Dataset
Use the MovieLens dataset and place it in:
./dataset/ml-latest-small/

Required files:
- movies.csv
- ratings.csv

### How to Run
1. Update DATA_DIR if needed.
2. Run the notebook from top to bottom.
3. Make sure all outputs are visible before submission.

### Tasks
- Task 1 uses TfidfVectorizer with movie genres.
- Task 2 builds user profiles as weighted averages of movie TF-IDF vectors.
- Task 5 uses manual SVD with scipy.
- Task 7 uses a meta-learning hybrid model.
- Task 12 uses SHAP for model-agnostic explainability.

### Design
This notebook is designed to run on a local Mac safely:
- Spark handles loading, preprocessing, aggregation, TF-IDF feature generation, and ALS.
- Local methods are restricted to bounded subsets only.
- Any conversion to pandas is intentionally limited.

### Local Safety Limits
- MAX_EVAL_USERS
- MAX_CF_USERS
- MAX_CF_MOVIES
- MAX_LOCAL_MOVIES
- MAX_LOCAL_RATINGS
- MAX_NEURAL_ROWS

These can be adjusted if the machine has more memory.

### Notes
- Manual SVD is run only on a reduced matrix because it is an assignment-specific local method.
- Neural training and explainability are also sampled to avoid driver-memory issues.