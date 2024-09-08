# FedClust: Distributed Computing using Federated Learning

## Project Description

FedClust is a novel approach to federated learning (FL) that addresses data heterogeneity by clustering local models. Traditional federated learning struggles with data heterogeneity due to non-IID (non-independent and identically distributed) data across clients. FedClust enhances the performance and generalization of local models by leveraging contrastive learning principles, forming clusters of similar models, and creating an unbiased global model through cluster aggregation.

This approach has been tested on datasets such as MNIST, Fashion MNIST, USPS, and SVHN, and has shown to outperform existing state-of-the-art algorithms in handling data heterogeneity.

## Features

- **Clustering-based Model Aggregation**: Groups similar local models to improve training efficiency and model performance.
- **Unbiased Global Model**: Reduces bias by averaging cluster models to generalize across clients.
- **Federated Learning with Data Heterogeneity**: Effectively manages diverse and non-IID data distributions.

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/yourusername/FedClust.git
    cd FedClust
    ```

2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3.  Set up the environment for federated learning experiments:

- Ensure that Python 3.8+ is installed.

## Usage

1.  **Running the Experiment**: To run the experiment, navigate to the project directory and execute the following command:
    ```bash
    python run_experiment.py --dataset MNIST --clusters 5 --rounds 50
    ```
2.  **Experiment Configuration**:

- **--dataset**: Specify the dataset to use (e.g., MNIST, FMNIST, USPS, SVHN).
- **--clusters**: Define the number of clusters for model aggregation.
- **--rounds**: Set the number of communication rounds between clients and the server.

3. **Customizing the Model**: The model architecture can be modified in the model.py file. You can switch between CNN, ResNet-18, and ResNet-50 as base encoders.

## Experiments

FedClust has been tested in both centralized and federated environments. The experimental results show that increasing the number of clusters improves model performance.

## Contact

For questions or support, please reach out to Adityasinh Chauhan - [202311063@daiict.ac.in](202311063@daiict.ac.in)
