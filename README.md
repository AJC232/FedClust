# FedClust: Distributed Computing using Federated Learning

🚀 **FedClust** is a cutting-edge approach to federated learning (FL) that tackles the challenges of data heterogeneity by clustering local models. Traditional FL methods often struggle with non-IID (non-independent and identically distributed) data across clients. FedClust enhances model performance and generalization by leveraging contrastive learning principles, forming clusters of similar models, and creating an unbiased global model through cluster aggregation.

FedClust has been rigorously tested on various datasets, including **MNIST**, **Fashion MNIST**, **USPS**, and **SVHN**, and has demonstrated superior performance compared to existing state-of-the-art algorithms.

## ✨ Features

- 🔗 **Clustering-based Model Aggregation**: Groups similar local models to enhance training efficiency and model performance.
- ⚖️ **Unbiased Global Model**: Reduces bias by averaging cluster models, ensuring better generalization across clients.
- 📊 **Federated Learning with Data Heterogeneity**: Effectively handles diverse and non-IID data distributions across clients.

## 🛠️ Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/FedClust.git
   cd FedClust
   ```

2. **Install Required Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up the Environment**:

   - Ensure Python 3.8+ is installed.

## 🚀 Usage

1. **Running the Experiment**:
   To start an experiment, navigate to the project directory and run:

   ```bash
   python main.py \
    --drive_path './experiments/' \
    --exp_name 'fmnist_cl2_dirichlet/' \
    --clients 10 \
    --dataset 'fmnist' \
    --distribution 'noniid-labeldir' \
    --rounds 50 \
    --clusters 2 \
    --epochs 10
   ```

2. **Experiment Configuration**:

- `--drive_path`: Directory where experiment results will be saved.

  - Example: --drive_path './experiments/'

- `--exp_name`: Name of the experiment. Results will be stored in a folder with this name.

  - Example: --exp_name 'fmnist_cl2_dirichlet/'

- `--clients`: Number of clients participating in the federated learning process.

  - Example: --clients 10

- `--dataset`: Dataset to be used for the experiment. Available options include mnist, fmnist, usps, svhn.

  - Example: --dataset 'fmnist'

- `--distribution`: Data distribution strategy among clients.

  - Example: --distribution 'noniid-labeldir'

- `--rounds`: Number of communication rounds between clients and the server.

  - Example: --rounds 50

- `--clusters`: Number of clusters for model aggregation.

  - Example: --clusters 2

- `--epochs`: Number of local epochs each client will train - their model for.
  - Example: --epochs 10

3. **Customizing the Model**:
   Modify the model architecture in the `model.py` file. You can switch between **CNN**, **ResNet-18**, and **ResNet-50** as base encoders.

## 📈 Experiments

FedClust has been evaluated in both centralized and federated learning environments. We initially assessed the baseline performance by training the model individually on a single client using conventional machine learning methods. This provided a reference for the model’s performance without federated learning or clustering.

In the federated learning setup, we utilized \( N = 20 \) clients and \( T = 20 \) communication rounds to train the model. We explored various cluster configurations (2, 5, and 10 clusters) to examine the impact of clustering on the global model’s performance and its ability to address data heterogeneity. Additionally, we conducted experiments with 10 clients and 50 communication rounds to further validate the effectiveness of our approach.

## 📬 Contact

For questions or support, feel free to reach out:

- **Adityasinh Chauhan**: [202311063@daiict.ac.in](mailto:202311063@daiict.ac.in)
