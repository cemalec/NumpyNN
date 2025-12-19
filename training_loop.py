from Model import Model
from Dataset import MNISTDataset, Dataset
from configs.hydrate_yaml import hydrate_model
from metrics import accuracy
import numpy as np
import argparse
import logging

# Create logger
logger = logging.getLogger(__name__)


# basic_model = Model.load('models/adam_model.npz')
def training_loop(model: Model, dataset: Dataset, epochs: int, batch_size: int):
    for epoch in range(epochs):
        i = 0
        batch_accuracies = []
        batch_losses = []
        for x_train, y_train in dataset.get_batches(batch_size):
            logger.debug(f"Epoch {epoch+1}, Batch {i+1}")
            logger.debug(
                f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}"
            )
            # Forward pass
            y_pred = model.forward(x_train)

            # Compute loss
            loss = model.compute_loss(y_train, y_pred)
            batch_losses.append(loss)

            # Backward pass (weights update as part of the optimizer step in Model.backward)
            model.backward(y_train, y_pred)

            # Compute accuracy
            batch_acc = accuracy(y_train, y_pred)
            logger.info(f"Batch {i+1}, Loss: {loss:.4f}, Accuracy: {batch_acc:.4f}")
            batch_accuracies.append(batch_acc)
            i += 1
        avg_acc = np.mean(batch_accuracies)
        avg_loss = np.mean(batch_losses)
        logger.info(
            f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Avg Accuracy: {avg_acc:.4f}"
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Train a simple neural network on MNIST"
    )
    argparser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs to train"
    )
    argparser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    argparser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer"
    )
    argparser.add_argument(
        "--model_save_path",
        type=str,
        default="models/bigger_model.npz",
        help="Path to save the trained model",
    )
    argparser.add_argument(
        "--model_load_path",
        type=str,
        default=None,
        help="Path to load a pre-trained model",
    )
    argparser.add_argument(
        "--model_config",
        type=str,
        default="configs/fully_connected.yaml",
        help="Path to model configuration YAML file",
    )
    argparser.add_argument("--log_level", type=str, default="INFO")
    # Parse command line arguments
    args = argparser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    log_level = args.log_level.upper()
    model_load_path = args.model_load_path
    model_config = args.model_config
    model_save_path = args.model_save_path
    if model_load_path:
        # Load pre-trained model
        basic_model = Model.load(model_load_path)
        logger.info(f"Loaded model from {model_load_path}")
    elif model_config:
        # Configure logging
        basic_model = hydrate_model(model_config)
        logger.info(f"Hydrated model from {model_config}")
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level, None),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load Dataset
    logger.info("Loading MNIST Dataset")
    train_dataset = MNISTDataset(split="train")
    test_dataset = MNISTDataset(split="test")
    X_train, y_train = train_dataset.X, train_dataset.y
    X_test, y_test = test_dataset.X, test_dataset.y
    logger.info(f"Training data shape: {X_train.shape}, {y_train.shape}")
    logger.info(f"Testing data shape: {X_test.shape}, {y_test.shape}")
    logger.info(f"Model architecture: {basic_model.to_dict()}")
    logger.info(
        f"Starting training for {epochs} epochs with batch size {batch_size} and learning rate {learning_rate}"
    )
    # Train Model
    training_loop(
        model=basic_model,
        dataset=train_dataset,
        epochs=epochs,
        batch_size=batch_size,
    )

    # Save Model
    basic_model.save(model_save_path)

    # Evaluate on training set
    y_train_pred = basic_model.predict(X_train)
    train_loss = basic_model.compute_loss(y_train, y_train_pred)
    logging.info(f"Train Loss: {train_loss:.4f}")
    logging.info(f"Train Accuracy: {accuracy(y_train, y_train_pred):.4f}")

    # Evaluate on test set
    y_test_pred = basic_model.predict(X_test)
    test_loss = basic_model.compute_loss(y_test, y_test_pred)
    test_accuracy = accuracy(y_test, y_test_pred)
    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Test Accuracy: {test_accuracy:.4f}")
