import argparse
from setproctitle import setproctitle
from src.data import get_dataset
from src.models import get_model
from src.runners import train
from src.model_tools import set_random_seed
from src.paths import get_model_folder


DEVICE = "cuda"
N_EPOCHS = 100
TRAINING_METHODS = [
    "regular",
    "step",
    "distillation",
    "fulldistillationreset",
    "alternatesteps",
]
TRAIN_BASELINE = True
KEEP_PROB_LEVELS = [0.8, 0.5, 0.3]


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-m",
        "--model-name",
        required=True,
        help="Name of the model to train.",
    )
    argparser.add_argument(
        "-d",
        "--dataset-name",
        required=True,
        help="Name of the dataset to train on.",
    )
    argparser.add_argument(
        "-s",
        "--seed",
        type=int,
        required=True,
        help="Random seed to use along all the experiment.",
    )
    return argparser.parse_args()


def main():
    args = parse_args()
    model_name = args.model_name
    dataset_name = args.dataset_name
    random_seed = args.seed
    setproctitle("python")

    # Set the random seed
    set_random_seed(random_seed)

    # Load data
    train_dl, test_dl, classes, input_size, input_channels = get_dataset(
        name=dataset_name, params={}
    )

    # Generate experiments
    experiments = []

    if TRAIN_BASELINE:
        # Append the baseline (No dropout)
        experiments.append((1.0, "regular"))

    for keep_prob_level in KEEP_PROB_LEVELS:
        for training_method in TRAINING_METHODS:
            experiments.append((keep_prob_level, training_method))

    for i, (keep_prob_level, training_method) in enumerate(experiments):
        alias = f"{dataset_name}_{model_name}_{training_method}_{keep_prob_level}_{random_seed}"
        model_folder = get_model_folder(alias)

        # If the training method is 'regular', double the epochs to match the training steps of other methods
        n_epochs = N_EPOCHS if training_method != "regular" else N_EPOCHS * 2

        print(
            f"[EXPERIMENT {i+1}/{len(experiments)}] Training model: '{alias}' for {n_epochs} epochs."
        )

        # Load model
        net, criterion, optimizer, lr_scheduler = get_model(
            name=model_name,
            params={
                "n_outputs": len(classes),
                "input_size": input_size,
                "input_channels": input_channels,
                "p": keep_prob_level,
                "device": DEVICE,
            },
            n_epochs=n_epochs,
        )

        # Train and eval
        train(
            net=net,
            training_method=training_method,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dataloader=train_dl,
            test_dataloader=test_dl,
            n_epochs=n_epochs,
            device=DEVICE,
            save_path=model_folder,
            save_every_n_epochs=999999,
        )


if __name__ == "__main__":
    main()
