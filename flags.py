import argparse


def train_arguments():
    """Commandline arguments for running DreamBooth training script.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        required=True,
        help="Path to a pretrained huggingface Stable Diffusion model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Directory where trained models are saved.",
    )
    parser.add_argument(
        "--instance_images_dir",
        type=str,
        default=None,
        required=True,
        help=(
            "Directory where a few images of the instance to be fine tuned "
            "into the model are saved."
        ),
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help=(
            "Prompt for creating the instance images."
        ),
    )
    parser.add_argument(
        "--class_images_dir",
        type=str,
        default=None,
        required=True,
        help=(
            "Directory where images of similar objects for preserving "
            "model priors are saved."
        ),
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        required=True,
        help=(
            "Prompt for creating the class images."
        ),
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Train batch size."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-6,
        help="Train learning rate."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=400,
        help="Maximum number of training steps."
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight for prior preservation loss."
    )

    return parser


def cache_model_flags():
    """Commandline arguments for running local model caching script.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        required=True,
        help="Directory to write the cached model files.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        help="Name of the huggingface model.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=True,
        help="Revision of the huggingface model repo to cache.",
    )


def run_model_flags():
    """Commandline arguments for running a tuned DreamBooth model.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        required=True,
        help="Directory of the tuned model files.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        required=True,
        help="Comma separated prompt strings for generating the images.",
    )
    parser.add_argument(
        "--num_samples_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate for each prompt.",
    )