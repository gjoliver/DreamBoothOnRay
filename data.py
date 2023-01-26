from ray.data import read_images
from ray.data.preprocessors import TorchVisionPreprocessor
from torchvision import transforms
from transformers import AutoTokenizer

def get_train_dataset(args, image_resolution=512):
    """Build a Ray Dataset for fine-tuning DreamBooth model.
    """
    # Load image sets.
    class_dataset = read_images(args.class_images_dir)
    instance_dataset = read_images(args.instance_images_dir)

    # Preprocessing the training images.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                image_resolution,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(image_resolution),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    preprocessor = TorchVisionPreprocessor(["image"], transform=transform)

    instance_dataset = preprocessor.transform(instance_dataset)
    class_dataset = preprocessor.transform(class_dataset)
    
    # Load tokenizer for tokenizing the image prompts.
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_dir, subfolder="tokenizer",
    )

    def _tokenize(prompt):
        return tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.numpy()

    # Get the token ids for both prompts.
    class_prompt_ids = _tokenize(args.class_prompt)
    instance_prompt_ids = _tokenize(args.instance_prompt)

    def _get_add_prompt_ids_fn(prompt_ids):
        def fn(batch):
            batch["prompt_ids"] = [prompt_ids] * len(batch)
            return batch
        return fn

    # Add class prompt ids to prompt data set.
    class_dataset = class_dataset.map_batches(
        _get_add_prompt_ids_fn(class_prompt_ids)
    )
    # Add instance prompt ids to instance data set.
    instance_dataset = instance_dataset.map_batches(
        _get_add_prompt_ids_fn(instance_prompt_ids)
    )

    # Join the two datasets. This is out training dataset.
    return class_dataset.union(instance_dataset).random_shuffle()
