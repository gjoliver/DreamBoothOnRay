# DreamBoothOnRay
Fine tune dream booth model using Ray AIR

Download and cache a pre-trained Stable-Diffusion model locally.
Default model and version are ``CompVis/stable-diffusion-v1-4``
at git hash ``3857c45b7d4e78b3ba0f39d4d7f50a2a05aa23d4``.
```
python cache_model.py --model_dir=<model_dir>
```
Note that actual model files will be downloaded into
``\<model_dir>\snapshots\<git_hash>\`` directory.

Create a regularization image set for a class of subjects:
```
python run_model.py                    \
  --model_dir=<model_dir>              \
  --output_dir=<output_dir>            \
  --prompts="photo of a <class_name>"  \
  --num_samples_per_prompt=200
```

Save a few (4 to 5) images of the subject being fine-tuned
in a local directory. Then launch the training job with:
```
python train.py                                   \
  --model_dir=<model_dir>                         \
  --output_dir=<output_dir>                       \
  --instance_images_dir=<train_images_dir>        \
  --instance_prompt="a photo of sks <class_name>" \
  --class_images_dir=<class_images_dir>           \
  --class_prompt="a photo of a <class_name>"
```
