# DreamBoothOnRay
Fine tune dream booth model using Ray AIR

Finally, to fine tune the model:
```
python test.py                                    \
  --model_dir=<model dir>                         \
  --output_dir=<output dir>                       \
  --instance_images_dir=<train images dir>        \
  --instance_prompt="a photo of sks <class name>" \
  --class_images_dir=<class images dir>           \
  --class_prompt="a photo of a <class name>"
```
