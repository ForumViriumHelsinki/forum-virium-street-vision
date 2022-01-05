# Forum Virium Street Object Recognition

## Installation
```shell
pip install -r requirements.txt
```

## Jupyter notebooks
Execute
```shell
jupyter-lab
```
to open jupyter-lab environment.

### Training
Training the model can be done with `train.ipynb`.

There are two options at the moment: 
    - use whole images (use "For image data" cells).
        - these are not perspective corrected
        - these have a lot of patches with no target classes in them
        - this takes a long time (~3-4 hours per epoch on Emblica bigrig).
    - use prepared .npzs
        - are perspectice corrected and sampled so there are more target classes in them
        - takes shorter time (~15mins per epoch on Emblica bigrig)
So unless you really want to try something, use .npzs :)
You can generate more of them by using `rearrange_dataset.ipynb`

### Prediction
There are multiple prediction pipeline options.

You can use `predict.ipynb` to do inference on a folder.

For inference on a single image file, you have multiple options.
`predict-single-file.ipynb` allows for most flexibility and exploring.
`python predict.py <target_image> <output>` is for commandline.
There is also a docker image which can be used! Build it (you'll need a trained model, and use a command similar to the example below to run inference on a single image.

Example docker run command:
```bash
docker run \
-v <absolute_path_to_folder_containing_target_images>:/app/input \
-v <absolute_path_to_output_folder>:/app/output \
forum-virium:latest \
python predict.py input/<target_image> output/<output_folder>
```

## Data
Data source of truth is https://cvat.apps.emblica.com/projects/2



