# Forum Virium Street Object Recognition

## Installation
```shell
pip install -r requirements.txt
```

## Running
```shell
jupyter-lab
```
(In this folder, then execute `train.ipynb`)


## Data
Data source of truth is https://cvat.apps.emblica.com/projects/2

# Docker

example run command:
```
docker run \
-v <absolute_path_to_folder_containing_target_images>:/app/input \
-v <absolute_path_to_output_folder>:/app/output \
forum-virium:latest \
python predict.py input/<target_image> output/<output_folder>
```
