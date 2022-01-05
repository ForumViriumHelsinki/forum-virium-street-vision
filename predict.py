import os
import sys
import cv2
import numpy as np
import glob
import json
import patchify
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow import keras
import skimage as ski
import re
import argparse

parser = argparse.ArgumentParser(description="Process Street Vision images")
parser.add_argument('image', type=str, help="Image you want to process")
parser.add_argument('outfolder', type=str, help="Specify output folder")

args = parser.parse_args()

target = args.image
outfolder = args.outfolder
image_size = (5760, 2880)
patch_size = 480
patches_horizontally = 2
patches_vertically = 4
rect_height = patch_size * patches_vertically # 1920
rect_width  = patch_size * patches_horizontally # 960 so one 5760px image fits in 5760 / 960 = 6 rectangles
equ_correction_steps = image_size[0] // rect_width
model_path = './2021-12-31-light-augments-val-loss-0_085'

label_rgbs = {
    "background": [0,0,0],
    "porras": [255,106,77],
    "portti": [51,221,255],
    "porttikäytävä": [255,204,51],
    "rapputunnus": [131,224,112],
    "sisäänkäynti": [61,61,245],
}

num_labels = len(label_rgbs.keys())
class_threshold = 0.8
min_bounding_box_size = 1000 # Theoretical max for 480x480 is 230400 pixels

class Equirectangular:
    def __init__(self, img_name):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        [self._height, self._width, _] = self._img.shape
    

    def GetPerspective(self, FOV=0, THETA=0, PHI=0, height=0, width=0):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        equ_h = self._height
        equ_w = self._width
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0

        wFOV = FOV
        hFOV = float(height) / width * wFOV

        w_len = np.tan(np.radians(wFOV / 2.0))
        h_len = np.tan(np.radians(hFOV / 2.0))


        x_map = np.ones([height, width], np.float32)
        y_map = np.tile(np.linspace(-w_len, w_len,width), [height,1])
        z_map = -np.tile(np.linspace(-h_len, h_len,height), [width,1]).T

        D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
        xyz = np.stack((x_map,y_map,z_map),axis=2)/np.repeat(D[:, :, np.newaxis], 3, axis=2)

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz).T
        lat = np.arcsin(xyz[:, 2])
        lon = np.arctan2(xyz[:, 1] , xyz[:, 0])

        lon = lon.reshape([height, width]) / np.pi * 180
        lat = -lat.reshape([height, width]) / np.pi * 180

        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90  * equ_cy + equ_cy



        persp = cv2.remap(self._img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        return persp

    
def perspective_correction(input_path, output_folder, steps=6):
    """
    FOV unit is degree
    theta is z-axis angle(right direction is positive, left direction is negative)
    phi is y-axis angle(up direction positive, down direction negative)
    height and width is output image dimension
    """
    
    filename = input_path.split("/")[-1].split(".")[:-1][0]
    if output_folder[-1] == '/':
        output_folder = output_folder[:-1]

    equ = Equirectangular(input_path)

    step = 360 // steps
    filepaths = []
    for i in range(0, 360, step):
        img = equ.GetPerspective(FOV=step, THETA=i, PHI=0, height=rect_height, width=rect_width)
        i_str = str(i).zfill(3)
        i_step_str = str(i+step).zfill(3)
        output = f'{output_folder}/{filename}-degrees-{i_str}-{i_step_str}.png'
        cv2.imwrite(output, img)
        filepaths.append(output)
    return filepaths

def create_patches_prime(filepath):
    img = Image.open(filepath)
    patches_in_column = rect_height // patch_size
    patches_in_row = rect_width // patch_size
    img_patched = patchify.patchify(np.asarray(img), (patch_size, patch_size, 3), step=patch_size).reshape(patches_in_column * patches_in_row, patch_size, patch_size, 3)
    filepaths = []
    for i in range(0, img_patched.shape[0]):
        outfilepath = f'{".".join(filepath.split(".")[:-1])}-{str(i).zfill(2)}.npz'
        np.savez_compressed(outfilepath, img_patched[i])
        filepaths.append(outfilepath)
    return filepaths

def create_patches(perspective_corrected_filepaths):
    npz_filepaths = []
    for perspective_corrected_filepath in perspective_corrected_filepaths:
        npz_filepaths_prime = create_patches_prime(perspective_corrected_filepath)
        npz_filepaths += npz_filepaths_prime
    return npz_filepaths

def predict_with(model, patched_npz_filepaths):
    outfilepaths = []
    for filepath in patched_npz_filepaths:
        outfilepath = f'{".".join(filepath.split(".")[:-1])}-prediction.npz'
        x = np.asarray([np.load(filepath)['arr_0'] / 255]) # Keras expects a batch
        y = model.predict(x, batch_size=1)
        np.savez_compressed(outfilepath, y)
        outfilepaths.append(outfilepath)
    return outfilepaths

def stich_back_single(folder, filename):
    target_filename_without_extension = ".".join(filename.split(".")[:-1])
    prediction_filenames = sorted([filename for filename in os.listdir(folder) if target_filename_without_extension in filename and "-prediction" in filename and "-prediction-full" not in filename])
    canvas = np.zeros([rect_height, rect_width, num_labels])
    for i, prediction_filename in enumerate(prediction_filenames):
        # Calculate proper point to insert prediction
        row = i // patches_horizontally
        column = i % patches_horizontally
        start_row_idx = row * patch_size
        end_row_idx = (row + 1) * patch_size
        start_column_idx = column * patch_size
        end_column_idx = (column + 1) * patch_size
        
        # Load and insert prediction
        prediction = np.load(f'{folder}/{prediction_filename}')['arr_0'].reshape(patch_size, patch_size, num_labels)
        canvas[start_row_idx:end_row_idx, start_column_idx:end_column_idx] = prediction
        
    outfilepath = f'{folder}/{target_filename_without_extension}-prediction-full.npz'
    np.savez_compressed(outfilepath, canvas)
    return outfilepath
        
def stich_back(perspective_corrected_filepaths):
    outfilepaths = []
    
    for filepath in perspective_corrected_filepaths:
        filepath_parts = filepath.split("/")
        folder = "/".join(filepath_parts[:-1])
        filename = filepath_parts[-1]
        outfilepath = stich_back_single(folder, filename)
        outfilepaths.append(outfilepath)
        
    return outfilepaths

def get_bounding_box(filepath, orig_filename, start_degree, end_degree):
    stiched_pred = np.load(filepath)['arr_0']
    boxes = []
    for cls_idx, cls_name in enumerate(label_rgbs):
        if cls_name == 'background':
            continue
        target = stiched_pred[:,:,cls_idx] 
        target = target > class_threshold
        target = ski.morphology.closing(target, ski.morphology.square(20))
        label_target = ski.measure.label(target, connectivity=2)
        regions = ski.measure.regionprops(label_target)
        
        for props in regions:
            minr, minc, maxr, maxc = props.bbox
            boxsize = (maxr-minr) * (maxc-minc)
            if boxsize > min_bounding_box_size:
                max_degrees = end_degree - start_degree
                degrees_start = start_degree + (minc / rect_width  * max_degrees)
                degrees_end = start_degree + (maxc / rect_width * max_degrees)
                boxes.append({ 'filename': orig_filename, 'class': cls_name, 'degrees': (degrees_start, degrees_end)})
    return boxes

def get_bounding_boxes(stiched_prediction_filepaths):
    boxes = []
    for filepath in stiched_prediction_filepaths:
        orig_filename = f'{filepath.split("/")[-1].split("-degrees")[0]}.JPG'
        pattern = re.compile('.*-degrees-(\d{3})-(\d{3})-prediction.*')
        res = re.match(pattern, filepath)
        start_degree = int(res.group(1))
        end_degree = int(res.group(2))
        new_boxes = get_bounding_box(filepath, orig_filename, start_degree, end_degree)
        boxes = boxes + new_boxes
    return boxes

def run_pipeline(target_filepath, output_folder='./temp'):
    print('Initializing...')
    
    # create temp folder
    target_filepath_split = target_filepath.split("/")
    target_filepath_parent = "/".join(target_filepath_split[:-1])
    out_folder_name = output_folder.split("/")[-1]
    if out_folder_name not in os.listdir(target_filepath_parent):
        os.mkdir(output_folder)

    model = keras.models.load_model(model_path)
    
    
    # setup
    """
    filenames = [filename for filename in os.listdir(target_filepath) if '.JPG' in filename]
    filenames_n = len(filenames)
    filenames_digits = len(str(filenames_n))
    
    print(f'Total number of files: {filenames_n}')
    print(f'Done: {"0".zfill(filenames_digits)}/{filenames_n}')
    """
    # The Pipeline
    print(f'Correcting perspective...')
    perspective_corrected_filepaths = perspective_correction(target_filepath, output_folder)
    print(f'Creating patches...')
    patched_npz_filepaths = create_patches(perspective_corrected_filepaths)
    print(f'Running prediction...')
    prediction_filepaths = predict_with(model, patched_npz_filepaths)
    print(f'Stitching prediction back...')
    stiched_back_filepaths = stich_back(perspective_corrected_filepaths)
    print(f'Calculating degrees...')
    degrees = get_bounding_boxes(stiched_back_filepaths)
    with open(f'{output_folder}/results.json', 'w+') as outfile:
        json.dump(degrees, outfile)
    print('Done!')

run_pipeline(target, outfolder)