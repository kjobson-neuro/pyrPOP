#!/usr/bin/env python3

import os
import numpy as np
import nibabel as nb
from scipy import ndimage
from scipy.optimize import minimize
from datetime import datetime
import nipype
import nipype.interfaces.afni as afni
import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe
import subprocess
import argparse
from nilearn import image

env = os.environ.copy()

# Import data
parser = argparse.ArgumentParser(description='Calculate centilloid with only PET data.')

# Set up parser for the PET data and its output
parser.add_argument('-pet', type=str, help="The path to the PET scan.")
parser.add_argument('-work', type=str, help="The path to the work directory.")
parser.add_argument('-templates', type=str, help="The path to the templates.")
parser.add_argument('-origin', type=str, help="Reset origin?")
parser.add_argument('-tpopt', type=int, help="Which template?")
parser.add_argument('-out', type=str, help="The path to the output directory.")
parser.add_argument('-exe', type=str, help="The path to the directory with executable scripts.")
args = parser.parse_args()

# Load the input options
input_file = args.pet
work_dir = args.work
temp_dir = args.templates
tpopt = args.tpopt
origin = args.origin
output_dir = args.out
exe_dir = args.exe

# Define function to load templates 
def load_images(paths):
    images = []
    for path in paths:
        img = nb.load(path)
        images.append(img.get_fdata())
    return images

# Define cost function to minimize
def cost_function(coefficients, templates, source_image):
    # Ensure source_image is a numpy array
    if isinstance(source_image, nb.Nifti1Image):
        source_image = source_image.get_fdata()
    combined_image = np.sum([coeff * template for coeff, template in zip(coefficients, templates)], axis=0)
    mse = np.mean((combined_image - source_image) ** 2)
    return mse

# Find best linear combination of templates
def find_best_combination(templates, source_image):
    initial_guess = np.array([1.0 / len(templates)] * len(templates))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    result = minimize(cost_function, initial_guess, args=(templates, source_image), method='SLSQP', constraints=constraints)
    return result.x

# Define the main function
def rPOP(input_file, output_dir, set_origin, template, work_dir, temp_dir):
    print("\n\n********** Welcome to PYrPOP v1.0 (January 2025) **********")
    print("PYrPOP is dependent on:")
    print("*1. AFNI Neuroimaging Suite (https://afni.nimh.nih.gov/)")
    print("*2. Python (https://www.python.org/)")
    print("*3. Advanced Normalization Tools (ANTs) (https://stnava.github.io/ANTs/)")
    
    print("*** PYrPOP is only distributed for academic/research purposes, with NO WARRANTY. ***")
    print("*** PYrPOP is not intended for any clinical or diagnostic purposes. ***")

    # Load the templates
    warptempl_all = [
       os.path.join(temp_dir, 'Template_FBP_all.nii'),
       os.path.join(temp_dir, 'Template_FBP_pos.nii'),
       os.path.join(temp_dir, 'Template_FBP_neg.nii'),   
       os.path.join(temp_dir, 'Template_FBB_all.nii'),
       os.path.join(temp_dir, 'Template_FBB_pos.nii'),
       os.path.join(temp_dir, 'Template_FBB_neg.nii'),
       os.path.join(temp_dir, 'Template_FLUTE_all.nii'),
       os.path.join(temp_dir, 'Template_FLUTE_pos.nii'),
       os.path.join(temp_dir, 'Template_FLUTE_neg.nii')
    ]

    warptempl_fbp = [
       os.path.join(temp_dir, 'Template_FBP_all.nii'),
       os.path.join(temp_dir, 'Template_FBP_pos.nii'),
       os.path.join(temp_dir, 'Template_FBP_neg.nii')
    ]

    warptemple_fbb = [
       os.path.join(temp_dir, 'Template_FBB_all.nii'),
       os.path.join(temp_dir, 'Template_FBB_pos.nii'),
       os.path.join(temp_dir, 'Template_FBB_neg.nii')
    ]

    warptempl_flute = [
       os.path.join(temp_dir, 'Template_FLUTE_all.nii'),
       os.path.join(temp_dir, 'Template_FLUTE_pos.nii'),
       os.path.join(temp_dir, 'Template_FLUTE_neg.nii')
    ]
     
    # Reset origin to center of image if it's set
    if set_origin == "Reset":
        center = np.array(data.shape) / 2
        affine = img.affine
        affine[:3, 3] = -center * affine[:3, :3].diagonal()
        img_centered = nb.Nifti1Image(data, affine)
        print(img_centered)
        # Save the centered image
        nb.save(img_centered, f'{work_dir}/img_centered.nii.gz')
    elif set_origin == "Keep":
        affine = img.affine
        img_uncentered = nb.Nifti1Image(data, affine)
        nb.save(img_uncentered, f'{work_dir}/img_centered.nii.gz')
        print(img_uncentered)

    # Template choice
    if tpopt == 1:
        warptempl = warptempl_all
    elif tpopt == 2:
        warptempl = warptempl_fbp
    elif tpopt == 3:
        warptempl = warptemple_fbb
    elif tpopt == 4:
        warptempl = warptempl_flute

    templates = load_images(warptempl)

    # Perform registration
    temp_reg = os.path.join(temp_dir, 'Template_FBB_all.nii')
    
    prefix = 'init_reg'    

    subprocess.run([f"{exe_dir}/init_ants_reg.sh",
        temp_reg,
        input_file,
        work_dir,
        prefix],
       env=env
    )
    
    warpedimg = f'{work_dir}/{prefix}.nii.gz'
    moving_img = nb.load(warpedimg)
    ANTswarpedimage = moving_img.get_fdata()
    
    # Find best linear combination
    coefficients = find_best_combination(templates, moving_img)
    
    # Create composite template
    composite_template = np.sum([coeff * template for coeff, template in zip(coefficients, templates)], axis=0)
    
    # Save composite template
    # Use the affine from one of the templates (e.g., the first one)
    template_affine = nb.load(warptempl[0]).affine
    composite_img = nb.Nifti1Image(composite_template, template_affine)
    nb.save(composite_img, f'{work_dir}/composite_template.nii.gz')

    # Warp the image to MNI space using ANTs
    full_prefix = 'w_pet'
    # Perform registration
    subprocess.run([f"{exe_dir}/full_ants_reg.sh", 
               f"{work_dir}/composite_template.nii.gz", f"{input_file}", f"{work_dir}/{prefix}0GenericAffine.mat", f"{work_dir}", f"{full_prefix}"])

    # Estimate FWHM using AFNI's 3dFWHMx
    afni_out = 'w_pet_afni'
    subprocess.run([f"{exe_dir}/afni.sh",
                f"{work_dir}", f"{afni_out}"])
    
    fwhm_file = f'{work_dir}/w_pet_afni_automask.txt'
    # Read FWHM estimations
    fwhm_data = np.loadtxt(fwhm_file)
    
    # Extract only the first row for old FWHM calc
    fwhm_x, fwhm_y, fwhm_z = fwhm_data[0, 0:3]

    # Calculate smoothing filters
    def calc_filter(fwhm):
        return np.sqrt(max(0, 10**2 - fwhm**2)) if fwhm < 10 else 0

    filter_x = calc_filter(fwhm_x)
    filter_y = calc_filter(fwhm_y)
    filter_z = calc_filter(fwhm_z)

    # Apply smoothing
    sigma = (filter_x / 2.355, filter_y / 2.355, filter_z / 2.355)

    smoothed_img = image.smooth_img(
    os.path.join(work_dir, f'{full_prefix}.nii.gz'),
    fwhm=[filter_x, filter_y, filter_z]  # Direct FWHM input in mm
    )

    nb.save(smoothed_img, f'{output_dir}/s_pet.nii.gz')

    # Save results to CSV
    results = {
        'Filename': ['s_pet.nii.gz'],
        'EstimatedFWHMx': [fwhm_x],
        'EstimatedFWHMy': [fwhm_y],
        'EstimatedFWHMz': [fwhm_z],
        'FWHMfilterappliedx': [filter_x],
        'FWHMfilterappliedy': [filter_y],
        'FWHMfilterappliedz': [filter_z],
        'AFNIEstimationRerunMod': ['0']
    }

    import pandas as pd
    print(results)
    df = pd.DataFrame(results)
    csv_file = os.path.join(output_dir, f'PYrPOP_{datetime.now().strftime("%m-%d-%Y_%H-%M-%S")}.csv')
    df.to_csv(csv_file, index=False)

    print("\nPYrPOP just finished! Warped and differentially smoothed AC PET images were generated.")
    print("Lookup the .csv database to assess FWHM estimations and filters applied.\n")

# Example usage:
rPOP(input_file, output_dir, origin, tpopt, work_dir, temp_dir)

