## Libraries and Requirements:

RUN

`pip3 install -r requirements.txt`

To install all the required libraries.

Note: ipywidgets and ipyevents packages need to be installed for mlab to run in jupyter.

## Data
Download the 'data.zip' from the Box drive, unzip, and place it in the same folder as this code. (BrainReg)

## Usage:

Interface.ipynb is the script the user will use to run the project and have outputs displayed.

Pipeline.ipynb simulates steps of the pipeline.

DataPreview.ipynb visualizes all the data we have available.


## Number of files
Filetype | Number Present | Patients Missing this file type | 
--- | --- | --- |
Fluoroscopy Images | 83 | ['DBS_bS21', 'DBS_bT15', 'DBS_rS02'] |
Postop Images | 76 | ['DBS_bG02', 'DBS_bG06', 'DBS_bG09', 'DBS_bG12', 'DBS_bG13', 'DBS_bG27', 'DBS_bG38', 'DBS_bG74', 'DBS_bS18', 'DBS_bT08']|
Preop Images | 79 |['DBS_bG71', 'DBS_bG72', 'DBS_bG74', 'DBS_bS18', 'DBS_bS21', 'DBS_bT15', 'DBS_rS02'] |
MRI T1 Files | 44 |['DBS_bG01', 'DBS_bG02', 'DBS_bG03', 'DBS_bG06', 'DBS_bG09', 'DBS_bG10', 'DBS_bG12', 'DBS_bG13', 'DBS_bG17', 'DBS_bG18', 'DBS_bG19', 'DBS_bG20', 'DBS_bG21', 'DBS_bG22', 'DBS_bG25', 'DBS_bG26', 'DBS_bG27', 'DBS_bG28', 'DBS_bG29', 'DBS_bG30', 'DBS_bG38', 'DBS_bG53', 'DBS_bG54', 'DBS_bG56', 'DBS_bG57', 'DBS_bG59', 'DBS_bG61', 'DBS_bG62', 'DBS_bG63', 'DBS_bG64', 'DBS_bG65', 'DBS_bG66', 'DBS_bG67', 'DBS_bG68', 'DBS_bG69', 'DBS_bG70', 'DBS_bG74', 'DBS_bG75', 'DBS_bG76', 'DBS_bS02', 'DBS_bS18', 'DBS_bT14'] |
hull.at Files | 82 | ['DBS_bS06', 'DBS_bS21', 'DBS_bT15', 'DBS_rS02']|
Fluoro aligned Files | 26 |['DBS_bG01', 'DBS_bG02', 'DBS_bG03', 'DBS_bG06', 'DBS_bG09', 'DBS_bG10', 'DBS_bG12', 'DBS_bG13', 'DBS_bG17', 'DBS_bG18', 'DBS_bG19', 'DBS_bG20', 'DBS_bG21', 'DBS_bG22', 'DBS_bG25', 'DBS_bG26', 'DBS_bG27', 'DBS_bG28', 'DBS_bG29', 'DBS_bG30', 'DBS_bG38', 'DBS_bG53', 'DBS_bG54', 'DBS_bG56', 'DBS_bG57', 'DBS_bG59', 'DBS_bG61', 'DBS_bG62', 'DBS_bG63', 'DBS_bG64', 'DBS_bG65', 'DBS_bG66', 'DBS_bG67', 'DBS_bG68', 'DBS_bG69', 'DBS_bG70', 'DBS_bG71', 'DBS_bG72', 'DBS_bG73', 'DBS_bG74', 'DBS_bG75', 'DBS_bS02', 'DBS_bS05', 'DBS_bS16', 'DBS_bS18', 'DBS_bS19', 'DBS_bS20', 'DBS_bS21', 'DBS_bT05', 'DBS_bT07', 'DBS_bT15', 'DBS_IG01', 'DBS_IS01', 'DBS_IS04', 'DBS_IS05', 'DBS_IS06', 'DBS_IT03', 'DBS_rG02', 'DBS_rS02', 'DBS_rT02'] |
Camera position Files | 45 | ['DBS_bG17', 'DBS_bG27', 'DBS_bG29', 'DBS_bG30', 'DBS_bG38', 'DBS_bG53', 'DBS_bG54', 'DBS_bG56', 'DBS_bG59', 'DBS_bG62', 'DBS_bG63', 'DBS_bG64', 'DBS_bG65', 'DBS_bG72', 'DBS_bS02', 'DBS_bS05', 'DBS_bS17', 'DBS_bS18', 'DBS_bS21', 'DBS_bT01', 'DBS_bT05', 'DBS_bT07', 'DBS_bT08', 'DBS_bT15', 'DBS_IG01', 'DBS_IS01', 'DBS_IS02', 'DBS_IS04', 'DBS_IT01', 'DBS_IT02', 'DBS_IT03', 'DBS_IT04', 'DBS_IT06', 'DBS_IT07', 'DBS_IT08', 'DBS_IT09', 'DBS_rG01', 'DBS_rG02', 'DBS_rS02', 'DBS_rT01', 'DBS_rT02'] |
GT Electrode location Files | 74 |['DBS_bG30', 'DBS_bS02', 'DBS_bS05', 'DBS_bS21', 'DBS_bT05', 'DBS_bT07', 'DBS_bT15', 'DBS_IG01', 'DBS_IT03', 'DBS_rG02', 'DBS_rS02', 'DBS_rT02']  |
GT Pin tips location Files | 74 |['DBS_bG30', 'DBS_bS02', 'DBS_bS05', 'DBS_bS21', 'DBS_bT05', 'DBS_bT07', 'DBS_bT15', 'DBS_IG01', 'DBS_IT03', 'DBS_rG02', 'DBS_rS02', 'DBS_rT02'] |

NOTE: DBS_bS02 has files in DICOM FOMAT!
study_id
|
├── fluoro.tif
│   
├── preop_ct.nii
|
├── postop_ct.nii
|
├── T1.nii
|
├── hull_rh.mat
|
├── pin_tips.npy
|
└── lead_coord.npy

