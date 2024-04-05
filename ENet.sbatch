#!/bin/bash
#SBATCH -p bullet
#SBATCH -t 72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mail-user=cnkwoch@okstate.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
module load cuda/12.1
#module load anaconda3/2022.10
module load apptainer
export APPTAINER_BIND=$PWD
#apptainer run --nv leoTainer_1.sif python3 init.py -lr 5e-4 -bs 4 -wd 2e-4 -e 50 -nc 2 -iptr /scratch/chijioke/Segmentation_models/ENet_model/CornRowsDataset1/XTrain/ -lptr /scratch/chijioke/Segmentation_models/ENet_model/CornRowsDataset1/yTrain/ -ipv /scratch/chijioke/Segmentation_models/ENet_model/CornRowsDataset1/XVal/ -lpv /scratch/chijioke/Segmentation_models/ENet_model/CornRowsDataset1/yVal/ --cuda cuda:0 --mode train
#apptainer run --nv leoTainer_1.sif python3 init.py -lr 5e-4 -bs 4 -wd 2e-4 -e 50 -nc 12 -iptr /scratch/chijioke/Segmentation_models/ENet_model/CamVid_dataset/train/ -lptr /scratch/chijioke/Segmentation_models/ENet_model/CamVid_dataset/trainannot/ -ipv /scratch/chijioke/Segmentation_models/ENet_model/CamVid_dataset/val/ -lpv /scratch/chijioke/Segmentation_models/ENet_model/CamVid_dataset/valannot/ --cuda cuda:0 --mode train
apptainer run --nv leoTainer_1.sif python3 init.py -lr 5e-4 -bs 4 -wd 2e-4 -e 50 -nc 2 -iptr /scratch/chijioke/Segmentation_models/ENet_model/GreenHouseDataset/XTrain/ -lptr /scratch/chijioke/Segmentation_models/ENet_model/GreenHouseDataset/yTrain/ -ipv /scratch/chijioke/Segmentation_models/ENet_model/GreenHouseDataset/XVal/ -lpv /scratch/chijioke/Segmentation_models/ENet_model/GreenHouseDataset/yVal/ --cuda cuda:0 --mode train
#apptainer run --nv leoTainer_1.sif python3 init.py -m /scratch/chijioke/Segmentation_models/ENet_model/ckpt-enet-50--1.0422000413990913e+23.pth  -i /scratch/chijioke/Segmentation_models/ENet_model/CornRowsDataset1/XTest/rgb_img_22.jpg --cuda cuda:0 --mode test
