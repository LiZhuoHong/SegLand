# SegLand: Discovering Novel Classes in Land Cover Mapping via Hybrid Semantic Segmentation Framework
Land-cover mapping is one of the vital applications in Earth observation. As natural and human activities change the landscape, the land-cover map needs to be rapidly updated. However, discovering newly appeared land-cover types in existing classification systems is still a non-trivial task hindered by various scales of complex land objects and insufficient labeled data over a wide-span geographic area. To address these limitations, we propose a generalized few-shot segmentation-based framework, named SegLand, to update novel classes in high-resolution land-cover mapping. 

The SegLand is accepted by the CVPR 2024 L3D-IVU workshop and score **:rocket:1st place in the OpenEarthMap Land Cover Mapping Few-Shot Challenge:rocket:**. See you in CVPR (Seattle, 17 June)!

Contact me at ashelee@whu.edu.cn
* [**Paper**](https://arxiv.org/abs/2403.02746)
* [**My homepage**](https://lizhuohong.github.io/lzh/)
  
Our previous works:
* [**Paraformer (L2HNet V2)**](https://arxiv.org/abs/2403.02746): accepted by CVPR 2024 (Highlight), the hybrid CNN-ViT framework for HR land-cover mapping using LR labels.[**Code**](https://github.com/LiZhuoHong/Paraformer/)
* [**L2HNet V1**](https://www.sciencedirect.com/science/article/abs/pii/S0924271622002180): accepted by ISPRS P&RS in 2022, The low-to-high network for HR land-cover mapping using LR labels.
* [**SinoLC-1**](https://essd.copernicus.org/articles/15/4749/2023/): accepted by ESSD in 2023, the first 1-m resolution national-scale land-cover map of China.[**Data**](https://zenodo.org/record/7821068)
* [**BuildingMap**](https://arxiv.org/abs/2403.02746): accepted by IGARSS 2024 (Oral), To identify every building's function in urban area.[**Data**](https://github.com/LiZhuoHong/BuildingMap/)


## Training Instructions

* **To train and test the SegLand on the contest dataset, follow these steps:**
1. Dataset and project preprocessing
*  Replace the `'YOUR_PROJECT_ROOT'` in `./scripts/train_oem.sh` with your POP project directory;
*  Download the OEM trainset and unzip the file, then replace the `'YOUR_PATH_FOR_OEM_TRAIN_DATA'` in `./scripts/train_oem.sh`;
*  Download the OEM testset and unzip the file, then replace the `'YOUR_PATH_FOR_OEM_TEST_DATA'` in `./scripts/evaluate_oem_base.sh and ./scripts/evaluate_oem.sh`;
(The train.txt, val.txt, all_5shot_seed123.txt (the list of support set), and test.txt have already been set according to the released data list, which do not need any modification)

2. Base class training and evaluation
*  Train the base model by running `CUDA_VISIBLE_DEVICES=0 bash ./scripts/train_oem.sh`, and the model together with the log file will be stored in ./model_saved_base;
*  Evaluate the trained base model by running `CUDA_VISIBLE_DEVICES=0 bash ./scripts/evaluate_oem_base.sh`, you shall replace the 'RESTORE_PATH' with your own saved checkpoint path, and the output prediction maps together with the log file will be stored in ./output;

3. Novel class updating and evaluation
*  Run `python gen_new_samples_for_new_class.py` to transform the samples generated with cutmixing operation, the generated samples and list are stored in 'YOUR_PATH_OF_CUTMIX_SAMPLES', and the samples should be copied to 'YOUR_PATH_FOR_OEM_TRAIN_DATA', while the list should be appended after all_5shot_seed123.txt;
*  Update the trained base model by running `CUDA_VISIBLE_DEVICES=0 bash ./scripts/ft_oem.sh`, you shall replace the 'RESTORE_PATH' with your own saved checkpoint path, and the model together with the log file will be stored in ./model_saved_ft;
*  Evaluate the trained base model by running `CUDA_VISIBLE_DEVICES=0 bash ./scripts/evaluate_oem_base.sh`, you shall replace the 'RESTORE_PATH' with your own saved checkpoint path, and the output prediction maps together with the log file will be stored in ./output;

4. Output transformation and probability map fusion
*  Run `python trans.py` to transform the output map to the format that matches the requirements of the competetion, the output will be stored in ./upload;
* (Optional) If multiple probability outputs (in *.mat format) are generated, these can be fused by running `python fusemat.py`, you shall replace all the 'PATH_OF_PROBABILITY_MAP_\*' with your own paths (which will be generated under ./output/prob)
