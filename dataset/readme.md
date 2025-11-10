## ReasonSeg-Hard
We propose REASONSEG-HARD, a new evaluation dataset for stress testing reasoning segmentation. Specifically, we constructed a dataset with 217 image-query pairs consisting of 79 images and their respective queries sampled from ReasonSeg Test Split with the rest consisting of our own examples.

<img width="700"  alt="image" src="https://github.com/user-attachments/assets/0b90247e-85f6-4cbe-9dd4-8688a7b07204" />


Download
---
Please download the dataset from the following sources:
- [OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/skao_connect_ust_hk/EtXL4CYt-RhArzSuzmQckyQB4FtonuQwq-mcZnSQzSrSjA?e=g2CYUK)
- [Google Cloud](https://drive.google.com/drive/folders/1tN_Stbuk1s9UZZNKyetmRJIGgSvFJcVt?usp=sharing)

API
---
To access the data, please use the class provided in `read_dataset.py`, entries will be in the format of `(img_name, query, mask_name)`.
