Implementation of our submited paper

"GEAP: Gradually Enhanced Adversarial Perturbations on Color Pixel Vectors for Image Steganography"

Directories and files included in the implementation:


'datas/' - Some of the datasets of our experiment.
    
    'datas/BOSS256-C' - It contains 10,000 256*256*3 color images.
        'boss256_color_train.txt' - The image name list of the training set, which contains 7,000 images.
        'boss256_color_val.txt' - The image name list of the validation set, which contains 1,000 images.
        'boss256_color_tst.txt' - The image name list of the test set, which contains 2,000 images.
    
    'datas/ALASKA256-C' - It contains 80,000 256*256*3 color images.
        'alaska256_v2_trn.txt' - The image name list of the training set, which contains 54,000 images. 
        'alaska256_v2_val.txt' - The image name list of the validation set, which contains 6,000 images.'
        alaska256_v2_tst.txt' - The image name list of the test set, which contains 20,000 images.

'models/' - Corresponding network model files and GEAP implementation files.

    'models/WISERNet' - WISRNet and corresponding GEAP implementation files:
        'WISERNet_train256.py' - The training model.
        'GEAP_element_WISERNet.py' - The element-wise embedding.
        'GEAP_vector_WISERNet.py' - The vector-wise embedding.
        'WISERNet_test256.py' - The testing model. 
        'WISERNet_adv.py' - ADVC with pixel-wise embedding.
        'WISERNet_adv_cpv0.py' - ADVC with vector-wise embedding.
        'WISERNet_aen.py' - AENC with pixel-wise embedding.
        
    'models/XuNet-C' - XuNet-C and corresponding GEAP implementation files:
        'XuNet_train256_color.py' - The training model.
        'XuNet_test256_color.py' - The testing model.
        'XuNet_adv256_color.py' - ADVC with the element-wise embedding.
        'XuNet_adv256_cpv0.py' - ADVC with the vector-wise embedding.
        'Xunet_ite256_v02.py' - GEAP with the element-wise embedding.
        'XuNet_adv256_cpv_v02.py' - GEAP with the vector-wise embedding.
        
    'models/SRNet' - SRNet for color images:
        'SRNet_train_color.py' - The training model.
        'SRNet_test_color.py' - The testing model.


'prep/' - Code to generate stego images and .mat files.



