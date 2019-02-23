# CycleGAN
A TensorFlow implementation of CycleGAN.

## Packages Needed
The packages needed to run the CycleGAN model include:
- numpy
- tensorflow
- sklearn
- pillow

## Downloading Datasets
To download a dataset to train/test CycleGAN on, run the following command:  
```
$ bash download_dataset.sh [dataset name]
```

The list of available datasets for download are:  
- apple2orange
- summer2winter_yosemite
- horse2zebra
- monet2photo
- cezanne2photo
- ukiyoe2photo
- vangogh2photo
- maps
- cityscapes
- facades
- iphone2dslr_flower
- ae_photos

## Training CycleGAN
To train CycleGAN, run the command:  
```
$ python train.py [arguments (e.g. --data_A=./data/apple2orange/trainA)]
```

To continue training where you left off, run the command:  
```
$ python train.py --load_model=[checkpoint directory (e.g. 20022019-0801)] \
                  [arguments (e.g. --data_A=./data/apple2orange/trainA)]
```

To get the list of arguments, run the command:  
```
$ python train.py -h
```

## Testing CycleGAN
To test a trained CycleGAN model, run the command:  
```
$ python test.py --load_model=[checkpoint directory (e.g. 20022019-0801)] \
                 [arguments (e.g. --data_A=./data/apple2orange/testA)]
```

To get the list of arguments, run the command:  
```
$ python test.py -h
```

## Original Paper
The original CycleGAN paper was written by Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros.  
The paper can be found here: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf).
