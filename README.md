#  [COMP6248-Reproducability-Challenge]Unsupervised and domain-adaptive pedestrian re-identification

*Based on the paper:*  [Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification](https://openreview.net/forum?id=rJlnOhVYPS)  which is accepted by  [ICLR-2020](https://iclr.cc/) .




## Environmental requirements
* ’Python==3.6.0‘
* ’PyTorch==1.1‘
* ‘numpy’, 
* ‘torch==1.1.0’, 
* ‘torchvision==0.2.2’, 
* ‘six’, 
* ‘h5py’, 
* ‘Pillow’, ‘
* scipy’,
* ‘scikit-learn’, 
* ‘metric-learn’
You can use the following code to deploy the environment

```
`python setup.py install`
```

## Data set preparation
The data structure we use is as follows,You can download the following dataset from my Google Cloud Disk.
[duke.tar - Google ](https://drive.google.com/file/d/17mHIip2x5DXWqDUT97aiqKsrTQvSI830/view?usp=sharing)
[market1501.tar - Google ](https://drive.google.com/file/d/1kbDAPetylhb350LX3EINoEtFsXeXB0uW/view?usp=sharing)
[MSMT17_V1_ps_label.tar.gz - Google ](https://drive.google.com/file/d/1WUDSTRmiXsUSbGaa9oKIQMWhawlBdBag/view?usp=sharing)


```
MMT
-examples
--data
---dukemtmc
----DukeMTMC-reID
---market1501
----Market-1501
---msmt17
----MSMT17
```












