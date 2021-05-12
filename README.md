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
[duke.tar - Google 云端硬盘](https://drive.google.com/file/d/17mHIip2x5DXWqDUT97aiqKsrTQvSI830/view?usp=sharing)
[market1501.tar - Google 云端硬盘](https://drive.google.com/file/d/1kbDAPetylhb350LX3EINoEtFsXeXB0uW/view?usp=sharing)
[MSMT17_V1_ps_label.tar.gz - Google 云端硬盘](https://drive.google.com/file/d/1WUDSTRmiXsUSbGaa9oKIQMWhawlBdBag/view?usp=sharing)


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


## Data set  load
In this project, the following four files are used to read the data set separately[datasets ](https://github.com/weifenglin/COMP6248-Reproducability-Challenge/tree/main/datasets)

It should be noted that you need to set the following code according to your data set path.
There are three things that need to be modified
* dataset_dir=The absolute path of the corresponding dataset file
* str=dir_path+'/*/*/*.jpg'
* pattern = re.compile(r'([\d]+)c(\d)')

```
msmt17.py
market1501.py
dukemtmc.py
....
dataset_dir = '/content/MMT/examples/data/dukemtmc/'
....
def _process_dir(self, dir_path, relabel=False):
        str=dir_path+'/*/*/*.jpg'  
# Here * represents a layer of directory structure, set here according to the location of the data set
        img_paths = glob.glob(str)
        print(dir_path)
        pattern = re.compile(r'([\d]+)c(\d)')
# returns a regular expression object
        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))
        return dataset

```

The following content is returned after successfully reading the data collection.
```
=> DukeMTMC-reID loaded
Dataset statistics:
  ----------------------------------------
  subset   | # ids | # images | # cameras
  ----------------------------------------
  train    |   702 |    16522 |         8
  query    |   702 |     2228 |         8
  gallery  |  1110 |    17661 |         8
  ----------------------------------------
/usr/local/lib/python3.7/dist-packages/torchvision/transforms/transforms.py:258: UserWarning: Argument interpolation should be of type InterpolationMode instead of int. Please, use InterpolationMode enum.
  "Argument interpolation should be of type InterpolationMode instead of int. "
=> Market1501 loaded
Dataset statistics:
  ----------------------------------------
  subset   | # ids | # images | # cameras
  ----------------------------------------
  train    |   751 |    12936 |         6
  query    |   750 |     3368 |         6
  gallery  |   751 |    15913 |         6
  ----------------------------------------
```

## Prepare Pre-trained Models
When /training with the backbone of  / [IBN-ResNet-50](https://arxiv.org/abs/1807.09441) , you need to download the  [ImageNet](http://www.image-net.org/)  pre-trained model from this  [link](https://drive.google.com/drive/folders/1thS2B8UOSBi_cJX6zRy6YYRwz_nVFI_S)  and save it under the path of logs/pretrained/.

```
mkdir logs && cd logs
mkdir pretrained
```

The file tree should be
```
MMT/logs
--pretrained
---esnet50_ibn_a.pth.tar
```





