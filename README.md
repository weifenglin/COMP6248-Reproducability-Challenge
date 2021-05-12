# COMP6248-Reproducability-Challenge
COMP6248-Reproducability-Challenge
# 计算机视觉OpenCV+TensorFlow
岗位：图形图像算法工程师
岗位需求：OpenCV+TensorFlow

*OpenCV入门+TensorFlow入门*

Open Source Computer Vision Library 跨平台的 [计算机视觉](https://zh.wikipedia.org/wiki/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89) 库

学习方式：原理+api+源码=100%掌握

网址：[ImageNet](http://image-net.org/)*ImageNet*is an image database organized according to the [WordNet](http://wordnet.princeton.edu/) hierarchy (currently only the nouns), in which each node of the hierarchy is depicted by hundreds and thousands of images. Currently we have an average of over five hundred images per node. We hope ImageNet will become a useful resource for researchers, educators, students and all of you who share our passion for pictures.
 [Click here](http://image-net.org/about-overview) to learn more about ImageNet, [Click here](http://image-net.org/about-join) to join the ImageNet mailing list.

OpenCV-和图像处理和计算机视觉相关的就可以用OpenCV

## 搭建OpenCV和TensorFlow的一站式开发环境Anaconda
1.下载安装Anaconda
2.下载安装TensorFlow和OpenCV
3.下载安装notebook

my环境：macOS Mojave   Python 3.6

[anaconda3+opencv安装方法 - ytrewq121的博客 - CSDN博客](https://blog.csdn.net/ytrewq121/article/details/78966309)



以后的开发都使用notebook进行。

[image:5A6180C9-C08A-4995-A7B3-A546DCB9C24E-352-00000161D8057856/06651456-D823-408C-BF50-63AF45EC5D8E.png]

Anaconda提供环境，然后在安装的环境中搭建notebook

[image:57F0CB13-FF08-4DBD-8263-E68BAEF9DE5D-352-0000017EA8F49ED7/E7CC9ED6-EFD1-4588-9ADB-5A4E1434E50F.png]

## helloworld程序

三步骤思路

1.import  
2.string
3.print


[image:0C7FF978-C0C8-4ABA-8530-E54E0AD3065A-352-000001C05152382D/46C638ED-D187-419A-AB23-3966ED74FA27.png]

`import tensorflow as tf`

`hello = tf.constant('hello tf!')`

`sess = tf.Session()`

`print(sess.run(hello))`
`#常量的定义  session 的使用  print 的使用`

[image:5921AAC6-8A18-442C-8E70-AB109178641E-352-0000061BE01FF2EB/AA8DC3FC-B0EF-482D-BDE3-6ECFF795C0AE.png]

上面测试了 OpenCV安装成功

## 通过开发环境-图片的读取和展示

`#1 引入OpenCV 2 API 3 stop 停止程序`

`import cv2`
`img = cv2.imread('123.png',1)# read image 1  name0 gray 1 color `
`cv2.imshow('image',img)#第一个参数秒回窗体名称  第二个描绘图片类型`
`cv2.waitKey(0)#执行程序的暂停`


[image:5C0D3AB1-155E-4C14-AB78-5F582BA930ED-352-000009DA608F0A13/184B4551-38A8-45F9-B203-28FD474164F4.png]


## OpenCV第三方开发库

 [https://opencv.org/](https://opencv.org/) 
[image:687D8D2B-7F02-448F-8B87-FB0363120274-352-000009E596566E2B/64F60316-A9A5-4112-A370-82E59052E7E4.png]

点击进入release

[image:7DBE3ED5-C58A-4BAF-8985-8B66FDE0DBEB-352-00000A91F31DD38D/8D095453-F6F6-4E5C-8B86-EE6F9EB99647.png]


core是最重要的文件

ml机器学习

dnn深度学习

photo和图片处理相关的

stitchiong拼接模块


## 图片的写入
`import cv2`
`img = cv2.imread('123.png',1)# read image 1  name0 gray 1 color `
`cv2.imshow('image',img)#第一个参数秒回窗体名称  第二个描绘图片类型`
`cv2.waitKey(0)#执行程序的暂停`


以上程序完成了

1.文件的读取
2.封装格式解析
3.数据解码
4.数据加载

那么2 3 步骤是怎么回事呢？

图片格式有JPG png 封装格式  包括两个bufen:1.文件头 2.文件数据

解码器可以根据1文件头中的信息解码文件数据

`import cv2`

`img = cv2.imread('123.png',1)`

`cv2.imwrite('456.jpg',img)# 1name2data `

以上代码实现了  imread读取图片到 img

然后使用 imwrite将图片输出


## 不同质量的图片的保存
实现不同压缩比的JPG 和 陪png的保存
`import cv2`
`#Duqu caise tupian`
`img = cv2.imread('123.png',1)`

`#shiyogn yixia tupian zhilaingde shezi 最后参数为0-100`
`cv2.imwrite('789.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,100])`
`#1M 压缩到100k  基本上还可以保证图片质量`

————
`#1.png 无损压缩 JPG 有损压缩  2.透明度属性`
`import cv2`
`#Duqu caise tupian`
`img = cv2.imread('123.png',1)`
`#shiyogn yixia tupian zhilaingde shezi 最后参数为0-100`
`cv2.imwrite('789.png',img,[cv2.IMWRITE_PNG_COMPRESSION,0])`
`#jpg  0 压缩比高  压缩范围 0-100 `
`#png 0 压缩比底 压缩范围 0-9`

## 像素的基本操作/基本概念
1.像素  就是一个个方块

2.RGB
像素点
rgb参数合成   

3.颜色深度
例如 RGB  203 203 9  颜色深度的范围 0-255 8bit
所以对于一个RGB能够表示 255的3次方种颜色

4.WH  640*480  高宽
所以高度为640  宽度为480. 表示的是像素点
5.1.14M=720*547*3*8/8(B)

以上是jpg图片的格式

6.对于png图片
还有一个分量 alpha     bgr

## 图片像素读取以及写入的demo

[image:CCBFA0A3-79B8-4D71-A178-B3A1B6CD6FD9-352-00001246BB228F67/577D44A6-EA0E-4C27-897A-C057B100CD8E.png]

使用cv2.imread()读取的img中存储的是三元组矩阵
对于RGB格式，三元组的存储是bgr的格式

使用for循环实现功能
从10 100-110 110结束

`import cv2`
`img = cv2.imread('123.png',1)`
`#以上是解析后的图像 img 是矩阵的结构`
`(b,g,r)=img[100,100]`
`print(b,g,r)`
`for i in range(1,100):`
`    img[10+i,100] = (255,0,0)`
`cv2.imshow('image',img)`
`cv2.waitKey(0)#dai代表1000s`

[image:994BF5DD-5745-4EB8-86C2-C50F020F5BC4-352-0000138C7C96757A/F03106BC-D975-44F7-9FB5-EBC890FFE3B4.png]

## TensorFlow常量变量

学习方法：类比 api  原理
数据类型：运算符  流程 字典 数组

[image:85BF386F-5AF4-433A-B87B-5480D70A5FDE-352-000013EEFFB3E874/ED701D5F-40D9-4CB3-AFA9-1FE6B30FE8D3.png]

`import tensorflow as tf`

`data1 = tf.constant(2.5)`

`data2 = tf.Variable(10,name='var')`

`print(data1)`
`print(data2)`

`Tensor(“Const_4:0”, shape=(), dtype=float32)`
`<tf.Variable ‘var_4:0’ shape=() dtype=int32_ref>`

并没有将数据打印出来 ，只是打印出来了数据类型！
应当使用session来处理



`import tensorflow as tf`
`data2 = tf.Variable(10,name='var')`
`#初始化`

`init = tf.global_variables_initializer()`
`sess.run(init)`
`print(sess.run(data2))`


## TensorFlow 运算实质
### 数据+计算图

TensorFlow本质：张量tensor + 数据运算（运算图）
tensor的本质：数据 （可以为常量、变量、一维、多维）

Session是执行的核心，相当于交互环境（op计算图）
TensorFlow中所有的数据必须初始化后才能使用，例如：
`init = tf.global_variables_initializer()`
`sess.run(init)`
`print(sess.run(data2))`


session 使用完成后都有关闭session.close()

或者使用
with sess:
  …

‘’‘ 表示多行注释’‘’

## TensorFlow的四则运算

`import tensorflow as  tf`

`data1=tf.constant(6)`
`data2=tf.Variable(2)`

`dataadd=tf.add(data1,data2)`
`datacopy=tf.assign(data2,dataadd) #dataadd->data2`
`datamul=tf.multiply(data1,data2)`
`datasub=tf.subtract(data1,data2)`
`datadiv=tf.divide(data1,data2)`

`init=tf.global_variables_initializer()`

`#使用session计算图`

`with tf.Session() as sess:`
`    sess.run(init)`
`    print(sess.run(dataadd))`
`    print(sess.run(datamul))`
`    print(sess.run(datasub))`
`    print(sess.run(datadiv))`
`    print('sess.run(datacopy)',sess.run(datacopy))  #已经将8（6+2）赋给了data2`
`    print('datacopy.eval()',datacopy.eval())   #将 14（6+8）赋给了data2`
`    print('',tf.get_default_session().run(datacopy))   #同上  datacopy.eval（）相当于tf.get_default_session().run(datacopy)使用系统默session`


`print('end!')`


## 矩阵基础（一）

实现placehlod

`#实现placehlod`
`import tensorflow as tf`
`data1=tf.placeholder(tf.float32)`
`data2=tf.placeholder(tf.float32)`
`dataAdd=tf.add(data1,data2)`
`with tf.Session() as sess:`
`    print(sess.run(dataAdd,feed_dict={data1:6,data2:2}))`
`    #1. dataAdd 2.  feed_dict={1 : canshu , 2:canshu }      `
`print('end!')`

`data2=tf.placeholder(tf.float32)`
然后使用：`dataAdd,feed_dict={data1:6,data2:2}` 对place的进行赋值


### 矩阵的定义

矩阵的学习 类比数组  M行N列  [] [] 
[[6,6]] 一行两列的矩阵表

`#矩阵的学习 类比数组  M行N列  [] [] `
`#[[6,6]] 一行两列的矩阵表`
`import tensorflow as tf`
`data1=tf.constant([[6,6]])`
`data2=tf.constant([[2],`
`                   [2]])`
`data3=tf.constant([[3,3]])`
`data4=tf.constant([[1,2],`
`                   [3,4],`
`                   [5,6]])`

`print(data4)#打印矩阵的维度`

`with tf.Session() as sess:`
`    print(sess.run(data4))#将整个data4打印出来`
`    print(sess.run(data4[0]))#打印某一行`
`    print(sess.run(data4[:,0]))#打印第一列`
`    print(sess.run(data4[0,0]))#打印第一行，第一列`

所有的矩阵下标都是从0开始的

## 矩阵基础（二）//矩阵运算

矩阵的乘法   M*N✖️N*B=M*B的矩阵
其余的加减乘除都很简单

`#矩阵的学习 类比数组  M行N列  [] [] `
`#[[6,6]] 一行两列的矩阵表`
`import tensorflow as tf`
`data1=tf.constant([[6,6]])`
`data2=tf.constant([[2],`
`                   [2]])`
`data3=tf.constant([[3,3]])`
`data4=tf.constant([[1,2],`
`                   [3,4],`
`                   [5,6]])`

`print(data4)#打印矩阵的维度`

`#矩阵的基本乘法与加法`
`matMul=tf.matmul(data1,data2)`
`matAdd=tf.add(data1,data3)`

`#矩阵的普通乘法，不满足矩阵的乘法。`
`matMul2=tf.multiply(data1,data2)`

`with tf.Session() as sess:`
`    print(sess.run(data4))#将整个data4打印出来`
`    print(sess.run(data4[0]))#打印某一行`
`    print(sess.run(data4[:,0]))#打印第一列`
`    print(sess.run(data4[0,0]))#打印第一行，第一列`
`    print('end!')`
`    print(sess.run(matAdd))`
`    print(sess.run(matMul))`
`    print(sess.run(matMul2))`
`    `

[image:6810AA8A-D697-4363-BFA3-C58D1EE82371-270-000005D0D33E788F/9FED7590-67B8-4379-B990-3DCC5F1E4E75.png]

` print(sess.run([matMul,matAdd]))`
通过上面中括号的形式可以一次打印多个内容

[image:C5A3E482-AFDB-433D-87EE-04BC9EBA476D-270-000005E5902348E1/87A8A969-A7C0-4CBA-BD00-683B2A08CB84.png]


## 矩阵基础（三）//特殊矩阵 
[image:770E2BA6-6AFF-408D-BD80-A0C1706B902E-270-0000060E31ED1BC5/D42C567E-FC02-47A7-9613-B0F31AB56945.png]

那么我们要定义矩阵特别大呢？

`import tensorflow as tf`

`mat0=tf.constant([[0,0,0],[0,0,0]])`
`with tf.Session() as sess:`
`    print(sess.run(mat0))`

[image:12F16B79-5CC4-472C-A889-D59D392626A6-270-000006277CE4F651/17FAFE09-9647-40F9-B6EC-DB16A0676FB6.png]

`mat1=tf.zeros([200,300])`
通过上面的语句定义了 200行 300列的空矩阵

定义为0的空矩阵的方法
`mat1=tf.zeros([2,3])`
定义为1的矩阵的方法、
`mat2=tf.ones([3,2])`
填充矩阵
`mat3=tf.fill([2,3],15)`

[image:12295FE5-013E-4EAC-8F70-71AEFE0A18B4-270-000006506ECA464C/C2B118F9-BF06-4D7E-8CA2-B7E5766D980F.png]


`mat1=tf.constant([[2],[3],[4]])`
`mat2=tf.zeros_like(mat1)`

使用zeros_like(),填充一个形状相同的矩阵

`mat3=tf.linspace(0.0,2.0,11)`将0.0-2.0之间的数据分成11等分
[image:3E759537-1118-4B4C-9FE7-95B09CC9C2CB-270-00000692EDC91A5A/FC25D740-23CE-41E7-8F98-E7D9361A72CC.png]

`mat4=tf.random_uniform([2,3],-1,2)`建立一个随机矩阵，矩阵的大小是2，3，范围是-1到2
[image:154E5BD6-52CC-47E2-A002-7E5BB35EE1F2-270-000006989D50366C/1471CA2E-0EA4-4514-B587-EBFDA724028B.png]


## Numpy模块的使用

    1.首先，检查numpy是否安装正常。
[image:97341FD0-4C11-42F3-ACE5-5A41600E590F-270-00000D57B5A13D2E/450892CB-9D52-4606-A202-23607EE686F7.png]
通过上图可以看到，numpy，安装正常。

 2 . 采用类比的方式，类比数据库。
[image:B417BBC7-DB35-4778-B1D3-3DE6A5EFD7A5-270-00000DD5795B357F/075A5ADB-F7ED-47F7-8100-703BFB9835C3.png]
`#CURD`
`import numpy as np`
`data1=np.array([1,2,3,4,5])`
`print(data1)`
`data2=np.array([[1,2],`
`                [3,4]])`
`print(data2)`
`#打印矩阵的维度`
`print(data1.shape,data2.shape)`
`#zero ones`
`print(np.zeros([2,3]),np.ones([2,2]))`
`#改查`
`data2[1,0]=5`
`print(data2)`
`print(data2[1,1])`
`#基本运算  加减乘除`
`data3=np.ones([2,3])#定义一个单位矩阵`
`print(data3*2)#矩阵乘以一个数字`
`print(data3+2)`
`print(data3/3)`
`#矩阵之间的加减乘除`
`data4=np.array([`
`    [1,2,3],`
`    [4,5,6]`
`])`
`print(data3+data4)`


## matplotlib绘图模块的使用

首先查看环境是否安装

[image:320F01A0-355F-4B82-BF94-40E9B8A86A8D-270-00000E2332466592/C28554EC-8060-4EC6-B828-F5BD32701E18.png]

`    import numpy as np`
`import matplotlib.pyplot as plt`
`x=np.array([1,2,3,4,5,6,7,8])`
`y=np.array([3,5,7,2,6,8,9,4])`
`#使用matplotlib 中的  plt.plot方法->绘制的是折现图`

`plt.plot(x,y,'r')#折线 1 x 2 y 3 color`



[image:066C1B4F-18A8-48C8-B68A-FD12D59BE41E-270-00000E5676BD78C0/16582875-C7DC-4677-A7F8-CF402F04554C.png]

    
`plt.plot(x,y,'g',lw=10)#4 line w  表示折线的宽度`

[image:A8160EA0-CCED-49F2-AFF3-193E120DA858-270-00000E6BE0797DDD/09BE5AA8-E251-40F4-A3E7-E6978B9D2CB9.png]

`x=np.array([1,2,3,4,5,6,7,8])`
`y=np.array([13,15,17,22,36,28,19,34])`

`plt.bar(x,y,0.9,alpha=1,color='b')#5 颜色  4 alpha透明度 3 描述的是占用宽度的比例  1 2 xy轴坐标`


[image:F9079372-B19E-481D-894D-C68EBA2E2CD1-270-00000E90E1C39A06/234423E9-43CB-4110-B021-603241B9B3B9.png]

matplotlib画廊，matplotlib.org/  中有实例画廊。
[image:20FEDEBD-824A-43B2-A5F8-C80C43FC293A-270-00000FE1BF28E22A/4EDB834B-5205-400A-BAEA-2D4C2EABDF08.png]
[Gallery — Matplotlib 3.0.3 documentation](https://matplotlib.org/gallery/index.html)




## 人工神经网络逼近股票价格
红色矩形表示股票上涨，蓝色表示股票价格下跌
突出来的表示最高价或者最低价

`data=np.linspace(1,15,15)  #从1 到 15`

[image:85CA961F-EDBD-4330-8593-53C8D71450B9-270-00000EE1581B71F4/CC6736F7-B67A-453D-B6CB-850F07424A22.png]


`#1.功能 数据的加载和`

`#首先导入数据`
`import tensorflow as tf`
`import numpy as np`
`import matplotlib.pyplot as plt`

`#日期`
`date=np.linspace(1,15,15)  #从1 到 15`

`#收盘价格`
`endPrice = np.array([2511.90,2538.26,2510.68,2591.66,2732.98,2701.69,2701.29,2678.67,2726.50,2681.50,2739.17,2715.07,2823.58,2864.90,2919.08]`
`)`

`#开盘价格`
`beginPrice = np.array([2438.71,2500.88,2534.95,2512.52,2594.04,2743.26,2697.47,2695.24,2678.23,2722.13,2674.93,2744.13,2717.46,2832.73,2877.40])`
`print(date) #打印日期`
`plt.figure()#定义一个绘图`

`for i in range(0,15):`
`    # 1 柱状图`
`    #用折线表示柱状  日期是二维的，因为开盘一个价格，收盘一个价格`
`    dateOne = np.zeros([2])`
`    dateOne[0]=i;`
`    dateOne[1]=i;`
`    `
`    princeOne=np.zeros([2])`
`    princeOne[0]=beginPrice[i]`
`    princeOne[1]=endPrice[i]`
`    `
`    #使用if判断应该用什么数据来描述？`
`    if endPrice[i]>beginPrice[i]:`
`        plt.plot(dateOne,princeOne,'r',lw=8)`
`    else:`
`        plt.plot(dateOne,princeOne,'g',lw=8)`
`        `
`plt.show()`
`    `



[image:E290AE57-EF20-4540-952D-6C0918AAB2D2-270-00000F57A5F60EFD/018155EE-68E2-4327-960C-568BBF74284F.png]



### 创建一个简单的人工神经网路

输入层->隐藏层->输出层

输入矩阵（15*1）   隐藏层矩阵（1*10）  输出层（15*1）

那么实现了什么功能？

输入：天数
输出：每天股价
*计算公式*
A（15*1）*w1（1*10）+b1（1*10）=B（15*10）  A为输入层   B为隐藏层  C输出层
B（15*10）*w2（10*1）+b2（15*1）=C（15*1）
…..
A为输入层   B为隐藏层  C输出层

注意矩阵的维度！

开始计算的时候，给定一个初始化的w1,w2,b1,b2

1次循环 A 1天 w1 w2 b1 b2 (0.1 0.1 0.2 0.3 )->c
     2400  2511  差 111点

2次循环 ”梯度下降法“  目的111减少  

终止条件：1. 通过for循环的次数  2. 差异 2% 输出最终的w1,w2,b1,b2


`# layer1：激励函数+乘加运算`
`import tensorflow as tf`
`import numpy as np`
`import matplotlib.pyplot as plt`
`date = np.linspace(1,15,15)`
`endPrice = np.array([2511.90,2538.26,2510.68,2591.66,2732.98,2701.69,2701.29,2678.67,2726.50,2681.50,2739.17,2715.07,2823.58,2864.90,2919.08]`
`)`
`beginPrice = np.array([2438.71,2500.88,2534.95,2512.52,2594.04,2743.26,2697.47,2695.24,2678.23,2722.13,2674.93,2744.13,2717.46,2832.73,2877.40])`
`print(date)`
`plt.figure()`
`for i in range(0,15):`
`    # 1 柱状图`
`    dateOne = np.zeros([2])`
`    dateOne[0] = i;`
`    dateOne[1] = i;`
`    priceOne = np.zeros([2])`
`    priceOne[0] = beginPrice[i]`
`    priceOne[1] = endPrice[i]`
`    if endPrice[i]>beginPrice[i]:`
`        plt.plot(dateOne,priceOne,'r',lw=8)`
`    else:`
`        plt.plot(dateOne,priceOne,'g',lw=8)`
`#plt.show()`
`# A(15x1)*w1(1x10)+b1(1*10) = B(15x10)`
`# B(15x10)*w2(10x1)+b2(15x1) = C(15x1)`
`# 1 A B C `
`dateNormal = np.zeros([15,1])`
`priceNormal = np.zeros([15,1])`
`for i in range(0,15):`
`    dateNormal[i,0] = i/14.0;`
`    priceNormal[i,0] = endPrice[i]/3000.0;`
`x = tf.placeholder(tf.float32,[None,1])`
`y = tf.placeholder(tf.float32,[None,1])`
`# B`
`w1 = tf.Variable(tf.random_uniform([1,10],0,1))`
`b1 = tf.Variable(tf.zeros([1,10]))`
`wb1 = tf.matmul(x,w1)+b1`
`layer1 = tf.nn.relu(wb1) # 激励函数`
`# C`
`w2 = tf.Variable(tf.random_uniform([10,1],0,1))`
`b2 = tf.Variable(tf.zeros([15,1]))`
`wb2 = tf.matmul(layer1,w2)+b2`
`layer2 = tf.nn.relu(wb2)`
`loss = tf.reduce_mean(tf.square(y-layer2))#y 真实 layer2 计算`
`train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)`
`with tf.Session() as sess:`
`    sess.run(tf.global_variables_initializer())`
`    for i in range(0,10000):`
`        sess.run(train_step,feed_dict={x:dateNormal,y:priceNormal})`
`    # w1w2 b1b2  A + wb -->layer2`
`    pred = sess.run(layer2,feed_dict={x:dateNormal})`
`    predPrice = np.zeros([15,1])`
`    for i in range(0,15):`
`        predPrice[i,0]=(pred*3000)[i,0]`
`    plt.plot(date,predPrice,'b',lw=1)`
`plt.show()`

[image:09D4D96D-B795-4FE9-A555-359735FFE09F-270-000012409C405946/34B88286-7D4C-4A89-AE74-5907C9C8EC54.png]


# 图片的几何变换
图片的几何变换是计算机视觉的基础。
本章案例：Hog+SVM小狮子识别

矩阵的仿射等变换->完成梯形到正方形的变换

包含内容：图片位移、缩放、镜像、剪切、仿射变换 

图片信息：图片 宽高、通道个数


## 图片缩放

`#1 load 加载 2 info信息 3 resize方法的使用  4 check`
`import cv2`
`img = cv2.imread('image0.jpg',1)  #加载 矩阵`
`imgInfo=img.shape`
`print(imgInfo)`
`height=imgInfo[0]`
`width=imgInfo[1]`
`mode=imgInfo[2]#图片的组成方式`
`#1 放大 缩小 两种方式   2等比例缩放和非等比例缩放`
`dstHeight=int(height*0.5)`
`dstWidth=int(width*0.5)`
`#最近邻域差值 双线性插值 橡塑关系差值 立方差值`
`dst=cv2.resize(img,(dstWidth,dstHeight))`
`cv2.imshow('image',dst)`
`cv2.waitKey(0)`

## 最近领域差值
越是简单的模型越适合用来举例子，我们就举个简单的图像：3X3 的256级灰度图，也就是高为3个象素，宽也是3个象素的图像，每个象素的取值可以是 0－255，代表该像素的亮度，255代表最亮，也就是白色，0代表最暗，即黑色。假如图像的象素矩阵如下图所示（这个原始图把它叫做源图，Source）：
234   38    22
67     44    12
89     65    63

这个矩阵中，元素坐标(x,y)是这样确定的，x从左到右，从0开始，y从上到下，也是从零开始，这是图象处理中最常用的坐标系，就是这样一个坐标：

  ----------------------＞X
  |
  |
  |
  |
  |
∨Y

如果想把这副图放大为 4X4大小的图像，那么该怎么做呢？那么第一步肯定想到的是先把4X4的矩阵先画出来再说，好了矩阵画出来了，如下所示，当然，矩阵的每个像素都是未知数，等待着我们去填充（这个将要被填充的图的叫做目标图,Destination）：
?        ?        ?       ?
?        ?        ?       ?
?        ?        ?       ?
?        ?        ?       ? 
               
        然后要往这个空的矩阵里面填值了，要填的值从哪里来来呢？是从源图中来，好，先填写目标图最左上角的象素，坐标为（0，0），那么该坐标对应源图中的坐标可以由如下公式得出：                                      
srcX=dstX* (srcWidth/dstWidth) , srcY = dstY * (srcHeight/dstHeight)
好了，套用公式，就可以找到对应的原图的坐标了(0*(3/4),0*(3/4))=>(0*0.75,0*0.75)=>(0,0)
,找到了源图的对应坐标,就可以把源图中坐标为(0,0)处的234象素值填进去目标图的(0,0)这个位置了。

接下来,如法炮制,寻找目标图中坐标为(1,0)的象素对应源图中的坐标,套用公式:
(1*0.75,0*0.75)=>(0.75,0)
结果发现,得到的坐标里面竟然有小数,这可怎么办?计算机里的图像可是数字图像,象素就是最小单位了,象素的坐标都是整数,从来没有小数坐标。这时候采用的一种策略就是采用四舍五入的方法（也可以采用直接舍掉小数位的方法），把非整数坐标转换成整数，好，那么按照四舍五入的方法就得到坐标（1，0），完整的运算过程就是这样的：
(1*0.75,0*0.75)=>(0.75,0)=>(1,0)
那么就可以再填一个象素到目标矩阵中了，同样是把源图中坐标为(1,0)处的像素值38填入目标图中的坐标。
         
依次填完每个象素，一幅放大后的图像就诞生了，像素矩阵如下所示：
234    38     22     22  
67      44     12     12  
89      65     63     63  
89      65     63     63  

这种放大图像的方法叫做最临近插值算法，这是一种最基本、最简单的图像缩放算法，效果也是最不好的，放大后的图像有很严重的马赛克，缩小后的图像有很严重的失真；效果不好的根源就是其简单的最临近插值方法引入了严重的图像失真，比如，当由目标图的坐标反推得到的源图的的坐标是一个浮点数的时候，采用了四舍五入的方法，直接采用了和这个浮点数最接近的象素的值，这种方法是很不科学的，当推得坐标值为 0.75的时候，不应该就简单的取为1，既然是0.75，比1要小0.25 ，比0要大0.75 ,那么目标象素值其实应该根据这个源图中虚拟的点四周的四个真实的点来按照一定的规律计算出来的，这样才能达到更好的缩放效果。双线型内插值算法就是一种比较好的图像缩放算法，它充分的利用了源图中虚拟点四周的四个真实存在的像素值来共同决定目标图中的一个像素值，因此缩放效果比简单的最邻近插值要好很多。



## 双线性插值

[image:CC4D1E9A-45C5-4381-8031-107E16A80952-270-000015AB494DF824/05BEBCFA-3685-4273-8C7B-D9D51A2A43FC.png]

[image:A94F9877-984C-4C98-A284-64B62E6B1317-270-000015AE25F49B45/2701989E-BA55-4CCE-B303-9749EA86BF0A.png]

[image:97FAF70C-6242-4794-8AB8-BB873BEC3583-270-000015B04F9C7C26/A2DEB205-B2EC-4F94-9513-FCEF0FF228C6.png]


## 使用源码来实现最近领域差值
`# 1 获取图片info信息 2 创建和缩放之后大小相同的空白模板  3 需要重新计算xy坐标`
`import cv2`
`import numpy as np`
`img=cv2.imread('image0.jpg',1)`
`imgInfo=img.shape`
`height=imgInfo[0]`
`width=imgInfo[1]`
`dstHight=int(height/2)`
`dstWidth=int(width/2)`
`dstImage=np.zeros((dstHight,dstWidth,3),np.uint8)#uint8的范围 0-255`
`for i in range(0,dstHight):#高度对应的是行信息`
`    for j in range(0,dstWidth) :`
`        iNew=int(i*(height*1.0/dstHight))`
`        jNew=int(j*(width*1.0/dstWidth))`
`        dstImage[i,j]=img[iNew,jNew]`
`        `
`cv2.imshow('dst',dstImage)`
`cv2.waitKey(0)`
`#1 opencv API resize 2 算法原理  3 用自己源码形式来实现图片缩放过程`
`    `

学习一门知识可以从以下三个方面
1 opencv API resize 
2 算法原理  
3 用自己源码形式来实现图片缩放过程

## 图片剪切
[image:0DCB0368-CB34-4077-82A2-B4E263D734F5-270-000016AEB778D11B/A2136E18-7E03-4920-A476-4711080A199A.png]
`import cv2`
`img=cv2.imread('image0.jpg',1)`
`imgInfo=img.shape`
`dst=img[100:200,100:300]`
`cv2.imshow('image',dst)`
`cv2.waitKey(0)`

## 图片的移位

1API 2算法原理 3源代码

`import cv2`
`import numpy as np`
`img=cv2.imread('image0.jpg',1)`
`cv2.imshow('src',img)`
`imgInfo=img.shape`
`height=imgInfo[0]`
`width=imgInfo[1]`
`####`
`matShift=np.float32([[1,0,100],[0,1,200]])#2*3  水平方向移动100像素  竖直方向移动200像素`

`dst=cv2.warpAffine(img,matShift,(height,width))#1  data信息 2  mat移位信息 3 info`
`cv2.imshow('dst',dst)`
`cv2.waitKey(0)`

2`#API级别`
`#[1,0,100],[0,1,200]  2*2 2*1`
`# [[1,0],[0,1]] 2*2  A`
`#[[100],[200]] 2*1  B`
`#xy C`
`#A*C+B = xy   [[1*x+0*y],[0*x+1*y]]+[[100],[200]]`
`# = [[x+100],[y+200]]`

`#像素级别`
`#（10，20）->(110,120)`


## 图片镜像
### 实现的思路步骤：
1 创建足够大的”画板“
2 将一幅图像分别从前向后、从后向前绘制
3 绘制中心分割线

## 图片的缩放
`#[[A1 A2 A3],[B1 B2 B3]]`
`#图片移位公式`
`#x->x*0.5 y->y*0.5`
`import cv2`
`import numpy as np`
`img=cv2.imread('image0.jpg',1)`
`cv2.imshow('src',img)`
`imgInfo=img.shape`
`height=imgInfo[0]`
`width=imgInfo[1]`
`#定义一个mat缩放矩阵`
`matScale=np.float32([[0.5,0,0],[0,0.5,0]])`
`dst=cv2.warpAffine(img,matScale,(int(width/2),int(height/2)))`
`cv2.imshow('dst',dst)`
`cv2.waitKey(0)`
[image:1B45F1E7-D6BC-49E0-AEBD-A3B28DBE66A2-270-000028697DFF3A7A/DC0B395C-ECFC-4A61-A671-88B0EB407121.png]

## 仿射变换

`import cv2`
`import numpy as np`
`img=cv2.imread('image0.jpg',1)`
`cv2.imshow('src',img)`
`imgInfo=img.shape`
`height=imgInfo[0]`
`width=imgInfo[1]`
`#src 原图像的三个点，映射到目标图像的三个位置上面 （左上角 左下角 右上角）`
`matSrc=np.float32([[0,0],[0,height-1],[width-1,0]])`
`matDst=np.float32([[50,50],[300,height-200],[width-300,100]])`
`#组合两个矩阵，定义一个放射变换矩阵`
`matAffine=cv2.getAffineTransform(matSrc,matDst)#本质上获取一个矩阵 矩阵组合  第一个参数 src 第二个参数 dst上面新的位置`

`dst=cv2.warpAffine(img,matAffine,(width,height))`
`cv2.imshow('dst',dst)`
`cv2.waitKey(0)`

确定三个点，就可以对矩阵进行任意的变化，任意的拉伸


[image:E776F0EE-5942-4FE6-A4AC-F9F2A48590B3-270-000028B58C933C08/38BBF7CB-8670-497B-BF42-5A87177E8EF9.png]




## 图片旋转



[image:66C6A661-5F53-41E6-917F-3E607E8DF1E5-636-000003D03612DD29/889E35A6-F7AE-48AE-9918-91F98A43C75F.png]

                      

`import cv2`
`import numpy as np`
`img=cv2.imread('image0.jpg',1)`
`cv2.imshow('src',img)`
`imgInfo=img.shape`
`height=imgInfo[0]`
`width=imgInfo[1]`

`#定义一个旋转矩阵，然后初始化矩阵`
`#2*3`

`matRotate=cv2.getRotationMatrix2D((height*0.5,width*0.5),45,0.5)#mat rotate 1 ceter图像的中心点 2 angle 3 sc `

`#100 *100`
`dst=cv2.warpAffine(img,matRotate,(height,width))`
`cv2.imshow('dst',dst)`
`cv2.waitKey(0)`

`matRotate=cv2.getRotationMatrix2D((height*0.5,width*0.5),45,0.5)`

函数getRotationMartrix2D（）函数，第一个参数是旋转的中心点，第二个参数是旋转的角度，第三个参数是缩放的大小


然后使用   函数  cv2.warpAffine（图片，变换，长宽）进行变换输出到dst矩阵中

最后使用imshow显示图片




[image:0933A7DB-4462-44D8-A99B-E7D3E30F2AC7-636-000004071A42E35D/CC6DB865-A3EC-4299-9F8E-28580BE3177B.png]


# 计算机视觉—机器学习

[image:3CE46167-22C5-4B76-99D7-497771947418-636-0000043675D6E5E1/04EA3BAB-74B3-4207-886C-69112286CAB3.png]

## 机器学习是什么？
 通过机器学习的方式，达到某种目的。
机器学习=样本+分类器+特征

深度学习=样本+神经网络模型

最大的区别是 有没有明确的特征 ，深度学习我们并不知道他把什么特征给提取出来了。

如何获取样本？  网络 公司内部  自己收集

如何通过视频获取我们的样本？

特征  Hog特征（主要用于行人检测）  

[image:C7461065-0827-4508-9C5F-7B15A38C8863-636-000004B70BF3668F/33E1B7AB-AD3A-4330-BECC-3DB7FD59E4DC.png]

Haar特征（主要用在人脸识别）

[image:08879D57-550B-4B78-96B2-920F51ACC235-636-00000484EF300C83/64868A35-EE1F-4D67-AE1B-573E896AD030.png]

[image:172B3F67-8401-4A64-B6D6-73498CE524DB-636-00000490D4E41A46/D7F4F072-EF5C-4E22-B91B-797DEE672246.png]


我们不知道人脸在图像的什么地方，什么大小，所以我们需要一个遍历过程来检测。

但是工作量太大，国外专家提出了积分图的方式。


[image:F11ABDD8-03BA-4731-81E0-CD50C9D878CC-636-0000049A3850E579/3F3DA0E7-A94B-4DD3-A9BB-40CB3EAC0B12.png]


首先使用  
* haar特征+adaboost分类器=人脸识别（adaboost分类器   强分类器、弱分类器）
* svm支持向量机+hog特征=小狮子识别

训练出来的数据我们需要进行检验，检验我们训练出来的是否有效

## 视频分解图片


[image:42C50001-1B21-4B10-B37A-0833FFCF1A07-636-000005C13547691D/9127F368-9214-4BD0-8D33-9E9E94AF1F83.png]

[image:3C31888B-5EDF-4EA2-AC9A-B5106A190B1C-636-000005C679A51FFD/83644189-931F-47F5-9275-A9FD3FE0B48D.png]


`import cv2`
`cap=cv2.VideoCapture("1.mp4")#获取一个视频打开cap  1 file name`
`isOpened=cap.isOpened#判断是否打开？`
`print(isOpened)`

`fps=cap.get(cv2.CAP_PROP_FPS) #贞信息`

`width =int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))`
`heigh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))`

`print(fps,width,heigh)`

`i=0`

`while(isOpened):`
`    if i==10:`
`        break;`
`    else:`
`        i=i+1`
`    (flag,frame)=cap.read()#读取每一帧，或者是读取每一张  flag表示是否读取成功  frame 图片的内容`
`    #保存读取进来的图片`
`    fileName='image'+str(i)+'.jpg'`
`    print(fileName)`
`    if flag==True:`
`        cv2.imwrite(fileName,frame,[cv2.IMWRITE_JPEG_LUMA_QUALITY,100])`
`print('end!')`


## 视频合成图片

`import cv2`
`img=cv2.imread('image1.jpg')`
`imgInfo=img.shape`
`size=(imgInfo[1],imgInfo[0])`
`print(size)`
`videoWrite=cv2.VideoWriter('2.mp4',-1,5,size)#写入对象的新创建   1  文件名称 2 选择一个可以使用的编码器 3 fps 4 size`

`for i in range(1,11):`
`    fileName='image'+str(i)+'.jpg'`
`    img=cv2.imread(fileName)`
`    videoWrite.write(img) #写入方法`
`    `
`print('end!')`


`videoWrite=cv2.VideoWriter('2.mp4',-1,5,size)#写入对象的新创建   1  文件名称 2 选择一个可以使用的编码器 3 fps 4 size`


## 基于Haad+Adaboost人脸识别

### 掌握和理解haad特征，什么是haad特征？

1.什么是特征？  特征=图像某个区域的像素点经过某种四则运算得到的特点->结果-》（具体值、向量、矩阵、多维）

所以，图像的特征实质就是某个数据（具体值、向量、矩阵、多维）

2.如何利用特征 区分目标?   阈（yu）值判决

3.如何得到这个判决？  使用机器学习，得到判决门限，根据计算特征进行阈值判决

[image:27CF9463-EFAD-458C-925A-1BDBEDBACB21-636-00000A9FFA9EC98F/01AF632D-9B72-4C4B-ADE7-F5D532493246.png]


[image:D30D0B16-A537-430F-968D-984069B16208-636-00000AA76D0717EF/FE59E54B-7FF9-4278-96D8-77B71A702230.png]

特征=白色-黑色

特征计算推导：

整个区域的权重设置为1  黑色部分的区域权重为-2

则第二种  特征=整个区域权重*权重1 + 黑色*权重2=（黑+白）*1+黑*（-2）
                          = 白色-黑色  = 第一种特征


### haad特征的遍历过程

haar模板需要 从上到下  从左到右 image size 模板  100*100 10*10 100次 step=10

1.考虑步长的问题  还需要缩放

举例  图片  1080*720  20  step 2 10*10

计算量 = 14（模板个数）*20（缩放个数）*（1080/2*720/2）*(100点+—运算  )  = 50-100亿次   
50-100亿   *  15  =1000亿次

然后在对haar特征进行判决  1000多亿次的计算。所以这种滑框的运算量太大啦。




## 积分图

[image:F29EEA8F-7FC0-44F2-8372-584C5FAB413A-636-00000CD1214D71C9/1F017F16-F8C8-4034-9356-D08A2B353D99.png]


A区域是第一小块
B区域是第一个横条
C区域是第一个竖条
D区域是第四个小方块

4=A-B-C+D=1+1+2+3+4-1-2-1-3=4  (3+-就可以算完)



## Adaboost分类器

haar特征+Adaboost分类器  用于人脸识别

苹果  苹果 苹果 香蕉
0.1   0.1  0.1   0.5

再次训练
正向样本减小，负样本进行减小

训练的终止条件：1迭代的最大次数  for count
2 最小的检测概率

1分类器的结构

2adaboost训练分类器的计算过程

3xml文件结构


---
  
haar》T1     and    haar》T2  ?  苹果：香蕉（由两个强分类器组成）
adaboost分类器的个数一般有15-20个
只有通过20个强分类器，才能够判断这是不是苹果

[image:3924FD94-DAF9-4B08-BB52-3BB56F6A1BB2-636-00001219FE248653/E3C686DD-51F9-4A68-B328-C7772EEC8E9B.png]



3个强分类器  1  x1  t1 2  x1 t2 3 x3 t3

x1>t1&&x2>t2&&x3>t3   ?   目标->苹果

一个强分类器又可以分成若干个弱分类器

一个弱分类器又可以分为若干个node特征

强分类器的作用：判决
弱分类器的作用：计算强分类器的特征

例如  x2 = sum(y1,y2,y3)  共同构成了强分类器的特征

y1  y2  y3 每一个弱分类器的tezheng de jisuan ?

y1(弱分类器)怎么计算？

3个haar特征构成一个弱分类器（opencv规定），每个Haar特征对应一个node节点

1node Haar1》nodeT z1=a1
1node Haar1《nodeT z1=a2

Z=z1+z2+z3>T   yi=AA
Z=z1+z2+z3<T   yi=BB

实际上在计算的时候，远远比上面更复杂。

## adaboost 分类器的训练

1  初始化数据权值分布
苹果 苹果 苹果 香蕉
0.1   0.1   0.1   0.1  （初试化的权值必须相等）

2  遍历阈值
minP t  （计算误差概率）

3 计算权重  G1（x）

OpenCV中使用exe文件，不需要编写代码，直接使用训练

4  权重分布  uodate
0.2 0.2 0.2 0.8（因为香蕉是错误的结果，所以被加强）

训练终止条件：  1 for count  2  查看是否满足终止条件

xml 文件的结构

[image:18EBBD7D-6C1C-46EA-961A-AC28CABB2B1E-636-00001351F008A62E/35982C91-F12F-4779-BCEE-149DE27A4205.png]




阈值门限的判决，解析过程OpenCV已经帮我们做了。

[image:DBCEC0A9-C9A5-4193-BE48-5B0380502BED-636-0000135819DFDBD4/C1D8510C-1643-464C-89FD-7A0447373A66.png]



## 使用Haar+adaboost分类器实现人脸识别


`#1  load 当前的xml文件（两个xml文件，一个描述人脸，一个检测眼睛）   2 load JPG  3  Haar计算  gray灰度处理  4 detect 检测`
`#5 draw  将人脸和眼睛绘制出来`

`import cv2`
`import numpy as np`

`#load xml   file name`
`face_xml = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')`
`eye_xml = cv2.CascadeClassifier('haarcascade_eye.xml')`
`#load jpg`
`img=cv2.imread('face.jpg')`
`cv2.imshow('src',img)`
`#计算Haar特征，这个过程OpenCV自己完成`
`#把bgr图片转换为灰度图片  灰度处理`
`gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)`

`#detect检测   1 灰度数据图片  2 缩放  3  目标大小 人脸大小最小不能小于  5`
`faces=face_xml.detectMultiScale(gray,1.3,5)`
`print('face=',len(faces))`
`#draw  为人脸画方框`

`for (x,y,w,h) in faces:`
`    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)`
`    roi_face=gray[y:y+h,x:x+w]`
`    roi_color=img[y:y+h,x:x+w]`
`        `
`#实现眼睛识别`
`    eyes=eye_xml.detectMultiScale(roi_face)`
`    print('eye=',len(eyes))`
`    for(e_x,e_y,e_w,e_h) in eyes:`
`        cv2.rectangle(roi_color,(e_x,e_y),(e_x+e_w,e_y+e_h),(0,255,0),2)`
`        `
`cv2.imshow('dst',img)`
`cv2.waitKey(0)`




[image:6F4BDFF1-3B43-4A5E-BB41-DFCD695626A5-636-0000153A5B991003/8953F2FD-A1C2-4451-A11D-A71862EEA573.png]



## Hog+SVM 小狮子识别
特征：Hog
分类器：SVM

如何通过训练得到分类器？


什么是SVM？

既然是分类器，那么就应该具有分类的功能

[image:B859894A-30B6-4A2E-A4F2-08A6EAE3A166-636-00001635A3E74489/06BBE1D2-CDF0-4EB8-A17D-BD0E1F371D5B.png]
二维平面

通过将这些点投影到一个超平面，寻求这个最优的超平面实现这个分类

SVM ：  line

使用SVM完成 ： 身高 + 体重 的分类   完成训练+预测

`import cv2`
`import numpy as np`
`import matplotlib.pyplot as plt`

`#准备数据`

`rand1=np.array([[155,48],[159,50],[164,53],[168,56],[172,60]])`
`rand2=np.array([[152,53],[156,55],[160,56],[172,64],[176,65]])`

`#laber标签的准备`

`label=np.array([[0],[0],[0],[0],[0],[1],[1],[1],[1],[1]])`

`# data  对数据进行一定的预处理，尤其是训练数据`

`data=np.vstack((rand1,rand2))#将两组数据合并到一起   这样每个数据都对应于一个laber标签`
`data=np.array(data,dtype='float32')`
`# svm训练的时候对数据必须有标签`
`#[155,48]  - > 0表示女生   [152,53] - >1表示男生`
`#监督学习！   0  称之为 负样本  1 称之为  正样本`
`#开始训练和预测`
`# 训练svm`
`svm=cv2.ml.SVM_create()  #创建了一个支持向量机 `
`#设置SVM的属性`
`svm.setType(cv2.ml.SVM_C_SVC)`
`svm.setKernel(cv2.ml.SVM_LINEAR)`
`svm.setC(0.01)`

`#训练`
`result=svm.train(data,cv2.ml.ROW_SAMPLE,label)`

`#预测`
`pt_data=np.vstack([[167,55],[162,57]])`
`pt_data=np.array(pt_data,dtype='float32')`
`print(pt_data)`
`(par1,par2)=svm.predict(pt_data)               `
`print(par1,par2)  `
`               `
`           `    

[image:6C130A5A-3839-475C-A322-DCE8E511AC58-636-00001757A1FE5414/B1A4155F-425B-4D70-BDEE-A6581E97D409.png]



总结SVM：

需要掌握
1. SVM的本质，实质还是一个分类器，使用寻求一个二维超平面来实现分类
2. SVM的核，主要的使用是line线性核
3. 数据 样本，需要包含正负样本，正负样本的个数不一定相同
           准备样本的时候一定需要的一个laber标签
4. 训练的使用方法，使用cv2.ml.SVM_create()
5. 设置SVM的属性，svm.setType(cv2.ml.SVM_C_SVC)
6. 设置SVM的核，svm.setKernel(cv2.ml.SVM_LINEAR)
7. 设置  svm.setC(0.01)
8. 对SVM进行训练  svm.train(data,cv2.ml.ROW_SAMPLE,label)
9. 使用SVM进行预测  SVM.predict（）

对SVM进行公示推导！


## Hog特征

1. 什么是Hog特征？  某个像素进行某种运算得到的某种特征

Haar是直接经过模板计算得出的特征

Hog特征则比Haar特征更加复杂

2.   模板划分  ， 梯度 方向  模板 ， 根据梯度方向进行bin投影  ，  计算每个模块的hog特征

3. 模块划分，hog也需要画框来进行计算。

[image:CFBDD8C7-DAB6-4753-A098-790DFFF62E5F-636-00001826B5F5D29B/5CA01666-4707-4698-AAED-A1D4CD22EAAA.png]

image是整个ppt

1. Windows 窗口  (size)  (蓝色的窗体)
2. block   (size)  （红色矩形就是block模块）
3. cell   (size)   （绿色的矩形就称之为cell）
大小关系，位置关系。 

 窗口在滑动的时候，有一个步长的概念。

窗体是特征计算的最顶层单元
1 win  一个窗口必须包含一个对象（描述信息），才能够描述这个obj  （win的宽  高 一般是整个block宽高的整数倍）  64*128
2  block  win的宽  高 一般是整个block宽高的整数倍   16*16
3  block  的步长，block对窗体进行遍历，block滑动的像素被称为block的步长  8*8
4  计算block的count = （（64-12）/8+1   水平方向上）*(（128-16）/8 + 竖直方向上的滑动 ) = 105个 block
5 cell的大小 如果 block的大小为16*16 那么cell的大小推荐为 8*8
6 block = ？ cell   16*16 = 2 * 2 cell  所以一个block中可以包括4个cell
7 cell bin  梯度（就是一个运算） 
    每个像素的梯度的计算的时候，有两个属性，一个是大小，一个是   方向 angle   0-360度  /40 =9块  bin  所以一个bin就是40度  总共有9个bin
一个cell必须包括9个bin

hog特征的维度：  （Haar特征是个具体的值，但是hog特征得到的是一个向量）
维度->必须完全描述一个obj信息，并且hog必须包括所有的obj信息

hog 维度=105（block）* 4（cell）* 9(bin)=3780 维度

梯度的方向和大小如何计算（必须以像素为单位）

所以每一个像素都有一个梯度，win下的所有像素->hog

特征模板  -》 和haar类似

水平方向【1 0 -1】  和 竖直方向【【1】【0】【-1】】

a=p1*1+p2 *0 + p3 *(-1) =相邻像素之差
b=上下像素之差
f=根号下（a 方 + b方）
angle = action （a/b）


计算bin的投影（主要依赖与梯度）
 bin 0-360  9 bin 0-40
bin1 0-20  180-200
 i j  f a =10  10度位于 0 - 20度 的中心 则认为投影在了bin1上面

25 bin1 bin2
f1=f * f(jiajaio)
f2 = f*(1-f(夹角))   0  -  1.0 的范围内
 +1  hog 

*颜色空间归一化—>梯度计算—>梯度方向直方图—>重叠块直方图归一化—>HOG特征*

[image:8073E705-8411-4C36-B866-F9C33CEBB09B-636-00001B47E00C04A8/31F67CA4-433C-49AE-B73B-DF631EC88DEE.png]


如何整个窗体的hog特征  cell的复用原理

3780维度  来源于 win 窗体（包括  block cell bin）

1   《 bin
cell  分为9份   cell1 -  cell9   bin 1- bin 9


2 cell 的复用

block 中 有 4个cell
【0】【】【】【3】

cell0 bin0-bin9
cellx0 cellx2 cellx4 
cellx0: i j -> bin bin +1
cellx2: i j - > cell 2 cell3 - > bin  bin + 1 
……????

what fuck ？

【cell 9】 【4 cell 】 【105】 =3780 维度

如何进行判决?  hog 

svm 线性分类器/  得到3780维度的向量

那么  hog *  3780的  得到一个值  如果大于  则是目标否则不是目标



  



































