# ImageCaptioning-Attention-PyQt5
ImageCaptioning improved with attention. Also a PyQt5 applications 

# Welcome !
- Hello guys, hope you are doing awesome these days !😄
- In my previous ImageCaption repository, I implemented a ImageCaption algorithm and I promised to upload an attention based version latter. And here it is ！😄
- Using the `ResNet50` pretrained on ImageNet as the backbone(no finetune) and also some attention, the model can describe image like human(most of the time).
- Moreover,`Beam Search` are also used during the inferrence part and this give another great improvment on the model's performence 
- Now, let's enjoy some funny stuff😎

# 1.Examples👍
## ①.doggy doggy, juicy doggy
- <img src="https://github.com/Mountchicken/ImageCaptioning-Attention-PyQt5/blob/main/github/dog.JPG" width="719" height="413" alt="😀"/><br/>

## ②.When did I become a hat ?
- <img src="https://github.com/Mountchicken/ImageCaptioning-Attention-PyQt5/blob/main/github/girl.JPG" width="719" height="413" alt="😀"/><br/>

## ③.You don't want to mess up with No. 1 shooter in the west
- <img src="https://github.com/Mountchicken/ImageCaptioning-Attention-PyQt5/blob/main/github/cowboy.JPG" width="719" height="413" alt="😀"/><br/>

## ④.🌶④💉💧🐮🍺
- <img src="https://github.com/Mountchicken/ImageCaptioning-Attention-PyQt5/blob/main/github/sun.JPG" width="719" height="413" alt="😀"/><br/>

## ⑤.Portland Timbers, Assemble!
- <img src="https://github.com/Mountchicken/ImageCaptioning-Attention-PyQt5/blob/main/github/champers.JPG" width="719" height="413" alt="😀"/><br/>

## ⑥.mountchicken must has something to do with mountain
- <img src="https://github.com/Mountchicken/ImageCaptioning-Attention-PyQt5/blob/main/github/mountain.JPG" width="719" height="413" alt="😀"/><br/>
 
# 2.Requirements
- skimage
- spacy
- PyQt5
- Pip install them

# 3.Train😣
## download flickr30k
- Download the flickr30k dataset, unpack all the images into the folder `flickr30k/flickr30k-images`. I have already preprocessed the captions.txt, and you don't need to download that 
- [flickr(提取码:hrf3)](https://pan.baidu.com/s/1r0RVUwctJsI0iNuVXHQ6kA)
## download my checkpoint(if you don't want to train it with 14h on GeForce2080ti)
- Put the downloaded checkpoint into the folder `checkpoint`
- [checkpoint(提取码:qny4)](https://pan.baidu.com/s/189u5i5vZbzIp9r4XoEYn6A)
## change some parameters
- `train.py` line20 - line26, set the dataset path
- `train.py` line31 - line34, `load_model`:load my checkpoint or not.
- Ok, you can train now

# 4.Inferrence😀
- `inferrence.py` line245, choose your predict image path

# 5.APP

## Run main_gui.py
### if you run the .py file succesfully, it should look like this
- <img src="https://github.com/Mountchicken/ImageCaptioning-Attention-PyQt5/blob/main/github/main.JPG" width="719" height="413" alt="😀"/><br/>

### Then, you need to push the initialize button to load the model, after that, just wait the `Finished` sign appers in the right.
- <img src="https://github.com/Mountchicken/ImageCaptioning-Attention-PyQt5/blob/main/github/initialized.JPG" width="719" height="413" alt="😀"/><br/>

### Finally, load the image with `Load Image` button and press `Detect`
- <img src="https://github.com/Mountchicken/ImageCaptioning-Attention-PyQt5/blob/main/github/taiquandao.JPG" width="719" height="413" alt="😀"/><br/>

# For more issue, contact me
- `Email Address` mountchicken@outlook.com
