# ImageCaptioning-Attention-PyQt5
ImageCaptioning improved with attention. Also a PyQt5 applications 

# Welcome !
- Hello guys, hope you are doing awesome these days !😄
- In my previous ImageCaption repository, I implemented a ImageCaption algorithm and I promised to upload an attention based version latter. And here it is ！😄
- Using the `ResNet50` pretrained on ImageNet as the backbone(no finetune) and also some attention, the model can describe image like human(most of the time).
- Moreover,`Beam Search` are also used during the inferrence part and this give another great improvment on the model's performence 
- Now, let's enjoy some funny stuff😎
-
# Examples👍
## doggy doggy, juicy doggy
- <img src="https://github.com/Mountchicken/ImageCaptioning-Attention-PyQt5/blob/main/github/dog.JPG" width="719" height="413" alt="😀"/><br/>

# How to use it
- I'm sorry guys, i haven't find a way to deploy it, and you have to run it in your compiler 🙇‍♂️(VScode, pycharm or...)

## download pretrained weights
- The weights are larger than the uploading limit(25M below😅). Download them using BaiduYun
- Put them in CTPN_weights [CTPN weights(提取码:vqih)](https://pan.baidu.com/s/1OP4H87hunibVOQK_TKH-OA)
- Put them in CRNN_weights [CRNN weights(提取码:k4r4)](https://pan.baidu.com/s/1Ie-X_5Z-JuypKzsD3bRkzA)

## Choose which model to use
- In `inferrence.py`, from line 27 to line 32
- `argument: crnn_weights`: the file location of crnn weigth downloaded in the previous step
- `argument: ctpn_basemodel`:choose a ctpb backbone: vgg16, resnet50, shufflenet_v2_x1_0, mobilenet_v3_large, mobilenet_v3_small
- `argument: ctpn_weights`:corresponding ctpn weights with ctpn_base model downloaded in the previous step

## Run main_gui.py
- if you run the .py file succesfully, it should look like this
- ![test_example](https://github.com/Mountchicken/CTPN_CRNN_ChineseOCR_PyQt5/blob/main/github/menu.JPG)
- Then, you need to push the initialize button to load the model, after that, just wait the `Finished` sign appers in the right.
- ![test_example](https://github.com/Mountchicken/CTPN_CRNN_ChineseOCR_PyQt5/blob/main/github/Initialized.JPG)
- Finally, load the image with `Load Image` button and press `Detect`
- ![test_example](https://github.com/Mountchicken/CTPN_CRNN_ChineseOCR_PyQt5/blob/main/github/detectd.JPG)

# For more issue, contact me
- `Email Address` mountchicken@outlook.com
