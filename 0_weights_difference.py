Missing key(s) in state_dict: "module.encoder.0.weight", "module.encoder.0.bias", "module.encoder.3.weight", "module.encoder.3.bias", "module.encoder.6.weight", "module.encoder.6.bias", "module.encoder.8.weight", "module.encoder.8.bias", "module.encoder.11.weight", "module.encoder.11.bias", "module.encoder.13.weight", "module.encoder.13.bias", "module.encoder.16.weight", "module.encoder.16.bias", "module.encoder.18.weight", "module.encoder.18.bias", "module.conv1.0.weight", "module.conv1.0.bias", "module.conv2.0.weight", "module.conv2.0.bias", "module.conv3.0.weight", "module.conv3.0.bias", "module.conv3.2.weight", "module.conv3.2.bias", "module.conv4.0.weight", "module.conv4.0.bias", "module.conv4.2.weight", "module.conv4.2.bias", "module.conv5.0.weight", "module.conv5.0.bias", "module.conv5.2.weight", "module.conv5.2.bias", "module.center.block.0.conv.weight", "module.center.block.0.conv.bias", "module.center.block.1.weight", "module.center.block.1.bias", "module.dec5.block.0.conv.weight", "module.dec5.block.0.conv.bias", "module.dec5.block.1.weight", "module.dec5.block.1.bias", "module.dec4.block.0.conv.weight", "module.dec4.block.0.conv.bias", "module.dec4.block.1.weight", "module.dec4.block.1.bias", "module.dec3.block.0.conv.weight", "module.dec3.block.0.conv.bias", "module.dec3.block.1.weight", "module.dec3.block.1.bias", "module.dec2.block.0.conv.weight", "module.dec2.block.0.conv.bias", "module.dec2.block.1.weight", "module.dec2.block.1.bias", "module.dec1.conv.weight", "module.dec1.conv.bias", 
"module.cls.0.weight", "module.cls.0.bias". 

Unexpected key(s) in state_dict: "encoder.0.weight", "encoder.0.bias", "encoder.3.weight", "encoder.3.bias", "encoder.6.weight", "encoder.6.bias", "encoder.8.weight", "encoder.8.bias", "encoder.11.weight", "encoder.11.bias", "encoder.13.weight", "encoder.13.bias", "encoder.16.weight", "encoder.16.bias", "encoder.18.weight", "encoder.18.bias", "conv1.0.weight", "conv1.0.bias", "conv2.0.weight", "conv2.0.bias", "conv3.0.weight", "conv3.0.bias", "conv3.2.weight", "conv3.2.bias", "conv4.0.weight", "conv4.0.bias", "conv4.2.weight", "conv4.2.bias", "conv5.0.weight", "conv5.0.bias", "conv5.2.weight", "conv5.2.bias", "center.block.0.conv.weight", "center.block.0.conv.bias", "center.block.1.weight", "center.block.1.bias", "dec5.block.0.conv.weight", "dec5.block.0.conv.bias", "dec5.block.1.weight", "dec5.block.1.bias", "dec4.block.0.conv.weight", "dec4.block.0.conv.bias", "dec4.block.1.weight", "dec4.block.1.bias", "dec3.block.0.conv.weight", "dec3.block.0.conv.bias", "dec3.block.1.weight", "dec3.block.1.bias", "dec2.block.0.conv.weight", "dec2.block.0.conv.bias", "dec2.block.1.weight", "dec2.block.1.bias", "dec1.conv.weight", "dec1.conv.bias", 
"final.weight", "final.bias". 

# Pytorch 中只导入部分层权重地方法
# https://blog.csdn.net/jackzhang11/article/details/108047586
# 此时如果将最后的全连接层都拿掉，再新添加一个conv3， 定义为 model2.
# 在新模型 model2 中，load 之前 model1 网络结构的权重参数，就会出现报错。
# ，在原来模型的参数"pretext.pth"中，并不存在新模型的conv3参数；与此同时，fc1和fc2的相关参数，对于新模型来说也是unexpected的。因此问题就出现在这里：原模型参数的键，不能完全和修改后的模型的key进行匹配。因此要解决这个问题，就是要抽取出"pretext.pth"中存在于新模型中的键值对。


net = model2()
pretext_model = torch.load('pretext.pth')
model2_dict = net.state_dict()
state_dict = {k:v for k,v in pretext_model.items() if k in model2_dict.keys()}
model2_dict.update(state_dict)
net.load_state_dict(model2_dict)


import torch
def load_from(self, config):
    pretrained_path = config.MODEL.PRETRAIN_CKPT
    if pretrained_path is not None:
        print("pretrained_path:{}".format(pretrained_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        if "model"  not in pretrained_dict:
            print("---start load pretrained modle by splitting---")
            pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
            for k in list(pretrained_dict.keys()):
                if "output" in k:
                    print("delete key:{}".format(k))
                    del pretrained_dict[k]
            msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
            # print(msg)
            return
        pretrained_dict = pretrained_dict['model']
        print("---start load pretrained modle of swin encoder---")

        model_dict = self.swin_unet.state_dict()
        full_dict = copy.deepcopy(pretrained_dict)
        for k, v in pretrained_dict.items():
            if "layers." in k:
                current_layer_num = 3-int(k[7:8])
                current_k = "layers_up." + str(current_layer_num) + k[8:]
                full_dict.update({current_k:v})
        for k in list(full_dict.keys()):
            if k in model_dict:
                if full_dict[k].shape != model_dict[k].shape:
                    print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                    del full_dict[k]

        msg = self.swin_unet.load_state_dict(full_dict, strict=False)
        # print(msg)
    else:
        print("none pretrain")