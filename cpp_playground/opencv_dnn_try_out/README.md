code from https://github.com/WeiZhenOoooo/yolov9-onnx-opencv

不知道为什么，始终只能识别最多一个结果，并且box的xywh一直全是0，改了半天没解决。

好了，可以确定是python代码那边导出onnx的时候的问题了，C++ opencv dnn这边用cpu推理是正常的，换cuda就炸。参考：https://www.bilibili.com/video/BV1RG411S7kj/?spm_id_from=333.337.search-card.all.click&vd_source=aba245c5d1c4487c2355023d2870a6f7

cudnn版本不能高于8.x，因为9.x删除了cudnnSetRNNDescriptor_v6()，该函数在8.x中被标记为弃用，vcpkg目前使用的opencv dnn-cuda源码依然在使用这个函数。
详见 https://docs.nvidia.com/deeplearning/cudnn/latest/api/overview.html?highlight=cudnnsetrnndescriptor_v6
另，编译好慢，尤其是nvidia cicc