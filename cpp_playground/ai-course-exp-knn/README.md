令人震惊的是，居然没有人提到OpenCV依赖qt这一问题，导致找了半天才发现需要装qt，才能用cv::imshow以及cv::waitkey，不然就会莫名爆炸，甚至debug都是直接被弹出来。
msys2的opencv包没有依赖qt，手动装个qt6就好了

~~1. AKAZE从图片中提取特征点和特征向量~~ AKAZE似乎在小图像或者过于平滑的图片上容易找不出特征点
~~1. ORB从图片中提取特征点和描述符~~
1. 对于28x28的FashionMinist图像，先放大
2. 然后使用AKAZE提取特征点
3. PCA将上述内容降维作为特征向量
4. KNN进行分类