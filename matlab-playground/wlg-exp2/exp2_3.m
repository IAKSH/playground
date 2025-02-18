% 读取图像文件 '300px-Lenna.jpg'，并将其存储在变量 raw_image_3 中
raw_image_3 = imread('OIP.jpg');

% 将彩色图像转换为灰度图像，并将其存储在变量 gray_image_3 中
gray_image_3 = rgb2gray(raw_image_3);

%figure % 这行代码被注释掉，未创建新图形窗口

% 在 3x2 网格的第一个子图中显示原始灰度图像
subplot(3,2,1);
imshow(gray_image_3);
title('原图');

% 在图像中加入椒盐噪声，噪声密度为0.04，并将结果存储在变量 noise_image 中
noise_image = imnoise(gray_image_3,'salt & pepper',0.04);

% 在 3x2 网格的第二个子图中显示加入噪声的图像
subplot(3,2,2);
imshow(noise_image);
title('噪声图像');

% 定义一个 3x3 的滑动窗口，值全为 1/9（均值滤波器）
sliding_window = ones(3,3)/9;

% 对噪声图像进行均值滤波，结果存储在变量 mean_filter_image 中
mean_filter_image = conv2(noise_image,sliding_window,'same');

% 在 3x2 网格的第三个子图中显示均值滤波后的图像
subplot(3,2,3);
imshow(mean_filter_image,[]);
title('均值滤波');

% 对噪声图像进行 3x3 大小的中值滤波，结果存储在变量 med_filter_image 中
med_filter_image = medfilt2(noise_image,[3,3]);

% 在 3x2 网格的第四个子图中显示中值滤波后的图像
subplot(3,2,4);
imshow(med_filter_image);
title('中值滤波');

% 定义一个 3x3 大小的高斯滤波器，标准差为0.5
gau_filter_1 = fspecial('gaussian',[3,3],0.5);

% 对灰度图像进行高斯滤波，结果存储在变量 gau_filter_image_1 中
gau_filter_image_1 = filter2(gau_filter_1,gray_image_3);

% 在 3x2 网格的第五个子图中显示高斯滤波后的图像
subplot(3,2,5);
imshow(gau_filter_image_1,[]);
title('高斯滤波');

% 定义一个运动滤波器，长度为20，方向为20度
motion_filter_1 = fspecial('motion',20,20);

% 对灰度图像进行运动滤波，结果存储在变量 motion_filter_image_1 中
motion_filter_image_1 = imfilter(gray_image_3,motion_filter_1,'replicate');

% 在 3x2 网格的第六个子图中显示运动滤波后的图像
subplot(3,2,6);
imshow(motion_filter_image_1);
title('图像模糊');
