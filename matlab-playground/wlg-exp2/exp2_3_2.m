% 读取图像并加入椒盐噪声
raw_image_3 = imread('onion.jpg');
gray_image_3 = rgb2gray(raw_image_3);
noise_image = imnoise(gray_image_3,'salt & pepper',0.04);

% 使用双边滤波
filtered_image_bilateral = imbilatfilt(noise_image);

% 显示原图、噪声图像和双边滤波后的图像
figure;
subplot(1,3,1);
imshow(gray_image_3);
title('原图');
subplot(1,3,2);
imshow(noise_image);
title('噪声图像');
subplot(1,3,3);
imshow(filtered_image_bilateral);
title('双边滤波');
