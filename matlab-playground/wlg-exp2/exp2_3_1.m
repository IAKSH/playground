raw_image_3 = imread('onion.jpg');
gray_image_3 = rgb2gray(raw_image_3);

figure

% 原图
subplot(4,2,1);
imshow(gray_image_3);
title('原图');

% 加入高斯噪声
gaussian_noise_image = imnoise(gray_image_3,'gaussian',0,0.01);
subplot(4,2,2);
imshow(gaussian_noise_image);
title('高斯噪声图像');

% 对高斯噪声图像进行均值滤波
mean_filter_image_gaussian = conv2(gaussian_noise_image, sliding_window, 'same');
subplot(4,2,3);
imshow(mean_filter_image_gaussian, []);
title('均值滤波（高斯噪声）');

% 对高斯噪声图像进行中值滤波
med_filter_image_gaussian = medfilt2(gaussian_noise_image, [3,3]);
subplot(4,2,4);
imshow(med_filter_image_gaussian);
title('中值滤波（高斯噪声）');

% 加入均匀噪声
uniform_noise_image = imnoise(gray_image_3,'speckle',0.04);
subplot(4,2,5);
imshow(uniform_noise_image);
title('均匀噪声图像');

% 对均匀噪声图像进行均值滤波
mean_filter_image_uniform = conv2(uniform_noise_image, sliding_window, 'same');
subplot(4,2,6);
imshow(mean_filter_image_uniform, []);
title('均值滤波（均匀噪声）');

% 对均匀噪声图像进行中值滤波
med_filter_image_uniform = medfilt2(uniform_noise_image, [3,3]);
subplot(4,2,7);
imshow(med_filter_image_uniform);
title('中值滤波（均匀噪声）');
