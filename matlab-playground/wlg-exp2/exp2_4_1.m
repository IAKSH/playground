raw_image = imread('OIP.jpg');
gray_image = rgb2gray(raw_image);
noise_image = imnoise(gray_image, 'salt & pepper', 0.04);

% 对噪声图像进行二维傅里叶变换
fft_noise = fft2(noise_image);
sfft_noise = fftshift(fft_noise);

% 创建低通滤波器
[M, N] = size(noise_image);
D0 = 30; % 截止频率
[u, v] = meshgrid(-floor(N/2):floor(N/2)-1, -floor(M/2):floor(M/2)-1);
D = sqrt(u.^2 + v.^2);
lowpass_filter = double(D <= D0);

% 应用低通滤波器
filtered_fft = sfft_noise .* lowpass_filter;

% 逆傅里叶变换
ifsfft_noise = ifftshift(filtered_fft);
filtered_image = ifft2(ifsfft_noise);
filtered_image = real(filtered_image);

% 显示结果
figure;
subplot(1,3,1);
imshow(noise_image);
title('噪声图像');

subplot(1,3,2);
imshow(log(1 + abs(sfft_noise)), []);
title('频域图像');

subplot(1,3,3);
imshow(filtered_image, []);
title('低通滤波后的图像');
