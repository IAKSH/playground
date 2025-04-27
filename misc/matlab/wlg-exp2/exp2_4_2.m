raw_image = imread('onion2.jpg');
gray_image = rgb2gray(raw_image);

% 使用Sobel算子进行边缘检测
edge_image = edge(gray_image, 'Sobel');

% 显示结果
figure;
subplot(1,2,1);
imshow(gray_image);
title('原图');

subplot(1,2,2);
imshow(edge_image);
title('Sobel边缘检测');
