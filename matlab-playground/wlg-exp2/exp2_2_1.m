row_image = imread('onion.jpg');
gray_image = rgb2gray(row_image);
[M,N] = size(gray_image);

figure

% 原图
subplot(2,2,1);
imshow(gray_image);
title('原图');

% 参数设置1：增强因子1.2，偏移量10
enhance_image1 = zeros(M,N);
for ii = 1:M
    for jj = 1:N
        enhance_image1(ii,jj) = gray_image(ii,jj)*1.2 + 10;
    end
end
enhance_image1 = uint8(enhance_image1);
subplot(2,2,2);
imshow(enhance_image1);
title('增强因子1.2，偏移量10');

% 参数设置2：增强因子1.5，偏移量20
enhance_image2 = zeros(M,N);
for ii = 1:M
    for jj = 1:N
        enhance_image2(ii,jj) = gray_image(ii,jj)*1.5 + 20;
    end
end
enhance_image2 = uint8(enhance_image2);
subplot(2,2,3);
imshow(enhance_image2);
title('增强因子1.5，偏移量20');

% 参数设置3：增强因子2.0，偏移量30
enhance_image3 = zeros(M,N);
for ii = 1:M
    for jj = 1:N
        enhance_image3(ii,jj) = gray_image(ii,jj)*2.0 + 30;
    end
end
enhance_image3 = uint8(enhance_image3);
subplot(2,2,4);
imshow(enhance_image3);
title('增强因子2.0，偏移量30');
