% 读取图像文件, 并将其存储在变量 row_image 中
row_image = imread('OIP.jpg');

% 创建一个新图形窗口
figure

% 在 2x2 网格的第一个子图中显示原始彩色图像
subplot(2,2,1);
imshow(row_image);
title('原图');

% 将彩色图像转换为灰度图像，并将其存储在变量 gray_image 中
gray_image = rgb2gray(row_image);

% 在 2x2 网格的第二个子图中显示灰度图像
subplot(2,2,2);
imshow(gray_image);
title('灰度图');

% 获取灰度图像的大小，并分别存储在变量 M 和 N 中
[M,N] = size(gray_image);

% 在 2x2 网格的第三个子图中绘制灰度直方图
subplot(2,2,3)
[counts,x] = imhist(gray_image,256);
counts = counts/M/N; % 归一化直方图计数
stem(x,counts);
title('灰度直方图');

% 创建一个大小与灰度图像相同的全零矩阵 enhance_image
enhance_image = zeros(M,N);

% 找到灰度图像的最小和最大灰度值
bmin = min(min(gray_image));
bmax = max(max(gray_image));

% 对每个像素进行亮度增强操作
for ii = 1:M
    for jj = 1:N
        enhance_image(ii,jj) = gray_image(ii,jj)*1.5 + 20;
    end
end

% 将增强后的图像转换为 8 位无符号整数类型
enhance_image=uint8(enhance_image);

% 在 2x2 网格的第四个子图中显示亮度增强后的图像
subplot(2,2,4);
imshow(enhance_image);
title('图像亮度增强');
