% 读取图像文件，并将其存储在变量 color_image 中
color_image = imread('OIP.jpg');

% 获取图像的行数和列数，并分别存储在变量 row 和 col 中
row = size(color_image,1);
col = size(color_image,2);

% 创建一个新图形窗口
figure

% 在 2x2 网格的第一个子图中显示原始彩色图像
subplot(2,2,1)
imshow(color_image)

% 创建一个与原始图像大小相同的三通道全零矩阵 red_image
red_image = zeros(row,col,3);
% 将原始图像的红色通道复制到 red_image 的红色通道
red_image(:,:,1) = color_image(:,:,1);
% 在 2x2 网格的第二个子图中显示红色通道图像
subplot(2,2,2)
imshow(uint8(red_image))

% 创建一个与原始图像大小相同的三通道全零矩阵 green_image
green_image = zeros(row,col,3);
% 将原始图像的绿色通道复制到 green_image 的绿色通道
green_image(:,:,2) = color_image(:,:,2);
% 在 2x2 网格的第三个子图中显示绿色通道图像
subplot(2,2,3)
imshow(uint8(green_image))

% 创建一个与原始图像大小相同的三通道全零矩阵 blue_image
blue_image = zeros(row,col,3);
% 将原始图像的蓝色通道复制到 blue_image 的蓝色通道
blue_image(:,:,3) = color_image(:,:,3);
% 在 2x2 网格的第四个子图中显示蓝色通道图像
subplot(2,2,4)
imshow(uint8(blue_image))
