% 读取图像文件 '300px-Lenna.jpg'，并将其存储在变量 raw_image_4 中
raw_image_4 = imread('OIP.jpg');  

% 将彩色图像转换为灰度图像，并将其存储在变量 gray_image_4 中
gray_image_4 = rgb2gray(raw_image_4);

% 在 1x2 网格的第一个子图中显示原始灰度图像
subplot(1,2,1);
imshow(gray_image_4); 
title('原图');                       

% 对灰度图像进行二维傅里叶变换，并将结果存储在变量 fftI1 中
fftI1 = fft2(gray_image_4);            

% 对傅里叶变换结果进行中心化，即将频谱中心移动到图像中心，并存储在变量 sfftI1 中
sfftI1 = fftshift(fftI1);              

% 提取傅里叶变换结果的实部，存储在变量 RR1 中
RR1 = real(sfftI1);                    

% 提取傅里叶变换结果的虚部，存储在变量 II1 中
II1 = imag(sfftI1);                   

% 计算傅里叶变换结果的幅值谱，存储在变量 A1 中
A1 = sqrt(RR1.^2 + II1.^2);             

% 对幅值谱进行归一化处理，使其值在 0 到 225 之间
A1 = (A1 - min(min(A1))) / (max(max(A1)) - min(min(A1))) * 225;

% 在 1x2 网格的第二个子图中显示傅里叶变换的频域图像
subplot(1,2,2);
imshow(A1);  
title('傅里叶变换频域图');   
