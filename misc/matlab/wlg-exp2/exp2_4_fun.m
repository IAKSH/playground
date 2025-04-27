% 原图
raw_image_4 = imread('onion.png');  
gray_image_4 = rgb2gray(raw_image_4);
subplot(1,3,1);
imshow(gray_image_4); 
title('原图');                       

fftI1=fft2(gray_image_4);            
sfftI1=fftshift(fftI1);              
RR1=real(sfftI1);                    
II1=imag(sfftI1);                   
A1=sqrt(RR1.^2+II1.^2);             
A1=(A1-min(min(A1)))/(max(max(A1))-min(min(A1)))*225;
subplot(1,3,2);imshow(A1);  
title('傅里叶变换频域图');       


% 逆傅里叶变换
isfftI1 = ifftshift(sfftI1);
ifftI1 = ifft2(isfftI1);
% 获取实部以显示图像
reconstructed_image = real(ifftI1);
subplot(1,3,3);
imshow(reconstructed_image, []);
title('逆傅里叶变换实部');
