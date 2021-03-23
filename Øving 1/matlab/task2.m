clear


filename = '../data/grass.jpg';
threshold = 0.4;
Im = im2double(imread(filename));
ImR = Im(:,:,1); ImG = Im(:,:,2); ImB = Im(:,:,3);

ImRnorm = ImR./(ImR + ImG + ImB);
ImGnorm = ImG./(ImR + ImG + ImB);
ImBnorm = ImB./(ImR + ImG + ImB);

ImTh = ImGnorm > threshold;

figure(1);
imshow(ImTh);
title('Threshold normalized Green channel')

figure(2);
title('RGB');
subplot(221); imshow(Im); title('Image');
subplot(222); imshow(ImRnorm); title('Red')
subplot(223); imshow(ImGnorm); title('Green')
subplot(224); imshow(ImBnorm); title('Blue')





