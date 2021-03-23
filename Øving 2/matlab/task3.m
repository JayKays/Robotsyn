% Note: the sample image is naturally grayscale
clear;
I = rgb_to_gray(im2double(imread('../data/calibration.jpg')));
%
% Task 3.1: Compute the Harris-Stephens measure
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sigma_D = 1;
sigma_I = 3;
alpha = 0.06;
[Ix,Iy,Im]     = derivative_of_gaussian(I, sigma_D);

A11 = gaussian(Ix.^2, sigma_I);
A12 = gaussian(Ix.*Iy, sigma_I);
A21 = gaussian(Ix.*Iy, sigma_I);
A22 = gaussian(Iy.^2, sigma_I);

response = A11.*A22 + A12.*A21 - alpha*(A11 + A22).^2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Task 3.4: Extract local maxima
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[corners_y, corners_x] = extract_local_maxima(response, 0.001);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Figure 3.1: Display Harris-Stephens corner strength
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1);
imshow(response, []); % Specifying [] makes Matlab auto-scale the intensity
cb = colorbar();
cb.Label.String = 'Corner strength';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Figure 3.4: Display extracted corners
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(2);
imshow(I); hold on;
scatter(corners_x, corners_y, 15, 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'yellow');
