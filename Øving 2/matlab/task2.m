clear;
close all;

% This bit of code is from HW1.
filename       = '../data/grid.jpg';
edge_threshold = 0.015;
blur_sigma     = 1;
I_rgb          = im2double(imread(filename));
I_gray         = rgb_to_gray(I_rgb);
[Ix,Iy,Im]     = derivative_of_gaussian(I_gray, blur_sigma);
[x,y,theta]    = extract_edges(Ix, Iy, Im, edge_threshold);

% You can adjust these for better results
line_threshold = 0.15;
N_rho          = 400;
N_theta        = 400;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Task 2.1: Determine appropriate ranges
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tip: theta is computed using atan2. Check that the range
% returned by atan2 matches your chosen ranges.
rho_max   = norm(size(I_rgb(:,:,1))); % Length of  image diagonal
rho_min   = -norm(size(I_rgb(:,:,1)));
theta_min = -pi; 
theta_max = pi; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Task 2.2: Compute the accumulator array
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Zero-initialize an array to hold our votes
H = zeros(N_rho, N_theta);

% 1) Compute rho for each edge (x,y,theta)
% Tip: You can do this without for-loops
rho = x.*cos(theta) + y.*sin(theta);

% 2) Convert to discrete row,column coordinates
% Tip: Use round(...) to round a number to an integer type
% Tip: Remember that Matlab indices start from 1, not 0
theta_d = ceil(N_theta*(theta- theta_min)/(theta_max - theta_min));
rho_d = ceil(N_rho*(rho - rho_min)/(rho_max - rho_min));

% 3) Increment H[row,column]
% Tip: Make sure that you don't try to access values at indices outside
% the valid range: [1,N_rho] and [1,N_theta]

for i = 1:size(rho_d,1)
    H(rho_d(i), theta_d(i)) = H(rho_d(i), theta_d(i)) + 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Task 2.3: Extract local maxima
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1) Call extract_local_maxima
[row, col] = extract_local_maxima(H,line_threshold);
% 2) Convert back to continuous rho,theta quantities

maxima_rho = row*(rho_max - rho_min)/N_rho + rho_min; % Placeholder
maxima_theta = col*(theta_max - theta_min)/N_theta + theta_min; % Placeholder


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Figure 2.2: Display the accumulator array and local maxima
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1);
imagesc(H, 'XData', [theta_min theta_max], 'YData', [rho_min rho_max]); hold on;
cb = colorbar();
cb.Label.String = 'Votes';
scatter(maxima_theta, maxima_rho, 100, 'MarkerEdgeColor', 'red', 'MarkerFaceColor', 'none', 'LineWidth', 1.5);
xlabel('\theta (radians)');
ylabel('\rho (pixels)');
title('Accumulator array');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Figure 2.3: Draw the lines back onto the input image
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(2);
imshow(I_rgb); hold on;
for i=1:size(maxima_rho)
    draw_line(maxima_theta(i), maxima_rho(i));
end
title('Dominant lines');
