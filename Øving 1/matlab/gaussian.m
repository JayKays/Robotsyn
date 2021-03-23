function I_blur = gaussian(I, sigma)
    % Applies a 2-D Gaussian blur with standard deviation sigma to
    % a grayscale image I.

    % Hint: The size of the kernel should depend on sigma. A common
    % choice is to make the half-width be 3 standard deviations. The
    % total kernel width is then 2*ceil(3*sigma) + 1.
    
    h = 2*sigma;
    gauss = normpdf(-h:1:h, 0, sigma);
    
    I_blur = zeros(size(I)); % Placeholder
    
    %Row convolution
    for i = 1:size(I,1)
        I_blur(i,:) = conv(I(i,:), gauss, 'same');
    end
    
    %Column convolution
    for j = 1:size(I_blur,2)
        I_blur(:,j) = conv(I_blur(:,j), gauss, 'same').';
    end
end
