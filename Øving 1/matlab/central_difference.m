function [Ix, Iy, Im] = central_difference(I)
    % Computes the gradient in the x and y direction using
    % a central difference filter, and returns the resulting
    % gradient images (Ix, Iy) and the gradient magnitude Im.
    kernel = [1/2, 0, -1/2];
    
    Ix = zeros(size(I)); % Placeholder
    Iy = zeros(size(I)); % Placeholder
    
    for i = 1:size(I,1)
        Ix(i,:) = conv(I(i,:), kernel, 'same');
    end
    
    for j = 1:size(I,2)
        Iy(:,j) = conv(I(:,j).', kernel, 'same').';
    end
    
    Im = sqrt(Ix.^2 + Iy.^2); % Placeholder
end
