function [y, x, theta] = extract_edges(Ix, Iy, Im, threshold)
    % Returns the x and y coordinates of pixels whose gradient
    % magnitude is greater than the threshold. Also, returns
    % the angle of the image gradient at each extracted edge.

    % Hint: use find() to extract above-threshold pixels.
    % Hint: sub2ind may be useful when computing the angle.

    [x, y] = find(Im>threshold);
    
    ind = sub2ind(size(Im),x,y);
    
    theta = atan2(Iy(ind),Ix(ind));
    
end
