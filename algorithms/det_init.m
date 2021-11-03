function  new_loc = det_init(posU, posV)

% Number of true molecules
R = length(posU);

% First step: building lists of preferences -------------------------------

% Computing "preferences" of true molecules
new_loc = zeros(R,2);
for r = 1:R
    
    posUd = posU;
    posUd(r)=[];
    posVd = posV;
    posVd(r)=[];
    
    % Computes all distances
    dist = sqrt((posU(r)-posUd).^2 + (posV(r)-posVd).^2);
    [~,ind_min] = min(dist);
    new_loc(r,1) = round((posUd(ind_min) + posU(r)+1)/2);
    new_loc(r,2) = round((posVd(ind_min) + posV(r)+1)/2);
end


