function  new_loc = stoc_init(posU, posV, radius)

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
    dir = atan2(posVd(ind_min)-posV(r),posUd(ind_min)-posU(r));
    [x, y]=cirrdnPJ( posU(r), posV(r),radius,dir);
    new_loc(r,1) = round(x+0.5);
    new_loc(r,2) = round(y+0.5);
end


