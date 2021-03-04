function [X, Xbx, totalmols] = gengroundtruth(Fluo, Cam, Grid, srf, ix, iy)

if nargin < 5
    ix = [1 srf*Grid.sx];
    iy = [1 srf*Grid.sx];
end

X = zeros(ix(2)-ix(1)+1, iy(2)-iy(1)+1, size(Fluo.emitter_brightness,1));

ix = ix(1):ix(2);
iy = iy(1):iy(2);

pos = 1 + fix(Fluo.emitters*srf); % for example if a moluecule has a position in [0 1), is in the pixel 1

totalmols = 0;

for n1 = 1:size(X,1)
    for n2 = 1:size(X,2)
        
        mols = (pos(:,1) == ix(n1)) & (pos(:,2) == iy(n2));
        if any(mols)
            totalmols = totalmols + sum(mols);
            X(n1, n2, :) = sum(Fluo.emitter_brightness(:, mols),2);
        end
        
    end
end

Xbx = nonneg(X - Fluo.background) + Fluo.background;

X = X * Cam.quantum_gain;
Xbx = Xbx * Cam.quantum_gain;

