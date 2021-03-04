function v = compcov(X, N, L, Gh0, SmSm, replambdah)

% This function computes kron(Gl, Gl) * Diag(X(:)) * kron(Gl, Gl)'

% (a) Computing GkGrx = kron(Gl, Gl) * Diag(rx)
GkGrx = zeros(N^2, L^2);
parfor l = 1 : numel(X) % find(X(:)' > 0)
    if (abs(X(l)) > 0)
        % Convolution of column from Diag(rx) having a single nonzero
        % component, followed by integration (downsampling)
        [i,j] = ind2sub([L L], l);
        GkGrx(:,l) = X(l) * vec(SmSm(circshift(Gh0, [i-1 j-1])));
    end
end

% (b) v = GkGrx * kron(Gl, Gl)'
v = zeros(N^2, N^2);
parfor n = 1 : N^2
    % 2D convolution performed on frequency domain
    row = ifft2(replambdah .* fft2(reshape(GkGrx(n,:), [L L])));

    % Integration (Sm * row * Sm')
    v(n,:) = vec(SmSm(row)).';
end   

