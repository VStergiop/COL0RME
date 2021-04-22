function A = set2zero(N, tol)

A = zeros(N^2);

for t = 1:N^2
    for r =t:N^2
        [i1,j1] = ind2sub([N N],t);
        [i2,j2] = ind2sub([N N],r);
       if sqrt((i1-i2).^2 + (j1-j2).^2)<tol
           a1 = sub2ind([N N], i1, j1);
           a2 = sub2ind([N N], i2, j2);
           A(a1, a2) = 1;
           A(a2, a1) = 1;
       end
    end
end