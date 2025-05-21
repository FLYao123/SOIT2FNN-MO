function w = Initialization_ETA (X, U, V, m)
c = size (V, 1);
w = zeros (c, 1);
mf = U.^m;
dist = Distance_Function (V, X);
dist = dist .^ 2;
w = sum((mf.*dist)') ./ sum(mf');
