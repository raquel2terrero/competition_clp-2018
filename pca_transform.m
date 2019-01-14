function [X_pca, W, m] = pca_transform(X)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
    m=mean(X);
    X_train2=X_train-ones(size(X,1),1)*m;
    COEFF = pca(X_train2);
    W=COEFF(:,1:size(X,2));
    X_pca=X_train2*W;
end

