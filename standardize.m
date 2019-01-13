function [X_scl, X_mean, X_std] = standardize(X)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    X_mean = mean(X);
    X_std = std(X);
    X_scl = (X - X_mean) ./ X_std;
end

