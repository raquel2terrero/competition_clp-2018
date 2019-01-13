function [ X_train, Labs_train, X_test, Labs_test, X_val, Labs_val ] = Train_Test_Val( X, Labels )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here

%% Generation of Training (60 %), Validation (20%) and Test (20%) sets
% Randomize vectors

N_datos = length(Labels);
indexperm=randperm(N_datos);
X=X(indexperm,:);
Labs=Labels(indexperm);

% Generation of Train, Validation and Test sets
N_train=round(0.6*N_datos);
N_val=round(0.8*N_datos)-N_train;
N_test=N_datos-N_train-N_val;

% Train
X_train=X(1:N_train,:);
Labs_train=Labs(1:N_train);

% Validation
X_val=X(N_train+1:N_train+N_val,:);
Labs_val=Labs(N_train+1:N_train+N_val);

% Test
X_test=X(N_train+N_val+1:N_datos,:);
Labs_test=Labs(N_train+N_val+1:N_datos);

clear indexperm
end

