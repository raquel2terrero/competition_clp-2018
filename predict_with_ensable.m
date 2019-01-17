function [pred] = predict_with_ensemble(Mdl_ens,X)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    pred = 0;
    for i = 1:length(Mdl_ens)
        pred = pred + predict(Mdl_ens{i}, X);
    end
    pred = pred / length(Mdl_ens);
    pred = double(pred > 0.5);
end

