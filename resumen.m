function [accuracy,f1_score] = resumen(labels,pred, quiet)
%resumen resultados prediccion para clasificador binario
    if nargin == 2
        quiet = 1;
    end
    accuracy = sum(labels==pred) / length(labels);
    C = confusionmat(labels,pred);
    precision = C(2,2) / sum(C(:,2));
    recall = C(2,2) / sum(C(2,:));
    f1_score = 2 * precision * recall / (precision + recall);
    if quiet ~= 1
        fprintf('Accuracy: %f\n', accuracy)
        fprintf('Precision: %f\n', precision)
        fprintf('Recall: %f\n', recall)
        fprintf('F1-score: %f\n', f1_score)
        C
    end
end
