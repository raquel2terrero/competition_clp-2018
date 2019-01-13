function [accuracy,f1_score] = resumen(labels,pred)
%resumen resultados prediccion para clasificador binario
    accuracy = sum(labels==pred) / length(labels)
    C = confusionmat(labels,pred)
    precision = C(2,2) / sum(C(:,2))
    recall = C(2,2) / sum(C(2,:))
    f1_score = 2 * precision * recall / (precision + recall)
end
