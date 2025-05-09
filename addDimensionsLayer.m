classdef addDimensionsLayer < nnet.layer.Layer & nnet.layer.Formattable
    % Layer that adds new singleton dimensions to the data with the given
    % label(s).
    
    properties
        NewDimsLabels (1,:) char = ''
    end
    
    methods
        function layer = addDimensionsLayer(addedDimLabels,NameValue)
            arguments
                addedDimLabels (1,:) char
                NameValue.Name = ''
            end
            % The added dimension labels should be one or more of 'S', 'C',
            % 'B', 'T', 'U'. Note that, in total, a dlarray can only have one 'C',
            % 'B', and 'T' dimensions each.
            layer.NewDimsLabels = addedDimLabels;
            layer.Name = NameValue.Name;
        end
        
        function Z = predict(layer,X)
            % Add the new dimension labels to the end of the existing
            % dimension labels.
            outputFormat = [dims(X), layer.NewDimsLabels];
            % Applying the new dimension labels generates trailing
            % singleton dimension for the added labels.
            Z = dlarray(X,outputFormat);
        end
    end
end