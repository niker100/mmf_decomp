function vector = extract(vector)
    if isa(vector, 'gpuArray')
        vector = gather(vector);
    end

    if isa(vector, 'dlarray')
        vector = extractdata(vector);
    end
end