function cData = cropData(data, n)

nSample = size(data)(1);
initSize = size(data)(2);
finalSize = (sqrt(initSize)-2*n)^2;
cData = zeros(nSample, finalSize);
cropped_input_layer_size = size(cData)(2);
for i=1:nSample
    temp = reshape( data(i,:), sqrt(initSize), sqrt(initSize) );
    cData(i,:) = reshape( temp(n+1:end-n, n+1:end-n), 1, finalSize );
end

end
