function savestack(filename, stack)

for k = 1 : size(stack, 3)
   imwrite(stack(:, :, k), filename, 'WriteMode', 'append', 'Compress', 'none');
end