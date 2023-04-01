function count = count_correl(N,level)

count = 0;
for k=1:level
   count = count+N-k; 
end