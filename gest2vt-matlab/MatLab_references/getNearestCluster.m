function indx = getNearestCluster(w,centers)

% bob = sum((centers-(ones(size(centers,1),1)*w(1:8)')).^2,2)
bob = (ones(size(centers,1),1)*w(1:8)')

[~,indx] = min(sqrt(sum((centers-(ones(size(centers,1),1)*w(1:8)')).^2,2)));

end