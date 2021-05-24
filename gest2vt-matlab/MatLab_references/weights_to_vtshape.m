function xy_data = weights_to_vtshape(weights, mean_vtshape,  U)
% bob = pinv(U(:,1:length(weights)));
xy_data = mean_vtshape + weights*pinv(U(:,1:length(weights)));

% xy_data=mean_vtshape;
% 
% for i=1:length(weights)
% 
%     xy_data = xy_data + weights(i)*pinv(U(:,i));
%     
% end;



