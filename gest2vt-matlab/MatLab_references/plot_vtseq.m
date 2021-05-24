function plot_vtseq(phi,points,contourdata)
%PLOT_VTSEQ Summary of this function goes here
%   Detailed explanation goes here

figure;
params = phi';
s=std(contourdata.weights);

for i=1:length(points)
        subplot(1,length(points),i);
        xy = weights_to_vtshape(params(points(i),:).*s(1:8), contourdata.mean_vtshape,contourdata.U_gfa);
        plot_from_xy(xy,contourdata.SectionsID,'k'); 
        axis([-35 30 -20 30]);
        axis off; 
        hold off;
        text(-30, -15,[num2str(points(i)), ' ms']);
end

shg;

