function [targets,omegas] = add_gesture(targets,omegas,startms,endms,constriction, degree, crit)
%ADD_GESTURE Summary of this function goes here
%   Detailed explanation goes here

targets(startms:endms,constriction) = ...
    repmat(degree,length(startms:endms),1);

omegas(startms:endms,constriction) = ...
    repmat(-1000*log(crit)/length(startms:endms),length(startms:endms),1);  

draw_box(startms,endms,constriction,degree,-1000*log(crit)/length(startms:endms));

end

