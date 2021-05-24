close all; clear
load ../contourdata.mat

tv_a        = [ 5.0  7.1 7.9 4.9 2.4 0.2];
tv_schwa    = [ 2.6  3.1 5.9 5.7 5.6 1.2];

N=500;

targets     = zeros(N,6);
omegas      = zeros(N,6);

inipars = zeros(8,1);

close all; figure; subplot(2,1,1);

[targets,omegas] =...
    add_gesture(targets,omegas,1,150,1,-1,0.01);

[targets,omegas] =...
    add_gesture(targets,omegas,1,350,2,tv_a(2),0.01);

[targets,omegas] =...
    add_gesture(targets,omegas,1,500,6,-1,0.0001);
    
[targets,omegas] =...
    add_gesture(targets,omegas,1,350,5,tv_a(5),0.01);
    
[targets,omegas] =...
    add_gesture(targets,omegas,151,350,1,tv_a(1),0.01);
    
[targets,omegas] =...
    add_gesture(targets,omegas,351,500,2,-1,0.01);

% additional

finish_score(N);

[phi,t] = simulate_vt(contourdata, targets, omegas, inipars);

plot_vtseq(phi,[50 150 250 350 450],contourdata);



