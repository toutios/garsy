close all; clear
load ../contourdata.mat

tv_a        = [ 5.0 7.1 7.9 4.9 2.4 0.2];
tv_schwa    = [ 2.6 3.1 5.9 5.7 5.6 1.2];
tv_a        = [ 5.0 7.1 7.9 4.9 2.4 0.2];
tv_schwa    = [ 2.6 3.1 5.9 5.7 5.6 1.2];

N=650;

targets     = zeros(N,6);
omegas      = zeros(N,6);

inipars = zeros(8,1);

close all; figure; subplot(2,1,1);

% from Browman and Goldstein (1992), pp. 25

[targets,omegas] =...
    add_gesture(targets,omegas,1,150,2,0.2,0.01); % TT crit alveolar

[targets,omegas] =...
    add_gesture(targets,omegas,51,250,1,-4,0.01); % LIPS clo
    
[targets,omegas] =...
    add_gesture(targets,omegas,1,650,5,tv_a(5),0.01); % TB wide pharyngeal

[targets,omegas] =...
    add_gesture(targets,omegas,451,650,2,-0.1,0.01); % TT clo alveolar

[targets,omegas] =...
    add_gesture(targets,omegas,271,650,6,5,0.01); % VEL wide

% additional controls

[targets,omegas] =...
    add_gesture(targets,omegas,251,451,1,tv_a(1),0.01); % lips open for a

[targets,omegas] =...
    add_gesture(targets,omegas,151,450,2,tv_a(3),0.01); % TT opens

[targets,omegas] =...
    add_gesture(targets,omegas,1,270,6,-0.1,0.01); % Velum closes

finish_score(N);

[phi,t] = simulate_vt(contourdata, targets, omegas, inipars);

plot_vtseq(phi,150:100:650,contourdata);



