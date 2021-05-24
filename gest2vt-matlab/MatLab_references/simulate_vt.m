function [phi,t] = simulate_vt(contourdata, targets, omegas, inipars)

[Nz,Nphi] = size(contourdata.jac{1});   % No. task variables z 
                                        % No. model articulators phi

% parameters of flow on task mass Z
omega = zeros(Nz,1);   % natural frequencies of task variables
w = ones(Nphi,1);
W = diag(w);

centers = contourdata.centers;
fwd = contourdata.fwd;
jac = contourdata.jac;
jacDot = contourdata.jacDot;

 zPhiRel = logical(...
   [1 0 0 0 0 1 1 0;...
    0 0 0 0 0 0 0 1;...
    0 0 0 1 0 0 0 0;...
    0 1 1 0 0 0 0 0;...
    1 1 1 1 1 0 0 0;...
    1 1 1 1 1 0 0 0]);

% parameters of the neutral gesture
% (see Saltzman & Munhall, 1989, Appendix A)
omega_N = 10;
B_N = 2*omega_N*eye(Nphi);
K_N = omega_N^2.*eye(Nphi);
G_N = diag(~(zPhiRel'*double(omega~=0)));

% test = double(omega~=0)

N=size(targets,1);

phi=zeros(Nphi,N);
phi(1:8,1) = inipars;
phi(1:8,2) = inipars;

h=0.001;
t = h:h:(N*h);

for i=3:N
    
    indx = getNearestCluster(phi(:,i-1),centers);
    F = fwd{indx};
    J = jac{indx};
    J_t = jacDot{indx};   
    %bob = diag(omega~=0)
    bob = size(F);
    Jstar = jacStar(J,W,diag(omega~=0),Nz);
    %Jstar = pinv(J);
    
    B = diag(2*omegas(i,:));
    K = diag(omegas(i,:).^2);
    
    z0 = targets(i,:)';
    
    x1 = phi(:,i-1);
    x2 = phi(:,i-2);
    
    b = - G_N*K_N - Jstar*K*F(:,2:9);
    a = - Jstar*B*J - Jstar*J_t - B_N + Jstar*J*B_N - G_N*B_N;
    c = Jstar*K*(z0 - F(:,1));
    bob = size(c);
    
    x = inv(eye(8) - a*h - b*h^2)*(- x2 + 2*x1 - a*h*x1 + c*h^2);
    
    %bob = inv(eye(8) - a*h - b*h^2)
    phi(:,i) = x;
    
   % pause()
    
end

labels = {'jaw','tng1','tng2','tng3','tng4','lip1','lip2','vel'};

figure;
for i=1:8
    subplot(8,1,i)   
    plot(t,phi(i,:));  
    legend(labels{i});
end
shg;

