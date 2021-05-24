function Jstar = jacStar(J,W,G_A,Nz)

Jt = J';
C = J/W*Jt;
Im = eye(Nz);
Jstar = (W\Jt)/(C+(Im-G_A));

% Jstar = pinv(J);
%Jstar = (W\J')/(J/W*J');

end