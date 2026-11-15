function result = test_panda_mex
% Check the rebuilt MEX on smooth vector problems with an identity prox.
problem = struct('dimension',2,'constraint_type','costum','constraint',@identity_prox);
options = struct('tolerance',1e-8,'buffer_size',10, ...
    'max_iterations',500,'max_stable_iter',80);

panda('init', problem, options);
x = randn(2,1);
[iterations,gamma] = panda('solve',x,@quadratic);
assert(norm(x)<1e-6 && iterations<500 && gamma<1, ...
    'Quadratic MEX regression failed.');
panda('cleanup');

panda('init', problem, options);
x = zeros(2,1);
[quartic_iterations,quartic_gamma] = panda('solve',x,@quartic);
[~,gradient] = quartic(x);
assert(norm(gradient)<1e-6 && quartic_iterations<500, ...
    'Quartic MEX regression failed.');
panda('cleanup');

result = struct('quadratic_iterations',iterations,'quadratic_gamma',gamma, ...
    'quartic_iterations',quartic_iterations,'quartic_gamma',quartic_gamma, ...
    'quartic_solution',x,'quartic_gradient',gradient);
disp('PANDA MEX regression checks passed.');
end

function [f,g] = quadratic(x)
f=2*sum(x.^2); g=4*x;
end

function [f,g] = quartic(x)
f=sum(0.5*x.^2+100*x.^4-x); g=x+400*x.^3-1;
end

function [z,g] = identity_prox(x,gamma)
z=x; g=0;
end
