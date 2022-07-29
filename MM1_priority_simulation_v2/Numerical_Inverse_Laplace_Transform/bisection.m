function [x,gx] = bisection(g,xl,xr,options)

% Compute intial function values
gr  = g(xr); gl  = g(xl); sl = sign(gl);

if gl*gr > 0
    fprintf(1,'The input data not suitable!'); 
    x = []; gx = [];
    return
end

if options.display
    fprintf(1,'\n- - - bisection algorithm; \n- - - [tol = %1.2e/ maxit = %4i]\n',options.tol,options.maxit);
    fprintf(1,'ITER ; X ; |G(X)| ; |XR-XL|\n');
end

for i = 1:options.maxit
    xm  = (xl + xr)/2; 
    gm  = g(xm);
    
    if options.display
        fprintf(1,'[%4i] ; %1.8e ; %1.2e ; %1.2e \n',i,xm,abs(gm),abs(xl-xr));
    end
    
    if abs(xl-xr) < options.tol % || abs(gm) < options.tol 
        x = xm; gx = gm;
        return
    end
    
    if gm > 0
        if sl < 0, xr = xm; else, xl = xm; end
    else
        if sl < 0, xl = xm; else, xr = xm; end
    end
end

