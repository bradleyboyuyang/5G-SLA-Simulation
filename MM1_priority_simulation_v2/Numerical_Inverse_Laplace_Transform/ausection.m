function x = ausection(f,xl,xr,options)

% Compute intial function values
phi = (3-sqrt(5))/2;

xln = phi*xr + (1-phi)*xl;
xrn = (1-phi)*xr + phi*xl;

fln = f(xln);
frn = f(xrn);

if options.display
    fprintf(1,'\n- - - golden section algorithm; \n- - - [tol = %1.2e/ maxit = %4i]\n',options.tol,options.maxit);
    fprintf(1,'ITER ; X ; F(X) ; |XR-XL|\n');
end

for i = 1:options.maxit
    
    if fln < frn
        xr  = xrn; xrn = xln;
        xln = phi*xr + (1-phi)*xl;
        frn = fln; fln = f(xln);
    else
        xl  = xln; xln = xrn;
        xrn = (1-phi)*xr + phi*xl;
        fln = frn; frn = f(xrn);
    end
    
    if options.display
        fprintf(1,'[%4i] ; %1.8e ; %1.2e ; %1.2e \n',i,(xl+xr)/2,f((xl+xr)/2),abs(xl-xr));
    end
        
    if abs(xl-xr) < options.tol
        x = (xl+xr)/2; 
        return
    end
end

