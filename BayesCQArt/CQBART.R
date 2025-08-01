CQBART = function(x,y,K=9,n.sampler=13000,n.burn=3000,thin=20) {
  
  #	x:		the design matrix
  #	y:		the response
  #	n.sample:	the length of the Markov chain
  #	n.burn:	the length of burn-in
  
  theta = (1:K)/(K+1)
  n = dim(x)[1]
  p = dim(x)[2]
  eps = y-x%*%solve(t(x)%*%x)%*%t(x)%*%y
  
  GAMMA_QUANT = (1-2*theta)/(theta*(1-theta))
  TAU_SQ_QUANT = (2/(theta*(1-theta)))



  GAMMA_QUANT <- 1.0  # Example value
  
  # Center and scale the vector
  ysorted <- sort(y)
  ycentered <- y - (ysorted[floor(0.5 * length(y))] )
  yscaled <- -0.5 + (ycentered - min(ycentered)) / (max(ycentered) - min(ycentered))
  
  
  ##The parameters with ".c" are the temporary ones that we use for updating.
  ##The parameters with ".p" are the recorded ones.
  ##Initialization
  
  alpha_D.c = rep(1,K) ## dirichilet prior
  zi.c = apply(rmultinom(n,1,alpha_D.c),2,which.max) ## For each i, index of k s.t C_ik==1 (cluster group of yi)
  xi1.c = GAMMA_QUANT[zi.c]
  xi2.c = TAU_SQ_QUANT[zi.c]
  pi.c = rep(1/K,K)

  nu.c = rep(1,n)
  tau.c = 1
  b.k = quantile(eps,prob=theta)
  b.c = b.k[zi.c]
  
  allfit.c= rep(0, n)
  
  #---	Iteration
  zi.p = matrix(0,n.sampler,n)
  pi.p = matrix(1/K,n.sampler,K) ##omega
  nu.p = matrix(0, n.sampler, n)
  b.p = matrix(0, n.sampler, K)
  tau.p = rep(0, n.sampler)
  
  
  
  for(iter in 1:n.sampler){
    if(iter/1000 == as.integer(iter/1000)) {
      print(paste("This is step ", iter, sep=""))
    }
    
    
    for(j in 1:m){ 
    #---	The full conditional for tz (nu)
    
      for(i in 1:n){
        rgig_chi = xi1.c[i]^2*tau.c/(xi2.c[i])+2*tau.c
        rgig_psi = (yscaled[i] - allfit.c[i])^2 * tau.c/(xi2.c[i])
        nu.c[i]=  do_rgig(0.5,rgig_chi,rgig_psi);
      }
      
      
      allfit.c =	fit(t[j],xi,di,ftemp) # current tree fit
      
      #---	The full conditional for tree
      
      
      
      
      #---	The full conditional for tau
      temp.shape = a/2+3/2*n
      temp.rate = sum(( yscaled - allfit.c -xi1.c*nu.c)^2/(2*xi2.c*nu.c)+nu.c)+b/2
      tau.c = rgamma(1,shape=temp.shape,rate=temp.rate)
      
      
      
      
      #---	The full conditional for zi (c)
      for(i in 1:n){
        temp.power = (yscaled[i] - allfit.c[i] -xi1.c*nu.c[i])^2*tau.c/(xi2*nu.c[i])
        temp.alpha = pi.c*exp(-0.5*temp.power)/sqrt(xi2)
        zi.c[i] = which.max(rmultinom(1,1,temp.alpha/sum(temp.alpha)))
      }
      xi1.c = xi1[zi.c]
      xi2.c = xi2[zi.c]
      
    }
      
    
    #---	The full conditional for pi (omega)
    n.c = rep(0,K)
    for(k in 1:K){
      if(!is.null(which(zi.c==k))){
        n.c[k] = length(which(zi.c==k))
      }
    }
    pi.c = rdirichlet(1,n.c+alpha_D.c)
    
 
    
    tz.p[iter,] = tz.c
    beta.p[iter,] = beta.c
    tau.p[iter] = tau.c
    eta2.p[iter] = eta2.c
    zi.p[iter,] = zi.c
    pi.p[iter,] = pi.c
    b.p[iter,] = b.k
    
    dic.p[iter] = dic.c/n
    
  }
  
}