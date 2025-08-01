// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

#include <ctime>
#include "RcppArmadillo.h"
#include "info.h"
#include "tree.h"
#include "functions.h"
#include "rgig.h"
#include "bd.h"
#include <cmath>

using namespace arma;
using namespace Rcpp;


// Dirichlet sampler
arma::vec rdirichlet(int K, arma::vec alpha) {
  arma::vec samples(K);
  for (int k = 0; k < K; ++k) {
    samples(k) = R::rgamma(alpha(k), 1.0);
  }
  return samples / sum(samples);
}
// Multinomial sampler
arma::uvec rmultinom_custom(int n, int size, const arma::vec& prob) {
  arma::uvec result(n);
  Environment stats("package:stats"); // Access the 'stats' namespace
  Function rmultinom = stats["rmultinom"]; // Get the 'rmultinom' function from 'stats'
  for (int i = 0; i < n; ++i) {
    NumericMatrix temp = rmultinom(1, size, Rcpp::wrap(prob)); // Call rmultinom
    NumericVector temp_col = temp(_, 0); // Extract the column as a NumericVector
    result(i) = std::distance(temp_col.begin(), std::max_element(temp_col.begin(), temp_col.end())) + 1; // Find max index and convert to 1-based index
  }
  return result;
}

// Custom function to simulate multinomial distribution and find the index of the maximum value
int rmultinom_custom2(const arma::vec& prob) {
  Environment stats("package:stats"); // Access the 'stats' namespace
  Function rmultinom = stats["rmultinom"]; // Get the 'rmultinom' function from 'stats'
  NumericMatrix temp = rmultinom(1, 1, Rcpp::wrap(prob)); // Call rmultinom with size = 1
  NumericVector temp_col = temp(_, 0); // Extract the column as a NumericVector
  return std::distance(temp_col.begin(), std::max_element(temp_col.begin(), temp_col.end())); // Find max index (0-based)
}



// via the depends attribute we tell Rcpp to create hooks for RcppArmadillo so that the build process will know what to do
// [[Rcpp::depends(RcppArmadillo)]]

// via the exports attribute we tell Rcpp to make this function available from R
// [[Rcpp::export]]
Rcpp::List BayesCQArt_binary(arma::vec const& y,
                     arma::mat const& X,
                     arma::mat const& Xtest,
                     int burn,
                     int nd,
                     int m,
                     int min_obs_node,
                     double aa_parm, //gamma prior
                     double bb_parm,
                     int nc,
                     double pbd,
                     double pb,
                     double alpha,
                     double betap,
                     double kappa,
                     int maxdepth,
                     int K) { // (Add)
  
  RNGScope scope;         // Initialize Random number generator
    
   // arma::ivec distincty = arma::unique(y);
  //  Rcpp::Rcout << "number of unique elements: " << distincty.size() << std::endl;
  //  if(distincty.size() > 2){
    //  Rcpp::stop("For classification  y should be binary.");
    //}
    
  size_t n = y.size();
  size_t p = X.size() / n;
  
  vec theta = linspace(1.0, K, K) / (K + 1.0); //(Added)
 // int n = x.n_rows;
//  int p = x.n_cols;
  vec xi1 = (1 - 2 * theta) / (theta % (1 - theta));
  vec xi2 = (2 / (theta % (1 - theta)));
 // vec eps = y - X * solve(X.t() * X, X.t() * y);
  vec alpha_c = ones(K);
  vec pi_init = alpha_c / sum(alpha_c);
  uvec zi_c = rmultinom_custom(n,1, pi_init); //For each i, index of k s.t C_ik==1 (cluster group of yi)
  vec GAMMA_QUANT = xi1.elem(conv_to<uvec>::from(zi_c - 1));
  vec TAU_SQ_QUANT = xi2.elem(conv_to<uvec>::from(zi_c - 1));
  vec pi_c = ones(K) / K;
  vec pi_ci = pi_c.elem(conv_to<uvec>::from(zi_c - 1));  //For each i, w_k s.t C_ik==1
//  vec b_k = quantile(eps, theta);
//  vec b_c = b_k.elem(conv_to<uvec>::from(zi_c - 1));
  
  
  
  
  Rcpp::Rcout << " number of observations: " << n << std::endl;
  Rcpp::Rcout << " number of covariates: " << p << std::endl;
  
  // scale min and max are -0.5 and 0.5
    arma::vec yscaled(n); // this is actually the latent
    yscaled.fill(0.0); //starting value of zeros

  vec eps = yscaled - X * solve(X.t() * X, X.t() * yscaled);
  vec b_k = quantile(eps, theta);
  vec b_c = b_k.elem(conv_to<uvec>::from(zi_c - 1));
  
  arma::vec v(n); // Storage for latent vector v
  sinfo allys_vs; // Sufficient stats for all of y, use to initialize the trees.
  v.fill(1.0);
  allys_vs.sum_y = arma::sum(yscaled);
  allys_vs.sum_v_inv = arma::sum(1 / v);
  allys_vs.sy2 = arma::sum(yscaled % yscaled);
  allys_vs.n = n;
  allys_vs.sum_y_div_v = arma::sum(yscaled / v);
  

  
  // Data for predictions
  dinfo dip;
  dip.x = Xtest;
  dip.n = 0;
  size_t np = dip.x.size() / p;
  
  if (np > 0) {
    dip.n = np;
    dip.p = p;
  }
  
  xinfo xi;
  makexinfo(p, n, X, xi, nc);
  std::vector<tree> t(m);
  for (size_t j = 0; j < (size_t)m; j++) t[j].setm(0.0);
  
  // Prior and MCMC
  pinfo pi;
  pi.pbd = pbd;
  pi.pb = pb;
  pi.alpha = alpha;
  pi.betap = betap;
  pi.sigma0 = sqrt(3 / (kappa * sqrt((double)m)));
  pi.phi = 1.0;

  
  arma::vec y_current(n);
  y_current.fill(0.0);
  arma::vec latent_current(n);
  latent_current.fill(1.0);
  arma::vec ftemp(n);
  arma::vec allfit(n);
  allfit.fill(0.0);
  
  dinfo di;
  di.n = n;
  di.p = p;
  di.x = X;
  di.y = y_current;
  di.vv = latent_current;
  double rgig_chi, rgig_psi;
  
  // Storage for output
  arma::vec pmean(n);
  pmean.fill(0.0);
    
    arma::vec pmean_temp(n); //posterior mean of in-sample fit, sum draws,then divide
      pmean_temp.fill(0);
    
  arma::vec ppredmean(1);
  ppredmean.fill(0.0);
    
    arma::vec ppredmean_temp(1);
    ppredmean_temp.fill(0.0);
    arma::vec ppredmean_temp2(1);
    ppredmean_temp2.fill(0.0);
    
  arma::vec fpredtemp(1);
  fpredtemp.fill(0.0);
  if (dip.n > 0) {
    ppredmean.resize(dip.n);
    ppredmean.fill(0.00);
    fpredtemp.resize(dip.n);
    fpredtemp.fill(0.00);
      
      ppredmean_temp.resize(dip.n);
      ppredmean_temp.fill(0.00);
      ppredmean_temp2.resize(dip.n);
      ppredmean_temp2.fill(0.00);
  }
  
  double cvptemp = 1.0 / (double)di.p;
  for (size_t i = 0; i < di.p; i++) {
    std::vector<double> cvvtemp;
    for (size_t j = 0; j < di.p; j++) {
      cvvtemp.push_back(cvptemp);
    }
    pi.cvpm.push_back(cvvtemp);
  }
  
  time_t tp;
  int time1 = time(&tp);
  arma::vec restemp;
  restemp.fill(0.0);
  
  std::vector<int> splitvars;
  
  // MCMC
  for (size_t i = 0; i < (nd + burn); i++) {

    
    if(i%1000==0) Rcpp::Rcout << "CQBART mcmc iteration: " << i << std::endl;
    
    
      for(size_t k=0; k<n;k++){
          yscaled(k) = rtrun(allfit(k) +  GAMMA_QUANT(k)*di.vv(k), sqrt(pi.phi*TAU_SQ_QUANT(k)*di.vv(k)),0.0, 1-y(k));
      }
      
    for(size_t j=0; j<(size_t)m; j++){ //for each j th tree?
 
//Rcpp::Rcout << "For each tree : " << j<< std::endl;
      
   
      for(size_t k=0;k<n;k++) { //k is i in the paper
        rgig_chi = (yscaled(k) -  b_c(k) - allfit(k))*(yscaled(k) -  b_c(k) - allfit(k))/(pi.phi*TAU_SQ_QUANT(k)); //detla_1
        rgig_psi = (2 + GAMMA_QUANT(k)*GAMMA_QUANT(k)/TAU_SQ_QUANT(k))/pi.phi; // delta_2
        latent_current(k) = do_rgig(0.5,rgig_chi,rgig_psi);
      }
      
     // Rcpp::Rcout << "yscaled: " << yscaled<< std::endl;
      //Rcpp::Rcout << "b_c+allfit " << b_c+allfit<< std::endl;
     // Rcpp::Rcout << "allfit: " << allfit<< std::endl;
   
      di.vv=latent_current; // v_i
      
    // Rcpp::Rcout << "vi: " << di.vv<< std::endl;
  
      fit(t[j],xi,di,ftemp);//current tree fit
 
      allfit -= ftemp; //allfit = allfit - ftemp;

      
      for(size_t k=0;k<n;k++) { //k is i in the paper
       y_current(k) = yscaled(k)-  b_c(k)   - allfit(k)- GAMMA_QUANT(k)*latent_current(k); //current y is the residual
      }
      //sample latent to integrate out
     
      di.y=y_current;
      
   
      if(unif_rand() > pi.pbd){ //If birth nor prune occur ( pi.pbd : prob that birth or prune occur )
        
        //Rcpp::Rcout << "Birth/Prune No" << pi.pbd<< std::endl;
     
        tree::tree_p tnew;
        tnew=new tree(t[j]);
        
      // calirotp(tnew, t[j], xi, di, pi, GAMMA_QUANT(1), TAU_SQ_QUANT(1), (size_t) min_obs_node);//SWAP move interchanges the splitting rule of a parent and child non- terminal nodes.
      calirotp_new(tnew, t[j], xi, di, pi, GAMMA_QUANT, TAU_SQ_QUANT, (size_t) min_obs_node);
        delete tnew;
      } else { //When birth or prune occur
     
     //Rcpp::Rcout << "Birth/Prune " << pi.pbd<< std::endl;
    
        bd(t[j],xi,di,pi,GAMMA_QUANT,TAU_SQ_QUANT, min_obs_node,splitvars,maxdepth);
      }
      
    //  calichgv_new(t[j], xi, di, pi, GAMMA_QUANT, TAU_SQ_QUANT, minperbot);

      
      //calichgv(t[j], xi, di, pi, GAMMA_QUANT(1), TAU_SQ_QUANT(1), minperbot);  // changes the splitting variable and value
    

      drmu_new(t[j],xi,di,pi, GAMMA_QUANT, TAU_SQ_QUANT);  // mu sample
      fit(t[j],xi,di,ftemp);
      allfit += ftemp;
 
   
    // Full conditional for pi (omega)
    vec n_c(K, fill::zeros);
    
    for(size_t k=0;k<n;k++) {
      ++n_c(zi_c(k) -1 );  // Decrement zi_c(i) by 1 to match the zero-based indexing of n_c
    }
    
    pi_c = rdirichlet(K, n_c + alpha_c);
    
    pi_ci = pi_c.elem(conv_to<uvec>::from(zi_c - 1));
    // Rcpp::Rcout << "length pi_ci " <<   pi_ci.n_elem<< std::endl;
    
    // 2. Sample tau (1/pi.phi)
    
    restemp= yscaled- allfit;
    
    for(size_t k = 0; k < n; k++) {
      
      restemp(k) = yscaled(k)-  b_c(k)  - allfit(k) - GAMMA_QUANT(k) * latent_current(k);
      
   // Rcpp::Rcout << " bc" <<  b_c<< std::endl;
      
      restemp(k) = restemp(k) * restemp(k);  // Squaring the difference
      restemp(k) *= std::pow(2 * TAU_SQ_QUANT(k) * latent_current(k), -1);  // Element-wise multiplication with inverse
    }
    
  // Rcpp::Rcout << " CQ_vi" <<  latent_current<< std::endl;
    
   pi.phi = 1/R::rgamma(n/2+aa_parm/2, (bb_parm/2 + arma::sum(restemp) /2 ) ); // tau^-1 distribution

  //Rcpp::Rcout << "Done for pi_cq" <<   pi.phi<< std::endl;
    
    
  
    }
 
    
 //Rcpp::Rcout << "Done for tree" << n<< std::endl;
   


    // Full conditional for zi (C)

    
    for (size_t k = 0; k < n; k++) {
      arma::vec restemp2 = yscaled(k) -  b_c(k) - allfit(k) - xi1 * latent_current(k); // length K vector
      
      restemp2 = restemp2 % restemp2; // element-wise multiplication
  
 
      restemp2 %= arma::pow(pi.phi * xi2 * latent_current(k), -1); // element-wise division
      

      arma::vec temp_alpha = pi_ci(k) * arma::exp(-0.5 * restemp2) / arma::sqrt(xi2);
  
 // Rcpp::Rcout << "Done for alpha" << temp_alpha<< std::endl;
      arma::vec norm_alpha = temp_alpha / arma::sum(temp_alpha);
      zi_c(k) = rmultinom_custom2(norm_alpha) + 1;
    }
    
  
    GAMMA_QUANT = xi1.elem(conv_to<uvec>::from(zi_c - 1));
    TAU_SQ_QUANT = xi2.elem(conv_to<uvec>::from(zi_c - 1));
   
  // Rcpp::Rcout << "Done for C" << TAU_SQ_QUANT<< std::endl;
   
   
   
   // Full conditional for b (length n)
   
   for (int k = 0; k < K; ++k) {
     uvec which_k = find(zi_c == (k +1 ));

     
     if (!which_k.is_empty()) {

       vec bc =  yscaled(which_k)-   b_c(which_k)- allfit(which_k)- GAMMA_QUANT.elem(which_k) % latent_current(which_k);
        vec sc = 1 / ( TAU_SQ_QUANT.elem(which_k) % latent_current.elem(which_k));
       double mean = sum(bc % sc) / sum(sc);
       double sd = 1 / sqrt(sum(sc));
  
       b_k(k) = R::rnorm(mean, sd);
     }
   }
   b_c = b_k.elem(conv_to<uvec>::from(zi_c - 1));
   
   //Rcpp::Rcout << "Done for intercept " << b_c<< std::endl;
   

    // Store results after burn-in
   // if (i >= burn) {
      //allfit += ftemp;
      //pmean += ftemp;
     // if (dip.n > 0) {
      //  ppredmean += fpredtemp;
    //  }
    // }
    
      if(i >= burn){
          for(size_t k=0; k<n; k++){
              if (allfit(k) >0) {
                  pmean_temp(k) = 1.0;
                  } else {
                  pmean_temp(k) = 0.0;
                  }
              }
          pmean += pmean_temp;
          if(dip.n) {
              ppredmean_temp.fill(0.0);
           for(size_t j=0;j<(size_t)m;j++) {
              fit(t[j],xi,dip,fpredtemp);
              ppredmean_temp += fpredtemp;
            }
            for(size_t k=0; k<dip.n; k++){
                double tmp_rexp = pi.phi*Rcpp::rexp(1)[0];
                double tmp_rnorm = Rcpp::rnorm(1)[0];
                
              if (ppredmean_temp(k) + GAMMA_QUANT(k)*tmp_rexp + sqrt(TAU_SQ_QUANT(k)*tmp_rexp*pi.phi)*tmp_rnorm  >0) {
                  ppredmean_temp2(k) = 1.0;
                  } else {
                  ppredmean_temp2(k) = 0.0;
                  }
              }
          ppredmean += ppredmean_temp2;
            
            
          }
      }
  
  
    
  }
  
  int time2 = time(&tp);
  Rcpp::Rcout << "time for MCMC loop: " << time2-time1 << " seconds" << std::endl;
  pmean /= (double)nd;
  ppredmean /= (double)nd;
  
    
    arma::ivec varsused = arma::conv_to< arma::icolvec >::from(splitvars);
 return Rcpp::List::create(
             Rcpp::Named("pred_train")= pmean,
             Rcpp::Named("pred_test") = ppredmean,
             Rcpp::Named("vars_used") = varsused
             );

  
}

