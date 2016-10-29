#include "Softmax.H"
#include "matrixToFile.H"
#include <armadillo>
#include <iostream>
#include <cassert>
#include <stdexcept>

//#define DEBUG_GRADIENTS //comment out when happy

using namespace arma;

Softmax::Softmax(double alpha, int niters, double lambda, int nlabels): mAlpha(alpha), mIters(niters), mLambda(lambda), mLabels(nlabels){ 

#ifdef DEBUG_GRADIENTS

    // create small test set
    
    int N = 10; // no. of samples
    int M = 20; // dimension of samples
    vec XUnrolled = zeros(N*M);
    
    for (int i = 0; i != N*M; ++i)
    {
	XUnrolled(i) = sin(i+1)/10;
    }
    
    mX = reshape(XUnrolled, N, M);
    my = zeros(N);
    
    for (int i = 0; i != N; ++i)
    {
	my(i) = (i+1)%nlabels;
    }

    mParams = randu<mat>(M, mLabels);
    
    checkGradients();
    
#endif
    
}

void Softmax::train(mat& X, vec& y){

    assert(y.n_rows == X.n_rows);
    mX = join_horiz(ones<mat>(X.n_rows), X); // add bias
    my = y;
  
    /* Initialise/reset remaining members. To be filled after training */
    mParams = zeros<mat>(mX.n_cols, mLabels); 
    mGrad = zeros<mat>(mX.n_cols, mLabels); 
    mCostHistory = zeros<mat>(mIters);

    gradientDescent();

}
  
void Softmax::gradientDescent(){

    for (int iter = 0; iter < mIters; ++iter)
    {
	mCostHistory(iter) = cost(mParams); // store cost
	mGrad = grad(); // update gradient 
	mParams = mParams - mAlpha*mGrad; // update params
	if ((iter+1)%100 == 0)
	{
	    std::cout << (iter+1) << " iterations completed" << std::endl;
	}
    }
}

vec Softmax::predict(const mat& X){

    assert(X.n_cols == mX.n_cols - 1);

    uword N = X.n_rows;
    mat X_test = join_horiz(ones<mat>(N), X);

    /* find class confidences */

    mat zs = X_test*mParams; // N by K
    zs = zs.t(); // K by N
    mat activations = exp(zs); // before normalisation of columns
    
    for (uword i = 0; i != N; ++i)
    {
	activations.col(i) = activations.col(i)/accu(activations.col(i)); // normalised

    }
    
    uvec predictions = zeros<uvec>(X_test.n_rows);

    for (size_t i = 0; i != activations.n_cols; ++i)
    {
	/* index of max confidence for each sample is class identifier. */
	/* store this in uword predictions. */

	activations.col(i).max(predictions(i)); 

    }

    return conv_to<vec>::from(predictions); 
}

double Softmax::score(const mat& X, const vec& y){

    size_t n = X.n_rows;
    assert(n == y.n_rows);
    
    vec predictions = predict(X);
    return accu(predictions == y)/(double)n;
    
}


mat Softmax::grad(){

    size_t N = mX.n_rows;

    mat zs = mX*mParams; // N by K
    zs = zs.t(); // K by N
    mat activations = exp(zs); // before normalisation of columns
    
    for (uword i = 0; i != N; ++i)
    {
	activations.col(i) = activations.col(i)/accu(activations.col(i)); // normalised

    }

    umat yFull = yToFull(my);

    
    mat temp = mParams;
    temp.row(0) = zeros<rowvec>(mLabels); // no regularisation for first element of each grad

    // mX is NxM; activations and yFull are KxN
    mat grad = 1/(double)N * mX.t()*(activations - yFull).t() + (mLambda/(double)N)*temp;

    return grad;
}

void Softmax::checkGradients(){

    /* checks gradients against numerical approx. */

    vec theta = vectorise(mParams);
    int len = theta.n_rows;
    mat numgrad = zeros<mat>(len);
    mat perturbed = zeros<mat>(len);
    double delta = 1e-4;

    
    for (int i = 0; i < len; ++i)
    {
	perturbed(i) = delta;
	mat theta1 = (theta - perturbed);
	mat theta2 = (theta + perturbed);

	theta1.set_size(mParams.n_rows, mParams.n_cols);
	theta2.set_size(mParams.n_rows, mParams.n_cols);

	double cost1 = cost(theta1);
	double cost2 = cost(theta2);

	numgrad(i) = (cost2 - cost1)/(2*delta);
	perturbed(i) = 0; // reset for next round
    }

    mat g = grad();
    std::cout << join_horiz(numgrad, vectorise(g)) << std::endl;

}



double Softmax::cost(const mat& theta){

    uword N = mX.n_rows;
    uword M = mX.n_cols;

    assert(theta.n_rows == M);
    
    mat zs = mX*theta; // N by K
    zs = zs.t(); // K by N
    mat activations = exp(zs); // before normalisation of columns
    
    for (uword i = 0; i != N; ++i)
    {
	activations.col(i) = activations.col(i)/accu(activations.col(i)); // normalised

    }

    umat yFull = yToFull(my);
    
    /* note no regularisation for bias. */

    double J = -1/(double)N * sum(sum(log(activations)%yFull)) + mLambda/(2*(double)N) * accu(theta.rows(1, M-1)%theta.rows(1, M-1)); 
       
    return J;
}



umat Softmax::yToFull(const vec& y){

    size_t nSamples = y.n_rows;
    umat yFull = zeros<umat>(mLabels, nSamples);
    for (size_t i = 0; i != nSamples; ++i)
    {
	yFull(y(i), i) = 1;
    }
    return yFull;

}



/* non-members */

mat sigmoid(const mat& z){

    mat g = zeros(size(z));
    g = 1/(1+exp(-z));
    return g;

}

