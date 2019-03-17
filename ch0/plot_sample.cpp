#include<iostream>
#include"matplotlibcpp.h"
using namespace std;

namespace plt = matplotlibcpp;

int main(){
  cout<<"matplotlib-cpp sample start"<<endl;

  int n = 5000;
  vector<double> x(n), y(n);
  for(int i=0; i<n; ++i) {
    x.at(i) = i;
    y.at(i) = sin(2*M_PI*i/240.0);
  }

  plt::plot(x, y, "--r");
  plt::save("image.png");

  return 0;
}