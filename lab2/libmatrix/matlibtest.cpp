#include"Matrix.hpp"
#include <iostream>


int main(void) {
    int m , n , k;
    std::cin >> m >> n >> k;
    Matrix A(m,n), B(n,k) , C(m,k), D(m,k);
    A.ProduceMat(1);
    B.ProduceMat(2);

    C = Strassen(A,B);
    D = A*B;

    printf("normal:");
    C.printMatrix();
    printf("Strassen:");
    D.printMatrix();
    
    return 0;
}