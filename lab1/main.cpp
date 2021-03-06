#include<iostream>
#include<cstdlib>
#include<ctime>
#include"Matrix.hpp"

//Strassen算法

Matrix sub_sum(const Matrix &A, const Matrix &B,const  Submartrix &a, const Submartrix &b)
{
    Matrix sum(a.size,a.size);
    for(int i=0;i<sum.row;i++)
        for(int j=0; j<sum.col;j++) 
            sum.data[i][j]=A.data[a.row+i][a.col+j] + B.data[b.row+i][b.col+j];
    return sum;
}

Matrix sub_minus(const Matrix &A, const Matrix &B,const  Submartrix &a, const Submartrix &b)
{
    Matrix sum(a.size,a.size);
    for(int i=0;i<sum.row;i++)
        for(int j=0; j<sum.col;j++) 
            sum.data[i][j]=A.data[a.row+i][a.col+j] - B.data[b.row+i][b.col+j];
    return sum;
}

Matrix sub_mat(const Matrix &A, const Submartrix &a)
{
        Matrix B(a.size,a.size);
        for(int i=0;i<B.row;i++)
            for( int j=0;j<B.col;j++) 
                B.data[i][j] = A.data[a.row+i][a.col+j];
        return B;
}

Matrix strassen(const Matrix &A, const Matrix &B, const Submartrix &a,const Submartrix &b)
{
    // 划分门槛，防止矩阵划分过小
    if (a.size <= 64)
    {
        Matrix C(a.size,a.size);
        for (int i = 0; i < a.size; i++)
        {
            for (int j = 0; j < a.size; j++)
            {
                int sum = 0;
                for (int k = 0; k < a.size; k++)
                {
                    sum += A.data[a.row + i][a.col + k] * B.data[b.row + k][b.col + j];
                }
                C.data[i][j] = sum;
            }
        }
        return C;
    }
    int sub_size = a.size / 2;
    Submartrix a11(a.row, a.col, sub_size);
    Submartrix a12(a.row, a.col + sub_size, sub_size);
    Submartrix a21(a.row + sub_size, a.col, sub_size);
    Submartrix a22(a.row + sub_size, a.col + sub_size, sub_size);
    Submartrix b11(b.row, b.col, sub_size);
    Submartrix b12(b.row, b.col + sub_size, sub_size);
    Submartrix b21(b.row + sub_size, b.col, sub_size);
    Submartrix b22(b.row + sub_size, b.col + sub_size, sub_size);
    Matrix S1(sub_size,sub_size), S2(sub_size,sub_size), S3(sub_size,sub_size), S4(sub_size,sub_size), S5(sub_size,sub_size);
    Matrix S6(sub_size,sub_size), S7(sub_size,sub_size);
    Submartrix D(0,0,sub_size);
    S1 = strassen(sub_mat(A,a11),sub_minus(B,B,b12,b22),D,D);
    S2 = strassen(sub_sum(A,A,a11,a12),sub_mat(B,b22),D,D);
    S3 = strassen(sub_sum(A,A,a21,a22),sub_mat(B,b11),D,D);
    S4 = strassen(sub_mat(A,a22),sub_minus(B,B,b21,b11),D,D);
    S5 = strassen(sub_sum(A,A,a11,a22),sub_sum(B,B,b11,b22),D,D);
    S6 = strassen(sub_minus(A,A,a12,a22),sub_sum(B,B,b21,b22),D,D);
    S7 = strassen(sub_minus(A,A,a11,a21),sub_sum(B,B,b11,b12),D,D);
    Matrix C11(sub_size,sub_size), C12(sub_size,sub_size), C21(sub_size,sub_size), C22(sub_size,sub_size);
    C11 = S5 + S4 - S2 + S6;
    C12 = S1 + S2;
    C21 = S3 + S4;
    C22 = S5 + S1 - S3 - S7;
    Matrix C(sub_size * 2,sub_size * 2);
    C11.copy(C, Submartrix(0, 0, sub_size));
    C12.copy(C, Submartrix(0, sub_size, sub_size));
    C21.copy(C, Submartrix(sub_size, 0, sub_size));
    C22.copy(C, Submartrix(sub_size, sub_size, sub_size));
    return C;
}

Matrix Strassen(const Matrix &A, const Matrix &B){
    Submartrix D(0,0,A.row);
    return strassen(A,B,D,D);
}

int main(void) {
    int m, n, k;
    std::cout<<"input m, n , k >> ";
    std::cin>>m>>n>>k;
    Matrix A(m,n), B(n,k);
    time_t start, end, normal, stra;
    A.ProduceMat(34);
    B.ProduceMat(1);
    start = clock();
    Matrix C = A*B;
    end = clock();
    normal = end - start;
    printf("normal :\n");
    C.printMatrix();
    start = clock();
    Matrix D = Strassen(A,B);
    end = clock();
    stra = end - start;

    printf("strassen :\n");
    D.printMatrix();

    printf("time : %4f ms\n", normal/(float)(CLOCKS_PER_SEC)*1000);
    printf("time : %4f ms\n", stra/(float)(CLOCKS_PER_SEC)*1000);
}




