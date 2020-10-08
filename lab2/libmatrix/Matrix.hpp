#ifndef MATRIX_HPP
#define MATRIX_HPP

#include<cstdlib>
#include<cstdio>

struct Submartrix //方阵
{   //子矩阵首元素在父矩阵中的位置 
        int row;
        int col;
        int size;//矩阵大小
        Submartrix(int row, int col, int size) : row(row), col(col), size(size) {}
};


class Matrix{

public:

    int ** data;        //数据     
    int row;            //行
    int col;            //列

    Matrix();

    Matrix(int row, int col);

    Matrix(const Matrix& A);

// 随机数生成矩阵数值
    void ProduceMat(const unsigned int & seed);
// 拷贝原矩阵的部分（子矩阵b）到矩阵B中
    void copy(const Matrix &B, const Submartrix &b);

    Matrix operator + (const Matrix &A);

    Matrix operator - (const Matrix &A);

    Matrix operator *(const Matrix& A);

    Matrix& operator=(const Matrix& A);
// 打印矩阵
    void printMatrix();
// 资源回收
    ~Matrix();
};

Matrix Strassen(const Matrix &A, const Matrix &B);


#endif