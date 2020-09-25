#include "Matrix.hpp"
#include <iostream>

Matrix::Matrix() : data(nullptr), row(0), col(0) {}

Matrix::Matrix(int row, int col) : row(row), col(col)
{

    if (row == 0 || col == 0)
        data = nullptr; //空矩阵

    else
    {
        data = new int *[row];
        for (int i = 0; i < row; i++)
        {
            data[i] = new int[col];
        }
    }
}

Matrix::Matrix(const Matrix &A) : row(A.row), col(A.col)
{
    if (row == 0 || col == 0)
        data = nullptr;
    else
    {
        data = new int *[row];
        for (int i = 0; i < row; i++)
        {
            data[i] = new int[col];
        }
        for (int i = 0; i < row; i++)
            for (int j = 0; j < col; j++)
                data[i][j] = A.data[i][j];
    }
}

void Matrix::ProduceMat(const unsigned int &seed)
{
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
        {
            srand(seed + i + j);
            data[i][j] = rand() % 10;
        }
}

void Matrix::copy(const Matrix &B, const Submartrix &b)
{
    for (int i = 0; i < b.size; i++)
        for (int j = 0; j < b.size; j++)
            B.data[b.row + i][b.col + j] = data[i][j];
}

Matrix Matrix::operator+(const Matrix &A)
{
    if (row != A.row || col != A.col)
        return Matrix(0, 0);
    Matrix sum(row, col);
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            sum.data[i][j] = data[i][j] + A.data[i][j];
    return sum;
}

Matrix Matrix::operator-(const Matrix &A)
{
    if (row != A.row || col != A.col)
        return Matrix(0, 0);
    Matrix sum(row, col);
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            sum.data[i][j] = data[i][j] - A.data[i][j];
    return sum;
}

Matrix Matrix::operator*(const Matrix &A)
{
    //两个矩阵不可相乘，返回空矩阵；
    if (col != A.row)
        return Matrix(0, 0);

    Matrix C(row, A.col);
    for (int i = 0; i < row; i++)
        for (int j = 0; j < A.col; j++)
        {
            C.data[i][j] = 0;
            for (int k = 0; k < col; k++)
                C.data[i][j] += data[i][k] * A.data[k][j];
        }
    return C;
}

Matrix &Matrix::operator=(const Matrix &A)
{
    if (row != A.row || col != A.col)
        return *this;

    for (int i = 0; i < A.row; i++)
    {
        for (int j = 0; j < A.col; j++)
            data[i][j] = A.data[i][j];
    }
    return *this;
}

void Matrix::printMatrix()
{
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
            printf("%d ", data[i][j]);
        printf("\n");
    }
}

Matrix::~Matrix()
{
    for (int i = 0; i < row; i++)
    {
        delete[] data[i];
    }
    delete[] data;
}