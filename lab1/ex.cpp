#include<iostream>

using namespace std;

//表示矩阵的划分
struct Submartrix
{   //子矩阵首元素在父矩阵中的位置
    int row;
    int col;
    int size;//矩阵大小
    Submartrix(int row, int col, int size) : row(row), col(col), size(size) {}
};

//矩阵数据封装
class  matrix
{
    public:
    int size = 0;//矩阵大小
    int** data;//矩阵数据

    matrix(int size) :size(size)
    {
        data = new int* [size];
        for (int i = 0; i < size; i++) data[i] = new int[size];
        for(int i=0;i<size;i++)
            for(int j=0;j<size;j++) data[i][j] = 1;
    }

    matrix(const matrix& A)
    {
        size=A.size;
        data = new int* [size];
        for (int i = 0; i < size; i++) data[i] = new int[size];
        for(int i=0;i<size;i++)
            for(int j=0;j<size;j++) data[i][j] = A.data[i][j];
    }
    //将矩阵复制到矩阵B的子矩阵区域
    void copy(const matrix &B, const Submartrix &b)
    {
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++) B.data[b.row + i][b.col + j] = data[i][j];
    }

/*     matrix sub_mat(const Submartrix &a)
    {
        matrix B(a.size);
        for(int i=0;i<B.size;i++)
            for( int j=0;j<B.size;j++) 
                B.data[i][j] = data[a.row+i][a.col+j];
        return B;
    } */

    void ProduceMat(const unsigned int & seed) {
        // srand(seed);
        for(int i=0; i<size; i++) 
            for(int j=0; j<size; j++) {
                srand(seed+i+j);
                data[i][j] = rand()%10; 
            }
    }

    //矩阵加法
    matrix operator+(const matrix &B)
    {
        matrix sum(size);
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++) sum.data[i][j] = data[i][j] + B.data[i][j];
        return sum;
    }
    //矩阵减法
    matrix operator-(const matrix& B)
    {
        matrix sum(size);
        for(int i=0;i<size;i++)
            for(int j=0;j<size;j++)
                sum.data[i][j] = data[i][j] - B.data[i][j];
        return sum;
    }
    
    //赋值运算法重载
    matrix& operator=(const matrix& B) {
        for (int i = 0; i < B.size; i++) {
            for (int j = 0; j < B.size; j++) data[i][j] = B.data[i][j];
        }
        return *this;
    }

     ~matrix()
    {
        for(int i=0;i<size;i++) delete data[i];
        delete data;
    } 
};
//////////////////////////////////////
//一般算法
void matrix_mul_normal(int &size)
{
    matrix A(size), B(size), C(size);
    for(int i=0;i<size;i++)
        for(int j=0;j<size;j++)
        {
            C.data[i][j] = 0;
            for(int k=0;k<size;k++)
                C.data[i][j] += A.data[i][j]*B.data[i][j];
        }
    return;
}

//////////////////////////////////////
//分治算法

matrix DivideAndConquer(matrix& A, matrix& B, Submartrix& a, Submartrix& b)
{
    if (a.size <= 64)
    {
        matrix C(a.size);
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
    matrix C11(sub_size), C12(sub_size), C21(sub_size), C22(sub_size);
    C11 = DivideAndConquer(A, B, a11, b11) + DivideAndConquer(A, B, a12, b21);
    C12 = DivideAndConquer(A, B, a11, b12) + DivideAndConquer(A, B, a12, b22);
    C21 = DivideAndConquer(A, B, a21, b11) + DivideAndConquer(A, B, a22, b21);
    C22 = DivideAndConquer(A, B, a21, b12) + DivideAndConquer(A, B, a22, b22);
    matrix C(sub_size * 2);
    C11.copy(C, Submartrix(0, 0, sub_size));
    C12.copy(C, Submartrix(0, sub_size, sub_size));
    C21.copy(C, Submartrix(sub_size, 0, sub_size));
    C22.copy(C, Submartrix(sub_size, sub_size, sub_size));
    return C;
}


////////////////////////////////////
//Strassen算法

matrix sub_sum(const matrix &A, const matrix &B,const  Submartrix &a, const Submartrix &b)
{
    matrix sum(a.size);
    for(int i=0;i<sum.size;i++)
        for(int j=0; j<sum.size;j++) 
            sum.data[i][j]=A.data[a.row+i][a.col+j] + B.data[b.row+i][b.col+j];
    return sum;
}

matrix sub_minus(const matrix &A, const matrix &B,const  Submartrix &a, const Submartrix &b)
{
    matrix sum(a.size);
    for(int i=0;i<sum.size;i++)
        for(int j=0; j<sum.size;j++) 
            sum.data[i][j]=A.data[a.row+i][a.col+j] - B.data[b.row+i][b.col+j];
    return sum;
}

matrix sub_mat(const matrix &A, const Submartrix &a)
{
        matrix B(a.size);
        for(int i=0;i<B.size;i++)
            for( int j=0;j<B.size;j++) 
                B.data[i][j] = A.data[a.row+i][a.col+j];
        return B;
}

matrix strassen(const matrix &A, const matrix &B, const Submartrix &a,const Submartrix &b)
{
    if (a.size <= 64)
    {
        matrix C(a.size);
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
    matrix S1(sub_size), S2(sub_size), S3(sub_size), S4(sub_size), S5(sub_size);
    matrix S6(sub_size), S7(sub_size);
    Submartrix D(0,0,sub_size);
    S1 = strassen(sub_mat(A,a11),sub_minus(B,B,b12,b22),D,D);
    S2 = strassen(sub_sum(A,A,a11,a12),sub_mat(B,b22),D,D);
    S3 = strassen(sub_sum(A,A,a21,a22),sub_mat(B,b11),D,D);
    S4 = strassen(sub_mat(A,a22),sub_minus(B,B,b21,b11),D,D);
    S5 = strassen(sub_sum(A,A,a11,a22),sub_sum(B,B,b11,b22),D,D);
    S6 = strassen(sub_minus(A,A,a12,a22),sub_sum(B,B,b21,b22),D,D);
    S7 = strassen(sub_minus(A,A,a11,a21),sub_sum(B,B,b11,b12),D,D);
    matrix C11(sub_size), C12(sub_size), C21(sub_size), C22(sub_size);
    C11 = S5 + S4 - S2 + S6;
    C12 = S1 + S2;
    C21 = S3 + S4;
    C22 = S5 + S1 - S3 - S7;
    matrix C(sub_size * 2);
    C11.copy(C, Submartrix(0, 0, sub_size));
    C12.copy(C, Submartrix(0, sub_size, sub_size));
    C21.copy(C, Submartrix(sub_size, 0, sub_size));
    C22.copy(C, Submartrix(sub_size, sub_size, sub_size));
    return C;
}


/////////////////////////

void matrix_mul_recursive(const int &size)
{
    matrix A(size), B(size), C(size);
    Submartrix D(0, 0, size);
    C = DivideAndConquer(A, B, D, D);
    return;
}

void matrix_mul_strassen(const int &size)
{
    matrix A(size), B(size), C(size);
    Submartrix D(0, 0, size);
    C = strassen(A, B, D, D);
    return;
}

int main(void)
{
    int size=512;
    cin>>size;
    matrix A(size), B(size), C(size),E(size);
    A.ProduceMat(1);
    B.ProduceMat(2);
    matrix Aa = A,Bb = B;

    Submartrix D(0, 0, size);
    E = strassen(Aa, Bb, D, D);
    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++)
        {
            C.data[i][j] = 0;
            for(int k=0;k<size;k++)
                C.data[i][j] += A.data[i][k]*B.data[k][j];
        }
    }
    printf("strassen:\n");
    for(int i=0;i<10;i++) cout<<E.data[0][i]<<" ";
    cout<<endl;
    printf("normal:\n");
    for(int i=0;i<10;i++) cout<<C.data[0][i]<<" ";
    cout<<endl;
    return 0;
}