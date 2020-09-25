
CC = g++ -std=c++11

OUT = main
OBJ = main.o Matrix.o


$(OUT) : $(OBJ)
	$(CC) -o $(OUT)  $(OBJ)

Matrix.o : Matrix.cpp
	$(CC) -c $< -o $@

main.o : main.cpp 
	$(CC) -c $< -o $@

clean:
	del *.o *.exe