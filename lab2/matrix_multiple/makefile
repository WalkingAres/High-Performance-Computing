
CC = mpic++ -std=c++11

OBJ =p2p.o 

OUT1 = p2p
OUT2 = group

.PHONY : rungroup , runp2p , lib

rungroup:$(OUT2)
	mpirun --mca btl self,tcp -np 8 $< 512 256 512 

$(OUT2) : group.cpp
	$(CC) -o $(OUT2) group.cpp 


runp2p: $(OUT1)
	mpirun --mca btl self,tcp -np 8 $< 1024 1024 1024 

$(OUT) : $(OBJ)
	$(CC) -o $(OUT1)  $(OBJ)

p2p.o :p2p.cpp 
	$(CC) -c $< -o $@


lib : libMatrix.so
libMatrix.so : Matrix.cpp 
	g++ -fPIC -shared Matrix.cpp -o libMatrix.so 


clean:
	rm -f -r *.o $(OUT1) $(OUT2)