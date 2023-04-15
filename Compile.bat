g++ -g -std=c++11 -mavx2 -mfma -march=native main_SER_flex.cpp -o SER_O0.exe
g++ -g -std=c++11 -mavx2 -mfma -march=native -O2 main_SER_flex.cpp -o SER_O2.exe

g++ -g -std=c++11 -mavx2 -mfma -march=native main_SSE_ALIGNED_flex.cpp -o SSE_A_O0.exe
g++ -g -std=c++11 -mavx2 -mfma -march=native -O2 main_SSE_ALIGNED_flex.cpp -o SSE_A_O2.exe

g++ -g -std=c++11 -mavx2 -mfma -march=native main_AVX2_ALIGNED_flex.cpp -o AVX2_A_O0.exe
g++ -g -std=c++11 -mavx2 -mfma -march=native -O2 main_AVX2_ALIGNED_flex.cpp -o AVX2_A_O2.exe