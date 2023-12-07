#include <iostream>
#include  "sha256.h"

using namespace std;

int main()
{
    cout << "hello, world" << endl;
    
    char* inputchar = "test";
    int* input = (int*) malloc(sizeof(int));
    int* output = (int*) malloc(8 * sizeof(int));
    hash(input, output, 9);
    return 0;
}
    
