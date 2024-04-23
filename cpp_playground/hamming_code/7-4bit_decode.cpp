#include<iostream>
#include<cmath>
using namespace std;

int main() {
    int hamming[7];
    cout << "enter 7-digit Hamming code (even parity): ";
    string input;
    cin >> input;
    for(int i = 0; i < 7; i++) {
        hamming[i] = input[i] - '0';
    }

    int errorBit = 0;
    errorBit += (hamming[6] ^ hamming[4] ^ hamming[2] ^ hamming[0]) * 1;
    errorBit += (hamming[5] ^ hamming[4] ^ hamming[1] ^ hamming[0]) * 2;
    errorBit += (hamming[3] ^ hamming[2] ^ hamming[1] ^ hamming[0]) * 4;

    if(errorBit == 0) {
        cout << "no error" << endl;
    } else {
        cout << "error at " << errorBit << endl;
        hamming[7-errorBit] = !hamming[7-errorBit];
        cout << "corrected: ";
        for(int i = 0; i < 7; i++) {
            cout << hamming[i];
        }
        cout << endl;
    }

    cout << "result: " << hamming[2] << hamming[4] << hamming[5] << hamming[6] << endl;
    return 0;    
}